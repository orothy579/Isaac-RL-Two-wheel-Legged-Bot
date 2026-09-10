"""Deterministic, all-episode benchmark using the checkpoint's saved configs.

Run through runtime.sh and run_bounded.py in a memory-limited service.
No training, transfer-load fallback, success filtering, or reward edits.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--mode', choices=['flat', 'stairs'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=240)
parser.add_argument('--seconds', type=float, default=12.0)
parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import csv
import faulthandler
import hashlib
import json
import math
import os
import pickle
import gymnasium as gym
import torch
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
import lab.flamingo.tasks
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
from scripts.co_rl.core.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True  # match train.py


class SavedConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Isaac Lab moved these configuration classes; preserve saved field values.
        if module == 'isaaclab.actuators.actuator_pd_cfg':
            module = 'isaaclab.actuators.actuator_cfg'
        return super().find_class(module, name)


@torch.no_grad()
def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    checkpoint = args.checkpoint.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    with (checkpoint.parent / 'params/env.pkl').open('rb') as f:
        cfg = SavedConfigUnpickler(f).load()
    with (checkpoint.parent / 'params/agent.pkl').open('rb') as f:
        agent = SavedConfigUnpickler(f).load()
    # Old recorder-only keys became invalid manager terms in newer Isaac Lab.
    # This evaluation writes its own CSV; no demonstrations are being recorded.
    cfg.recorders = RecorderManagerBaseCfg()
    cfg.export_io_descriptors = False
    cfg.io_descriptors_output_dir = None
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    agent.device = args.device
    cfg.seed = args.seeds[0]
    cfg.episode_length_s = args.seconds + 1.0
    # Freeze evaluation terrain and optional reward adaptation at saved settings.
    for name in ('terrain_levels', 'climb_units', 'adaptive_reward', 'roger_threshold', 'weight_schedule'):
        if hasattr(cfg.curriculum, name):
            setattr(cfg.curriculum, name, None)
    command_cfg = cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.initial_phase_time = 0.0
    command_cfg.resampling_time_range = (1e9, 1e9)
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if args.mode == 'stairs':
        # Fixed physical heights and treads, independent of a training curriculum.
        cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=101, size=(10., 10.), border_width=7.5,
            num_rows=1, num_cols=3, curriculum=True, use_cache=False,
            sub_terrains={f'h{h}': terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1/3, step_height_range=(h/100, h/100),
                step_width=w, platform_width=5., border_width=1., holes=False)
                for h, w in [(5, .5), (10, .4), (15, .3)]})
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.commands.stair_event.diagnostics_enabled = False
        cases = [{'height_m':h/100, 'tread_m':w, 'yaw_deg':yaw, 'vx':.7, 'wz':0., 'type':i}
                 for i,(h,w) in enumerate([(5,.5),(10,.4),(15,.3)]) for yaw in [-30,-15,0,15,30]]
        task = 'Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo'
    else:
        cases = [{'name':name, 'vx':vx, 'wz':wz} for name,vx,wz in
                 [('forward',.7,0.), ('backward',-.7,0.), ('left',0.,.5),
                  ('right',0.,-.5), ('stop',0.,0.)]]
        task = 'Isaac-Velocity-Flat-Flamingo-Light-v1-ppo'
    if args.num_envs % len(cases):
        raise ValueError('num-envs must be divisible by the number of cases')
    env = CoRlVecEnvWrapper(gym.make(task, cfg=cfg), agent)
    raw = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)  # strict: never randomize a mismatched layer
    policy = runner.get_inference_policy(device=args.device)
    robot = raw.scene['robot']
    contacts = raw.scene.sensors['contact_forces']
    wheel_ids, _ = contacts.find_bodies(['left_wheel_link', 'right_wheel_link'])
    if len(wheel_ids) != 2:
        raise ValueError('expected two wheel contact bodies')
    near = raw.scene.sensors['base_height_scanner'] if args.mode == 'stairs' else None
    device = raw.device
    groups = torch.arange(args.num_envs, device=device) % len(cases)
    desired = torch.tensor([[c['vx'], 0., c['wz'], 0.] for c in cases], device=device)[groups]
    cmd = raw.command_manager.get_term('base_velocity')
    if args.mode == 'stairs':
        terrain = raw.scene.terrain
        terrain.terrain_levels.zero_()
        terrain.terrain_types[:] = torch.tensor([c['type'] for c in cases],device=device)[groups]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
    rows = []
    for seed in args.seeds:
        print(f'[BENCH] resetting seed={seed}', flush=True)
        env.seed(seed)
        env.reset()
        print(f'[BENCH] reset complete seed={seed}', flush=True)
        if args.mode == 'stairs':
            yaw = torch.tensor([math.radians(c['yaw_deg']) for c in cases],device=device)[groups]
            pose = robot.data.root_link_pose_w.clone()
            pose[:,3:7] = quat_from_euler_xyz(torch.zeros_like(yaw),torch.zeros_like(yaw),yaw)
            robot.write_root_pose_to_sim(pose)
            raw.sim.forward()
            raw.scene.update(raw.step_dt)
        cmd.vel_command_b[:] = desired
        cmd.time_elapsed.fill_(1.)
        cmd.is_standing_env.zero_()
        env.policy_state_handler.stack_buffer = None
        env.critic_state_handler.stack_buffer = None
        obs, _ = env.get_observations()
        alive = torch.ones(args.num_envs,dtype=torch.bool,device=device)
        failed = torch.zeros_like(alive)
        max_gain = torch.zeros(args.num_envs,device=device)
        stable_time = torch.zeros_like(max_gain)
        success = torch.zeros_like(alive)
        vx_error = torch.zeros_like(max_gain)
        wz_error = torch.zeros_like(max_gain)
        samples = torch.zeros_like(max_gain)
        start_ground = (torch.nanmedian(near.data.ray_hits_w[:,:,2],dim=1).values.clone()
                        if near is not None else torch.zeros_like(max_gain))
        peak_torque = torch.zeros_like(max_gain)
        peak_speed = torch.zeros_like(max_gain)
        for step in range(round(args.seconds/raw.step_dt)):
            if step % 100 == 0:
                print(f'[BENCH] seed={seed} step={step}', flush=True)
                faulthandler.dump_traceback_later(120, repeat=True)
            # Each env contributes exactly its first episode, including failures.
            # Post-reset observations/actions from a failed env are never scored.
            with torch.no_grad():
                actions = policy(obs)
                if runner.action_clip is not None:
                    actions = actions.clamp(-runner.action_clip, runner.action_clip)
                obs, _, done, _ = env.step(actions)
            terminated = done.bool() & alive
            failed |= terminated
            alive &= ~done.bool()
            cmd.vel_command_b[:] = desired
            cmd.is_standing_env.zero_()
            cmd.time_elapsed.fill_(1.)
            peak_torque = torch.maximum(peak_torque,torch.where(alive,robot.data.applied_torque.abs().amax(-1),0.))
            peak_speed = torch.maximum(peak_speed,torch.where(alive,robot.data.joint_vel.abs().amax(-1),0.))
            if step * raw.step_dt >= 1.0:
                vx_error += torch.where(alive,(robot.data.root_link_lin_vel_b[:,0]-desired[:,0]).abs(),0.)
                wz_error += torch.where(alive,(robot.data.root_link_ang_vel_b[:,2]-desired[:,2]).abs(),0.)
                samples += alive
            if args.mode == 'stairs':
                ground = torch.nanmedian(near.data.ray_hits_w[:,:,2],dim=1).values
                gain = ground-start_ground
                max_gain = torch.maximum(max_gain,torch.where(alive & torch.isfinite(gain),gain,0.))
                roll,pitch,_ = euler_xyz_from_quat(robot.data.root_link_quat_w)
                both = (contacts.data.current_contact_time[:,wheel_ids] > 0).all(-1)
                heights = torch.tensor([c['height_m'] for c in cases],device=device)[groups]
                stable = alive & both & (gain >= .85*heights) & (roll.abs()<.35) & (pitch.abs()<.35)
                stable_time = torch.where(stable,stable_time+raw.step_dt,0.)
                success |= stable_time >= .5
        for i in range(args.num_envs):
            count = int(samples[i])
            rows.append(dict(seed=seed, env=i, case=int(groups[i]), failed=bool(failed[i]),
                success=bool(success[i]), max_ground_gain_m=float(max_gain[i]),
                vx_mae_mps=float(vx_error[i]/samples[i]) if count else None,
                wz_mae_radps=float(wz_error[i]/samples[i]) if count else None,
                scored_samples=count, peak_torque_nm=float(peak_torque[i]),peak_joint_speed_radps=float(peak_speed[i])))
    with (args.output/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for index, case in enumerate(cases):
        selected=[r for r in rows if r['case']==index]
        def avg(key):
            values=[r[key] for r in selected if r[key] is not None]
            return sum(values)/len(values) if values else None
        summary.append(dict(case,episodes=len(selected),failure_rate=avg('failed'),
            success_rate=avg('success') if args.mode=='stairs' else None,
            mean_max_ground_gain_m=avg('max_ground_gain_m') if args.mode=='stairs' else None,
            vx_mae_mps=avg('vx_mae_mps'),wz_mae_radps=avg('wz_mae_radps'),
            peak_torque_nm=max(r['peak_torque_nm'] for r in selected),
            peak_joint_speed_radps=max(r['peak_joint_speed_radps'] for r in selected)))
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'strict_load':True,'deterministic_policy':True,'mode':args.mode,'seconds':args.seconds,
            'seeds':args.seeds,'cases':summary,'reward_edits':False,
            'notes':['Saved observation, action, noise and dynamics configs retained; current repository implementation.',
                     'Compatibility: actuator_pd_cfg relocated to actuator_cfg; recorder disabled with current empty config.',
                     'Stairs: exact fixed 5/10/15 cm risers with 50/40/30 cm treads; yaw -30/-15/0/15/30 deg.',
                     'Success: >=85% riser gain with both wheel contact and |roll,pitch|<0.35 rad for 0.5 s.',
                     'Success is supported climb, not proof of airborne jump or real-world transfer.',
                     'Velocity errors exclude first second and post-termination data; read with failure rate.']}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(result,indent=2),flush=True)
    sys.stderr.flush()
    # Dedicated batch process: avoid known native Kit teardown hang after flushing.
    os._exit(0)


if __name__ == '__main__':
    main()
