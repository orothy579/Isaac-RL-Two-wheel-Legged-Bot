# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record the full state of every robot of a stair-step rollout, to render it off the GPU.

For the H200, whose missing RT cores rule out Isaac's RTX camera: physics only, then
``scripts/co_rl/render_states_mujoco.py`` draws the recorded states. Protocol of ``record_stair_step.py
--yaw_spread`` / ``eval_stair_step.py``: curriculum frozen on each row, fixed forward command, deterministic
policy, every env's first episode, spawn headings ``linspace(-yaw_spread, yaw_spread, num_envs)``.

Saved per row (``<out_dir>/states_L<row>.npz``): per control step and env the root link position (relative to the
env's tile origin), orientation (w, x, y, z), all joint positions and the trailing-wheel climb so far [risers];
per env its tile (row, column), spawn yaw, risers climbed, end term and end step (states after the end are not
valid: a terminated env is already reset).

    bash scripts/co_rl/h200/run.sh -u scripts/co_rl/record_states_stair_step.py \\
        --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-Play-v1-ppo \\
        --checkpoint <model.pt> --levels 0 3 6 9 --num_envs 64 --yaw_spread 0.6 --out_dir logs/co_rl/_videos/<name>
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record stair-step rollout states for off-GPU rendering.")
parser.add_argument("--task", default="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-Play-v1-ppo")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--yaw_spread", type=float, default=0.6)
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--command", type=float, default=0.35)
parser.add_argument("--num_policy_stacks", type=int, default=2, help="49 for the blind student")
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import lab.flamingo.tasks  # noqa: E402, F401
from scripts.co_rl.core.runners import OnPolicyRunner  # noqa: E402
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper  # noqa: E402


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.command, args.command)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.initial_phase_time = 0.0  # see eval_stair_step.py
    cfg.episode_length_s = args.seconds + 1.0
    agent = load_cfg_from_registry(args.task, "co_rl_cfg_entry_point")
    agent.num_policy_stacks = args.num_policy_stacks
    agent.num_critic_stacks = 2
    env = CoRlVecEnvWrapper(gym.make(args.task, cfg=cfg), agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load_transfer(args.checkpoint, widen_inputs=True, std_scale=1.0)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    robot, terrain = base.scene["robot"], base.scene.terrain
    n, dev = base.num_envs, base.device
    ids = torch.arange(n, device=dev)
    steps = int(args.seconds / base.step_dt)
    yaws = torch.linspace(-args.yaw_spread, args.yaw_spread, n, device=dev)
    gen = terrain.cfg.terrain_generator
    sub = next(iter(gen.sub_terrains.values()))
    h_lo, h_hi = sub.step_height_range
    rows = gen.num_rows
    # what the renderer needs to rebuild each env's tile (TerrainGenerator draws one difficulty per tile from seed)
    terrain_json = json.dumps({
        "seed": gen.seed, "num_rows": gen.num_rows, "num_cols": gen.num_cols, "size": list(gen.size),
        "difficulty_range": list(gen.difficulty_range), "step_height_range": list(sub.step_height_range),
        "step_width_range": list(getattr(sub, "step_width_range", (sub.step_width, sub.step_width))),
        "difficulty_scale": getattr(sub, "difficulty_scale", 1.0), "platform_width": sub.platform_width,
        "border_width": sub.border_width})
    terms = base.termination_manager.active_terms
    summary = {"checkpoint": args.checkpoint, "task": args.task, "yaw_spread": args.yaw_spread, "levels": {}}
    for level in args.levels:
        riser = h_lo + min((level + 0.5) / rows * rows / (rows - 1), 1.0) * (h_hi - h_lo)
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        base.episode_length_buf[:] = 0
        base._reset_idx(ids)
        root = robot.data.default_root_state.clone()
        root[:, :3] += terrain.env_origins
        root[:, 3:7] = math_utils.quat_from_euler_xyz(torch.zeros_like(yaws), torch.zeros_like(yaws), yaws)
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
        base.scene.write_data_to_sim()
        base.sim.forward()
        env.policy_state_handler.stack_buffer = None
        env.critic_state_handler.stack_buffer = None
        obs, _ = env.get_observations()
        origin = terrain.env_origins.clone()
        pos = np.zeros((steps + 1, n, 3), np.float32)
        quat = np.zeros((steps + 1, n, 4), np.float32)
        q = np.zeros((steps + 1, n, robot.num_joints), np.float32)
        climb_t = np.zeros((steps + 1, n), np.float32)
        pos[0] = (root[:, :3] - origin).cpu().numpy()
        quat[0] = root[:, 3:7].cpu().numpy()
        q[0] = robot.data.default_joint_pos.cpu().numpy()
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        climb = torch.zeros(n, device=dev)
        end_step = np.full(n, steps, np.int64)
        end = ["time"] * n
        for k in range(steps):
            obs, _, dones, _ = env.step(torch.clamp(policy(obs), -1.0, 1.0))
            s = base._zmp_gait_state
            done = dones.bool() & alive
            # a terminated env's data is already its reset state: keep its last valid state instead
            keep = (alive & ~done).unsqueeze(1)
            pos[k + 1] = torch.where(keep, robot.data.root_link_pos_w - origin, torch.from_numpy(pos[k]).to(dev)).cpu().numpy()
            quat[k + 1] = torch.where(keep, robot.data.root_link_quat_w, torch.from_numpy(quat[k]).to(dev)).cpu().numpy()
            q[k + 1] = torch.where(keep, robot.data.joint_pos, torch.from_numpy(q[k]).to(dev)).cpu().numpy()
            climb = torch.where(alive & ~done, torch.maximum(climb, s.max_trailing), climb)
            climb_t[k + 1] = (climb / riser).cpu().numpy()
            for j in done.nonzero().flatten().tolist():
                end_step[j] = k + 1
                end[j] = "+".join(m for m in terms if base.termination_manager.get_term(m)[j]) or "?"
            alive &= ~done
            if not alive.any():
                break
        risers = (climb / riser).cpu().numpy()
        out = os.path.join(args.out_dir, f"states_L{level}.npz")
        np.savez_compressed(
            out, pos=pos, quat=quat, joint_pos=q, climb=climb_t, joint_names=np.array(robot.joint_names),
            row=terrain.terrain_levels.cpu().numpy(), col=terrain.terrain_types.cpu().numpy(), yaw=yaws.cpu().numpy(),
            risers=risers, end=np.array(end), end_step=end_step, step_dt=base.step_dt, riser_nominal=riser,
            checkpoint=args.checkpoint, terrain_json=terrain_json)
        lv = {"riser_m": round(riser, 4), "climbed_1plus": round(float((risers >= 0.9).mean()), 3),
              "climbed_3plus": round(float((risers >= 2.9).mean()), 3), "risers_mean": round(float(risers.mean()), 2),
              "ends": {e: end.count(e) for e in sorted(set(end))}}
        summary["levels"][str(level)] = lv
        print(f"[states] level {level}: {json.dumps(lv)} -> {out}", flush=True)
    with open(os.path.join(args.out_dir, "states_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    code = 0
    try:
        with torch.inference_mode():
            main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
