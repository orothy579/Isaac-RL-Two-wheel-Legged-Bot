# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record the state bank of the Robust-Reset stair task (``stair_step/reset_bank.py``, after OmniReset).

Rolls a policy out on every terrain row (protocol of ``record_states_stair_step.py``: curriculum frozen per row,
fixed forward command, deterministic policy, every env's first episode, spawn headings
``linspace(-yaw_spread, yaw_spread, num_envs)``) and keeps up to ``--per_label`` states per row of each kind:

* ``approach``     - 0.2-1.0 s before the first riser-face contact of the episode
* ``face_contact`` - a wheel pressed against a riser face (``zmp.riser_face_contacts``)
* ``one_wheel_up`` - the two wheels stand on treads at least half a riser apart
* ``on_stairs``    - the trailing wheel has climbed a riser or more (and both are level)
* ``pre_failure``  - 0.5-1.5 s before a termination that is not the time-out, still below 0.5 rad of tilt

The first four only where the policy had margin - tilt below ``--max_tilt``, base turning slower than 4 rad/s, and
alive ``--min_future`` s later. The student climbs leaning 0.35-0.5 rad (the limit is 0.6): restored there, the
one-frame glitch of a reset (last action 0, actuator lag drawn anew) tipped it over within 0.1 s.

Per state: the root link pose relative to the tile origin, the root COM velocity (world), every joint position and
velocity, the tile (row, column), the label, and the ``stack_policy`` frames of the ``--history`` control steps up to
it (newest first; the co_rl wrapper restores them as the policy's observation history). ``--merge`` joins banks
recorded from several policies.

    bash scripts/co_rl/h200/run.sh -u scripts/co_rl/build_reset_bank.py \\
        --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-Play-v1-ppo \\
        --checkpoint <teacher model.pt> --source teacher2749 --out logs/co_rl/_reset_bank/teacher2749.npz
    python3 scripts/co_rl/build_reset_bank.py --merge a.npz b.npz --out logs/co_rl/_reset_bank/stair_bank_v1.npz
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description="Record the Robust-Reset state bank.")
parser.add_argument("--task", default="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-Play-v1-ppo")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--source", default="policy", help="tag stored with every state of this run")
parser.add_argument("--levels", type=int, nargs="+", default=list(range(10)))
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--yaw_spread", type=float, default=0.6)
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--command", type=float, default=0.35)
parser.add_argument("--num_policy_stacks", type=int, default=2, help="49 for the blind student")
parser.add_argument("--per_label", type=int, default=300, help="states kept per row and label")
parser.add_argument("--history", type=int, default=50, help="policy frames kept per state (blind student: 50)")
parser.add_argument("--max_tilt", type=float, default=0.4, help="[rad] for every label but pre_failure")
parser.add_argument("--min_future", type=float, default=1.0, help="[s] alive after the state, every label but pre_failure")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--merge", nargs="+", default=None, help="join these banks into --out (no simulation)")
parser.add_argument("--out", required=True)


def merge(paths: list[str], out: str) -> None:
    import numpy as np

    banks = [np.load(p) for p in paths]
    for key in ("joint_names", "labels", "terrain_json"):
        if any(str(b[key]) != str(banks[0][key]) for b in banks):
            raise ValueError(f"banks differ in {key}")
    arrays = {k: np.concatenate([b[k] for b in banks]) for k in
              ("rel_pos", "quat", "vel", "joint_pos", "joint_vel", "history", "row", "col", "label", "source")}
    np.savez_compressed(out, **arrays, joint_names=banks[0]["joint_names"], labels=banks[0]["labels"],
                        terrain_json=banks[0]["terrain_json"], checkpoints=np.concatenate([b["checkpoints"] for b in banks]))
    print(f"[bank] merged {len(arrays['row'])} states from {paths} -> {out}")


if __name__ == "__main__" and "--merge" in sys.argv:
    a = parser.parse_args()
    merge(a.merge, a.out)
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

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
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_step import (  # noqa: E402
    reset_bank,
    zmp_rewards as zmp,
)
from scripts.co_rl.core.runners import OnPolicyRunner  # noqa: E402
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper  # noqa: E402


def pick(cand: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Up to k random (step, env) pairs of a boolean (steps, envs) mask."""
    idx = cand.nonzero()
    if len(idx) > k:
        idx = idx[torch.randperm(len(idx), generator=gen, device="cpu")[:k].to(idx.device)]
    return idx


def main():
    if not args.checkpoint:
        sys.exit("--checkpoint is required to record")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
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
    gen_cfg = terrain.cfg.terrain_generator
    sub = next(iter(gen_cfg.sub_terrains.values()))
    h_lo, h_hi = sub.step_height_range
    rows = gen_cfg.num_rows
    sample_gen = torch.Generator().manual_seed(args.seed)
    J = robot.num_joints
    terms = base.termination_manager.active_terms
    out = {k: [] for k in ("rel_pos", "quat", "vel", "joint_pos", "joint_vel", "history", "row", "col", "label")}
    back = torch.arange(args.history, device=dev)  # history frame i of the state after step k: frame k + 1 - i
    summary = {}
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
        obs, extras = env.get_observations()
        origin = terrain.env_origins.clone()
        state = torch.zeros(steps, n, 13 + 2 * J, device=dev)
        # stack_policy frame 0 = after the reset (the wrapper repeats it as the history before), k + 1 = after step k
        first = extras["observations"]["stack_policy"]
        frames = torch.zeros(steps + 1, n, first.shape[1], device=dev)
        frames[0] = first
        valid = torch.zeros(steps, n, dtype=torch.bool, device=dev)
        tilt = torch.zeros(steps, n, device=dev)
        calm = torch.zeros_like(valid)  # base turning slower than 4 rad/s
        face = torch.zeros_like(valid)
        one_up = torch.zeros_like(valid)
        on_stairs = torch.zeros_like(valid)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        end_step = torch.full((n,), steps, dtype=torch.long, device=dev)
        failed = torch.zeros(n, dtype=torch.bool, device=dev)
        for k in range(steps):
            obs, _, dones, extras = env.step(torch.clamp(policy(obs), -1.0, 1.0))
            frames[k + 1] = extras["observations"]["stack_policy"]
            s = zmp.riser_face_contacts(base)
            done = dones.bool() & alive
            ok = alive & ~done  # a terminated env's data is already its reset state
            state[k] = torch.cat([robot.data.root_link_pos_w - origin, robot.data.root_link_quat_w,
                                  robot.data.root_com_vel_w, robot.data.joint_pos, robot.data.joint_vel], dim=1)
            valid[k] = ok
            tilt[k] = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
            calm[k] = robot.data.root_link_ang_vel_w.norm(dim=1) < 4.0
            face[k] = (s.face_force > 5.0).any(dim=1)
            known = s.support_known.all(dim=1)
            gap = (s.support_z[:, 0] - s.support_z[:, 1]).abs()
            one_up[k] = known & (gap >= 0.5 * riser)
            on_stairs[k] = known & (gap < 0.25 * riser) & (s.max_trailing >= 0.9 * riser)
            end_step = torch.where(done, torch.full_like(end_step, k), end_step)
            failed |= done & ~base.termination_manager.time_outs
            alive &= ~done
            if not alive.any():
                break
        # first face contact of each env -> its approach window
        first = torch.where(face.any(dim=0), face.float().argmax(dim=0), torch.full_like(end_step, -1))
        t = torch.arange(steps, device=dev).unsqueeze(1)
        approach = (first >= 0) & (t >= first - int(1.0 / base.step_dt)) & (t <= first - int(0.2 / base.step_dt))
        pre_fail = (failed & (t >= end_step - int(1.5 / base.step_dt)) & (t <= end_step - int(0.5 / base.step_dt))
                    & (tilt < 0.5))
        # margin: upright enough, not spinning, and still alive min_future s later (a failure, or the end of the run)
        lasts = (~failed & (end_step >= steps - 1)) | (t <= end_step - int(args.min_future / base.step_dt))
        margin = (tilt < args.max_tilt) & calm & lasts
        masks = {"approach": approach & margin, "face_contact": face & margin, "one_wheel_up": one_up & margin,
                 "on_stairs": on_stairs & ~one_up & margin, "pre_failure": pre_fail}
        lv = {}
        for li, name in enumerate(reset_bank.LABELS):
            sel = pick(masks[name] & valid, args.per_label, sample_gen)
            kk, ee = sel[:, 0], sel[:, 1]
            x = state[kk, ee].cpu().numpy()
            out["rel_pos"].append(x[:, 0:3])
            out["quat"].append(x[:, 3:7])
            out["vel"].append(x[:, 7:13])
            out["joint_pos"].append(x[:, 13:13 + J])
            out["joint_vel"].append(x[:, 13 + J:])
            hk = (kk.unsqueeze(1) + 1 - back.unsqueeze(0)).clamp(min=0)  # (len, history)
            out["history"].append(frames[hk, ee.unsqueeze(1)].cpu().numpy())
            out["row"].append(np.full(len(sel), level))
            out["col"].append(terrain.terrain_types[ee].cpu().numpy())
            out["label"].append(np.full(len(sel), li))
            lv[name] = [len(sel), int((masks[name] & valid).sum())]  # kept, candidates
        lv["failed"] = int(failed.sum())
        summary[level] = lv
        print(f"[bank] level {level} ({riser * 100:.1f} cm): kept / candidates {json.dumps(lv)}", flush=True)
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["row"], arrays["col"], arrays["label"] = (arrays[k].astype(np.int64) for k in ("row", "col", "label"))
    np.savez_compressed(
        args.out, **arrays, source=np.array([args.source] * len(arrays["row"])), joint_names=np.array(robot.joint_names),
        labels=np.array(reset_bank.LABELS), terrain_json=json.dumps(reset_bank.terrain_signature(gen_cfg)),
        checkpoints=np.array([args.checkpoint]))
    print(f"[bank] {len(arrays['row'])} states -> {args.out}; terminations seen: {terms}", flush=True)


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
