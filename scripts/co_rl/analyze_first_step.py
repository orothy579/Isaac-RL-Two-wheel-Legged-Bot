# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Which wheel goes up the FIRST riser, and does the climb then succeed? (stair-step tasks)

Runs the deterministic policy on fixed terrain rows (first episode of every env) and records:

* ``lead``  - side whose wheel the riser blocks first (``blocked`` of the gait state; yaw < 0
  puts the left wheel ahead), and the yaw at that moment
* ``first`` - wheel that first stands on the first tread (support >= 0.7 riser above the floor)
* ``ok``    - the trailing wheel climbed >= 0.9 riser
* ``end``   - termination term (or ``time``)

It found that the sweep-v3 policy started every staircase with the right wheel (docs section 8).

    ./run_gb10.sh -u scripts/co_rl/analyze_first_step.py --task <...-Play-v1-ppo> --checkpoint <model.pt> \\
        --levels 0 3 --num_envs 256 --out first_step.json
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="First-riser wheel order and outcome of a stair-step policy.")
parser.add_argument("--task", default="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Perceptive-Play-v1-ppo")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--levels", type=int, nargs="+", default=[0, 3])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--command", type=float, default=0.35, help="forward velocity command [m/s]")
parser.add_argument("--num_policy_stacks", type=int, default=2)
parser.add_argument("--out", default=None, help="write the per-env JSON here")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import lab.flamingo.tasks  # noqa: E402, F401
from scripts.co_rl.core.runners import OnPolicyRunner  # noqa: E402
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper  # noqa: E402

SIDES = ["L", "R", "both", "none"]


def main() -> dict:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.curriculum.terrain_levels = None  # freeze the riser height per evaluated row
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.command, args.command)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.episode_length_s = args.seconds + 1.0
    agent = load_cfg_from_registry(args.task, "co_rl_cfg_entry_point")
    agent.num_policy_stacks = args.num_policy_stacks
    agent.num_critic_stacks = 2
    env = CoRlVecEnvWrapper(gym.make(args.task, cfg=cfg), agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load_transfer(args.checkpoint, widen_inputs=True, std_scale=1.0)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    terrain = base.scene.terrain
    n, dev = base.num_envs, base.device
    steps = int(args.seconds / base.step_dt)
    ids = torch.arange(n, device=dev)
    h_lo, h_hi = next(iter(terrain.cfg.terrain_generator.sub_terrains.values())).step_height_range
    rows = terrain.cfg.terrain_generator.num_rows
    terms = base.termination_manager.active_terms
    report = {"checkpoint": args.checkpoint, "task": args.task, "levels": {}}

    with torch.inference_mode():
        for level in args.levels:
            riser = h_lo + min((level + 0.5) / rows * rows / (rows - 1), 1.0) * (h_hi - h_lo)
            terrain.terrain_levels[:] = level
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            base.episode_length_buf[:] = 0
            base._reset_idx(ids)
            base.scene.write_data_to_sim()
            base.sim.forward()
            env.policy_state_handler.stack_buffer = None
            env.critic_state_handler.stack_buffer = None
            obs, _ = env.get_observations()

            alive = torch.ones(n, dtype=torch.bool, device=dev)
            lead = torch.full((n,), 3, dtype=torch.long, device=dev)
            lead_yaw = torch.zeros(n, device=dev)
            first = torch.full((n,), 3, dtype=torch.long, device=dev)
            best = torch.zeros(n, device=dev)
            end_reason = torch.full((n,), -1, dtype=torch.long, device=dev)
            for _ in range(steps):
                obs, _, dones, _ = env.step(torch.clamp(policy(obs), -1.0, 1.0))
                s = base._zmp_gait_state
                q = base.scene["robot"].data.root_link_quat_w
                yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                blocked = s.blocked
                new_lead = (lead == 3) & blocked.any(dim=1) & alive
                side = torch.where(blocked.all(dim=1), 2, torch.where(blocked[:, 1], 1, 0))
                lead = torch.where(new_lead, side, lead)
                lead_yaw = torch.where(new_lead, yaw, lead_yaw)
                up = (s.foot_max_z - s.floor_z.unsqueeze(1)) >= 0.7 * riser
                new_first = (first == 3) & up.any(dim=1) & alive
                first = torch.where(new_first, torch.where(up.all(dim=1), 2, torch.where(up[:, 1], 1, 0)), first)
                best = torch.where(alive, torch.maximum(best, s.max_trailing), best)
                done = dones.bool() & alive
                if done.any():
                    for k, name in enumerate(terms):
                        hit = done & base.termination_manager.get_term(name).bool() & (end_reason < 0)
                        end_reason = torch.where(hit, k, end_reason)
                alive &= ~done
            ok = (best / riser) >= 0.9

            envs = [{"lead": SIDES[int(lead[j])], "lead_yaw": round(float(lead_yaw[j]), 3),
                     "first": SIDES[int(first[j])], "ok": bool(ok[j]), "risers": round(float(best[j] / riser), 2),
                     "end": terms[int(end_reason[j])] if end_reason[j] >= 0 else "time"} for j in range(n)]
            summary = {}
            for key in ("lead", "first"):
                for side in SIDES:
                    sel = [e for e in envs if e[key] == side]
                    if sel:
                        hits = sum(e["ok"] for e in sel)
                        summary[f"{key}={side}"] = {"n": len(sel), "ok": hits, "ok_rate": round(hits / len(sel), 3)}
            report["levels"][str(level)] = {"riser_m": round(riser, 4), "summary": summary, "envs": envs}
            print(f"[first_step] level {level} ({riser * 100:.1f} cm): climbed {int(ok.sum())}/{n}", flush=True)
            for key, v in summary.items():
                print(f"[first_step]   {key:12s} n={v['n']:4d} climbed={v['ok']:4d} ({v['ok_rate'] * 100:4.1f} %)", flush=True)
            for side in ("L", "R"):
                ys = [e["lead_yaw"] for e in envs if e["lead"] == side]
                if ys:
                    print(f"[first_step]   lead {side}: yaw at first block mean {sum(ys) / len(ys):+.2f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    main()
    os._exit(0)  # Isaac Sim 5.1 can hang in app.close()
