# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a policy on the alternating-feet stair task and report gait / ZMP statistics.

Runs the deterministic policy (mean action) with the terrain curriculum frozen at fixed riser
heights and reports per level: risers climbed (trailing wheel), clean single-support step-ups,
alternation ratio, jump (flight) time, single-support time, terminations, plus ZMP checks
(dynamic ZMP vs contact CoP in double support, stability margin). Prints a JSON summary and
optionally writes it to ``--out``.

    ./run_gb10.sh scripts/co_rl/eval_stair_step.py --checkpoint <model.pt> --levels 0 3 6 9 \\
        --num_envs 64 --seconds 20
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the flamingo-light stair_step (alternating feet) task.")
parser.add_argument("--task", default="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Play-v1-ppo")
parser.add_argument("--checkpoint", required=True, help="model_*.pt (stair_step or a stand-drive warm start)")
parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9], help="terrain rows (riser 5..15 cm)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=20.0, help="evaluation time per level [s]")
parser.add_argument("--command", type=float, default=0.35, help="forward velocity command [m/s]")
parser.add_argument("--num_policy_stacks", type=int, default=2)
parser.add_argument("--num_critic_stacks", type=int, default=2)
parser.add_argument("--out", default=None, help="write the JSON summary here")
parser.add_argument("--stochastic", action="store_true", help="sample actions with the learned std (as in training)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import lab.flamingo.tasks  # noqa: E402, F401
from scripts.co_rl.core.runners import OnPolicyRunner  # noqa: E402
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper  # noqa: E402


def main() -> dict:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.curriculum.terrain_levels = None  # freeze the riser height per evaluated level
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command, args_cli.command)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    # the shared velocity command zeroes itself 0.5 s into an episode and keeps 0 until its first resample
    # (5-10 s later); a direct _reset_idx does not restart that clock, so before this line only the first
    # evaluated level of a process ran with a 0 command (and stood in front of the riser)
    env_cfg.commands.base_velocity.initial_phase_time = 0.0
    env_cfg.episode_length_s = args_cli.seconds + 1.0
    agent_cfg = load_cfg_from_registry(args_cli.task, "co_rl_cfg_entry_point")
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # std_scale=1.0: keep the learned per-action std (used by --stochastic)
    runner.load_transfer(args_cli.checkpoint, widen_inputs=True, std_scale=1.0)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    terrain = base.scene.terrain
    steps = int(args_cli.seconds / base.step_dt)
    all_ids = torch.arange(base.num_envs, device=base.device)

    report = {"checkpoint": args_cli.checkpoint, "seconds": args_cli.seconds, "command": args_cli.command,
              "stochastic": args_cli.stochastic, "levels": {}}
    for level in args_cli.levels:
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        base.episode_length_buf[:] = 0
        base._reset_idx(all_ids)
        base.scene.write_data_to_sim()
        base.sim.forward()
        obs, _ = env.get_observations()
        # re-stack the history so no frame of the previous level leaks in
        env.policy_state_handler.stack_buffer = None
        env.critic_state_handler.stack_buffer = None
        obs, _ = env.get_observations()

        alive = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        stats = {k: torch.zeros(base.num_envs, device=base.device) for k in (
            "step_ups", "alt_step_ups", "flight_time", "single_time", "time", "lifted", "lifted_alt", "dist")}
        ups_lr = torch.zeros(base.num_envs, 2, device=base.device)
        lifted_lr = torch.zeros(base.num_envs, 2, device=base.device)
        wheel_best = torch.zeros(base.num_envs, 2, device=base.device)
        zc_err, margins = [], []
        best_climb = torch.zeros(base.num_envs, device=base.device)
        terminated = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        for _ in range(steps):
            if args_cli.stochastic:
                actions = runner.alg.actor_critic.act(obs)
            else:
                actions = torch.clamp(policy(obs), -1.0, 1.0)
            obs, _, dones, _ = env.step(actions)
            s = base._zmp_gait_state
            live = alive.float()
            stats["step_ups"] += s.step_up.float().sum(dim=1) * live
            stats["alt_step_ups"] += s.step_up_alternating.float().sum(dim=1) * live
            stats["flight_time"] += s.flight.float() * base.step_dt * live
            stats["single_time"] += (s.contact.sum(dim=1) == 1).float() * base.step_dt * live
            stats["time"] += base.step_dt * live
            stats["lifted"] += s.lifted_step_up.float().sum(dim=1) * live
            stats["lifted_alt"] += s.lifted_step_up_alternating.float().sum(dim=1) * live
            stats["dist"] += base.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(min=0.0) * base.step_dt * live
            ups_lr += s.step_up.float() * live.unsqueeze(1)
            lifted_lr += s.lifted_step_up.float() * live.unsqueeze(1)
            gain = (s.foot_max_z - s.floor_z.unsqueeze(1)).clamp(min=0.0, max=10.0)
            wheel_best = torch.maximum(wheel_best, gain * live.unsqueeze(1))
            best_climb = torch.maximum(best_climb, s.max_trailing * live)
            double = s.contact.all(dim=1) & alive
            if double.any():
                zc_err.append((s.zmp[double] - s.cop[double]).abs())
            margins.append(s.margin[alive])
            done = dones.bool() & alive
            terminated |= done & ~base.termination_manager.time_outs
            alive &= ~done  # count each env's first episode only

        h_lo, h_hi = next(iter(terrain.cfg.terrain_generator.sub_terrains.values())).step_height_range
        rows = terrain.cfg.terrain_generator.num_rows
        riser = h_lo + min((level + 0.5) / rows * rows / (rows - 1), 1.0) * (h_hi - h_lo)
        ups = stats["step_ups"].sum().item()
        zc = torch.cat(zc_err) if zc_err else torch.zeros(1, 2, device=base.device)
        mg = torch.cat(margins)
        report["levels"][str(level)] = {
            "riser_height_m": round(riser, 4),
            "risers_climbed_mean": round((best_climb / riser).mean().item(), 3),
            "risers_climbed_max": round((best_climb / riser).max().item(), 3),
            "envs_climbed_1plus": round(((best_climb / riser) >= 0.9).float().mean().item(), 3),
            "step_ups_per_env": round(stats["step_ups"].mean().item(), 3),
            "alternating_ratio": round(stats["alt_step_ups"].sum().item() / max(ups, 1.0), 3),
            "step_ups_left_right": [round(v, 3) for v in ups_lr.mean(dim=0).tolist()],
            # strict detector: leg retracted >= lift_required, no contact during the swing, landed on a tread
            "lifted_step_ups_per_env": round(stats["lifted"].mean().item(), 3),
            "lifted_step_ups_left_right": [round(v, 3) for v in lifted_lr.mean(dim=0).tolist()],
            "lifted_alternating_ratio": round(stats["lifted_alt"].sum().item() / max(stats["lifted"].sum().item(), 1.0), 3),
            "lifted_share": round(stats["lifted"].sum().item()
                                  / max(torch.floor(wheel_best / riser + 0.3).sum().item(), 1.0), 3),
            "mean_forward_speed": round((stats["dist"].sum() / stats["time"].sum().clamp(min=1e-6)).item(), 3),
            "flight_frac": round((stats["flight_time"].sum() / stats["time"].sum().clamp(min=1e-6)).item(), 4),
            "single_support_frac": round((stats["single_time"].sum() / stats["time"].sum().clamp(min=1e-6)).item(), 4),
            "terminated_frac": round(terminated.float().mean().item(), 3),
            "zmp_vs_cop_double_support_mean_abs_m": [round(v, 4) for v in zc.mean(dim=0).tolist()],
            "zmp_margin_mean_m": round(mg.mean().item(), 4),
            "zmp_margin_p10_m": round(mg.quantile(0.1).item() if mg.numel() < 16_000_000 else mg.mean().item(), 4),
        }
        print(f"[eval] level {level}: {json.dumps(report['levels'][str(level)])}", flush=True)
    return report


if __name__ == "__main__":
    code = 0
    try:
        # everything (also the per-level resets) in inference mode: env buffers written during
        # env.step() are inference tensors and cannot be modified in place outside of it
        with torch.inference_mode():
            result = main()
        print("[eval] SUMMARY " + json.dumps(result), flush=True)
        if args_cli.out:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump(result, f, indent=2)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Isaac Sim 5.1 may hang in app.close() on this machine
