# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record a video of a stair_step (alternating feet) policy with a camera that follows one robot.

    ./run_gb10.sh scripts/co_rl/record_stair_step.py --checkpoint <model.pt> --level 2 --yaw 0.5 \\
        --seconds 12 --out logs/co_rl/_videos/stair_step.mp4 [--task ...-Perceptive-Play-v1-ppo]

``--pick_best`` (with ``--yaw_spread``): first roll out every env once (deterministic policy, spawn
yaws spread over [-spread, spread]), then roll out again from the same spawn poses and film the env
that climbed the most in the first pass. ``--trace`` also saves that env's time series (npz + png).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Play-v1-ppo")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--yaw", type=float, default=0.5)
parser.add_argument("--yaw_spread", type=float, default=None, help="spawn yaws linspace(-s, s, num_envs)")
parser.add_argument("--pick_best", action="store_true", help="film the env that climbed most in a first pass")
parser.add_argument("--retries", type=int, default=4,
                    help="with --pick_best: GPU physics is not repeatable and success is all-or-nothing, so if the "
                         "filmed env climbs less than half of its first-pass risers, film the next best (this many times)")
parser.add_argument("--retry_until", type=float, default=None,
                    help="with --pick_best: keep filming (cycling through the first-pass envs that climbed >= 1 riser, "
                         "up to --retries + 1 attempts) until the filmed env climbs at least this many risers")
parser.add_argument("--command", type=float, default=0.35)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", default="logs/co_rl/_videos/stair_step.mp4")
parser.add_argument("--trace", default=None, help="save the filmed env's time series to <trace>.npz/.png")
parser.add_argument("--view", choices=("chase", "side"), default="chase",
                    help="chase: behind-left of the robot; side: from -y (the stair profile for a +x climb)")
parser.add_argument("--solo", action="store_true", help="hide the other robots (several envs share a tile)")
parser.add_argument("--num_policy_stacks", type=int, default=2, help="49 for the distilled blind student (50 frames)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import lab.flamingo.tasks  # noqa: E402, F401
from scripts.co_rl.core.runners import OnPolicyRunner  # noqa: E402
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper  # noqa: E402


def spawn(env, yaws: torch.Tensor):
    """Reset every env to the requested level with the given spawn yaws; return the observations."""
    base = env.unwrapped
    t = base.scene.terrain
    robot = base.scene["robot"]
    ids = torch.arange(base.num_envs, device=base.device)
    t.terrain_levels[:] = args.level
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
    base._reset_idx(ids)
    root = robot.data.default_root_state.clone()
    root[:, :3] += t.env_origins
    root[:, 3:7] = math_utils.quat_from_euler_xyz(torch.zeros_like(yaws), torch.zeros_like(yaws), yaws)
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    base.scene.write_data_to_sim()
    base.sim.forward()
    # re-stack the history so no frame of a previous rollout leaks in
    env.policy_state_handler.stack_buffer = None
    env.critic_state_handler.stack_buffer = None
    obs, _ = env.get_observations()
    return obs


def rollout(env, policy, obs, steps: int, follow: int | None = None):
    """Deterministic rollout of every env's first episode. With ``follow``, render frames (camera on
    that env) and record its time series until its episode ends."""
    base = env.unwrapped
    robot = base.scene["robot"]
    shoulder_ids, _ = robot.find_joints(["left_shoulder_joint", "right_shoulder_joint"], preserve_order=True)
    n = base.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=base.device)
    climb = torch.zeros(n, device=base.device)
    ups = torch.zeros(n, device=base.device)
    end_step = torch.full((n,), steps, dtype=torch.long, device=base.device)
    reason = ["time"] * n
    frames, trace = [], []
    origin = base.scene.terrain.env_origins.clone()
    for k in range(steps):
        actions = torch.clamp(policy(obs), -1.0, 1.0)
        obs, _, dones, _ = env.step(actions)
        s = base._zmp_gait_state
        live = alive.float()
        ups += s.step_up.float().sum(dim=1) * live
        climb = torch.maximum(climb, s.max_trailing * live)
        done = dones.bool() & alive
        if follow is not None:
            i = follow
            # robot data of a terminated env is already its reset state: stop the trace before that
            if alive[i] and not done[i]:
                rpy = math_utils.euler_xyz_from_quat(robot.data.root_link_quat_w[i : i + 1])
                trace.append(np.concatenate([
                    [k * base.step_dt],
                    (robot.data.root_link_pos_w[i] - origin[i]).cpu().numpy(),
                    [float(math_utils.wrap_to_pi(r)[0]) for r in rpy],  # roll, pitch, yaw
                    (s.wheel_z[i] - s.cfg.wheel_radius - origin[i, 2]).cpu().numpy(),  # wheel bottoms
                    s.contact[i].float().cpu().numpy(),
                    robot.data.joint_pos[i, shoulder_ids].cpu().numpy(),
                    actions[i].cpu().numpy(),
                    s.step_up[i].float().cpu().numpy(),
                    [float(s.max_trailing[i])],
                    s.zmp[i].cpu().numpy(), s.cop[i].cpu().numpy(), [float(s.margin[i])],
                    s.riser_height[i].cpu().numpy(),
                    s.lifted_step_up[i].float().cpu().numpy(),  # strict detector (leg lift, no edge contact)
                    s.rel_lift[i].cpu().numpy(),
                    s.touching[i].float().cpu().numpy(),
                    [float(robot.data.root_link_lin_vel_b[i, 0])],
                ]))
            frame = base.render()
            if frame is not None:
                frames.append(frame)
        if done.any():
            for j in done.nonzero().flatten().tolist():
                end_step[j] = k
                names = [m for m in base.termination_manager.active_terms if base.termination_manager.get_term(m)[j]]
                reason[j] = "+".join(names) or "?"
        alive &= ~done
        if follow is not None and not alive[follow]:
            break
        if follow is None and not alive.any():
            break
    return climb, ups, end_step, reason, frames, trace


TRACE_COLS = (["t", "x", "y", "z", "roll", "pitch", "yaw", "wheel_bottom_l", "wheel_bottom_r", "contact_l",
               "contact_r", "shoulder_l", "shoulder_r"] + [f"action_{a}" for a in range(4)]
              + ["step_up_l", "step_up_r", "climb", "zmp_x", "zmp_y", "cop_x", "cop_y", "margin", "riser_l",
                 "riser_r", "lifted_l", "lifted_r", "rel_lift_l", "rel_lift_r", "touch_l", "touch_r", "speed_x"])


def save_trace(path: str, trace: list, title: str):
    data = np.asarray(trace)
    np.savez(path + ".npz", data=data, cols=np.asarray(TRACE_COLS))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = {name: data[:, i] for i, name in enumerate(TRACE_COLS)}
    t = c["t"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for side, col in (("l", "tab:blue"), ("r", "tab:red")):
        ax[0].plot(t, c[f"wheel_bottom_{side}"], color=col, label=f"wheel bottom {side.upper()}")
        ax[0].fill_between(t, -0.02, -0.005, where=c[f"contact_{side}"] < 0.5, color=col, alpha=0.3, step="mid")
        ax[1].plot(t, c[f"shoulder_{side}"], color=col, label=f"shoulder {side.upper()}")
        for ts in t[c[f"step_up_{side}"] > 0.5]:
            for a in ax:
                a.axvline(ts, color=col, ls="--", lw=0.8)
    ax[0].plot(t, c["climb"], color="k", lw=1.5, label="climbed (trailing wheel)")
    ax[0].set_ylabel("height above spawn [m]\n(bars: wheel in air)")
    ax[1].set_ylabel("shoulder [rad]")
    ax[2].plot(t, c["x"], label="x"), ax[2].plot(t, c["y"], label="y")
    ax2 = ax[2].twinx()
    ax2.plot(t, c["yaw"], color="tab:green", label="yaw")
    ax2.set_ylabel("yaw [rad]", color="tab:green")
    ax[2].set_ylabel("base xy [m]")
    ax[3].plot(t, c["zmp_y"], label="ZMP y"), ax[3].plot(t, c["cop_y"], label="CoP y")
    ax[3].plot(t, c["margin"], label="lateral margin", color="k", lw=0.8)
    ax[3].set_ylim(-0.3, 0.3)
    ax[3].set_ylabel("heading frame [m]")
    ax[3].set_xlabel("t [s] (dashed: clean step-up, blue = left, red = right)")
    for a in (ax[0], ax[1], ax[2], ax[3]):
        a.legend(loc="upper left", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=100)


def main():
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.command, args.command)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    # the shared velocity command zeroes itself 0.5 s into an episode and keeps 0 until its first resample
    # (5-10 s later); a direct _reset_idx does not restart that clock, so before this line only the first
    # evaluated level of a process ran with a 0 command (and stood in front of the riser)
    cfg.commands.base_velocity.initial_phase_time = 0.0
    cfg.episode_length_s = args.seconds + 2.0
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (args.yaw, args.yaw)}
    # no debug markers (height-scan dots, command arrows/spheres) in the video
    cfg.scene.height_scanner.debug_vis = False
    for term in ("base_velocity", "integral_position"):
        if getattr(cfg.commands, term, None) is not None:
            getattr(cfg.commands, term).debug_vis = False
    # chase camera on one robot
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    if args.view == "side":
        cfg.viewer.eye = (0.25, -1.7, 0.35)
        cfg.viewer.lookat = (0.25, 0.0, 0.12)
    else:
        cfg.viewer.eye = (0.9, -1.3, 0.55)
        cfg.viewer.lookat = (0.15, 0.0, 0.05)
    cfg.viewer.resolution = (1280, 720)
    agent = load_cfg_from_registry(args.task, "co_rl_cfg_entry_point")
    agent.num_policy_stacks = args.num_policy_stacks
    agent.num_critic_stacks = 2
    env = CoRlVecEnvWrapper(gym.make(args.task, cfg=cfg, render_mode="rgb_array"), agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load_transfer(args.checkpoint, widen_inputs=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    steps = int(args.seconds / base.step_dt)
    if args.yaw_spread is not None:
        yaws = torch.linspace(-args.yaw_spread, args.yaw_spread, base.num_envs, device=base.device)
    else:
        yaws = torch.full((base.num_envs,), args.yaw, device=base.device)
    with torch.inference_mode():
        candidates, target = [0], 0.0
        if args.pick_best:
            climb, ups, end_step, reason, _, _ = rollout(env, policy, spawn(env, yaws), steps)
            risers = climb / max(_riser(base), 1e-6)
            for j in range(base.num_envs):
                print(f"[pass1] env {j:3d} yaw {float(yaws[j]):+.2f}: risers {float(risers[j]):4.2f} "
                      f"step-ups {int(ups[j])} end {float(end_step[j]) * base.step_dt:5.2f}s ({reason[j]})")
            order = torch.argsort(climb + 0.01 * ups, descending=True)
            candidates = [int(j) for j in order[: 1 + max(args.retries, 0)]]
            climbers = [int(j) for j in order if float(risers[j]) >= 0.9]
            if args.retry_until is not None and climbers:
                candidates = [climbers[k % len(climbers)] for k in range(1 + max(args.retries, 0))]
            print(f"[pass1] film env {candidates[0]}: risers {float(risers[candidates[0]]):.2f}, "
                  f"step-ups {int(ups[candidates[0]])}; >=1 riser {float((risers >= 0.9).float().mean()):.2f} of envs",
                  flush=True)
        best = None
        for attempt, follow in enumerate(candidates):
            base.viewport_camera_controller.set_view_env_index(follow)
            if args.solo:
                _hide_other_robots(base, follow)
            out = rollout(env, policy, spawn(env, yaws), steps, follow=follow)
            got = float(out[0][follow]) / max(_riser(base), 1e-6)
            if best is None or got > best[0]:
                best = (got, follow, out)
            if not args.pick_best:
                break
            want = float(risers[follow])
            print(f"[film] attempt {attempt}: env {follow} climbed {got:.2f} risers (first pass {want:.2f})", flush=True)
            if (got >= args.retry_until) if args.retry_until is not None else (got >= 0.5 * want):
                break
        risers, follow, (climb, ups, end_step, reason, frames, trace) = best
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=int(round(1.0 / base.step_dt)))
    summary = (f"env {follow} yaw {float(yaws[follow]):+.2f}: risers {risers:.2f}, clean step-ups {int(ups[follow])}, "
               f"episode end {float(end_step[follow]) * base.step_dt:.2f}s ({reason[follow]})")
    print(f"[record] {len(frames)} frames -> {args.out} ({summary})", flush=True)
    if args.trace:
        save_trace(args.trace, trace, f"{os.path.basename(args.checkpoint)} level {args.level}: {summary}")
        print(f"[record] trace -> {args.trace}.npz/.png", flush=True)


def _hide_other_robots(base, keep: int):
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for j in range(base.num_envs):
        if j != keep:
            prim = stage.GetPrimAtPath(f"/World/envs/env_{j}/Robot")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
    # the cloned envs inherit env_0's prims: hiding env_0 hides every robot unless the filmed one
    # has its own (stronger, local) visibility opinion
    prim = stage.GetPrimAtPath(f"/World/envs/env_{keep}/Robot")
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)


def _riser(base) -> float:
    """Nominal riser height [m] of ``args.level`` (same formula as eval_stair_step.py)."""
    rows = base.scene.terrain.cfg.terrain_generator.num_rows
    h_lo, h_hi = next(iter(base.scene.terrain.cfg.terrain_generator.sub_terrains.values())).step_height_range
    return h_lo + min((args.level + 0.5) / rows * rows / (rows - 1), 1.0) * (h_hi - h_lo)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
