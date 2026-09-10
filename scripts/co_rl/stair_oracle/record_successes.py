#!/usr/bin/env python3
"""Replay a stair policy and retain only oracle-labelled successful 10 s clips."""

from __future__ import annotations

import argparse
import os
import sys

# Make the repository's namespace packages importable even when this nested
# script is launched by absolute path or from another working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

import scripts.co_rl.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Registered stair-jump task name.")
parser.add_argument("--num-clips", type=int, default=5, help="Number of successful clips to retain.")
parser.add_argument("--terrain-level", type=int, default=0, help="Fixed curriculum level to replay.")
parser.add_argument("--terrain-type", type=int, default=5, help="Fixed terrain column/type to replay.")
parser.add_argument("--clip-seconds", type=float, default=10.0, help="Saved clip duration.")
parser.add_argument("--pre-seconds", type=float, default=4.0, help="Maximum approach context before trigger.")
parser.add_argument("--fps", type=int, default=20, help="Saved video frame rate.")
parser.add_argument("--search-seconds", type=float, default=600.0, help="Maximum simulated search time.")
parser.add_argument("--command-vx", type=float, default=0.7, help="Fixed forward command during replay [m/s].")
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use mean actions. By default the checkpoint's learned exploration std is retained.",
)
parser.add_argument("--resolution", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
parser.add_argument("--output-root", default="scripts/co_rl/stair_oracle", help="Data/video archive root.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_policy_stacks", type=int, default=2)
parser.add_argument("--num_critic_stacks", type=int, default=2)
cli_args.add_co_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint must be an explicit model_*.pt path")
if args_cli.num_clips < 1 or args_cli.clip_seconds <= 0 or args_cli.fps < 1 or args_cli.search_seconds <= 0:
    parser.error("--num-clips, --clip-seconds, --fps, and --search-seconds must be positive")
if not 0.0 <= args_cli.pre_seconds < args_cli.clip_seconds:
    parser.error("--pre-seconds must be in [0, clip-seconds)")

# RGB viewport rendering is required even for a headless run.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below runs after Isaac Sim starts."""

import json
import math
from collections import deque
from datetime import datetime

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import lab.flamingo.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from scripts.co_rl.core.runners import OnPolicyRunner
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg, CoRlVecEnvWrapper


def _scalar(value: torch.Tensor) -> float:
    return float(value[0].detach().cpu())


def _attempt_start_metadata(diagnostics, trigger_time_s: float, checkpoint: str) -> dict:
    return {
        "checkpoint": checkpoint,
        "trigger_time_s": trigger_time_s,
        "attempt_id": int(_scalar(diagnostics.attempt_id)),
        "terrain_level": int(_scalar(diagnostics.start_terrain_level)),
        "terrain_type": int(_scalar(diagnostics.start_terrain_type)),
        "terrain_step_height_m": _scalar(diagnostics.start_terrain_step_height),
        "terrain_step_width_m": _scalar(diagnostics.start_terrain_step_width),
        "success_climb_threshold_m": _scalar(diagnostics.start_success_climb_threshold),
        "oracle_distance_along_heading_m": _scalar(diagnostics.start_distance_along),
        "oracle_distance_to_plane_m": _scalar(diagnostics.start_distance_normal),
        "oracle_alignment_error_rad": _scalar(diagnostics.start_alignment),
        "oracle_alignment_error_deg": math.degrees(_scalar(diagnostics.start_alignment)),
        "start_vx_b_mps": _scalar(diagnostics.start_base_velocity[:, 0]),
        "start_vy_b_mps": _scalar(diagnostics.start_base_velocity[:, 1]),
        "start_vz_w_mps": _scalar(diagnostics.start_base_velocity[:, 2]),
        "command_vx_mps": _scalar(diagnostics.start_command[:, 0]),
        "command_vy_mps": _scalar(diagnostics.start_command[:, 1]),
        "command_yaw_rate_radps": _scalar(diagnostics.start_command[:, 2]),
    }


def _write_clip(path: str, frames: list[np.ndarray], fps: int):
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main():
    checkpoint = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    repo_root = _REPO_ROOT
    output_root = args_cli.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(repo_root, output_root)
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_name = f"video_eval_{run_stamp}_level{args_cli.terrain_level}_type{args_cli.terrain_type}"
    data_dir = os.path.join(output_root, "data", eval_name)
    video_dir = os.path.join(output_root, "videos", eval_name)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.seed = args_cli.seed
    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks
    if agent_cfg.to_dict()["algorithm"]["class_name"] != "PPO":
        raise ValueError("record_successes.py currently supports PPO checkpoints only")

    # Evaluation must stay on one requested curriculum level. This changes replay only,
    # never training or the checkpoint being evaluated.
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.climb_units = None
    env_cfg.viewer.resolution = tuple(args_cli.resolution)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-3.2, -4.0, 2.0)
    env_cfg.viewer.lookat = (0.8, 0.0, 0.25)
    stair_cfg = env_cfg.commands.stair_event
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_vx, args_cli.command_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    stair_cfg.diagnostics_enabled = True
    stair_cfg.diagnostics_max_envs = 1
    stair_cfg.diagnostics_output_dir = data_dir
    stair_cfg.diagnostics_training_log_dir = os.path.dirname(checkpoint)

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = gym_env.unwrapped
    terrain = raw_env.scene.terrain
    max_level = terrain.terrain_origins.shape[0] - 1
    max_type = terrain.terrain_origins.shape[1] - 1
    if not 0 <= args_cli.terrain_level <= max_level:
        gym_env.close()
        raise ValueError(f"terrain level must be in [0, {max_level}], got {args_cli.terrain_level}")
    if not 0 <= args_cli.terrain_type <= max_type:
        gym_env.close()
        raise ValueError(f"terrain type must be in [0, {max_type}], got {args_cli.terrain_type}")
    terrain.terrain_levels.fill_(args_cli.terrain_level)
    terrain.terrain_types.fill_(args_cli.terrain_type)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    env = CoRlVecEnvWrapper(gym_env, agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        runner.load(checkpoint)
    except RuntimeError as error:
        print(f"[VIDEO] strict load failed ({error}); using shape-compatible transfer load")
        runner.load_transfer(checkpoint)
    if args_cli.deterministic:
        policy = runner.get_inference_policy(device=raw_env.device)
    else:
        runner.eval_mode()
        actor_critic = runner.alg.actor_critic.to(raw_env.device)
        if agent_cfg.empirical_normalization:
            runner.obs_normalizer.to(raw_env.device)
            policy = lambda value: actor_critic.act(runner.obs_normalizer(value))
        else:
            policy = actor_critic.act
    obs, _ = env.get_observations()

    stair_term = raw_env.command_manager.get_term("stair_event")
    diagnostics = stair_term.diagnostics
    desired_frames = max(1, round(args_cli.clip_seconds * args_cli.fps))
    pre_frames = round(args_cli.pre_seconds * args_cli.fps)
    ring: deque[np.ndarray] = deque(maxlen=pre_frames)
    active_frames: list[np.ndarray] | None = None
    active_metadata: dict | None = None
    active_success = False
    capture_phase = 1.0
    step_dt = raw_env.step_dt
    max_steps = math.ceil(args_cli.search_seconds / step_dt)
    saved = 0
    triggers = 0

    print(
        f"[VIDEO] searching level {args_cli.terrain_level}, type {args_cli.terrain_type} "
        f"for {args_cli.num_clips} successes"
    )
    print(f"[VIDEO] data: {data_dir}")
    print(f"[VIDEO] clips: {video_dir}")
    for step in range(max_steps):
        if not simulation_app.is_running() or saved >= args_cli.num_clips:
            break
        with torch.inference_mode():
            actions = torch.clamp(policy(obs), -1.0, 1.0)
            obs, _, _, _ = env.step(actions)

        sim_time_s = (step + 1) * step_dt
        started = bool(_scalar(stair_term._last_attempt_started) > 0.5)
        succeeded = bool(_scalar(diagnostics.success_pulse) > 0.5)
        if started and active_frames is None:
            triggers += 1
            active_frames = list(ring)
            active_metadata = _attempt_start_metadata(diagnostics, sim_time_s, checkpoint)
            active_success = False

        if active_frames is not None and succeeded:
            active_success = True
            active_metadata.update(
                {
                    "oracle_success": True,
                    "success_observed_time_s": sim_time_s,
                    "max_ground_gain_m": _scalar(diagnostics.max_ground_gain),
                    "max_base_vz_w_mps": _scalar(diagnostics.max_base_vz),
                    "final_roll_rad": _scalar(diagnostics.final_roll),
                    "final_pitch_rad": _scalar(diagnostics.final_pitch),
                    "final_both_wheels_contact": bool(_scalar(diagnostics.final_both_contact)),
                }
            )

        capture_phase += args_cli.fps * step_dt
        if capture_phase >= 1.0:
            capture_phase -= 1.0
            frame = np.asarray(gym_env.render(), dtype=np.uint8)
            if active_frames is None:
                ring.append(frame.copy())
            else:
                active_frames.append(frame.copy())

        if active_frames is not None and len(active_frames) >= desired_frames:
            active_frames = active_frames[-desired_frames:]
            if active_success:
                saved += 1
                height_mm = round(active_metadata["terrain_step_height_m"] * 1000)
                width_mm = round(active_metadata["terrain_step_width_m"] * 1000)
                stem = f"success_{saved:03d}_attempt_{active_metadata['attempt_id']}_h{height_mm}mm_w{width_mm}mm"
                video_path = os.path.join(video_dir, f"{stem}.mp4")
                metadata_path = os.path.join(video_dir, f"{stem}.json")
                active_metadata["video_duration_s"] = desired_frames / args_cli.fps
                active_metadata["video_fps"] = args_cli.fps
                active_metadata["video_frames"] = desired_frames
                _write_clip(video_path, active_frames, args_cli.fps)
                with open(metadata_path, "w", encoding="utf-8") as file:
                    json.dump(active_metadata, file, indent=2)
                print(f"[VIDEO] saved {saved}/{args_cli.num_clips}: {video_path}")
            active_frames = None
            active_metadata = None
            active_success = False
            ring.clear()

    diagnostics.close()
    env.close()
    print(f"[VIDEO] finished: saved={saved}, triggers_examined={triggers}, simulated={min(max_steps, step + 1) * step_dt:.1f}s")
    if saved < args_cli.num_clips:
        print("[VIDEO] search limit reached before collecting every requested success")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
