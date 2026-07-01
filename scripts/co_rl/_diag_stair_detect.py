"""Diagnostic: verify StairDetectEventCommand fires near a step.

Teleports each env's robot to a different forward offset from the pit center, steps
once so the height scan updates, and prints the detected step_ahead per offset.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app = AppLauncher(args).app

import torch
import gymnasium as gym

import lab.flamingo  # noqa: F401  (registers the gym tasks)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo"
N = 16

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()

robot = env.scene["robot"]
origins = env.scene.env_origins.clone()
term = env.command_manager.get_term("stair_event")

# per-env forward offset from pit center: 0.10 .. 0.85 m
offsets = torch.linspace(0.10, 0.85, N, device=env.device)

zero_action = torch.zeros(N, env.action_manager.total_action_dim, device=env.device)

for trial in range(3):  # a few steps to let sensors settle
    new_pos = origins.clone()
    new_pos[:, 0] += offsets
    new_pos[:, 2] += 0.31  # nominal standing height above pit floor
    quat = torch.zeros(N, 4, device=env.device)
    quat[:, 0] = 1.0  # identity (facing +x)
    robot.write_root_pose_to_sim(torch.cat([new_pos, quat], dim=1))
    robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=env.device))
    env.step(zero_action)

step_ahead = term._last_step_ahead if hasattr(term, "_last_step_ahead") else None
flag = term.command[:, 0]

print("\n==== STAIR DETECT PROFILE (forward_band=%s, step_threshold=%.3f) ====" % (str(term.cfg.forward_band), term.cfg.step_threshold))
for i in range(N):
    sa = float(step_ahead[i]) if step_ahead is not None else float("nan")
    print(f"  offset={offsets[i]:.2f}m  step_ahead={sa:+.3f}  detected_flag={float(flag[i]):.0f}")
if step_ahead is not None:
    print("MAX step_ahead = %.3f  | any detected this step = %d/%d" % (step_ahead.max().item(), int((flag > 0.5).sum().item()), N))

env.close()
app.close()
