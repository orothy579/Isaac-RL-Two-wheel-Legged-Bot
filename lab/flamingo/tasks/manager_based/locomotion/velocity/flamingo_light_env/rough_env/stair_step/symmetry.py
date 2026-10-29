# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right mirror of the stair-step policy observation and action (for the PPO mirror loss).

The robot is symmetric about its sagittal (x-z) plane. Reflecting y -> -y swaps the left and right
joints (shoulder angles and wheel speeds keep their sign: both shoulders use [-0.75, 0] rad and both
wheels spin positive when driving forward), negates the roll and yaw rates, the lateral gravity
component, the lateral velocity and yaw-rate commands, and flips the wheel-area height scan
across the robot's centre line.

Sweep v3 (trial 3) learned to start a staircase with the RIGHT wheel only: approaching with the left
wheel ahead it half-lifts the left leg and stalls. The rewards are symmetric (the first lift of an
episode counts whichever leg it is), so the mirror loss transfers the right-first skill.

The policy input is built by the co_rl wrapper: ``num_policy_stacks + 1`` frames of ``stack_policy``
(newest first), then ``none_stack_policy``. Every term below must have a mirror rule.
"""

from __future__ import annotations

import torch

# term -> (permutation within the term, sign per mirrored element)
_RULES: dict[str, tuple[list[int], list[float]]] = {
    "joint_pos": ([1, 0], [1.0, 1.0]),  # [L, R] shoulder
    "joint_vel": ([1, 0, 3, 2], [1.0] * 4),  # [L, R] shoulder, [L, R] wheel
    "actions": ([1, 0, 3, 2], [1.0] * 4),
    "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),  # roll rate, pitch rate, yaw rate
    "base_projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
    "velocity_commands": ([0, 1, 2, 3], [1.0, -1.0, -1.0, 1.0]),  # vx, vy, yaw rate, height
}
# action layout: joint_pos [left, right shoulder], wheel_vel [left, right wheel]
_ACTION_PERM = [1, 0, 3, 2]


def _scan_rule(env, sensor_name: str) -> tuple[list[int], list[float]]:
    """Ray i -> the ray at the same x and opposite y (from the sensor's own ray pattern)."""
    starts = env.scene.sensors[sensor_name].ray_starts[0]  # (num_rays, 3), sensor frame
    x, y = starts[:, 0], starts[:, 1]
    d = (x.unsqueeze(1) - x.unsqueeze(0)).abs() + (y.unsqueeze(1) + y.unsqueeze(0)).abs()
    perm = d.argmin(dim=1)
    if not torch.all(d[torch.arange(len(perm)), perm] < 1.0e-4):
        raise ValueError(f"{sensor_name}: ray pattern is not symmetric about y = 0")
    return perm.tolist(), [1.0] * len(perm)


def _group_map(env, group: str, extra: dict) -> tuple[list[int], list[float]]:
    om = env.observation_manager
    perm, sign, offset = [], [], 0
    for name, dim, term_cfg in zip(om.active_terms[group], om.group_obs_term_dim[group], om._group_obs_term_cfgs[group]):
        if name in extra:
            p, s = extra[name]
        elif name in _RULES:
            p, s = _RULES[name]
        else:
            raise KeyError(f"no mirror rule for observation term '{group}/{name}'")
        if len(p) != dim[0]:
            raise ValueError(f"mirror rule for '{group}/{name}' has {len(p)} entries, the term has {dim[0]}")
        perm += [offset + i for i in p]
        sign += s
        offset += dim[0]
    return perm, sign


def policy_mirror_maps(env, frames: int, feet_scan: bool):
    """``(mirror_obs, mirror_actions)`` for a policy input of ``frames`` stacked ``stack_policy`` frames
    (newest first) followed by ``none_stack_policy``, with or without its trailing ``feet_scan`` term
    (the blind student of :mod:`scripts.co_rl.distill_stair_step` drops the scan of the perceptive env)."""
    extra = {}
    terms = env.observation_manager.active_terms["none_stack_policy"]
    if "feet_scan" in terms:
        extra["feet_scan"] = _scan_rule(env, "feet_scanner")
    frame_perm, frame_sign = _group_map(env, "stack_policy", extra)
    rest_perm, rest_sign = _group_map(env, "none_stack_policy", extra)
    if "feet_scan" in terms and not feet_scan:
        if terms[-1] != "feet_scan":
            raise ValueError("feet_scan must be the last none_stack_policy term to drop it")
        keep = len(rest_perm) - env.observation_manager.group_obs_term_dim["none_stack_policy"][-1][0]
        rest_perm, rest_sign = rest_perm[:keep], rest_sign[:keep]
    fdim = len(frame_perm)
    perm, sign = [], []
    for k in range(frames):
        perm += [k * fdim + i for i in frame_perm]
        sign += frame_sign
    perm += [frames * fdim + i for i in rest_perm]
    sign += rest_sign
    dev = env.device
    obs_index = torch.tensor(perm, dtype=torch.long, device=dev)
    obs_sign = torch.tensor(sign, dtype=torch.float32, device=dev)
    act_index = torch.tensor(_ACTION_PERM, dtype=torch.long, device=dev)
    if env.action_manager.total_action_dim != len(_ACTION_PERM):
        raise ValueError(f"action dim {env.action_manager.total_action_dim} != {len(_ACTION_PERM)}")

    def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
        return obs[..., obs_index] * obs_sign

    def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
        return actions[..., act_index]

    return mirror_obs, mirror_actions


def stair_step_policy_mirror(wrapped_env):
    """``(mirror_obs, mirror_actions)`` for the co_rl-wrapped stair-step env (blind or perceptive)."""
    env = wrapped_env.unwrapped
    return policy_mirror_maps(env, wrapped_env.num_policy_stacks + 1, feet_scan=True)
