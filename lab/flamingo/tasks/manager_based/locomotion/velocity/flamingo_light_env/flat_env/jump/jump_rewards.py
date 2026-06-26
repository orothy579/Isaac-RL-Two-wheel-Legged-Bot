# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reward terms for the flamingo-light flat *jump* task.

The jump is triggered by the ``event`` command (see
``mdp.EventCommandCfg`` / ``EventCommand``). That command exposes two channels:

* ``event_command[:, 0]`` – active flag (1.0 while the jump window is open).
* ``event_command[:, 1]`` – elapsed time [s] since the window opened, in
  ``[0, event_during_time]``.

All terms below are written from scratch for ``flamingo_light_v1`` and use only
the robot root state / wheel contacts (no hip/leg joints, which this light robot
does not have), so they are independent of the ``track_jump`` rewards in
``flamingo_env``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_z_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.5),
    max_up_vel: float = 4.0,
    up_vel_coef: float = 20.0,
    down_vel_coef: float = 0.0,
    temperature: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward an upward (z) take-off velocity during the jump window.

    Timeline within an active event (``event_time`` in seconds):

    * ``event_time < event_time_range[0]``  -> *pre-jump*: penalize falling so
      the robot loads its legs instead of dropping.
    * ``event_time in event_time_range``    -> *take-off*: reward matching the
      target upward velocity, weighted by how well the velocity is aligned with
      +z (a pure vertical jump).
    * just after the window                 -> optional reward for a controlled
      descent (``down_vel_coef``, off by default).
    """
    event_command = env.command_manager.get_command(event_command_name)
    flag = event_command[:, 0]
    event_time = event_command[:, 1]

    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_w
    lin_vel_z = lin_vel[:, 2]
    lin_vel_mag = torch.norm(lin_vel, dim=1) + 1e-6

    # cosine alignment between velocity and the world +z axis
    alignment = torch.abs(lin_vel_z / lin_vel_mag)

    target_up_vel = max_up_vel

    pre_jump = (event_time < event_time_range[0]).float()
    jump_phase = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()
    after_jump = torch.logical_and(
        event_time > event_time_range[1], event_time <= event_time_range[1] + 0.4
    ).float()

    # penalize uncontrolled descent before take-off
    descent_vel = torch.clamp(-lin_vel_z, min=0.0)
    descent_penalty = torch.clamp(descent_vel - 1.0, min=0.0)

    up_vel_reward = torch.exp(-torch.abs((target_up_vel - lin_vel_z) / max_up_vel) * temperature)
    down_vel_reward = torch.exp(-torch.abs((-target_up_vel - lin_vel_z) / max_up_vel) * temperature)

    reward = up_vel_reward * up_vel_coef * flag * jump_phase * alignment
    reward += down_vel_reward * down_vel_coef * flag * after_jump * alignment
    reward -= descent_penalty * flag * pre_jump
    return reward


def feet_off_ground_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.5),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
) -> torch.Tensor:
    """Reward both wheels leaving the ground during the take-off window.

    Encourages a genuine jump (full air phase) rather than just a vertical
    body bob while staying in contact.
    """
    event_command = env.command_manager.get_command(event_command_name)
    flag = event_command[:, 0]
    event_time = event_command[:, 1]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    airborne = (current_contact_time <= 0.0).all(dim=1).float()

    window = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()
    return airborne * flag * window


def push_ground_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "event",
    event_time_range: tuple = (0.3, 0.5),
    max_force: float = 300.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
) -> torch.Tensor:
    """Reward a strong, *symmetric* vertical push-off through both wheels.

    A large total z-force at take-off is good; a big left/right imbalance is
    discouraged so the robot jumps straight up.
    """
    event_command = env.command_manager.get_command(event_command_name)
    flag = event_command[:, 0]
    event_time = event_command[:, 1]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # [B, N, 3]

    z_force = torch.clamp(foot_force[..., 2], min=0.0)  # [B, N]
    total_force = z_force.sum(dim=1).clamp(max=max_force)  # [B]
    force_diff = torch.abs(z_force[:, 0] - z_force[:, 1])  # [B]

    reward = total_force * torch.exp(-force_diff / 20.0)

    window = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()
    return reward * flag * window


def base_height_when_not_jumping(
    env: ManagerBasedRLEnv,
    target_height: float,
    event_command_name: str = "event",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """L2 base-height penalty that is disabled while a jump is active.

    The stand/drive base-height reward would otherwise fight the jump (it pulls
    the body back to its nominal standing height). Gating it with ``(1 - flag)``
    keeps the nominal posture between jumps while letting the body rise freely
    during the event window.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    event_command = env.command_manager.get_command(event_command_name)
    not_active = 1.0 - event_command[:, 0]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        adjusted_target = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        adjusted_target = target_height

    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target) * not_active
