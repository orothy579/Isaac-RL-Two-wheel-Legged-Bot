# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reward terms for the flamingo-light stair-climbing ("coin") task.

These pair with ``mdp.CoinSequenceCommand``: a dense distance term pulls the robot
toward the active coin, a sparse pulse rewards each coin collected, and a larger
pulse rewards reaching the top required coin. ``heading_to_coin_exp`` encourages the
robot to face the coin (which, by construction, is straight ahead up the stairs).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_coin_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "coin",
    temperature: float = 1.0,
    scaler: float = 1.0,
) -> torch.Tensor:
    """Dense reward: exp(-temp * xy-distance) to the active coin (base frame)."""
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(des_pos_b / scaler, dim=1)
    return torch.exp(-temperature * distance)


def track_coin_xyz_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "coin",
    temperature: float = 1.0,
    scaler: float = 1.0,
) -> torch.Tensor:
    """Dense reward: exp(-temp * 3D-distance) to the active coin (base frame).

    Including the height gap (command z) means the robot cannot maximize the reward
    by leaning forward at the base of a step — it must actually climb to reduce z.
    """
    des_pos_b = env.command_manager.get_command(command_name)[:, :3]
    distance = torch.norm(des_pos_b / scaler, dim=1)
    return torch.exp(-temperature * distance)


def coin_collected_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "coin",
) -> torch.Tensor:
    """Sparse one-step pulse (1.0) on every step a coin is collected."""
    term = env.command_manager.get_term(command_name)
    return term.just_collected


def reach_top_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "coin",
) -> torch.Tensor:
    """Sparse one-step pulse (1.0) when the *top* required coin is collected."""
    term = env.command_manager.get_term(command_name)
    return term.just_reached_top


def hop_up_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "stair_event",
    event_time_range: tuple = (0.25, 0.6),
    target_up_vel: float = 2.5,
    up_vel_coef: float = 20.0,
    temperature: float = 2.0,
    load_penalty_coef: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strong take-off reward for the hop window — flat-jump *strength*, but WITHOUT
    the vertical-alignment factor (stairs need up-AND-forward to land on the next step).

    Phases (``event_time`` since the window opened):
    * ``< event_time_range[0]`` (load): penalize dropping, so the robot loads its legs.
    * inside ``event_time_range`` (take-off): reward reaching ``target_up_vel`` upward,
      scaled by ``up_vel_coef`` (this is the impulse the weak ``hop_up`` was missing —
      run 1 climbed because it kept this coefficient).

    NOTE: this term is farmable in place (a vertical hop also earns it), so keep its
    weight modest and let the non-farmable coin/`reach_top` progress dominate.
    """
    cmd = env.command_manager.get_command(event_command_name)
    flag = cmd[:, 0]
    event_time = cmd[:, 1]

    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_z = asset.data.root_lin_vel_w[:, 2]

    pre_jump = (event_time < event_time_range[0]).float()
    window = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()

    up_reward = torch.exp(-torch.abs((target_up_vel - lin_vel_z) / target_up_vel) * temperature)
    # discourage uncontrolled falling before take-off (load the legs instead)
    descent_vel = torch.clamp(-lin_vel_z, min=0.0)
    load_penalty = torch.clamp(descent_vel - 1.0, min=0.0) * load_penalty_coef

    return up_reward * up_vel_coef * flag * window - load_penalty * flag * pre_jump


def foot_clearance_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "stair_event",
    event_time_range: tuple = (0.25, 0.7),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_wheel_link"),
    ground_sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
) -> torch.Tensor:
    """Reward folding the legs up (wheel clearance) WHILE moving forward during a hop.

    Encourages the robot to retract its wheels above the ground so they clear the riser
    — the user's "legs fold to go higher". Scaled by forward speed so it pays only when
    hopping *forward over* the step, not when bobbing in place (anti-farming).
    """
    cmd = env.command_manager.get_command(event_command_name)
    flag = cmd[:, 0]
    event_time = cmd[:, 1]

    asset: RigidObject = env.scene[asset_cfg.name]
    wheel_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, n_wheels)

    sensor = env.scene.sensors[ground_sensor_cfg.name]
    ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)  # (N,1)
    clearance = torch.clamp(wheel_z - ground_z, min=0.0).mean(dim=1)  # avg wheel lift [m]

    fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)  # forward only
    window = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()
    return clearance * fwd_speed * flag * window


def heading_to_coin_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "coin",
    temperature: float = 2.0,
) -> torch.Tensor:
    """Reward facing the active coin: exp(-temp * heading_error^2).

    Heading error is the bearing of the (base-frame) coin vector; it is ~0 when the
    coin is straight ahead.
    """
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    heading = torch.atan2(des_pos_b[:, 1], des_pos_b[:, 0])
    return torch.exp(-temperature * torch.square(heading))
