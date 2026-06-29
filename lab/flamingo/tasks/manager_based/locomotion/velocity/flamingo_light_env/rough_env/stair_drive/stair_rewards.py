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
