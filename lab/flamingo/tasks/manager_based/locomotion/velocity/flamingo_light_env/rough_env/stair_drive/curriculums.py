# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Curriculum for the stair-climbing ("coin") task.

Advances difficulty *one step at a time*: an env that reached its top required coin
this episode needs one more step next time; an env that collected nothing drops one.
Step height stays fixed (set on the terrain) — only the number of required steps grows.

Runs on env reset, before the coin command resamples (same ordering the built-in
``terrain_levels_vel`` relies on), so ``reached_top`` still reflects this episode.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def coin_count_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "coin",
) -> torch.Tensor:
    """Increase required coins for envs that reached the top, decrease for no-progress."""
    term = env.command_manager.get_term(command_name)

    reached_top = term.reached_top[env_ids]
    no_progress = term.active_idx[env_ids] == 0

    move_up = reached_top
    move_down = no_progress & (~reached_top)

    level = term.coin_level[env_ids]
    level = level + move_up.float() - move_down.float()
    term.coin_level[env_ids] = level.clamp(min=1.0, max=float(term.n_steps))

    return torch.mean(term.coin_level)
