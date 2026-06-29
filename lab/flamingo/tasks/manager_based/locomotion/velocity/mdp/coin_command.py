# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for a Mario-style "coin" sequence used by the stair-climbing task.

A ladder of coins is placed one per stair step, straight ahead of the spawn (along
+x). The robot is guided toward the *current* (active) coin; when it gets within
``collect_radius`` (in xy) the coin is collected, a one-step ``just_collected`` pulse
is emitted (read by the sparse bonus reward) and the active coin advances to the next
(higher) step. ``coin_level`` (managed by the curriculum) controls how many steps the
episode requires; reaching the top coin sets ``reached_top`` (read by the curriculum).

The exposed command is the **base-frame** (yaw-only) vector from the robot to the
active coin, so a positive x means "the coin is ahead".
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CoinSequenceCommand(CommandTerm):
    """Generates a per-step "coin" ladder and tracks collection / progress."""

    cfg: CoinSequenceCommandCfg

    def __init__(self, cfg: CoinSequenceCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # --- derive the inverted-pyramid stair geometry (must match the terrain cfg) ---
        # see isaaclab.terrains.trimesh.mesh_terrains.inverted_pyramid_stairs_terrain
        terrain_size = cfg.tile_size - 2.0 * cfg.border_width
        num_steps = int(
            (cfg.tile_size - 2.0 * cfg.border_width - cfg.platform_width) // (2.0 * cfg.step_width) + 1
        )
        # number of coins = number of real steps, optionally capped by cfg.max_steps
        self.n_steps = min(num_steps, cfg.max_steps)
        # half-width of the flat pit floor the robot spawns on (env_origin sits here)
        self.pit_half_width = terrain_size / 2.0 - num_steps * cfg.step_width

        n = self.num_envs
        m = self.n_steps

        # world-frame coin positions, shape (num_envs, n_steps, 3)
        self.coins_w = torch.zeros(n, m, 3, device=self.device)
        # base-frame (yaw-only) vector to the active coin, shape (num_envs, 3)
        self.position_command_b = torch.zeros(n, 3, device=self.device)
        # how many steps/coins this episode requires (curriculum-controlled)
        self.coin_level = torch.full((n,), float(cfg.start_level), device=self.device)
        # index of the current target coin
        self.active_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        # one-step pulse: 1.0 on the step any coin was collected
        self.just_collected = torch.zeros(n, device=self.device)
        # one-step pulse: 1.0 on the step the *top* required coin was collected
        self.just_reached_top = torch.zeros(n, device=self.device)
        # whether the top (last required) coin has been reached this episode
        self.reached_top = torch.zeros(n, dtype=torch.bool, device=self.device)

        # metrics buffers (mean-logged + zeroed on reset by the base class)
        self.metrics["coin_level"] = torch.zeros(n, device=self.device)
        self.metrics["active_idx"] = torch.zeros(n, device=self.device)
        self.metrics["dist_to_coin"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        msg = "CoinSequenceCommand:\n"
        msg += f"\tmax_steps: {self.cfg.max_steps}\n"
        msg += f"\tcollect_radius: {self.cfg.collect_radius}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Base-frame (yaw-only) vector to the active coin. Shape (num_envs, 3)."""
        return self.position_command_b

    # -- helpers

    def _compute_coins(self, env_ids: Sequence[int]):
        """Place the coin ladder along +x from each env origin, one coin per step.

        Coin i (climbing outward from the pit center) sits on the tread of the
        (i+1)-th step: x offset = pit_half_width + (i+0.5)*step_width from the
        spawn center, height = (i+1)*step_height above the pit floor (= env origin).
        """
        origins = self.env.scene.env_origins[env_ids]  # (k, 3)
        idx = torch.arange(self.n_steps, device=self.device).float()  # (m,)
        dx = self.pit_half_width + (idx + 0.5) * self.cfg.step_width  # (m,)
        dz = (idx + 1.0) * self.cfg.step_height  # (m,)
        coins = torch.zeros(origins.shape[0], self.n_steps, 3, device=self.device)
        coins[:, :, 0] = origins[:, 0:1] + dx.unsqueeze(0)
        coins[:, :, 1] = origins[:, 1:2]
        coins[:, :, 2] = origins[:, 2:3] + dz.unsqueeze(0)
        self.coins_w[env_ids] = coins

    def _active_coin_w(self) -> torch.Tensor:
        """World position of the active coin for every env. Shape (num_envs, 3)."""
        return self.coins_w[torch.arange(self.num_envs, device=self.device), self.active_idx]

    def _refresh_command(self):
        """Recompute the base-frame (yaw-only) vector to the active coin."""
        delta_w = self._active_coin_w() - self.robot.data.root_link_pos_w
        self.position_command_b[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_link_quat_w), delta_w
        )

    # -- CommandTerm interface

    def _resample_command(self, env_ids: Sequence[int]):
        # (re)build the coin ladder and reset per-episode progress
        self._compute_coins(env_ids)
        self.active_idx[env_ids] = 0
        self.just_collected[env_ids] = 0.0
        self.just_reached_top[env_ids] = 0.0
        self.reached_top[env_ids] = False
        self._refresh_command()

    def _update_command(self):
        active_w = self._active_coin_w()
        root = self.robot.data.root_link_pos_w
        dist_xy = torch.norm(active_w[:, :2] - root[:, :2], dim=1)
        dz = (active_w[:, 2] - root[:, 2]).abs()

        top_idx = (self.coin_level - 1.0).long().clamp(min=0)
        # require BOTH horizontal proximity AND height match, so the robot cannot
        # "grab" a coin on a higher step just by leaning its body forward.
        within = (dist_xy < self.cfg.collect_radius) & (dz < self.cfg.collect_z_tol)
        is_top_active = self.active_idx >= top_idx

        collect = within & (~self.reached_top)
        advance = collect & (~is_top_active)  # collected an intermediate coin
        top_hit = collect & is_top_active     # collected the top required coin

        self.active_idx = torch.where(advance, self.active_idx + 1, self.active_idx)
        self.reached_top = self.reached_top | top_hit
        self.just_collected = collect.float()
        self.just_reached_top = top_hit.float()

        self._refresh_command()

    def _update_metrics(self):
        active_w = self._active_coin_w()
        dist_xy = torch.norm(active_w[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=1)
        # copy into the metric buffers in place (the base class zeroes these on reset,
        # so do NOT alias the live state tensors here)
        self.metrics["coin_level"][:] = self.coin_level
        self.metrics["active_idx"][:] = self.active_idx.float()
        self.metrics["dist_to_coin"][:] = dist_xy

    # -- debug visualization

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "coin_visualizer"):
                self.coin_visualizer = VisualizationMarkers(self.cfg.coin_visualizer_cfg)
            self.coin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "coin_visualizer"):
                self.coin_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        coin = self._active_coin_w().clone()
        coin[:, 2] += 0.1  # lift slightly so it sits above the tread
        self.coin_visualizer.visualize(coin)


GOLD_COIN_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/coin_command",
    markers={
        "coin": sim_utils.SphereCfg(
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.84, 0.0)),
        ),
    },
)


@configclass
class CoinSequenceCommandCfg(CommandTermCfg):
    """Configuration for :class:`CoinSequenceCommand`."""

    class_type: type = CoinSequenceCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    max_steps: int = 16
    """Upper bound on the number of coins (the real count is min(this, terrain steps))."""

    start_level: int = 1
    """Initial number of required steps/coins per episode."""

    # --- stair geometry: MUST match the terrain sub-terrain cfg ---
    tile_size: float = 10.0
    """Sub-terrain tile size [m] (terrain_generator ``size[0]``)."""

    border_width: float = 1.0
    """Sub-terrain border width [m] (the *sub-terrain* ``border_width``, not the generator's)."""

    platform_width: float = 2.5
    """Central platform width [m] (terrain ``platform_width``; affects the step count)."""

    step_width: float = 0.4
    """Stair tread depth [m] (terrain ``step_width``)."""

    step_height: float = 0.05
    """Stair step height [m] (terrain ``step_height``)."""

    collect_radius: float = 0.3
    """xy distance [m] within which a coin is collected."""

    collect_z_tol: float = 0.12
    """height gap [m] the robot must be within to collect a coin (prevents leaning to grab a higher coin)."""

    coin_visualizer_cfg: VisualizationMarkersCfg = GOLD_COIN_MARKER_CFG
