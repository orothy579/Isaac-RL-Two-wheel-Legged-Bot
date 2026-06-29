# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Perception-triggered jump event for stair climbing.

Unlike the time-based :class:`EventCommand`, this raises the jump flag when the
forward height-scan detects an upward step of at least ``step_threshold`` ahead of
the robot. Because the trigger is a deterministic function of the (real) height
scanner, the resulting flag is deployable — the real robot computes the same signal
from its own sensor.

Command layout (matches the flat jump rewards):
* ``command[:, 0]`` – active flag (1.0 while a hop window is open),
* ``command[:, 1]`` – elapsed time [s] since the window opened.

A rising-edge detector opens a hop window of ``event_during_time`` seconds; after it
closes there is a ``cooldown`` before the next hop can start, so the robot performs
discrete hops as it meets successive steps.
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
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StairDetectEventCommand(CommandTerm):
    """Jump event triggered by detecting an upward step ahead via the height scan."""

    cfg: StairDetectEventCommandCfg

    def __init__(self, cfg: StairDetectEventCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.sensor: RayCaster = env.scene.sensors[cfg.sensor_name]
        self.near_sensor: RayCaster = env.scene.sensors[cfg.near_sensor_name]

        n = self.num_envs
        self.event_command = torch.zeros(n, 2, dtype=torch.float32, device=self.device)
        self.active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.elapsed = torch.zeros(n, device=self.device)
        self.cooldown_timer = torch.zeros(n, device=self.device)

        self.metrics["detected"] = torch.zeros(n, device=self.device)
        self.metrics["active"] = torch.zeros(n, device=self.device)
        self.metrics["step_ahead"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return (
            "StairDetectEventCommand:\n"
            f"\tstep_threshold: {self.cfg.step_threshold}\n"
            f"\tevent_during_time: {self.cfg.event_during_time}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.event_command

    # -- detection

    def _detect_step_ahead(self) -> torch.Tensor:
        """Return (forward terrain height) - (terrain under the robot), per env [m].

        Forward height comes from the forward FOV height scan (camera-visible cells in
        a base-frame strip ahead of the robot); the under-robot reference comes from the
        dedicated downward ``base_height_scanner`` (the forward camera cannot see
        directly beneath the robot). Both are real sensors, so this is deployable.
        """
        hits = self.sensor.data.ray_hits_w  # (N, R, 3) world-frame ray hits
        base_xy = self.robot.data.root_link_pos_w[:, :2]
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
        cos, sin = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)  # (N,1)

        dx = hits[:, :, 0] - base_xy[:, 0:1]
        dy = hits[:, :, 1] - base_xy[:, 1:2]
        # rotate world delta into the yaw-aligned base frame
        bx = cos * dx + sin * dy
        by = -sin * dx + cos * dy
        bz = hits[:, :, 2]

        finite = torch.isfinite(bz) & torch.isfinite(bx) & torch.isfinite(by)
        valid_mask = getattr(self.sensor.data, "valid_mask", None)
        if valid_mask is not None:
            finite = finite & (valid_mask > 0.5)

        fwd = finite & (by.abs() < self.cfg.y_halfwidth)
        fwd = fwd & (bx > self.cfg.forward_band[0]) & (bx < self.cfg.forward_band[1])
        fwd_count = fwd.sum(dim=1)
        fwd_z = (bz * fwd).sum(dim=1) / fwd_count.clamp(min=1)

        # under-robot ground from the downward base scanner
        near_hits = self.near_sensor.data.ray_hits_w[:, :, 2]
        near_finite = torch.isfinite(near_hits)
        near_z = (near_hits * near_finite).sum(dim=1) / near_finite.sum(dim=1).clamp(min=1)

        valid = (fwd_count > 0) & (near_finite.sum(dim=1) > 0)
        return torch.where(valid, fwd_z - near_z, torch.zeros_like(fwd_z))

    # -- CommandTerm interface

    def _resample_command(self, env_ids: Sequence[int]):
        self.event_command[env_ids] = 0.0
        self.active[env_ids] = False
        self.elapsed[env_ids] = 0.0
        self.cooldown_timer[env_ids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        step_ahead = self._detect_step_ahead()
        detected = step_ahead > self.cfg.step_threshold

        # rising edge: open a hop window when a step appears and we're idle + off cooldown
        can_start = (~self.active) & (self.cooldown_timer <= 0.0) & detected
        self.active = self.active | can_start
        self.elapsed = torch.where(can_start, torch.zeros_like(self.elapsed), self.elapsed)

        # advance the open window; close it after event_during_time and start cooldown
        self.elapsed = torch.where(self.active, self.elapsed + dt, self.elapsed)
        closing = self.active & (self.elapsed >= self.cfg.event_during_time)
        self.active = self.active & (~closing)
        self.cooldown_timer = torch.where(closing, torch.full_like(self.cooldown_timer, self.cfg.cooldown), self.cooldown_timer)
        # tick cooldown down while idle
        self.cooldown_timer = torch.where(~self.active, (self.cooldown_timer - dt).clamp(min=0.0), self.cooldown_timer)

        self.event_command[:, 0] = self.active.float()
        self.event_command[:, 1] = self.elapsed * self.active.float()

        # cache for metrics
        self._last_detected = detected.float()
        self._last_step_ahead = step_ahead

    def _update_metrics(self):
        if hasattr(self, "_last_detected"):
            self.metrics["detected"][:] = self._last_detected
            self.metrics["step_ahead"][:] = self._last_step_ahead
        self.metrics["active"][:] = self.active.float()

    # -- debug visualization

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "active_vis"):
                self.active_vis = VisualizationMarkers(self.cfg.active_visualizer_cfg)
                self.inactive_vis = VisualizationMarkers(self.cfg.inactive_visualizer_cfg)
            self.active_vis.set_visibility(True)
            self.inactive_vis.set_visibility(True)
        else:
            if hasattr(self, "active_vis"):
                self.active_vis.set_visibility(False)
                self.inactive_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        pos = self.robot.data.root_link_pos_w.clone()
        pos[:, 2] += 0.75
        act = (self.event_command[:, 0] > 0.5).nonzero(as_tuple=True)[0]
        inact = (self.event_command[:, 0] <= 0.5).nonzero(as_tuple=True)[0]
        if act.numel() > 0:
            self.active_vis.visualize(pos[act])
        if inact.numel() > 0:
            self.inactive_vis.visualize(pos[inact])


_GREEN = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/stair_event_active",
    markers={"s": sim_utils.SphereCfg(radius=0.07, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))},
)
_RED = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/stair_event_inactive",
    markers={"s": sim_utils.SphereCfg(radius=0.07, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
)


@configclass
class StairDetectEventCommandCfg(CommandTermCfg):
    """Configuration for :class:`StairDetectEventCommand`."""

    class_type: type = StairDetectEventCommand

    asset_name: str = MISSING
    sensor_name: str = "height_scanner"
    """Forward FOV height-scan sensor used to read the terrain ahead."""

    near_sensor_name: str = "base_height_scanner"
    """Downward scanner under the base, used as the under-robot ground reference."""

    step_threshold: float = 0.03
    """Minimum forward-minus-near terrain height [m] that triggers a hop."""

    forward_band: tuple[float, float] = (0.15, 0.45)
    """Base-frame x range [m] of the 'ahead' cells used to measure the step."""

    y_halfwidth: float = 0.2
    """Base-frame |y| limit [m] of cells considered (keeps it to a forward strip)."""

    event_during_time: float = 0.5
    """Duration [s] of one hop window once triggered."""

    cooldown: float = 0.3
    """Idle time [s] required after a hop before the next one can start."""

    active_visualizer_cfg: VisualizationMarkersCfg = _GREEN
    inactive_visualizer_cfg: VisualizationMarkersCfg = _RED
