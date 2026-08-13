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

    _IDLE = 0
    _HOP = 1
    _WAIT_RESULT = 2
    _RETREAT = 3
    _RUNUP = 4

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
        self.state = torch.zeros(n, dtype=torch.long, device=self.device)
        self.state_timer = torch.zeros(n, device=self.device)
        self.clear_timer = torch.zeros(n, device=self.device)
        self.armed = torch.ones(n, dtype=torch.bool, device=self.device)
        self.takeoff_ground_z = torch.zeros(n, device=self.device)
        self.ground_gain = torch.zeros(n, device=self.device)

        self.metrics["detected"] = torch.zeros(n, device=self.device)
        self.metrics["active"] = torch.zeros(n, device=self.device)
        self.metrics["step_ahead"] = torch.zeros(n, device=self.device)
        self.metrics["phase"] = torch.zeros(n, device=self.device)
        self.metrics["attempt_started"] = torch.zeros(n, device=self.device)
        self.metrics["attempt_failed"] = torch.zeros(n, device=self.device)
        self.metrics["recovery_active"] = torch.zeros(n, device=self.device)
        self.metrics["runup_active"] = torch.zeros(n, device=self.device)
        self.metrics["forward_speed"] = torch.zeros(n, device=self.device)
        self.metrics["ground_gain"] = torch.zeros(n, device=self.device)

        if self.cfg.trigger_mode not in {"baseline", "one_shot", "recovery_runup"}:
            raise ValueError(
                "StairDetectEventCommand.trigger_mode must be one of "
                f"['baseline', 'one_shot', 'recovery_runup'], got {self.cfg.trigger_mode!r}"
            )

    def __str__(self) -> str:
        return (
            "StairDetectEventCommand:\n"
            f"\ttrigger_mode: {self.cfg.trigger_mode}\n"
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
        # NOTE: RayCasterFOV keeps the FULL grid in ray_hits_w and only marks the
        # camera-FOV subset in valid_mask. The near-forward ground strip we measure is
        # often OUTSIDE the camera FOV, so applying valid_mask zeroes every forward cell
        # -> step never detected -> no jump ever. Use the full grid by default; only
        # apply the FOV mask if explicitly requested.
        if self.cfg.use_fov_mask:
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
        self.state[env_ids] = self._IDLE
        self.state_timer[env_ids] = 0.0
        self.clear_timer[env_ids] = 0.0
        self.armed[env_ids] = True
        self.takeoff_ground_z[env_ids] = 0.0
        self.ground_gain[env_ids] = 0.0

    def _ground_height(self) -> torch.Tensor:
        """Mean finite ground height under the base, per environment."""
        hits = self.near_sensor.data.ray_hits_w[:, :, 2]
        finite = torch.isfinite(hits)
        return (hits * finite).sum(dim=1) / finite.sum(dim=1).clamp(min=1)

    def _update_baseline(self, detected: torch.Tensor, commanded: torch.Tensor, dt: float):
        """Original level-triggered behavior, retained as the exact ablation baseline."""
        can_start = (~self.active) & (self.cooldown_timer <= 0.0) & detected & commanded
        self.active = self.active | can_start
        self.elapsed = torch.where(can_start, torch.zeros_like(self.elapsed), self.elapsed)

        self.elapsed = torch.where(self.active, self.elapsed + dt, self.elapsed)
        closing = self.active & (self.elapsed >= self.cfg.event_during_time)
        self.active = self.active & (~closing)
        self.cooldown_timer = torch.where(
            closing,
            torch.full_like(self.cooldown_timer, self.cfg.cooldown),
            self.cooldown_timer,
        )
        self.cooldown_timer = torch.where(
            ~self.active,
            (self.cooldown_timer - dt).clamp(min=0.0),
            self.cooldown_timer,
        )
        self.state = torch.where(
            self.active,
            torch.full_like(self.state, self._HOP),
            torch.full_like(self.state, self._IDLE),
        )
        return can_start, torch.zeros_like(can_start)

    def _update_one_shot(self, detected: torch.Tensor, commanded: torch.Tensor, dt: float):
        """One trigger per detection encounter; re-arm only after the step clears."""
        eligible_clear = (~detected) & (~self.active)
        self.clear_timer = torch.where(
            eligible_clear, self.clear_timer + dt, torch.zeros_like(self.clear_timer)
        )
        self.armed |= self.clear_timer >= self.cfg.rearm_clear_time

        can_start = (
            (~self.active)
            & self.armed
            & (self.cooldown_timer <= 0.0)
            & detected
            & commanded
        )
        self.armed = self.armed & (~can_start)
        self.active = self.active | can_start
        self.elapsed = torch.where(can_start, torch.zeros_like(self.elapsed), self.elapsed)

        self.elapsed = torch.where(self.active, self.elapsed + dt, self.elapsed)
        closing = self.active & (self.elapsed >= self.cfg.event_during_time)
        self.active = self.active & (~closing)
        self.cooldown_timer = torch.where(
            closing,
            torch.full_like(self.cooldown_timer, self.cfg.cooldown),
            self.cooldown_timer,
        )
        self.cooldown_timer = torch.where(
            ~self.active,
            (self.cooldown_timer - dt).clamp(min=0.0),
            self.cooldown_timer,
        )
        self.state = torch.where(
            self.active,
            torch.full_like(self.state, self._HOP),
            torch.full_like(self.state, self._IDLE),
        )
        return can_start, torch.zeros_like(can_start)

    def _update_recovery_runup(
        self,
        detected: torch.Tensor,
        commanded: torch.Tensor,
        ground_z: torch.Tensor,
        forward_speed: torch.Tensor,
        dt: float,
    ):
        """Attempt-aware hop with failed-attempt retreat and accelerated re-approach."""
        # A cleared detector re-arms the next physical encounter. This prevents the same
        # continuously visible riser from opening repeated hop windows.
        eligible_clear = (~detected) & (self.state != self._HOP) & (
            self.state != self._WAIT_RESULT
        )
        self.clear_timer = torch.where(
            eligible_clear, self.clear_timer + dt, torch.zeros_like(self.clear_timer)
        )
        self.armed |= self.clear_timer >= self.cfg.rearm_clear_time
        self.cooldown_timer = (self.cooldown_timer - dt).clamp(min=0.0)
        self.state_timer = torch.where(
            self.state != self._IDLE, self.state_timer + dt, self.state_timer
        )

        idle_start = (
            (self.state == self._IDLE)
            & self.armed
            & (self.cooldown_timer <= 0.0)
            & detected
            & commanded
        )
        runup_start = (
            (self.state == self._RUNUP)
            & self.armed
            & (self.cooldown_timer <= 0.0)
            & (self.state_timer >= self.cfg.runup_min_time)
            & detected
            & commanded
        )
        can_start = idle_start | runup_start
        self.state = torch.where(
            can_start, torch.full_like(self.state, self._HOP), self.state
        )
        self.state_timer = torch.where(can_start, torch.zeros_like(self.state_timer), self.state_timer)
        self.elapsed = torch.where(can_start, torch.zeros_like(self.elapsed), self.elapsed)
        self.takeoff_ground_z = torch.where(can_start, ground_z, self.takeoff_ground_z)
        self.ground_gain = torch.where(can_start, torch.zeros_like(self.ground_gain), self.ground_gain)
        self.armed = self.armed & (~can_start)

        hopping = self.state == self._HOP
        self.elapsed = torch.where(hopping, self.elapsed + dt, self.elapsed)
        closing = hopping & (self.elapsed >= self.cfg.event_during_time)
        self.state = torch.where(
            closing, torch.full_like(self.state, self._WAIT_RESULT), self.state
        )
        self.state_timer = torch.where(closing, torch.zeros_like(self.state_timer), self.state_timer)
        self.cooldown_timer = torch.where(
            closing,
            torch.full_like(self.cooldown_timer, self.cfg.cooldown),
            self.cooldown_timer,
        )

        self.ground_gain = torch.where(
            self.state == self._WAIT_RESULT,
            ground_z - self.takeoff_ground_z,
            self.ground_gain,
        )
        waiting = self.state == self._WAIT_RESULT
        success = waiting & (self.ground_gain >= self.cfg.success_height_threshold)
        ready_to_judge = waiting & (self.state_timer >= self.cfg.post_hop_wait)
        stuck = (
            ready_to_judge
            & detected
            & (forward_speed < self.cfg.stuck_speed_threshold)
        )
        timed_out = waiting & (self.state_timer >= self.cfg.failure_timeout)
        failed = (~success) & (stuck | timed_out)
        cleared_without_stall = ready_to_judge & (~detected) & (~success)

        go_idle = success | cleared_without_stall
        self.state = torch.where(
            go_idle, torch.full_like(self.state, self._IDLE), self.state
        )
        self.state_timer = torch.where(go_idle, torch.zeros_like(self.state_timer), self.state_timer)
        self.state = torch.where(
            failed, torch.full_like(self.state, self._RETREAT), self.state
        )
        self.state_timer = torch.where(failed, torch.zeros_like(self.state_timer), self.state_timer)

        retreat_done = (self.state == self._RETREAT) & (
            self.state_timer >= self.cfg.retreat_time
        )
        self.state = torch.where(
            retreat_done, torch.full_like(self.state, self._RUNUP), self.state
        )
        self.state_timer = torch.where(
            retreat_done, torch.zeros_like(self.state_timer), self.state_timer
        )

        # If the robot failed to re-encounter the riser, retreat and try the approach again
        # instead of remaining in an unbounded run-up state.
        runup_timeout = (self.state == self._RUNUP) & (
            self.state_timer >= self.cfg.runup_timeout
        )
        self.state = torch.where(
            runup_timeout, torch.full_like(self.state, self._RETREAT), self.state
        )
        self.state_timer = torch.where(
            runup_timeout, torch.zeros_like(self.state_timer), self.state_timer
        )

        self.active = self.state == self._HOP
        return can_start, failed

    def _update_command(self):
        dt = self._env.step_dt
        step_ahead = self._detect_step_ahead()
        detected = step_ahead > self.cfg.step_threshold

        # only hop when the robot is actually COMMANDED to move: gate the window start by
        # the velocity command so a zero command never triggers a hop (the robot then
        # stands still instead of lurching forward at every step it sees). Deployable —
        # the operator's forward command is the same signal on the real robot.
        vel_cmd_xy = self._env.command_manager.get_command(self.cfg.vel_command_name)[:, :2]
        commanded = torch.norm(vel_cmd_xy, dim=1) > self.cfg.min_cmd_speed
        ground_z = self._ground_height()
        forward_speed = self.robot.data.root_lin_vel_b[:, 0]

        if self.cfg.trigger_mode == "baseline":
            can_start, failed = self._update_baseline(detected, commanded, dt)
        elif self.cfg.trigger_mode == "one_shot":
            can_start, failed = self._update_one_shot(detected, commanded, dt)
        else:
            can_start, failed = self._update_recovery_runup(
                detected, commanded, ground_z, forward_speed, dt
            )

        self.event_command[:, 0] = self.active.float()
        self.event_command[:, 1] = self.elapsed * self.active.float()
        # Preserve the two-channel observation used by sweep012. Negative elapsed values
        # encode recovery phases without changing the policy input dimension:
        #   -1 = retreat, -2 = run-up.
        self.event_command[:, 1] = torch.where(
            self.state == self._RETREAT,
            torch.full_like(self.elapsed, -1.0),
            self.event_command[:, 1],
        )
        self.event_command[:, 1] = torch.where(
            self.state == self._RUNUP,
            torch.full_like(self.elapsed, -2.0),
            self.event_command[:, 1],
        )

        # cache for metrics
        self._last_detected = detected.float()
        self._last_step_ahead = step_ahead
        self._last_attempt_started = can_start.float()
        self._last_attempt_failed = failed.float()
        self._last_forward_speed = forward_speed

    def _update_metrics(self):
        if hasattr(self, "_last_detected"):
            self.metrics["detected"][:] = self._last_detected
            self.metrics["step_ahead"][:] = self._last_step_ahead
        self.metrics["active"][:] = self.active.float()
        self.metrics["phase"][:] = self.state.float()
        self.metrics["attempt_started"][:] = getattr(
            self, "_last_attempt_started", torch.zeros_like(self.elapsed)
        )
        self.metrics["attempt_failed"][:] = getattr(
            self, "_last_attempt_failed", torch.zeros_like(self.elapsed)
        )
        self.metrics["recovery_active"][:] = (self.state == self._RETREAT).float()
        self.metrics["runup_active"][:] = (self.state == self._RUNUP).float()
        self.metrics["forward_speed"][:] = getattr(
            self, "_last_forward_speed", torch.zeros_like(self.elapsed)
        )
        self.metrics["ground_gain"][:] = self.ground_gain

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

    use_fov_mask: bool = False
    """If True, restrict detection to camera-FOV cells (valid_mask). Default False —
    the near-forward ground strip is usually outside the camera FOV, so masking it
    would prevent any step from being detected."""

    vel_command_name: str = "base_velocity"
    """Velocity command term used to gate the hop trigger (only hop when commanded to move)."""

    min_cmd_speed: float = 0.05
    """Minimum |xy velocity command| [m/s] required to open a hop window (0 command -> no hop)."""

    trigger_mode: str = "baseline"
    """Attempt logic: ``baseline`` keeps the original level trigger, ``one_shot`` requires
    the detector to clear before re-arming, and ``recovery_runup`` adds a failed-attempt
    retreat and accelerated re-approach. All modes keep the same two-channel observation."""

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

    rearm_clear_time: float = 0.10
    """How long the detector must remain clear before a one-shot attempt re-arms."""

    post_hop_wait: float = 0.35
    """Time [s] after the hop window closes before judging a failed attempt."""

    failure_timeout: float = 0.90
    """Maximum result-wait time [s] before an unsuccessful attempt enters retreat."""

    success_height_threshold: float = 0.035
    """Minimum under-base ground-height increase [m] that marks the attempt successful."""

    stuck_speed_threshold: float = 0.15
    """Forward root speed [m/s] below which a still-visible riser is considered a stall."""

    retreat_time: float = 0.60
    """Duration [s] of the rewarded backward retreat after a failed attempt."""

    runup_min_time: float = 0.15
    """Minimum acceleration time [s] before a run-up may trigger the next hop."""

    runup_timeout: float = 1.50
    """Maximum run-up time [s] before returning to retreat and trying again."""

    active_visualizer_cfg: VisualizationMarkersCfg = _GREEN
    inactive_visualizer_cfg: VisualizationMarkersCfg = _RED
