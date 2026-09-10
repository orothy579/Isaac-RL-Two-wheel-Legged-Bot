"""Low-overhead, training-only stair jump attempt diagnostics."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat

from .stair_oracle_geometry import infer_curriculum_stair_geometry, square_stair_oracle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_JOINT_NAMES = (
    "left_shoulder_joint",
    "right_shoulder_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)
_ACTION_NAMES = ("left_shoulder", "right_shoulder", "left_wheel", "right_wheel")
_RESULT_REASONS = {
    1: "result_window_elapsed",
    2: "episode_reset",
    3: "superseded",
}


class StairAttemptDiagnostics:
    """Collect one compact row per sampled jump attempt.

    The collector never changes commands, observations, rewards, actions, or reset
    decisions.  GPU state is accumulated while an attempt is active and copied to
    the CPU only when the attempt is finalized.
    """

    def __init__(
        self,
        cfg,
        env: ManagerBasedEnv,
        robot: Articulation,
        ground_sensor: RayCaster,
    ):
        self.cfg = cfg
        self.env = env
        self.robot = robot
        self.ground_sensor = ground_sensor
        self.device = env.device
        self.num_envs = env.num_envs
        self.enabled = bool(cfg.diagnostics_enabled and cfg.diagnostics_output_dir)

        # These tensors also provide zero-valued TensorBoard pulses when disabled.
        self.completed_pulse = torch.zeros(self.num_envs, device=self.device)
        self.success_pulse = torch.zeros(self.num_envs, device=self.device)
        self.start_distance_pulse = torch.zeros(self.num_envs, device=self.device)
        self.start_alignment_abs_deg_pulse = torch.zeros(self.num_envs, device=self.device)
        if not self.enabled:
            return

        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.diagnostics_contact_sensor_name]
        self.joint_ids, self.joint_names = robot.find_joints(list(_JOINT_NAMES), preserve_order=True)
        self.wheel_body_ids, self.wheel_body_names = self.contact_sensor.find_bodies(
            list(cfg.diagnostics_wheel_body_names), preserve_order=True
        )
        if len(self.joint_ids) != len(_JOINT_NAMES):
            raise RuntimeError(
                f"stair diagnostics expected joints {_JOINT_NAMES}, resolved {self.joint_names}"
            )
        if len(self.wheel_body_ids) != 2:
            raise RuntimeError(
                "stair diagnostics expected two wheel contact bodies, resolved "
                f"{self.wheel_body_names}"
            )

        tracked_count = min(int(cfg.diagnostics_max_envs), self.num_envs)
        tracked_ids = torch.linspace(0, self.num_envs - 1, tracked_count, device=self.device).round().long()
        self.tracked_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.tracked_mask[tracked_ids] = True
        n = self.num_envs
        self.attempt_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.pending_write = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.pending_reason = torch.zeros(n, dtype=torch.long, device=self.device)
        self.env_attempt_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.attempt_id = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(n, device=self.device)
        self.sample_count = torch.zeros(n, device=self.device)
        self.start_ground_z = torch.zeros(n, device=self.device)
        self.start_step_ahead = torch.zeros(n, device=self.device)
        self.start_terrain_level = torch.zeros(n, dtype=torch.long, device=self.device)
        self.start_terrain_type = torch.zeros(n, dtype=torch.long, device=self.device)
        self.start_terrain_step_height = torch.zeros(n, device=self.device)
        self.start_terrain_step_width = torch.zeros(n, device=self.device)
        self.start_success_climb_threshold = torch.zeros(n, device=self.device)
        self.start_distance_along = torch.full((n,), torch.nan, device=self.device)
        self.start_distance_normal = torch.full((n,), torch.nan, device=self.device)
        self.start_alignment = torch.full((n,), torch.nan, device=self.device)
        self.start_normal = torch.zeros(n, 2, device=self.device)
        self.start_root_pose = torch.zeros(n, 6, device=self.device)
        self.start_base_velocity = torch.zeros(n, 4, device=self.device)
        self.start_command = torch.zeros(n, 3, device=self.device)
        self.start_joint_pos = torch.zeros(n, 4, device=self.device)
        self.start_joint_vel = torch.zeros(n, 4, device=self.device)
        self.start_action = torch.zeros(n, 4, device=self.device)
        self.max_ground_gain = torch.zeros(n, device=self.device)
        self.max_base_vz = torch.full((n,), -torch.inf, device=self.device)
        self.max_abs_roll = torch.zeros(n, device=self.device)
        self.max_abs_pitch = torch.zeros(n, device=self.device)
        self.peak_abs_torque = torch.zeros(n, 4, device=self.device)
        self.abs_torque_impulse = torch.zeros(n, 4, device=self.device)
        self.mean_action_sum = torch.zeros(n, 4, device=self.device)
        self.peak_abs_action = torch.zeros(n, 4, device=self.device)
        self.peak_contact_force = torch.zeros(n, device=self.device)
        self.contact_impulse = torch.zeros(n, device=self.device)
        self.final_roll = torch.zeros(n, device=self.device)
        self.final_pitch = torch.zeros(n, device=self.device)
        self.final_both_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.final_climbed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.final_stable = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.final_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._update_counter = 0

        output_dir = os.path.abspath(cfg.diagnostics_output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, cfg.diagnostics_csv_name)
        self.metadata_path = os.path.join(output_dir, cfg.diagnostics_metadata_name)
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(self._columns())
        self._csv_file.flush()
        self._write_metadata(tracked_ids)
        print(
            f"[INFO] Stair attempt diagnostics: tracking {tracked_count}/{self.num_envs} envs -> {self.csv_path}"
        )

    @staticmethod
    def _columns() -> list[str]:
        columns = [
            "attempt_id",
            "env_id",
            "terrain_level",
            "terrain_type",
            "terrain_step_height_m",
            "terrain_step_width_m",
            "success_climb_threshold_m",
            "result_reason",
            "success",
            "climbed",
            "stable_landing",
            "duration_s",
            "oracle_distance_along_heading_m",
            "oracle_distance_to_plane_m",
            "oracle_alignment_error_rad",
            "oracle_alignment_error_deg",
            "oracle_normal_x_w",
            "oracle_normal_y_w",
            "detected_step_height_m",
            "start_ground_z_w_m",
            "start_local_x_m",
            "start_local_y_m",
            "start_base_z_w_m",
            "start_roll_rad",
            "start_pitch_rad",
            "start_yaw_rad",
            "start_vx_b_mps",
            "start_vy_b_mps",
            "start_vz_w_mps",
            "start_yaw_rate_b_radps",
            "command_vx_mps",
            "command_vy_mps",
            "command_yaw_rate_radps",
        ]
        for name in _JOINT_NAMES:
            columns.extend((f"start_{name}_pos_rad", f"start_{name}_vel_radps"))
        for name in _ACTION_NAMES:
            columns.append(f"trigger_action_{name}")
        columns.extend(
            (
                "max_ground_gain_m",
                "max_base_vz_w_mps",
                "max_abs_roll_rad",
                "max_abs_pitch_rad",
                "final_roll_rad",
                "final_pitch_rad",
                "final_both_wheels_contact",
            )
        )
        for name in _JOINT_NAMES:
            columns.extend((f"peak_abs_{name}_torque_nm", f"abs_{name}_torque_impulse_nms"))
        for name in _ACTION_NAMES:
            columns.extend((f"mean_action_{name}", f"peak_abs_action_{name}"))
        columns.extend(("peak_wheel_contact_force_n", "wheel_contact_impulse_ns"))
        return columns

    def _write_metadata(self, tracked_ids: torch.Tensor):
        metadata = {
            "purpose": "training-only oracle diagnostics; not an actor observation or reward",
            "csv": os.path.basename(self.csv_path),
            "training_log_dir": self.cfg.diagnostics_training_log_dir,
            "tracked_env_count": int(tracked_ids.numel()),
            "tracked_env_ids": tracked_ids.cpu().tolist(),
            "step_height_range_m": [float(v) for v in self.cfg.diagnostics_step_height_range],
            "step_width_range_m": [float(v) for v in self.cfg.diagnostics_step_width_range],
            "tile_size_m": float(self.cfg.diagnostics_tile_size),
            "terrain_border_width_m": float(self.cfg.diagnostics_terrain_border_width),
            "platform_width_m": float(self.cfg.diagnostics_platform_width),
            "result_wait_time_s": float(self.cfg.diagnostics_result_wait_time),
            "success_height_threshold_m": float(self.cfg.success_height_threshold),
            "success_min_step_fraction": float(self.cfg.diagnostics_success_min_step_fraction),
            "landing_max_abs_roll_rad": float(self.cfg.diagnostics_landing_max_abs_roll),
            "landing_max_abs_pitch_rad": float(self.cfg.diagnostics_landing_max_abs_pitch),
            "joint_order": list(_JOINT_NAMES),
            "action_order": list(_ACTION_NAMES),
            "success_definition": (
                "ground gain >= max(success_height_threshold, actual step height * min step fraction) "
                "AND both wheels contact AND final roll/pitch within limits"
            ),
            "contact_impulse_note": "sum(norm(net normal wheel contact force) * environment step_dt)",
        }
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def _ground_height(self) -> torch.Tensor:
        hits = self.ground_sensor.data.ray_hits_w[:, :, 2]
        finite = torch.isfinite(hits)
        return (hits * finite).sum(dim=1) / finite.sum(dim=1).clamp(min=1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor):
        """Finalize sampled attempts interrupted by an episode reset."""
        if not self.enabled:
            return
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask[ids] = True
        self._mark_finalize(reset_mask & self.attempt_active, reason_code=2)

    def update(
        self,
        can_start: torch.Tensor,
        ground_z: torch.Tensor,
        step_ahead: torch.Tensor,
        dt: float,
    ):
        """Capture starts, accumulate loads, and finalize mature attempts."""
        self.completed_pulse.zero_()
        self.success_pulse.zero_()
        self.start_distance_pulse.zero_()
        self.start_alignment_abs_deg_pulse.zero_()
        if not self.enabled:
            return

        starts = can_start & self.tracked_mask
        superseded = starts & self.attempt_active
        self._mark_finalize(superseded, reason_code=3)
        # A pending row is held for at most diagnostics_flush_interval_steps. Ignore
        # a new start in that tiny interval rather than overwrite an unwritten result.
        starts = starts & (~self.attempt_active) & (~self.pending_write)
        self._begin(starts, ground_z, step_ahead)
        self._accumulate(self.attempt_active, ground_z, dt)
        result_time = float(self.cfg.event_during_time + self.cfg.diagnostics_result_wait_time)
        completed = self.attempt_active & (self.elapsed >= result_time)
        self._mark_finalize(completed, reason_code=1)

        # nonzero() synchronizes CUDA with the host, so perform CSV transfers at a
        # bounded cadence rather than on every 50 Hz environment step.
        self._update_counter += 1
        if self._update_counter % int(self.cfg.diagnostics_flush_interval_steps) == 0:
            self._flush_pending()

    def _begin(self, mask: torch.Tensor, ground_z: torch.Tensor, step_ahead: torch.Tensor):
        self.env_attempt_count += mask.long()
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        unique_attempt_id = self.env_attempt_count * self.num_envs + env_ids
        self.attempt_active |= mask
        self.attempt_id[mask] = unique_attempt_id[mask]
        self.elapsed[mask] = 0.0
        self.sample_count[mask] = 0.0
        self.start_ground_z[mask] = ground_z[mask]
        self.start_step_ahead[mask] = step_ahead[mask]
        terrain = self.env.scene.terrain
        self.start_terrain_level[mask] = terrain.terrain_levels[mask]
        self.start_terrain_type[mask] = terrain.terrain_types[mask]

        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
        step_height, step_width, riser_radii = infer_curriculum_stair_geometry(
            self.env.scene.env_origins[:, 2],
            self.cfg.diagnostics_step_height_range,
            self.cfg.diagnostics_step_width_range,
            self.cfg.diagnostics_tile_size,
            self.cfg.diagnostics_terrain_border_width,
            self.cfg.diagnostics_platform_width,
            self.cfg.diagnostics_max_stair_steps,
        )
        distance_along, distance_normal, alignment, normal = square_stair_oracle(
            self.robot.data.root_link_pos_w[:, :2],
            yaw,
            self.env.scene.env_origins[:, :2],
            riser_radii,
        )
        self.start_terrain_step_height[mask] = step_height[mask]
        self.start_terrain_step_width[mask] = step_width[mask]
        climb_threshold = torch.maximum(
            torch.full_like(step_height, float(self.cfg.success_height_threshold)),
            step_height * float(self.cfg.diagnostics_success_min_step_fraction),
        )
        self.start_success_climb_threshold[mask] = climb_threshold[mask]
        self.start_distance_along[mask] = distance_along[mask]
        self.start_distance_normal[mask] = distance_normal[mask]
        self.start_alignment[mask] = alignment[mask]
        self.start_normal[mask] = normal[mask]
        local_pos = self.robot.data.root_link_pos_w - self.env.scene.env_origins
        self.start_root_pose[mask] = torch.stack(
            (local_pos[:, 0], local_pos[:, 1], self.robot.data.root_link_pos_w[:, 2], roll, pitch, yaw),
            dim=1,
        )[mask]
        self.start_base_velocity[mask] = torch.stack(
            (
                self.robot.data.root_link_lin_vel_b[:, 0],
                self.robot.data.root_link_lin_vel_b[:, 1],
                self.robot.data.root_link_lin_vel_w[:, 2],
                self.robot.data.root_link_ang_vel_b[:, 2],
            ),
            dim=1,
        )[mask]
        self.start_command[mask] = self.env.command_manager.get_command(self.cfg.vel_command_name)[mask, :3]
        self.start_joint_pos[mask] = self.robot.data.joint_pos[mask][:, self.joint_ids]
        self.start_joint_vel[mask] = self.robot.data.joint_vel[mask][:, self.joint_ids]
        self.start_action[mask] = self.env.action_manager.action[mask]
        self.max_ground_gain[mask] = 0.0
        self.max_base_vz[mask] = -torch.inf
        self.max_abs_roll[mask] = 0.0
        self.max_abs_pitch[mask] = 0.0
        self.peak_abs_torque[mask] = 0.0
        self.abs_torque_impulse[mask] = 0.0
        self.mean_action_sum[mask] = 0.0
        self.peak_abs_action[mask] = 0.0
        self.peak_contact_force[mask] = 0.0
        self.contact_impulse[mask] = 0.0
        self.start_distance_pulse[mask] = torch.nan_to_num(distance_along[mask], nan=0.0)
        self.start_alignment_abs_deg_pulse[mask] = torch.nan_to_num(
            alignment[mask].abs() * (180.0 / math.pi), nan=0.0
        )

    def _accumulate(self, mask: torch.Tensor, ground_z: torch.Tensor, dt: float):
        self.elapsed[mask] += dt
        self.sample_count[mask] += 1.0
        ground_gain = ground_z - self.start_ground_z
        self.max_ground_gain[mask] = torch.maximum(self.max_ground_gain[mask], ground_gain[mask])
        self.max_base_vz[mask] = torch.maximum(
            self.max_base_vz[mask], self.robot.data.root_link_lin_vel_w[mask, 2]
        )
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
        self.max_abs_roll[mask] = torch.maximum(self.max_abs_roll[mask], roll[mask].abs())
        self.max_abs_pitch[mask] = torch.maximum(self.max_abs_pitch[mask], pitch[mask].abs())

        torque = self.robot.data.applied_torque[:, self.joint_ids]
        self.peak_abs_torque[mask] = torch.maximum(self.peak_abs_torque[mask], torque[mask].abs())
        self.abs_torque_impulse[mask] += torque[mask].abs() * dt
        action = self.env.action_manager.action
        self.mean_action_sum[mask] += action[mask]
        self.peak_abs_action[mask] = torch.maximum(self.peak_abs_action[mask], action[mask].abs())
        wheel_force = torch.linalg.vector_norm(
            self.contact_sensor.data.net_forces_w[:, self.wheel_body_ids], dim=-1
        ).sum(dim=1)
        self.peak_contact_force[mask] = torch.maximum(self.peak_contact_force[mask], wheel_force[mask])
        self.contact_impulse[mask] += wheel_force[mask] * dt

    def _mark_finalize(self, mask: torch.Tensor, reason_code: int):
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
        wheel_contact = self.contact_sensor.data.current_contact_time[:, self.wheel_body_ids] > 0.0
        both_contact = wheel_contact.all(dim=1)
        climbed = self.max_ground_gain >= self.start_success_climb_threshold
        stable = (
            both_contact
            & (roll.abs() <= self.cfg.diagnostics_landing_max_abs_roll)
            & (pitch.abs() <= self.cfg.diagnostics_landing_max_abs_pitch)
        )
        success = climbed & stable & (reason_code == 1)
        self.final_roll[mask] = roll[mask]
        self.final_pitch[mask] = pitch[mask]
        self.final_both_contact[mask] = both_contact[mask]
        self.final_climbed[mask] = climbed[mask]
        self.final_stable[mask] = stable[mask]
        self.final_success[mask] = success[mask]
        self.pending_reason[mask] = reason_code
        self.pending_write |= mask
        self.attempt_active &= ~mask
        self.completed_pulse[mask] = 1.0
        self.success_pulse[mask] = success[mask].float()

    def _flush_pending(self):
        ids = self.pending_write.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return
        mean_action = self.mean_action_sum[ids] / self.sample_count[ids].clamp(min=1.0).unsqueeze(1)

        matrices = {
            "attempt_id": self.attempt_id[ids],
            "env_id": ids,
            "terrain_level": self.start_terrain_level[ids],
            "terrain_type": self.start_terrain_type[ids],
            "terrain_step_height": self.start_terrain_step_height[ids],
            "terrain_step_width": self.start_terrain_step_width[ids],
            "success_climb_threshold": self.start_success_climb_threshold[ids],
            "reason": self.pending_reason[ids],
            "success": self.final_success[ids].float(),
            "climbed": self.final_climbed[ids].float(),
            "stable": self.final_stable[ids].float(),
            "elapsed": self.elapsed[ids],
            "distance_along": self.start_distance_along[ids],
            "distance_normal": self.start_distance_normal[ids],
            "alignment": self.start_alignment[ids],
            "normal": self.start_normal[ids],
            "step_ahead": self.start_step_ahead[ids],
            "start_ground": self.start_ground_z[ids],
            "start_pose": self.start_root_pose[ids],
            "start_velocity": self.start_base_velocity[ids],
            "start_command": self.start_command[ids],
            "start_joint_pos": self.start_joint_pos[ids],
            "start_joint_vel": self.start_joint_vel[ids],
            "start_action": self.start_action[ids],
            "max_ground_gain": self.max_ground_gain[ids],
            "max_base_vz": self.max_base_vz[ids],
            "max_abs_roll": self.max_abs_roll[ids],
            "max_abs_pitch": self.max_abs_pitch[ids],
            "final_roll": self.final_roll[ids],
            "final_pitch": self.final_pitch[ids],
            "both_contact": self.final_both_contact[ids].float(),
            "peak_abs_torque": self.peak_abs_torque[ids],
            "torque_impulse": self.abs_torque_impulse[ids],
            "mean_action": mean_action,
            "peak_abs_action": self.peak_abs_action[ids],
            "peak_contact_force": self.peak_contact_force[ids],
            "contact_impulse": self.contact_impulse[ids],
        }
        cpu = {key: value.detach().cpu() for key, value in matrices.items()}
        for row_index in range(ids.numel()):
            alignment = float(cpu["alignment"][row_index])
            reason = _RESULT_REASONS[int(cpu["reason"][row_index])]
            row = [
                int(cpu["attempt_id"][row_index]),
                int(cpu["env_id"][row_index]),
                int(cpu["terrain_level"][row_index]),
                int(cpu["terrain_type"][row_index]),
                float(cpu["terrain_step_height"][row_index]),
                float(cpu["terrain_step_width"][row_index]),
                float(cpu["success_climb_threshold"][row_index]),
                reason,
                int(cpu["success"][row_index]),
                int(cpu["climbed"][row_index]),
                int(cpu["stable"][row_index]),
                float(cpu["elapsed"][row_index]),
                float(cpu["distance_along"][row_index]),
                float(cpu["distance_normal"][row_index]),
                alignment,
                math.degrees(alignment),
                *cpu["normal"][row_index].tolist(),
                float(cpu["step_ahead"][row_index]),
                float(cpu["start_ground"][row_index]),
                *cpu["start_pose"][row_index].tolist(),
                *cpu["start_velocity"][row_index].tolist(),
                *cpu["start_command"][row_index].tolist(),
            ]
            for joint_index in range(4):
                row.extend(
                    (
                        float(cpu["start_joint_pos"][row_index, joint_index]),
                        float(cpu["start_joint_vel"][row_index, joint_index]),
                    )
                )
            row.extend(cpu["start_action"][row_index].tolist())
            row.extend(
                (
                    float(cpu["max_ground_gain"][row_index]),
                    float(cpu["max_base_vz"][row_index]),
                    float(cpu["max_abs_roll"][row_index]),
                    float(cpu["max_abs_pitch"][row_index]),
                    float(cpu["final_roll"][row_index]),
                    float(cpu["final_pitch"][row_index]),
                    int(cpu["both_contact"][row_index]),
                )
            )
            for joint_index in range(4):
                row.extend(
                    (
                        float(cpu["peak_abs_torque"][row_index, joint_index]),
                        float(cpu["torque_impulse"][row_index, joint_index]),
                    )
                )
            for action_index in range(4):
                row.extend(
                    (
                        float(cpu["mean_action"][row_index, action_index]),
                        float(cpu["peak_abs_action"][row_index, action_index]),
                    )
                )
            row.extend(
                (
                    float(cpu["peak_contact_force"][row_index]),
                    float(cpu["contact_impulse"][row_index]),
                )
            )
            self._writer.writerow(row)
        self._csv_file.flush()
        self.pending_write[ids] = False
        self.pending_reason[ids] = 0
        self._clear(ids)

    def _clear(self, ids: torch.Tensor):
        self.attempt_active[ids] = False
        self.attempt_id[ids] = -1
        self.elapsed[ids] = 0.0
        self.sample_count[ids] = 0.0

    def close(self):
        if self.enabled and hasattr(self, "_csv_file") and not self._csv_file.closed:
            active = self.attempt_active.clone()
            self._mark_finalize(active, reason_code=2)
            self._flush_pending()
            self._csv_file.flush()
            self._csv_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
