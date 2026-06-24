# integral_position_command.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence
from dataclasses import MISSING # MISSING import
import math # for math.pi, if needed in utils
# quat_rotate_inverse를 사용할 경우, isaaclab.utils.math에서 가져와야 합니다.
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat , yaw_quat, quat_apply_inverse

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg # SceneEntityCfg 추가
from isaaclab.utils import configclass
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class IntegralPositionCommand(CommandTerm):
    def __init__(self, cfg: IntegralPositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.robot: RigidObject = env.scene[cfg.asset_name]
        
        # 발 에셋 가져오기
        self.feet_cfg = cfg.feet_cfg 
        self.feet_asset: RigidObject = env.scene[self.feet_cfg.name]
        self.feet_ids = self.feet_cfg.body_ids
        
        # 상태 변수들
        self.global_target_state = torch.zeros(self.num_envs, 4, device=self.device)
        self.virtual_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_error_b = torch.zeros(self.num_envs, 2, device=self.device)
        
        self.velocity_command_name = cfg.velocity_command_name
        self.max_acc = getattr(cfg, "max_acceleration", 1.0) 
        self.turn_threshold = getattr(cfg, "turn_threshold", 0.1) 
        
        # [신규] 하이브리드 가중치 (1.0=Base Only, 0.0=Feet Only, 0.5=Hybrid)
        self.pos_weight = getattr(cfg, "pos_weight", 0.5)

        self.dt = env.step_dt

    @property
    def command(self) -> torch.Tensor:
        return self.pos_error_b

    def _get_hybrid_robot_pos(self, env_ids: slice | Sequence[int] = slice(None)):
        """로봇의 현재 위치를 Base와 Feet의 가중 평균으로 계산"""
        # 1. Base 위치
        base_pos = self.robot.data.root_pos_w[env_ids]
        
        # 2. Feet 중심 위치
        if isinstance(env_ids, slice):
            foot_pos = self.feet_asset.data.body_link_pos_w[:, self.feet_ids, :]
        else:
            foot_pos = self.feet_asset.data.body_link_pos_w[env_ids, self.feet_ids, :]
        
        feet_center = torch.mean(foot_pos, dim=1)
        
        # 3. 가중 평균 (Z축은 Base를 따라가거나 무시)
        hybrid_pos = self.pos_weight * base_pos + (1.0 - self.pos_weight) * feet_center
        
        # Z축은 보통 지형 적응을 위해 Base나 Feet 중 하나를 따르거나 무시하지만,
        # XY 평면 추적이 핵심이므로 XY는 섞고, Z는 Base를 씁니다.
        hybrid_pos[:, 2] = base_pos[:, 2] 
        
        return hybrid_pos

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0: return
        if not isinstance(env_ids, torch.Tensor):
             env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 1. 하이브리드 위치로 가상 타겟 리셋
        start_pos = self._get_hybrid_robot_pos(env_ids)
        
        # 2. Heading은 Base를 따름
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids])
        
        self.global_target_state[env_ids, 0:3] = start_pos
        self.global_target_state[env_ids, 3] = yaw
        
        self.virtual_vel_b[env_ids] = 0.0
        self.pos_error_b[env_ids] = 0.0

    def _update_command(self):
        vel_cmd = self._env.command_manager.get_command(self.velocity_command_name)
        target_v_norm = torch.norm(vel_cmd[:, :2], dim=1)
        target_w_z = vel_cmd[:, 2]
        
        is_turning = torch.abs(target_w_z) > self.turn_threshold
        is_stopped = target_v_norm <= 0.05
        should_sync = is_turning | is_stopped
        
        # 로봇 상태 (하이브리드 위치 사용!)
        current_hybrid_pos = self._get_hybrid_robot_pos()
        _, _, robot_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        
        # --- A. 동기화 (Sync) ---
        if should_sync.any():
            self.global_target_state[should_sync, 0:3] = current_hybrid_pos[should_sync]
            self.global_target_state[should_sync, 3] = robot_yaw[should_sync]
            self.virtual_vel_b[should_sync] = 0.0

        # --- B. 적분 (Integrate) ---
        is_going_straight = ~should_sync
        if is_going_straight.any():
            # 가상 속도 Ramp
            target_v_xy = vel_cmd[is_going_straight, :2]
            current_v_xy = self.virtual_vel_b[is_going_straight, :2]
            diff_v = target_v_xy - current_v_xy
            change_v = torch.clamp(diff_v, -self.max_acc * self.dt, self.max_acc * self.dt)
            self.virtual_vel_b[is_going_straight, :2] += change_v
            
            # 위치 적분
            current_yaw = self.global_target_state[is_going_straight, 3]
            cos_yaw = torch.cos(current_yaw)
            sin_yaw = torch.sin(current_yaw)
            
            v_b_x = self.virtual_vel_b[is_going_straight, 0]
            v_b_y = self.virtual_vel_b[is_going_straight, 1]
            v_w_x = v_b_x * cos_yaw - v_b_y * sin_yaw
            v_w_y = v_b_x * sin_yaw + v_b_y * cos_yaw
            
            self.global_target_state[is_going_straight, 0] += v_w_x * self.dt
            self.global_target_state[is_going_straight, 1] += v_w_y * self.dt

        # --- C. 오차 계산 (Target - Hybrid_Pos) ---
        target_vec_w = self.global_target_state[:, 0:3] - current_hybrid_pos
        
        robot_yaw_quat = yaw_quat(self.robot.data.root_quat_w)
        from isaaclab.utils.math import quat_apply_inverse
        target_vec_b = quat_apply_inverse(robot_yaw_quat, target_vec_w)
        
        self.pos_error_b[:, 0] = target_vec_b[:, 0]
        self.pos_error_b[:, 1] = target_vec_b[:, 1]

    def _update_metrics(self):
        error_xy = torch.norm(self.pos_error_b, dim=1)
        self.metrics["position_error"] = error_xy
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vis"):
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import VisualizationMarkersCfg
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/integral_goal",
                    markers={
                        "far": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                        "near": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
                    }
                )
                self.goal_vis = VisualizationMarkers(marker_cfg)
            self.goal_vis.set_visibility(True)
        else:
            if hasattr(self, "goal_vis"):
                self.goal_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "goal_vis"): return
        pos = self.global_target_state[:, 0:3].clone()
        pos[:, 2] += 0.5 
        current_hybrid_pos = self._get_hybrid_robot_pos()
        dist = torch.norm(pos[:, :2] - current_hybrid_pos[:, :2], dim=1)
        marker_indices = torch.where(dist > 0.5, 0, 1)
        self.goal_vis.visualize(translations=pos, marker_indices=marker_indices)


@configclass
class IntegralPositionCommandCfg(CommandTermCfg):
    class_type: type = IntegralPositionCommand
    asset_name: str = "robot" 
    velocity_command_name: str = "base_velocity"
    
    max_acceleration: float = 3.0 
    turn_threshold: float = 0.1
    
    feet_cfg: SceneEntityCfg = MISSING
    
    # [신규] 하이브리드 가중치 (0.5 권장)
    # 1.0에 가까울수록 몸통 추종 (피칭 심함)
    # 0.0에 가까울수록 발 추종 (추진력 약함)
    pos_weight: float = 0.5