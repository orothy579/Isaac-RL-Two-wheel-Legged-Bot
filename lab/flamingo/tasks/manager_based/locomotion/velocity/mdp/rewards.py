# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from lab.flamingo.tasks.manager_based.locomotion.velocity.sensors import LiftMask
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
import torch.nn.functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Velocity-tracking rewards.
"""


def _decode_base_velocity_pos_z_command(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Map the normalized z command in [0, 1] back to the configured physical height range."""
    command_pos_z_norm = env.command_manager.get_command(command_name)[:, 3]
    command_term = env.command_manager.get_term(command_name)
    z_min, z_max = command_term.cfg.ranges.pos_z

    if z_max <= z_min:
        return torch.full_like(command_pos_z_norm, z_min)

    return z_min + command_pos_z_norm * (z_max - z_min)


def _get_event_command(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """Fetch the event command, inferring the common command name when omitted."""
    candidate_names = [command_name] if command_name is not None else ["yk_jump_command", "event"]

    for candidate_name in candidate_names:
        try:
            return env.command_manager.get_command(candidate_name)
        except Exception:
            continue

    raise ValueError(
        "No event command found. Expected one of ['yk_jump_command', 'event'] "
        "or pass an explicit command_name."
    )


def track_lin_vel_xy_link_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_xy_link_exp_cmd(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 1e-6,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) only when cmd[3] is non-zero."""
    asset: RigidObject = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)
    is_cmd3_nonzero = torch.abs(cmd[:, 3]) > cmd_threshold

    lin_vel_error = torch.sum(
        torch.square(cmd[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )

    reward = torch.exp(-lin_vel_error / std**2)
    return reward * is_cmd3_nonzero.float()

def track_lin_vel_xy_still_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # commands: (N, >=2)
    cmd = env.command_manager.get_command(command_name)

    # x_cmd: abs(x_cmd) <= 0.2 이면 0으로 매핑
    x_cmd = cmd[:, 0]
    x_cmd = torch.where(torch.abs(x_cmd) <= 0.5, torch.zeros_like(x_cmd), x_cmd)

    # 수정된 x_cmd를 반영한 (x, y) 커맨드 구성
    cmd_xy = torch.stack((x_cmd, cmd[:, 1]), dim=1)

    lin_vel_error = torch.sum(
        torch.square(cmd_xy - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_link_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def track_ang_vel_z_link_exp_cmd(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 1e-6,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) only when cmd[3] is non-zero."""
    asset: RigidObject = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)
    is_cmd3_nonzero = torch.abs(cmd[:, 3]) > cmd_threshold

    ang_vel_error = torch.square(
        cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    )

    reward = torch.exp(-ang_vel_error / std**2)
    return reward * is_cmd3_nonzero.float()

def error_track_pos_integral(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "integral_position", # 새로 만든 커맨드 이름
    kernel: str = "tanh",
    delta : float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize accumulated XY position error from an integral-position command.

    The integral command already computes target-minus-robot XY error in the
    robot/body frame. This term turns that 2-D error into one scalar penalty per
    environment so it can be used with a negative reward weight.
    """
    del asset_cfg  # kept for config compatibility with other reward terms

    pos_error_xy = env.command_manager.get_command(command_name)[:, :2]

    if delta > 0.0:
        error = F.huber_loss(
            pos_error_xy,
            torch.zeros_like(pos_error_xy),
            reduction="none",
            delta=delta,
        ).sum(dim=1)
    else:
        error = torch.linalg.norm(pos_error_xy, dim=1)

    return apply_kernel(error, kernel, scale)

def apply_kernel(error: torch.Tensor, kernel: str = "linear", scale: float = 1.0, temperature : float = 4.0) -> torch.Tensor:
    if kernel == "linear":
        return error * scale
    elif kernel == "tanh":
        return torch.tanh(error * temperature) * scale
    elif kernel == "exp":
        return (1.0 - torch.exp(-error * temperature)) * scale
    else:
        return error * scale
    
def lin_vel_z_link_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])

def ang_vel_xy_link_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)

def ang_vel_z_link_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_ang_vel_b[:, 2])


def uncommanded_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_cmd_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize yaw rotation only when the yaw command is close to zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_cmd = env.command_manager.get_command(command_name)[:, 2]
    no_yaw_cmd = torch.abs(yaw_cmd) <= yaw_cmd_threshold
    return torch.square(asset.data.root_link_ang_vel_b[:, 2]) * no_yaw_cmd.float()


def track_pos_z_exp(
    env: ManagerBasedRLEnv,
    temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,

) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current z position of the robot's base
    current_pos_z = asset.data.root_link_pos_w[:, 2]

    # Get the command z position relative to wheels from the command manager
    command_pos_z = env.command_manager.get_command("base_velocity")[:, 3]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = command_pos_z + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = command_pos_z

    # Compute the error between the current height difference and the commanded height difference
    pos_z_error = torch.square(adjusted_target_height - current_pos_z)

    return torch.exp(-pos_z_error * temperature)
'''
def track_pos_z_exp_v2(
    env: ManagerBasedRLEnv,
    temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,

) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current z position of the robot's base
    current_pos_z = asset.data.root_link_pos_w[:, 2]

    # Get the command z position relative to wheels from the command manager
    command_pos_z = _decode_base_velocity_pos_z_command(env, "base_velocity")

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = command_pos_z + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = command_pos_z

    # Compute the error between the current height difference and the commanded height difference
    pos_z_error = torch.abs(adjusted_target_height - current_pos_z)

    return torch.exp(-pos_z_error * temperature)
'''

def track_base_roll_pitch_exp(
    env: ManagerBasedRLEnv,
    temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi

    # Get the command z position relative to wheels from the command manager
    command = env.command_manager.get_command("roll_pitch")

    # Compute the error between the current height difference and the commanded height difference
    position_error = torch.norm(torch.square(command - torch.stack((roll, pitch), dim=1)), dim = 1)

    return torch.exp(-position_error * temperature)

def flat_euler_angle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi

    rp = torch.stack((roll, pitch), dim=-1)
    return torch.sum(torch.square(rp), dim=1)
    
def flat_euler_angle_exp(env: ManagerBasedRLEnv, temperature: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw).
    
     torch.exp(-temperature *  torch.sum(torch.abs(self.base_euler[:2]), dim=0))
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi

    rp = torch.stack((roll, pitch), dim=-1)
    return torch.exp(-temperature * torch.sum(torch.abs(rp), dim=1))

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def safe_landing_motion(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function to minimize air time and encourage smooth landings by ensuring wheel contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check contact force to determine if wheels are touching the ground
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.1

    # Force minimization reward: penalize higher forces to encourage smooth landing
    force_magnitude = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    total_landing_force = torch.sum(force_magnitude, dim=-1)  # Sum over all contact points
    force_minimization_reward = total_landing_force[:, -1] # Encourage lower landing forces

    return force_minimization_reward

def feet_air_time_positive_biped(env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > threshold
    return reward


def feet_air_time_positive_biped_forward(env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward swing completion events for bipeds only while moving forward.

    This prevents the agent from collecting air-time reward while stalling or rocking backward on a step.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene["robot"]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > 0.1
    forward_progress = asset.data.root_link_lin_vel_b[:, 0] > 0.05

    reward = torch.sum(torch.clamp(last_air_time, max=threshold) * first_contact, dim=1)
    reward *= single_stance.float()
    reward *= (forward_cmd & forward_progress).float()
    return reward

def feet_air_time_lift_mask(env: ManagerBasedRLEnv,
                            sensor_cfg: SceneEntityCfg,
                            mask_sensor_cfg_left: SceneEntityCfg,
                            mask_sensor_cfg_right: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds, encouraging locomotion when lift masks are active.

    This function rewards the agent for taking steps up to a specified threshold and also keeps one foot at
    a time in the air. Rewards are further enhanced when lift masks for the left or right foot are active.

    If the commands are small (i.e., the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # Lift mask sensors
    left_lift_mask = left_mask_sensor.data.mask
    right_lift_mask = right_mask_sensor.data.mask

    # Compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    # Base reward for alternating stance and air time
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # reward = torch.clamp(reward, max=threshold)

    # Apply lift mask to enhance rewards for locomotion
    lift_mask_bonus = left_lift_mask + right_lift_mask  # Shape: [N]
    reward *= lift_mask_bonus  # Amplify reward if lift masks are active

    # No reward for zero command
    reward *= torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1

    return reward


def both_feet_contact_on_stair_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    mask_sensor_cfg_left: SceneEntityCfg,
    mask_sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
    forward_cmd_threshold: float = 0.1,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize keeping both feet in contact while commanded forward and a stair is detected."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    command = env.command_manager.get_command(command_name)
    forward_cmd = command[:, 0] > forward_cmd_threshold

    stair_detected = torch.logical_or(
        left_mask_sensor.data.mask.bool(),
        right_mask_sensor.data.mask.bool(),
    )

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_contact_forces = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0]
    both_feet_contact = torch.all(max_contact_forces > contact_threshold, dim=1)

    return (forward_cmd & stair_detected & both_feet_contact).float()


def foot_clearance_lift_mask(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg_left: SceneEntityCfg,
    height_sensor_cfg_right: SceneEntityCfg,
    mask_sensor_cfg_left: SceneEntityCfg,
    mask_sensor_cfg_right: SceneEntityCfg,
    target_height: float = 0.3,
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a dynamically calculated target height off the ground,
    with separate rewards for left and right feet based on lift_mask.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        asset_cfg (SceneEntityCfg): Configuration for the asset (feet).
        sensor_cfg_left (SceneEntityCfg): Configuration for the left foot sensor.
        sensor_cfg_right (SceneEntityCfg): Configuration for the right foot sensor.
        target_height (float): Target height for foot clearance relative to the sensor.
        tanh_mult (float): Multiplication factor for velocity term in tanh.

    Returns:
        torch.Tensor: Reward values for the current step.
    """
    # Extract asset and sensor data
    asset: RigidObject = env.scene[asset_cfg.name]
    left_height_sensor: RayCaster = env.scene[height_sensor_cfg_left.name]
    right_height_sensor: RayCaster = env.scene[height_sensor_cfg_right.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # Lift mask sensor
    left_lift_mask = left_mask_sensor.data.mask
    right_lift_mask = right_mask_sensor.data.mask

    # Compute dynamic target heights (sensor-based)
    dynamic_target_height_left = target_height + torch.mean(left_height_sensor.data.ray_hits_w[..., 2], dim=1)  # Shape: [N]
    dynamic_target_height_right = target_height + torch.mean(right_height_sensor.data.ray_hits_w[..., 2], dim=1)  # Shape: [N]

    # Compute foot height errors
    left_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]  # Left foot height
    right_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[1], 2]  # Right foot height

    left_foot_error = torch.square(left_foot_height - dynamic_target_height_left)  # Shape: [N]
    right_foot_error = torch.square(right_foot_height - dynamic_target_height_right)  # Shape: [N]

    # left_foot_reward = torch.exp(-1.5 * left_foot_error)  # Shape: [N]
    # right_foot_reward = torch.exp(-1.5 * right_foot_error)  # Shape: [N]

    # Combine rewards for left and right feet
    total_reward = left_lift_mask * left_foot_error + right_lift_mask * right_foot_error  # Shape: [N]

    # No reward for zero command
    total_reward *= torch.norm(env.command_manager.get_command("base_velocity")[:, :4], dim=1) > 0.1

    return total_reward


class Trajectory_reward(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg,
                 env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.left_height_sensor: RayCaster = env.scene[cfg.params["height_sensor_cfg_left"].name]
        self.right_height_sensor: RayCaster = env.scene[cfg.params["height_sensor_cfg_right"].name]
        self.left_mask_sensor: LiftMask = env.scene[cfg.params["mask_sensor_cfg_left"].name]
        self.right_mask_sensor: LiftMask = env.scene[cfg.params["mask_sensor_cfg_right"].name]

        self.rad_2_deg = 57.2958
        self.phi = torch.pi / 2

        """
            from base coordinate
        """
        # base to hip
        self.left_joint1 = np.array([-0.02305, 0.08, 0.034], dtype=np.float64)
        self.right_joint1 = np.array([-0.02305, -0.08, 0.034], dtype=np.float64)
        # hip to shoulder
        self.left_joint2 = np.array([-0.083025, 0.08, 0.034], dtype=np.float64)
        self.right_joint2 = np.array([-0.083025, -0.08, 0.034], dtype=np.float64)
        # leg lengths
        self.l_1 = np.float64(0.183551 - 0.08)
        self.l_2 = np.sqrt((-0.083025 + 0.221034) ** 2 + (0.034 + 0.137321) ** 2)
        self.l_3 = np.sqrt((-0.221034 + 0.0569727) ** 2 + (-0.137321 + 0.28389) ** 2)
        self.l_4 = np.float64(0.24355 - 0.183551)

        # absolute foot point
        x = 0.1
        y = self.l_1
        z = 0.35
        v_x = 0.15
        v_y = 0.05
        v_z = 0.3


        self.start_point = np.array([[-x, y, -z],
                                     [x/10, -y, -z]], dtype=np.float64)
        self.end_point = np.array([[x/10, y, -z],
                                   [-x, -y, -z]], dtype=np.float64)
        self.start_vel1 = np.array([[v_x, -v_y, v_z/5 ],
                                    [-v_x, -v_y, 0 ]], dtype=np.float64)
        self.end_vel1 = np.array([[-v_x , v_y, -v_z / 10],
                                  [-v_x , v_y, 0]], dtype=np.float64)
        self.start_vel2 = np.array([[-v_x, v_y, 0],
                                    [v_x, v_y, v_z /5 ]], dtype=np.float64)
        self.end_vel2 = np.array([[-v_x , -v_y, 0],
                                  [-v_x , -v_y, -v_z / 10]], dtype=np.float64)


        self.T = 1
        self.async_flag = 0
        self.is_left = 1
        self.is_right = 1 - self.is_left

        #print(env.step_dt) : 0.02

        # 0.005
        self.dt = env.physics_dt
        self.step = 0
        self.last_step = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        height_sensor_cfg_left: SceneEntityCfg,
        height_sensor_cfg_right: SceneEntityCfg,
        mask_sensor_cfg_left: SceneEntityCfg,
        mask_sensor_cfg_right: SceneEntityCfg,
    ) -> torch.Tensor:

        # Height sensor
        left_height = torch.mean(self.left_height_sensor.data.ray_hits_w[..., 2], dim=1)
        right_height = torch.mean(self.right_height_sensor.data.ray_hits_w[..., 2], dim=1)
        # Lift mask sensor
        left_lift_mask = self.left_mask_sensor.data.mask
        right_lift_mask = self.right_mask_sensor.data.mask


        # [4096]
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        # [4096, 6]

        current_joint_pos = self.asset.data.joint_pos[:, asset_cfg.joint_ids]

        self.step = env.episode_length_buf
        t = self.step * self.dt #* env.decimation

        mask2 = t < 2 * self.T
        t[~mask2] = 0
        mask = t < self.T  # mask: [4096], True/False
        t[~mask] -= self.T
        t_normalized = t / self.T

        left_P = torch.empty((t.shape[0], 3), dtype=torch.float32, device=t.device)  # [4096, 3]
        right_P = torch.empty((t.shape[0], 3), dtype=torch.float32, device=t.device)  # [4096, 3]

        # t < self.T
        left_P[mask] = self.cubic_hermite_spline(
            self.start_point[0], self.end_point[0], self.start_vel1[0], self.end_vel1[0], t_normalized[mask]
        ).to(left_P.dtype)

        # t >= self.T
        left_P[~mask] = self.cubic_hermite_spline(
            self.end_point[0], self.start_point[0], self.start_vel2[0], self.end_vel2[0], t_normalized[~mask]
        ).to(left_P.dtype)

        # t < self.T
        right_P[mask] = self.cubic_hermite_spline(
            self.start_point[1], self.end_point[1], self.start_vel1[1], self.end_vel1[1], t_normalized[mask]
        ).to(right_P.dtype)

        # t >= self.T
        right_P[~mask] = self.cubic_hermite_spline(
            self.end_point[1], self.start_point[1], self.start_vel2[1], self.end_vel2[1], t_normalized[~mask]
        ).to(right_P.dtype)

        left_th1, left_th2, left_th3 = self.IK_3dof_leg(left_P[:,0], left_P[:,1], left_P[:,2], self.is_left)
        right_th1, right_th2, right_th3 = self.IK_3dof_leg(right_P[:,0], right_P[:,1], right_P[:,2],self.is_right)
        # compensate init joint pos
        left_th2 -= torch.pi/4
        left_th3 += torch.pi/2
        right_th2 -= torch.pi/4
        right_th3 += torch.pi/2

        left_target_joint_pos = torch.stack([left_th1, left_th2, left_th3], dim = 1).to(current_joint_pos.device)
        right_target_joint_pos = torch.stack([right_th1, right_th2, right_th3], dim = 1).to(current_joint_pos.device)

        left_error_joint_pos = torch.square(current_joint_pos[:, :3] - left_target_joint_pos)
        right_error_joint_pos = torch.square(current_joint_pos[:, 3:] - right_target_joint_pos)

        left_reward = right_lift_mask * torch.exp(-torch.sum(left_error_joint_pos, dim=1) / 0.25**2)
        right_reward = left_lift_mask * torch.exp(-torch.sum(right_error_joint_pos, dim=1) / 0.25**2)

        reward = left_reward + right_reward

        return reward

    def cubic_hermite_spline(self,
                             A,
                             D,
                             U,
                             V,
                             t):
        """
            A : start point
            D : end point
            U : start velocity
            V : end velocity
            U = 3*(B-A)
            V = 3*(D-C)
        """

        t = t.unsqueeze(-1)
        A = torch.from_numpy(A).to(t.device)
        D = torch.from_numpy(D).to(t.device)
        U = torch.from_numpy(U).to(t.device)
        V = torch.from_numpy(V).to(t.device)

        h00 = (2 * t ** 3) - (3 * t ** 2) + 1
        h10 = t ** 3 - (2 * t ** 2) + t
        h01 = (-2 * t ** 3) + (3 * t ** 2)
        h11 = t ** 3 - t ** 2

        return (
                h00 * A + h10 * U +
                h01 * D + h11 * V
        )


    def Rotation_X(self, theta):
        # rotation matrix
        theta = theta.to(dtype=torch.float64)  # float64

        # cos(theta), sin(theta)
        cos_theta = torch.cos(theta)  # [4096]
        sin_theta = torch.sin(theta)  # [4096]

        # row to batch
        R_x = torch.zeros((theta.shape[0], 3, 3), dtype=torch.float64, device=theta.device)  # [4096, 3, 3]

        R_x[:, 0, 0] = 1.0  # first row
        R_x[:, 1, 1] = cos_theta  # second row
        R_x[:, 1, 2] = -sin_theta
        R_x[:, 2, 1] = sin_theta
        R_x[:, 2, 2] = cos_theta

        return R_x

    def IK_3dof_leg(self, x, y, z, is_left: bool):

        """
            z-y plane
        """
        # [num_envs]
        d3 = torch.sqrt(y ** 2 + z ** 2)

        gamma_2 = torch.arcsin((self.l_1 / d3)) #* torch.sin(self.phi))=1
        gamma_3 = torch.pi - gamma_2 - self.phi
        gamma_1 = torch.arctan2(z, y)

        """
            x-z' plane j2, j4
        """

        if is_left == 1:
            theta_1 = -(gamma_3 + gamma_1)
            R = -theta_1 + self.phi - torch.pi / 2
            c = torch.stack([
                torch.full((theta_1.shape[0],), 0.0, dtype=torch.float64, device=theta_1.device),
                self.l_1 * torch.cos(-theta_1),
                self.l_1 * torch.sin(-theta_1)], dim=1)
            j2 = torch.tensor(self.left_joint2).unsqueeze(0).to(theta_1.device) + c
        else:
            theta_1 = gamma_3 - (torch.pi + gamma_1)
            R = theta_1 - self.phi + torch.pi / 2
            c = torch.stack([
                torch.full((theta_1.shape[0],), 0.0, dtype=torch.float64, device=theta_1.device)
                , -self.l_1 * torch.cos(theta_1),
                self.l_1 * torch.sin(theta_1)], dim=1 )
            j2 = torch.tensor(self.right_joint2).unsqueeze(0).to(theta_1.device) + c

        j4 = torch.stack([x, y, z], dim=1)
        # [num_envs, 3]
        j4_2_vec = j4 - j2

        # [num_envs, 3, 1]
        p_2 = torch.matmul(self.Rotation_X(R), j4_2_vec.unsqueeze(-1))

        x_2, z_2 = p_2[:,0], p_2[:,2]
        # [num_envs]
        x_2 = x_2.squeeze(1)
        z_2 = z_2.squeeze(1)

        theta_3 = torch.arccos((x_2 ** 2 + z_2 ** 2 - self.l_2 ** 2 - self.l_3 ** 2) / (2 * self.l_2 * self.l_3)) - torch.pi

        alpha = torch.arctan(self.l_3 * torch.sin(abs(theta_3)) / (self.l_2 + self.l_3 * torch.cos(abs(theta_3))))
        beta = torch.arctan2(z_2, -x_2) + torch.pi / 2
        theta_2 = alpha + beta

        return theta_1, theta_2, theta_3

def adaptive_terrain_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg_left: SceneEntityCfg,
    height_sensor_cfg_right: SceneEntityCfg,
    mask_sensor_cfg_left: SceneEntityCfg,
    mask_sensor_cfg_right: SceneEntityCfg,
    target_clearance: float = 0.3,
    clearance_margin: float = 0.05,
    smoothness_penalty_weight: float = 0.1,
    success_reward: float = 10.0,
) -> torch.Tensor:
    """
    Reward function for adaptive locomotion to handle stairs and flat terrain.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        asset_cfg (SceneEntityCfg): Configuration for the asset (feet and wheels).
        height_sensor_cfg_left (SceneEntityCfg): Configuration for the left foot sensor.
        height_sensor_cfg_right (SceneEntityCfg): Configuration for the right foot sensor.
        mask_sensor_cfg_left (SceneEntityCfg): Configuration for the left lift mask sensor.
        mask_sensor_cfg_right (SceneEntityCfg): Configuration for the right lift mask sensor.
        target_clearance (float): Desired height clearance for stairs.
        clearance_margin (float): Allowable error margin for clearance.
        smoothness_penalty_weight (float): Weight for penalizing erratic motions.
        success_reward (float): Reward for successfully clearing an obstacle.

    Returns:
        torch.Tensor: Reward values for the current step.
    """
    # Extract asset and sensor data
    asset: RigidObject = env.scene[asset_cfg.name]
    left_height_sensor: RigidObject = env.scene[height_sensor_cfg_left.name]
    right_height_sensor: RigidObject = env.scene[height_sensor_cfg_right.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # Lift mask sensor
    left_lift_mask = left_mask_sensor.data.mask
    right_lift_mask = right_mask_sensor.data.mask

    # Calculate dynamic target heights
    dynamic_target_height_left = target_clearance + torch.mean(left_height_sensor.data.ray_hits_w[..., 2], dim=1)
    dynamic_target_height_right = target_clearance + torch.mean(right_height_sensor.data.ray_hits_w[..., 2], dim=1)

    # Calculate foot height errors
    left_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]
    right_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[1], 2]

    left_foot_error = torch.abs(left_foot_height - dynamic_target_height_left)
    right_foot_error = torch.abs(right_foot_height - dynamic_target_height_right)

    # Reward for maintaining clearance within margin
    left_clearance_reward = torch.exp(-((left_foot_error - clearance_margin) ** 2))
    right_clearance_reward = torch.exp(-((right_foot_error - clearance_margin) ** 2))

    # Smoothness penalty for erratic z-velocity
    left_velocity_z = torch.abs(asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], 2])
    right_velocity_z = torch.abs(asset.data.body_lin_vel_w[:, asset_cfg.body_ids[1], 2])
    smoothness_penalty = smoothness_penalty_weight * (left_velocity_z + right_velocity_z)

    # Lift mask activation bonus (encourage proper activation)
    lift_mask_bonus = left_lift_mask * left_clearance_reward + right_lift_mask * right_clearance_reward

    # Success reward for clearing the obstacle
    clearance_success = (
        (left_foot_error < clearance_margin) & (right_foot_error < clearance_margin)
    ).float()
    success_bonus = clearance_success * success_reward

    # Combine rewards
    total_reward = lift_mask_bonus + success_bonus - smoothness_penalty

    return total_reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.exp(-body_vel.norm(dim=-1) * contacts)
    reward = torch.sum(reward, dim=1)  # Sum over both feet
    return reward

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive, i.e., not falling over."""
    return torch.ones(env.num_envs, device=env.device)

def reward_same_foot_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize X-axis displacement difference of two feet in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_apply_inverse(base_quat, foot_base[:,i,:])
    dx = foot_base[:,0,0] - foot_base[:,1,0]
    return torch.abs(dx)  # penalize both feet being too far apart and too close together


def action_rate_clip_l2(
    env: ManagerBasedRLEnv,
    max_delta: float = 10.0,
    max_penalty: float = 100.0,
) -> torch.Tensor:
    """Penalize bounded action-rate changes with final penalty clipping."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    action = torch.nan_to_num(action, nan=0.0, posinf=max_delta, neginf=-max_delta)
    prev_action = torch.nan_to_num(prev_action, nan=0.0, posinf=max_delta, neginf=-max_delta)

    action_diff = torch.clamp(action - prev_action, min=-max_delta, max=max_delta)
    penalty = torch.mean(torch.square(action_diff), dim=1)

    return torch.clamp(penalty, max=max_penalty)

def reward_same_shoulder_z_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize Z-axis displacement difference of two shoulders in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    shoulder_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    shoulder_base = shoulder_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        shoulder_base[:,i,:] = quat_apply_inverse(base_quat, shoulder_base[:,i,:])
    dz = shoulder_base[:,0,2] - shoulder_base[:,1,2]
    return torch.abs(dz)  # penalize both shoulders being too far apart and too close together

def diff_action_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """왼/오른쪽 숄더 관절의 '액션' 차이에 대한 L2 패널티.

    기대사항:
      - asset_cfg.joint_names 에는 정확히 두 관절(좌/우 숄더)이 들어와야 함
        예: ["left_shoulder_joint", "right_shoulder_joint"]
      - env.action_manager.action 의 스케일/범위는 상위에서 정규화되어 있다고 가정

    Returns:
      (N,) 텐서. 각 환경별 (left_action - right_action)^2
    """
    # 선택된 두 관절의 인덱스
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or len(joint_ids) != 2:
        raise RuntimeError(
            "diff_action_penalty: asset_cfg.joint_names(또는 joint_ids)로 정확히 두 관절을 지정하세요 "
            "(예: left_shoulder_joint, right_shoulder_joint)."
        )

    # 현재 스텝의 액션에서 두 관절 성분만 추출: shape (N, 2)
    pair_actions = env.action_manager.action[:, joint_ids]

    # 좌/우 액션 차이의 제곱 (L2)
    diff = pair_actions[:, 0] - pair_actions[:, 1]
    return torch.square(diff)

class FlamingoAirTimeReward(ManagerTermBase):
    """Reward for longer feet air and contact time with stuck detection and reward for locomotion."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.stuck_threshold: float = cfg.params.get("stuck_threshold", 0.1)
        self.stuck_duration: int = cfg.params.get("stuck_duration", 5)
        self.threshold: float = cfg.params.get("threshold", 0.2)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.stuck_counter = torch.zeros(self.asset.data.root_lin_vel_b.shape[0], device=self.asset.device)

        if not self.contact_sensor.cfg.track_air_time:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stuck_threshold: float,
        stuck_duration: int,
        threshold: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward calculates the air-time for the feet and applies a reward when the robot is stuck.

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # Extract the necessary sensor data
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

        # Compute the base movement command and its progress
        base_velocity_tensor = env.command_manager.get_command("base_velocity")[:, :3]
        progress = torch.norm(base_velocity_tensor - self.asset.data.root_lin_vel_b, dim=1)
        is_stuck = progress > self.stuck_threshold  # Detect lack of progress

        # Manage the stuck counter and determine stuck status
        self.stuck_counter = torch.where(is_stuck, self.stuck_counter + 1, torch.zeros_like(self.stuck_counter))
        stuck = self.stuck_counter >= self.stuck_duration
        stuck = stuck.unsqueeze(1)

        # Compute the reward based on air time and first contact when stuck
        stuck_air_time_reward = torch.sum((last_air_time - self.threshold) * first_contact * stuck.float(), dim=1)
        # Ensure no reward is given if there is no movement command
        stuck_air_time_reward *= torch.norm(base_velocity_tensor[:, :2], dim=1) > 0.1

        # # Foot clearance reward
        # foot_z_target_error = torch.square(self.asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - self.target_height)
        # foot_velocity_tanh = torch.tanh(
        #     tanh_mult * torch.norm(self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
        # )
        # foot_clearance_reward = (
        #     torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std) * stuck.float()
        # )

        # Final reward: Encourage lifting legs when stuck
        reward = stuck_air_time_reward  # + foot_clearance_reward

        return reward


def stand_origin_base(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity on x or y when the command is zero, encouraging the robot to stand still."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate linear and angular velocity errors
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel), dim=1)

    ang_vel = asset.data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(ang_vel)

    # Penalize the linear and angular velocity errors
    velocity_penalty = lin_vel_error + ang_vel_error

    # Calculate deviation from origin position
    current_pos = asset.data.root_pos_w[:, :2]
    position_error = torch.sum(torch.square(current_pos - env.scene.env_origins[:, :2]), dim=1)

    # Penalize the deviation from the origin position
    position_penalty = position_error

    # Apply the penalty only when the command is zero
    penalty = (velocity_penalty + position_penalty) * is_zero_command.float()

    return penalty


def stand_still_base(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity on x or y when the command is zero, encouraging the robot to stand still."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :2]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate linear and angular velocity errors
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel), dim=1)

    # Penalize the linear and angular velocity errors
    velocity_penalty = (lin_vel_error) / std**2
    return velocity_penalty * is_zero_command.float()


def stand_still(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for standing still when the command is zero, penalizing movement, especially backward movement."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :3]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate wheel velocity error
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    wheel_vel_error = torch.sum(torch.abs(wheel_vel), dim=1)

    # Penalize backward movement by adding a higher penalty for negative velocities
    # backward_movement_penalty = torch.sum(torch.clamp(wheel_vel, max=0), dim=1)

    # Calculate the reward
    reward = wheel_vel_error / std**2

    # Make sure to only give non-zero reward where command is zero
    reward = reward * is_zero_command.float()

    return reward


def joint_align_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    """Penalize joint mis-alignments.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    if cmd_threshold != -1.0:
        mis_aligned = torch.where(
            cmd <= cmd_threshold,
            torch.abs(
                asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
            ),
            torch.tensor(0.0),
        )
    else:
        mis_aligned = torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
        )

    return mis_aligned


def joint_soft_pos_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] * soft_ratio
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1] * soft_ratio
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_smoothness_hard(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using smoothing term."""
    sm1 = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    sm2 = torch.sum(
        torch.square(env.action_manager.action + env.action_manager.prev_action - 2 * env.action_manager.prev2_action),
        dim=1,
    )
    sm3 = 0.05 * torch.sum(torch.abs(env.action_manager.action), dim=1)

    return sm1 + sm2 + sm3


def force_action_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = -1.0,
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    if cmd_threshold != -1.0 or velocity_threshold != -1.0:
        force_action_zero = torch.where(
            torch.logical_or(cmd.unsqueeze(1) <= cmd_threshold, body_vel.unsqueeze(1) <= velocity_threshold),
            torch.tensor(0.0),
            torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]),
        )
    else:
        force_action_zero = torch.abs(env.action_manager.action[:, asset_cfg.joint_ids])
    return torch.sum(force_action_zero, dim=1)


def cliped_joint_applied_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float = 10.0,
) -> torch.Tensor:
    """Penalize applied torque-limit excess with a bounded per-step penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1).clamp(max=max_penalty)


def clipped_joint_applied_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float = 10.0,
) -> torch.Tensor:
    return cliped_joint_applied_torque_limits(env, asset_cfg, max_penalty)


def cliped_action_rate_l2(env: ManagerBasedRLEnv, max_penalty: float = 10.0) -> torch.Tensor:
    """Penalize action rate with a bounded per-step L2 penalty."""
    action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return action_rate.clamp(max=max_penalty)


def clipped_action_rate_l2(env: ManagerBasedRLEnv, max_penalty: float = 10.0) -> torch.Tensor:
    return cliped_action_rate_l2(env, max_penalty)


def base_height_adaptive_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.clip(torch.mean(sensor.data.ray_hits_w[..., 2], dim=1), -10.0, 10.0)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)


def base_height_adaptive_events_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    event_target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from one of two targets selected by the event command.

    When ``event_command[:, 0]`` is 0, the reward tracks ``target_height``.
    When ``event_command[:, 0]`` is 1, the reward tracks ``event_target_height``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    event_cmd = _get_event_command(env)
    use_event_target = event_cmd[:, 0] > 0.5

    selected_target_height = torch.where(
        use_event_target,
        torch.full_like(asset.data.root_link_pos_w[:, 2], event_target_height),
        torch.full_like(asset.data.root_link_pos_w[:, 2], target_height),
    )

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        terrain_height = torch.clip(torch.mean(sensor.data.ray_hits_w[..., 2], dim=1), -10.0, 10.0)
        adjusted_target_height = selected_target_height + terrain_height
    else:
        adjusted_target_height = selected_target_height

    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)

def shoulder_action_zero_when_event_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
) -> torch.Tensor:
    """Penalize shoulder actions only when event command is 0.

    Expected:
      - asset_cfg.joint_names or joint_ids should point to shoulder joints
        e.g. ".*_shoulder_joint"
    Returns:
      Tensor of shape (N,): sum of squared shoulder actions when event == 0, else 0
    """
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or len(joint_ids) == 0:
        raise RuntimeError(
            "shoulder_action_zero_when_event_zero: "
            "asset_cfg.joint_names(or joint_ids) must specify shoulder joints."
        )

    # current policy action for selected shoulder joints: (N, num_shoulders)
    shoulder_actions = env.action_manager.get_term("joint_pos").raw_actions

    # event == 0 인 경우만 패널티 적용
    event_cmd = _get_event_command(env, command_name)
    is_event_zero = event_cmd[:, 0] < 0.5

    penalty = torch.sum(torch.square(shoulder_actions), dim=1)
    return penalty * is_event_zero.float()

def low_posture_speed_penalty(
    env: ManagerBasedRLEnv,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    event_command_name: str | None = None,
) -> torch.Tensor:
    """Penalize excessive base xy speed only when event == 0 (low posture)."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # event == 0 -> low posture mode
    event_cmd = _get_event_command(env, event_command_name)
    is_low_posture = event_cmd[:, 0] < 0.5

    # actual base xy speed
    speed_xy = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)

    # penalty only for speed above threshold
    excess_speed = torch.clamp(speed_xy - max_speed, min=0.0)

    return torch.square(excess_speed) * is_low_posture.float()

def low_posture_wheel_speed_penalty(
    env: ManagerBasedRLEnv,
    max_wheel_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    event_command_name: str | None = None,
) -> torch.Tensor:
    """Penalize excessive wheel joint speed only when event == 0."""
    asset: Articulation = env.scene[asset_cfg.name]

    event_cmd = _get_event_command(env, event_command_name)
    is_low_posture = event_cmd[:, 0] < 0.5

    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(wheel_vel) - max_wheel_vel, min=0.0)

    return torch.sum(torch.square(excess), dim=1) * is_low_posture.float()

def base_target_range_height_v2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        min_target_height=0.33126,
        max_target_height=0.37126,
        minimum_height=0.2607,
        sharpness=2.0,
    ):
    # Ensure reward is zero when current height is at or below the minimum height
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:,2]

    # print(current_height)
    reward = torch.where(min_target_height <= current_height , torch.ones_like(current_height), 1 * (current_height - minimum_height) / (min_target_height - minimum_height))
    reward = torch.where(current_height <= max_target_height, reward, 1 * (1.0 - (current_height - max_target_height) / (min_target_height - minimum_height)))

    return (reward.clamp(min=0.0) ** sharpness)

def track_pos_z(
    env: ManagerBasedRLEnv,
    sharpness: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:, 2]

    target_height = _decode_base_velocity_pos_z_command(env, "base_velocity")

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        current_height_rel = current_height - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        current_height_rel = current_height

    # Compute reward based on current height
    reward = torch.zeros_like(current_height_rel)

    # If below minimum height, return zero reward
    below_minimum = current_height_rel <= minimum_height
    reward[below_minimum] = 0.0

    # If below target height but above minimum height
    below_target = current_height_rel <= target_height
    reward[below_target] = (current_height_rel[below_target] - minimum_height) / (target_height[below_target] - minimum_height)

    # If above target height
    above_target = current_height_rel > target_height
    reward[above_target] = 1.0 - (current_height_rel[above_target] - target_height[above_target]) / (target_height[above_target] - minimum_height)

    # Ensure reward is non-negative and apply sharpness
    reward = torch.clamp(reward, min=0.0) ** sharpness

    return reward

def base_height_range_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height is within a specified range and penalize deviations."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_z = asset.data.root_link_pos_w[:, 2]

    # Check if the height is within the specified range
    in_range = (root_pos_z >= min_height) & (root_pos_z <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(root_pos_z - torch.where(root_pos_z < min_height, max_height, min_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(root_pos_z), -out_of_range_penalty)

    return reward

def base_height_range_relative_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height is within a specified range and penalize deviations."""
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]

    root_pos_z = root_asset.data.root_link_pos_w[:, 2]
    # Get the mean z position of the wheels
    # wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2].mean(dim=1)
    # Get the minimum z position of the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2].max(dim=1).values

    # Calculate the height difference
    height_diff = root_pos_z - wheel_pos_z

    # Check if the height difference is within the specified range
    in_range = (height_diff >= min_height) & (height_diff <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(height_diff - torch.where(height_diff < min_height, max_height, min_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(height_diff), -out_of_range_penalty)

    return reward


def base_height_dynamic_wheel_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height relative to the furthest wheel is within a specified range and penalize deviations."""
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]

    root_pos_z = root_asset.data.root_link_pos_w[:, 2]
    # Get the z positions of all the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2]

    # Calculate the height differences for all wheels
    height_diffs = root_pos_z.unsqueeze(1) - wheel_pos_z

    # Find the maximum height difference for each instance (both positive and negative)
    max_height_diff, _ = torch.max(height_diffs, dim=1)
    min_height_diff, _ = torch.min(height_diffs, dim=1)

    # Choose the larger absolute value between max and min height differences
    furthest_height_diff = torch.where(
        torch.abs(max_height_diff) > torch.abs(min_height_diff), max_height_diff, min_height_diff
    )

    # Check if the furthest height difference is within the specified range
    in_range = (furthest_height_diff >= min_height) & (furthest_height_diff <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(
        furthest_height_diff - torch.where(furthest_height_diff < min_height, max_height, min_height)
    )

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(furthest_height_diff), -out_of_range_penalty)

    return reward

def link_x_vel_deviation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    weel_vel_b = torch.mean(quat_apply_inverse(asset.data.body_link_quat_w[:, asset_cfg.body_ids], asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]), dim=1)
    root_vel_b = asset.data.root_lin_vel_b
    
    # minimize difference between root and wheel velocities
    return torch.square(weel_vel_b[:, 0] - root_vel_b[:, 0])

def joint_target_deviation_range_l1(
    env: ManagerBasedRLEnv,
    min_angle: float,
    max_angle: float,
    in_range_reward: float,
    cmd_threshold: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the joint angle is within a specified range and penalize deviations."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    # Get the current joint positions
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Check if the joint angles are within the specified range
    in_range = (current_joint_pos >= min_angle) & (current_joint_pos <= max_angle)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.abs(current_joint_pos - max_angle)

    if cmd_threshold != -1.0:
        joint_deviation_range = torch.where(
            cmd.unsqueeze(1) <= cmd_threshold,
            torch.where(in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty),
            torch.tensor(0.0),
        )
    else:
        # Assign a fixed reward if in range, and a negative penalty if out of range
        joint_deviation_range = torch.where(
            in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty
        )

    # Sum the rewards over all joint ids
    return torch.sum(joint_deviation_range, dim=1)


def joint_target_deviation_range_l1_inv(
    env: ManagerBasedRLEnv,
    min_angle: float,
    max_angle: float,
    in_range_reward: float,
    cmd_threshold: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the joint angle is within a specified range and penalize deviations."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    # Get the current joint positions
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Check if the joint angles are within the specified range
    in_range = (current_joint_pos >= min_angle) & (current_joint_pos <= max_angle)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.abs(current_joint_pos - min_angle)

    if cmd_threshold != -1.0:
        joint_deviation_range = torch.where(
            cmd.unsqueeze(1) <= cmd_threshold,
            torch.where(in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty),
            torch.tensor(0.0),
        )
    else:
        # Assign a fixed reward if in range, and a negative penalty if out of range
        joint_deviation_range = torch.where(
            in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty
        )

    # Sum the rewards over all joint ids
    return torch.sum(joint_deviation_range, dim=1)


def joint_deviation_zero_leg_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - 0.7
    return torch.sum(torch.abs(angle), dim=1)

def joint_deviation_zero_shoulder_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] + 0.6
    return torch.sum(torch.abs(angle), dim=1)

def joint_deviation_zero_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def joint_deviation_zero_l1_cmd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    cmd_threshold: float = 1e-6,
) -> torch.Tensor:
    """Penalize joint positions that deviate from zero only when cmd[3] is zero."""
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)
    is_cmd3_zero = torch.abs(cmd[:, 3]) <= cmd_threshold

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(angle), dim=1)

    return penalty * is_cmd3_zero.float()

def joint_deviation_zero_l1_ang_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.01
) -> torch.Tensor:
    """
    Penalize joint positions that deviate from the default one, 
    BUT ONLY when the angular velocity command (yaw) is active.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. Get the angular velocity command (absolute value)
    # Assuming index 2 is Yaw (z-axis rotation) based on your track_ang_vel_z_link_exp function
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    
    # 2. Compute joint deviation (L1 norm)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(angle), dim=1)
    
    # 3. Apply penalty only when command > threshold
    # If rotating, return deviation penalty. If not, return 0.0.
    return torch.where(
        ang_vel_cmd > threshold, 
        deviation, 
        torch.zeros_like(deviation)
    )

def wheel_joint_deviation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the robot if the left and right wheel joints have opposite signs"""
    asset: RigidObject = env.scene[asset_cfg.name]
    wheel_joints = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # to_move = torch.abs(env.command_manager.get_command("base_velocity")[:, 0]) > 0.0
    
    # 왼쪽과 오른쪽 바퀴 속도 추출
    left_wheel_joint = wheel_joints[:, 0]
    right_wheel_joint = wheel_joints[:, 1]

    # 부호가 반대인 경우 1의 보상, 그렇지 않으면 0
    opposite_sign = torch.sign(left_wheel_joint) * torch.sign(right_wheel_joint) < 0
    reward = torch.where(opposite_sign, torch.tensor(1.0, device=wheel_joints.device), torch.tensor(0.0, device=wheel_joints.device))
    # reward = torch.where(to_move, reward, torch.tensor(0.0, device=wheel_joints.device))
    return reward


def wheel_static_link_lin_vel_z_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    clip_min: float = 0.0,
    clip_max: float | None = None,
) -> torch.Tensor:
    """Reward positive z-axis linear velocity of wheel static links."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_lin_vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2]
    reward = torch.clamp(body_lin_vel_z, min=clip_min)
    if clip_max is not None:
        reward = torch.clamp(reward, max=clip_max)
    return torch.mean(reward, dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)

def wheel_joint_velocity_zero_when_cmd_z_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_threshold: float = 1e-6,
) -> torch.Tensor:
    """Penalize wheel joint velocity only when cmd[3] is zero.

    This encourages wheel joints to stop when the 4th command component,
    usually posture/height command, is zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # command shape: (num_envs, >=4)
    cmd = env.command_manager.get_command(command_name)

    # cmd[3] == 0 인 환경에서만 penalty 적용
    is_cmd3_zero = torch.abs(cmd[:, 3]) <= cmd_threshold

    # wheel joints only
    wheel_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 각 env별 wheel velocity norm
    penalty = torch.linalg.norm(wheel_joint_vel, dim=1)

    return penalty * is_cmd3_zero.float()

class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for bipeds.

    This reward penalizes contact timing differences between the two feet to bias the policy towards a natural walking gait.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.cmd_threshold: float = cfg.params["cmd_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        # Parse and validate synced feet pair names
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if len(synced_feet_pair_names) != 2:
            raise ValueError("This reward requires exactly two pairs of feet for bipedal walking.")

        # Convert foot names to body IDs
        self.foot_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        self.foot_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        cmd_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward enforces that one foot is in the air while the other is in contact with the ground.

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # Calculate the asynchronous reward for the two feet
        async_reward = self._async_reward_func(self.foot_0, self.foot_1)

        # only enforce gait if the command velocity or body velocity is above a certain threshold
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > self.cmd_threshold, body_vel > self.velocity_threshold),
            async_reward,
            torch.tensor(0.0),
        )

    """
    Helper functions.
    """

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time

        # Ensure the tensors are properly broadcasted by selecting only the relevant dimensions
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)

        # Summing over the appropriate axis to reduce to the correct size
        return torch.exp(-(se_act_0 + se_act_1) / self.std).squeeze()


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    if cmd_threshold != -1.0:
        foot_clearance = torch.where(
            cmd <= cmd_threshold,
            torch.tensor(0.0),
            torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std),
        )
    else:
        foot_clearance = torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)

    return foot_clearance

def yaw_no_wheel_reward_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    yaw_threshold: float = 0.1,
    wheel_metric: str = "l1",  # "l1" or "l2"
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw 회전 중 바퀴를 안 쓰면 높아지는 exp 스코어(0~1)."""
    asset: RigidObject = env.scene[asset_cfg.name]

    yaw_cmd = env.command_manager.get_command(command_name)[:, 2]
    yaw_active = (torch.abs(yaw_cmd) > yaw_threshold).float()

    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]/40.0

    if wheel_metric == "l2":
        wheel_use_err = torch.square(torch.linalg.norm(wheel_vel, dim=1))  # (N,)
    else:
        wheel_use_err = torch.sum(torch.abs(wheel_vel), dim=1)  # (N,)

    score = torch.exp(-wheel_use_err / (std**2))
    return score * yaw_active

def feet_air_time_spot_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,                 # 예: "base_velocity"
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    yaw_threshold: float = 0.1,
    xy_threshold: float = 0.1,          # "제자리" 판정용
) -> torch.Tensor:
    """제자리 yaw 회전 시( yaw 크고 xy 작음 ) feet air time 보상."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    cmd = env.command_manager.get_command(command_name)
    yaw_active = torch.abs(cmd[:, 2]) > yaw_threshold
    xy_small = torch.norm(cmd[:, :2], dim=1) < xy_threshold
    spot_yaw = yaw_active & xy_small

    return torch.where(spot_yaw, reward, torch.zeros_like(reward))

def yaw_min_wheel_contact_exp(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    yaw_threshold: float = 0.1,
    xy_threshold: float = -1.0,         # >=0이면 spot-yaw(제자리 회전) 조건 추가
    contact_threshold: float = 1.0,     # 이 이상이면 "접촉"으로 판정
    temperature: float = 1.0,           # exp 민감도 (클수록 접촉 조금만 있어도 reward 급감)
    use_force: bool = False,            # False면 "접촉 개수", True면 "접촉 힘 합" 기반
) -> torch.Tensor:
    """회전(yaw) 중 휠-지면 contact를 최소화하는 exp 보상(0~1).

    - use_force=False: 접촉된 휠 개수(0~W)를 최소화 (가장 직관적)
    - use_force=True : 접촉 힘의 합을 최소화 (더 연속적이지만 스케일 튜닝 필요)

    반환:
      (N,) 텐서. yaw가 활성일 때만 0~1 보상, 아니면 0.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # (N, H, W, 3) -> force magnitude -> (N, H, W) -> history max -> (N, W)
    force_mag = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1).values
    )

    if use_force:
        # 힘 기반(연속): (N,)
        contact_measure = torch.sum(force_mag, dim=1)
    else:
        # 접촉 개수 기반(이산): (N,)
        in_contact = force_mag > contact_threshold
        contact_measure = torch.sum(in_contact.float(), dim=1)

    # exp로 0~1 보상 (contact_measure=0이면 1)
    score = torch.exp(-temperature * contact_measure)

    # yaw 활성 조건
    cmd = env.command_manager.get_command(command_name)
    yaw_active = torch.abs(cmd[:, 2]) > yaw_threshold

    # (옵션) 제자리 회전 조건까지 추가하고 싶으면
    if xy_threshold >= 0.0:
        xy_small = torch.norm(cmd[:, :2], dim=1) < xy_threshold
        yaw_active = yaw_active & xy_small

    return score * yaw_active.float()

class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)

def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)

def track_yaw_front_leg_trajectory_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    step_freq: float = 2.0,
    lift_height: float = 0.1,
    rest_height: float = 0.05,
    yaw_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward tracking a stepping trajectory (lifting front legs) when rotating in the yaw direction, accounting for terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. Yaw 커맨드 활성화 여부 확인
    yaw_cmd = env.command_manager.get_command(command_name)[:, 2]
    yaw_active = (torch.abs(yaw_cmd) > yaw_threshold).float()

    # 2. 시간에 따른 스텝 위상(Phase) 계산 (좌/우 교차)
    t = env.episode_length_buf * env.step_dt
    phase_left = t * step_freq * 2.0 * math.pi
    phase_right = phase_left + math.pi

    # 3. 평지 기준 목표 높이 (Z축 궤적) 생성
    base_target_z_left = rest_height + torch.clamp(torch.sin(phase_left), min=0.0) * lift_height
    base_target_z_right = rest_height + torch.clamp(torch.sin(phase_right), min=0.0) * lift_height

    # 4. 지형 변동성 고려 (RayCaster 센서 활용)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 센서가 감지한 지형의 평균 World Z 좌표를 더해 타겟 높이 보정
        terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        target_z_left = base_target_z_left + terrain_height
        target_z_right = base_target_z_right + terrain_height
    else:
        target_z_left = base_target_z_left
        target_z_right = base_target_z_right

    # 5. 현재 앞다리(발/바퀴)의 World Z축 높이
    left_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    # 6. 궤적 추종 오차 계산 및 지수 커널 보상 산출
    err_left = torch.square(left_foot_z - target_z_left)
    err_right = torch.square(right_foot_z - target_z_right)

    reward = torch.exp(-(err_left + err_right) / (std**2))

    # Yaw 커맨드가 임계치 이상일 때만 보상 부여
    return reward * yaw_active

def maximize_wheel_clearance_yaw_mask(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    mask_sensor_cfg_left: SceneEntityCfg,
    mask_sensor_cfg_right: SceneEntityCfg,
) -> torch.Tensor:
    """
    Reward function that maximizes the height clearance of the wheels from the ground 
    when their respective lift masks and yaw commands are active.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. 센서 데이터 로드 (지형 인식용 RayCaster 및 LiftMask)
    height_sensor: RayCaster = env.scene[height_sensor_cfg.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # 마스크 활성화 상태 (0.0 또는 1.0)
    left_mask = left_mask_sensor.data.mask
    right_mask = right_mask_sensor.data.mask

    # 2. 지형의 World Z축 평균 높이 계산
    terrain_z = torch.mean(height_sensor.data.ray_hits_w[..., 2], dim=1)

    # 3. 바퀴(End-effector)의 현재 World Z축 높이
    # asset_cfg.body_ids는 [좌측_앞바퀴_ID, 우측_앞바퀴_ID] 순서여야 합니다.
    wheel_z_left = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    wheel_z_right = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    # 4. 지면으로부터의 실제 높이 (Clearance)
    clearance_left = wheel_z_left - terrain_z
    clearance_right = wheel_z_right - terrain_z

    # 5. 마스크가 켜져 있을 때 Clearance 자체를 보상으로 사용 (값이 클수록 선형적으로 보상 증가)
    # 지면보다 아래로 뚫고 들어가는 경우(음수)를 대비해 clamp 적용
    reward_left = left_mask * torch.clamp(clearance_left, min=0.0)
    reward_right = right_mask * torch.clamp(clearance_right, min=0.0)

    total_reward = reward_left + reward_right

    return total_reward

def base_orientation_euler_l2(
    env: ManagerBasedRLEnv, 
    target_euler: list[float] | torch.Tensor, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """로봇 베이스의 현재 오일러 각도와 목표 오일러 각도 사이의 오차를 L2 제곱으로 계산합니다.
    
    Args:
        target_euler: [roll, pitch, yaw] 형태의 목표 각도 (단위: radian)
    """
    asset = env.scene[asset_cfg.name]
    
    # 1. 현재 쿼터니언을 오일러 각도(XYZ)로 변환
    # r, p, y는 각각 [-pi, pi] 범위의 텐서
    current_roll, current_pitch, current_yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
    current_euler = torch.stack((current_roll, current_pitch, current_yaw), dim=-1)

    # 2. 목표 오일러 각도 텐서화 및 확장
    if not isinstance(target_euler, torch.Tensor):
        target_euler = torch.tensor(target_euler, device=env.device, dtype=torch.float32)
    target_euler = target_euler.repeat(env.num_envs, 1)

    # 3. 각도 차이 계산 (단순 차이 계산 후 필요시 각도 Wrap-around 처리 가능)
    # 일반적인 RL 보상에서는 단순 차이의 제곱을 사용해도 충분합니다.
    error = current_euler - target_euler
    
    # 4. 각도 차이가 pi를 넘지 않도록 정규화 (선택 사항이나 권장됨)
    error = (error + math.pi) % (2 * math.pi) - math.pi
    # 5. 모든 축(Roll, Pitch, Yaw)의 오차 제곱합 반환
    return torch.sum(torch.square(error), dim=1)

def base_lin_vel_z_positive(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """로봇 베이스가 위쪽(Z축 양의 방향)으로 움직일 때만 보상을 반환합니다.
    
    이 함수는 로봇이 누워있다가 상체를 들어 올리는 '기립 시도'를 격려합니다.
    """
    asset = env.scene[asset_cfg.name]
    # root_lin_vel_w: 세계 좌표계 기준 선속도 (N, 3) -> [X, Y, Z]
    vel_z = asset.data.root_lin_vel_w[:, 2]
    
    # Z속도가 0보다 클 때만 속도값을 반환하고, 내려가거나 가만히 있으면 0을 반환
    return torch.clamp(vel_z, min=0.0)

def penalize_wheel_use_when_standing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.1,
    wheel_vel_threshold: float = 0.5,
    strong_scale: float = 5.0,
) -> torch.Tensor:
    """Penalize wheel usage strongly when the robot is supposed to stand still.

    Args:
        command_name: usually "base_velocity"
        asset_cfg: wheel joint names must be passed in asset_cfg.joint_names
        cmd_threshold: below this command magnitude, regard as standing
        wheel_vel_threshold: wheel speed above this gets extra penalty
        strong_scale: extra multiplier for wheel speeds above threshold

    Returns:
        (N,) tensor penalty term
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # standing condition: translational + yaw command all small
    cmd = env.command_manager.get_command(command_name)[:, :3]
    is_standing = torch.linalg.norm(cmd, dim=1) < cmd_threshold
    
    # selected wheel joint velocities
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # base penalty: absolute wheel usage
    base_penalty = torch.sum(torch.abs(wheel_vel), dim=1)


    penalty = base_penalty

    return penalty * is_standing.float()



class TerrainLevelDeltaReward(ManagerTermBase):
    """Reward terrain level increase, penalize decrease if needed."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        terrain = getattr(env.scene, "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is None:
            self.prev_terrain_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        else:
            self.prev_terrain_levels = terrain_levels.clone().to(env.device)

    def __call__(
        self,
        env,
        level_up_reward: float = 1.0,
        level_down_penalty: float = 0.0,
    ) -> torch.Tensor:
        terrain = getattr(env.scene, "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is None:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        current_levels = terrain_levels.to(env.device)
        delta = (current_levels - self.prev_terrain_levels).float()

        reward = torch.clamp(delta, min=0.0) * level_up_reward
        reward -= torch.clamp(-delta, min=0.0) * level_down_penalty

        self.prev_terrain_levels[:] = current_levels
        return reward


def terrain_level_reward(
    env: ManagerBasedRLEnv,
    normalize: bool = True,
) -> torch.Tensor:
    """Reward the current terrain curriculum level, normalized to [0, 1]."""
    terrain = getattr(env.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain is None or terrain_levels is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    levels = terrain_levels.to(device=env.device, dtype=torch.float32)
    if not normalize:
        return levels

    terrain_generator = getattr(terrain.cfg, "terrain_generator", None)
    num_rows = getattr(terrain_generator, "num_rows", None)
    if num_rows is None or num_rows <= 1:
        return torch.zeros_like(levels)

    return torch.clamp(levels / float(num_rows - 1), min=0.0, max=1.0)


def height_exp_reward(
    env: ManagerBasedRLEnv,
    alpha: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_sensor_cfg: SceneEntityCfg | None = None,
    left_wheel_sensor_cfg: SceneEntityCfg | None = None,
    right_wheel_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Exponential reward that increases as the base and wheel heights (z-values) grow.

    Computes the relative height of the base link w.r.t. each sensor's ground
    measurement, averages them, and returns ``exp(alpha * mean_height)``.

    Args:
        env: The RL environment instance.
        alpha: Exponential scaling factor. Larger values make the reward
            grow more steeply with height.
        asset_cfg: Configuration for the robot asset (default: ``"robot"``).
        base_sensor_cfg: Scene entity config for the base height ray-caster sensor.
        left_wheel_sensor_cfg: Scene entity config for the left wheel height
            ray-caster sensor.
        right_wheel_sensor_cfg: Scene entity config for the right wheel height
            ray-caster sensor.

    Returns:
        Tensor of shape ``(num_envs,)`` with the exponential reward value.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]  # (num_envs,)

    heights = []

    # base height: root_z - ground_z measured by base_height_scanner
    if base_sensor_cfg is not None:
        base_sensor: RayCaster = env.scene[base_sensor_cfg.name]
        ground_z_base = torch.mean(base_sensor.data.ray_hits_w[..., 2], dim=1)
        heights.append(root_z - ground_z_base)

    # left wheel height: root_z - ground_z measured by left_wheel_height_scanner
    if left_wheel_sensor_cfg is not None:
        left_sensor: RayCaster = env.scene[left_wheel_sensor_cfg.name]
        ground_z_left = torch.mean(left_sensor.data.ray_hits_w[..., 2], dim=1)
        heights.append(root_z - ground_z_left)

    # right wheel height: root_z - ground_z measured by right_wheel_height_scanner
    if right_wheel_sensor_cfg is not None:
        right_sensor: RayCaster = env.scene[right_wheel_sensor_cfg.name]
        ground_z_right = torch.mean(right_sensor.data.ray_hits_w[..., 2], dim=1)
        heights.append(root_z - ground_z_right)

    if len(heights) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    # average height across all provided sensors
    mean_height = torch.stack(heights, dim=0).mean(dim=0)  # (num_envs,)

    return torch.exp(alpha * mean_height)
