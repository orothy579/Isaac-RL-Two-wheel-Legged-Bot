# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


_IMU_DEBUG_PRINT_EVERY = 50
_imu_debug_print_counts: dict[tuple[str, str], int] = {}
_body_debug_print_counts: dict[tuple[str, str], int] = {}


def _debug_print_imu_value(sensor_name: str, value_name: str, value: torch.Tensor) -> None:
    if sensor_name not in ("lower_imu", "upper_imu"):
        return

    key = (sensor_name, value_name)
    count = _imu_debug_print_counts.get(key, 0) + 1
    _imu_debug_print_counts[key] = count
    if count % _IMU_DEBUG_PRINT_EVERY != 1:
        return

    first_env_value = value[0].detach().cpu().tolist()
    print(f"[IMU DEBUG] {sensor_name}.{value_name}[0] = {first_env_value}", flush=True)


def _debug_print_body_value(asset: Articulation, asset_cfg: SceneEntityCfg, value_name: str, value: torch.Tensor) -> None:
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        return

    body_ids_list = body_ids.tolist() if isinstance(body_ids, torch.Tensor) else list(body_ids)
    body_names = tuple(asset.body_names[body_id] for body_id in body_ids_list)
    key = (",".join(body_names), value_name)
    count = _body_debug_print_counts.get(key, 0) + 1
    _body_debug_print_counts[key] = count
    if count % _IMU_DEBUG_PRINT_EVERY != 1:
        return

    first_env_values = value[0].detach().cpu().tolist()
    for body_name, body_value in zip(body_names, first_env_values):
        print(f"[BODY OBS DEBUG] {body_name}.{value_name}[0] = {body_value}", flush=True)


def base_lin_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b

def base_lin_vel_x_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 0].unsqueeze(-1)

def base_lin_vel_y_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)

def base_lin_vel_z_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)

def base_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def body_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body angular velocity in the selected body's link frame.

    If multiple bodies are selected, the result is flattened in the configured body order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    ang_vel_b = quat_apply_inverse(quat_w.reshape(-1, 4), ang_vel_w.reshape(-1, 3)).reshape(env.num_envs, -1, 3)
    # _debug_print_body_value(asset, asset_cfg, "ang_vel_b", ang_vel_b)
    if ang_vel_b.shape[1] == 1:
        return ang_vel_b[:, 0]
    return ang_vel_b.flatten(start_dim=1)


def body_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gravity direction projected into the selected body's link frame.

    If multiple bodies are selected, the result is flattened in the configured body order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    gravity_w = asset.data.GRAVITY_VEC_W[:, None, :].expand(-1, quat_w.shape[1], -1)
    projected_gravity_b = quat_apply_inverse(quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)).reshape(env.num_envs, -1, 3)
    # _debug_print_body_value(asset, asset_cfg, "projected_gravity_b", projected_gravity_b)
    if projected_gravity_b.shape[1] == 1:
        return projected_gravity_b[:, 0]
    return projected_gravity_b.flatten(start_dim=1)
        
def base_pos_z_rel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
    else:
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
    
def base_pos_z_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
    else:
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)

def imu_ang_vel(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """IMU angular velocity in IMU/body frame. Shape: (num_envs, 3)."""
    imu = env.scene[sensor_cfg.name]
    value = imu.data.ang_vel_b
    _debug_print_imu_value(sensor_cfg.name, "ang_vel_b", value)
    return value


def imu_lin_acc(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """IMU linear acceleration in IMU/body frame. Shape: (num_envs, 3)."""
    imu = env.scene[sensor_cfg.name]
    value = imu.data.lin_acc_b
    _debug_print_imu_value(sensor_cfg.name, "lin_acc_b", value)
    return value


def imu_projected_gravity(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gravity direction projected into IMU frame. Shape: (num_envs, 3)."""
    imu = env.scene[sensor_cfg.name]
    value = imu.data.projected_gravity_b
    _debug_print_imu_value(sensor_cfg.name, "projected_gravity_b", value)
    return value

def terrain_level(
    env: ManagerBasedEnv,
    normalize: bool = True,
) -> torch.Tensor:
    """Current terrain curriculum level for each environment."""
    terrain = getattr(env.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain is None or terrain_levels is None:
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    levels = terrain_levels.to(device=env.device, dtype=torch.float32).unsqueeze(-1)
    if not normalize:
        return levels

    terrain_generator = getattr(terrain.cfg, "terrain_generator", None)
    num_rows = getattr(terrain_generator, "num_rows", None)
    if num_rows is None or num_rows <= 1:
        return levels

    return levels / float(num_rows - 1)


def current_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current reward value. Returns zeros if the reward manager is not initialized."""
    if not hasattr(env, "reward_manager") or env.reward_manager is None:
        # Assuming the shape should be (num_envs,) based on the environment
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    try:
        return env.reward_buf.unsqueeze(-1)
    except AttributeError:
        # Fallback to zeros if the reward_manager is initialized but compute isn't ready
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def is_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()


def lift_mask_by_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    
    """
    Generate a lift mask for the robot's legs based on row-wise height scan gradients from separate left and right sensors.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        sensor_cfg_left (SceneEntityCfg): Configuration for the left raycast sensor.
        sensor_cfg_right (SceneEntityCfg): Configuration for the right raycast sensor.
        command_name (str): Command name to check movement intention.
        gradient_threshold (float): Threshold for row-wise height gradient to detect steps.

    Returns:
        torch.Tensor: Lift mask for left and right legs. Shape: [num_envs, 2].
    """
    #* Step 1: Extract ray hit positions (Z coordinates) from left and right sensors
    left_lift_mask_sensor = env.scene.sensors[sensor_cfg_left.name]
    right_lift_mask_sensor = env.scene.sensors[sensor_cfg_right.name]

    left_mask= left_lift_mask_sensor.data.mask 
    right_mask = right_lift_mask_sensor.data.mask  
    
    lift_mask = torch.stack([left_mask, right_mask], dim=1) 

    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)  # Shape: [num_envs]
    lift_mask *= (command_norm > 0.1).unsqueeze(-1).float()  # Apply movement condition

    return lift_mask

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def base_euler_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_com_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy

def base_euler_angle_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy


def joint_pos_rel_sin(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as sine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_sin = torch.sin(current_value)
    return current_value_sin


def joint_pos_rel_cos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as cosine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_cos = torch.cos(current_value)
    return current_value_cos


def height_scan_raw(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.ray_hits_w[..., 2]

def masked_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.5,
    base_height: float = 0.0,
    noise_std: float = 0.0,
    noise_std_per_meter: float = 0.0,
    random_noise_range: tuple[float, float] = (0.0, 0.0),
    dropout_prob: float = 0.0,
    sample_hold_steps: int = 1,
    use_sensor_fov_mask: bool = True,
    frame_dropout_prob: float = 0.0,
    valid_indices: Sequence[int] | None = None,
    valid_probability_scale: float = 1.0,
    valid_probabilities_by_index: Sequence[tuple[int, float]] | dict[int, float] | None = None,
    height_bias_by_index: Sequence[tuple[int, float]] | dict[int, float] | None = None,
) -> torch.Tensor:
    """Camera-FOV masked height scan for sim-to-real matching.

    Inside camera FOV, this returns the base-link-frame height scan:
        -hit_point_base_z - offset

    Outside camera FOV, it returns base_height.

    Optional RealSense-style noise can be added inside the valid FOV only:
        base_height: value used for invalid-FOV cells and dropped valid cells.
        noise_std: constant Gaussian noise standard deviation in meters.
        noise_std_per_meter: depth-proportional Gaussian noise standard deviation.
        random_noise_range: uniform noise range in meters.
        dropout_prob: probability that a valid cell is replaced with base_height.
        sample_hold_steps: policy steps for which the whole simulated sensor image is held constant.
        use_sensor_fov_mask: if False, use only the empirical mask/probabilities for visibility.
        frame_dropout_prob: probability that the whole held image is replaced with base_height.
        valid_indices: optional empirical cell indices that are allowed to remain valid.
        valid_probability_scale: multiplier for valid_probabilities_by_index before clamping to [0, 1].
        valid_probabilities_by_index: optional per-cell visibility probability pairs keyed by flattened cell index.
        height_bias_by_index: optional per-cell height-bias pairs in meters keyed by flattened cell index.
    """
    hold_steps = max(1, int(sample_hold_steps))
    cache_key = (id(env), sensor_cfg.name)
    step_count = int(getattr(env, "common_step_counter", 0))
    hold_bucket = step_count // hold_steps
    cache = getattr(masked_height_scan, "_sample_hold_cache", {})
    cached = cache.get(cache_key)
    if (
        cached is not None
        and cached["bucket"] == hold_bucket
        and cached["device"] == env.device
    ):
        return cached["value"].clone()

    sensor = env.scene.sensors[sensor_cfg.name]
    base_pos_w = sensor.data.pos_w
    base_quat_w = getattr(sensor.data, "quat_w", None)
    if base_quat_w is None:
        asset: Articulation = env.scene[asset_cfg.name]
        base_pos_w = asset.data.root_link_pos_w
        base_quat_w = asset.data.root_link_quat_w

    num_envs, num_rays, _ = sensor.data.ray_hits_w.shape
    hit_pos_b = quat_apply_inverse(
        base_quat_w.repeat_interleave(num_rays, dim=0),
        (sensor.data.ray_hits_w - base_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(num_envs, num_rays, 3)
    depth = -hit_pos_b[..., 2]
    height = depth - offset
    valid_mask = sensor.data.valid_mask.bool() if use_sensor_fov_mask else torch.ones_like(height, dtype=torch.bool)
    finite_mask = torch.isfinite(height) & torch.isfinite(depth) & torch.isfinite(hit_pos_b).all(dim=-1)
    valid_mask = valid_mask & finite_mask
    height = torch.where(finite_mask, height, torch.full_like(height, float(base_height)))
    depth = torch.where(finite_mask, depth, torch.zeros_like(depth))

    if valid_indices is not None:
        empirical_mask = torch.zeros_like(valid_mask)
        empirical_mask[:, torch.as_tensor(valid_indices, device=valid_mask.device, dtype=torch.long)] = True
        valid_mask = valid_mask & empirical_mask

    if valid_probabilities_by_index is not None:
        valid_probability_items = (
            valid_probabilities_by_index.items()
            if isinstance(valid_probabilities_by_index, dict)
            else valid_probabilities_by_index
        )
        valid_probability_items = tuple(valid_probability_items)
        visibility_prob = torch.zeros_like(height)
        indices = torch.as_tensor(
            tuple(index for index, _ in valid_probability_items),
            device=height.device,
            dtype=torch.long,
        )
        probabilities = torch.as_tensor(
            tuple(probability for _, probability in valid_probability_items),
            device=height.device,
            dtype=height.dtype,
        ).mul(float(valid_probability_scale)).clamp_(0.0, 1.0)
        visibility_prob[:, indices] = probabilities
        valid_mask = valid_mask & (torch.rand_like(height) < visibility_prob)

    if frame_dropout_prob > 0.0:
        keep_frame = torch.rand(height.shape[0], 1, device=height.device) >= float(frame_dropout_prob)
        valid_mask = valid_mask & keep_frame

    if noise_std > 0.0 or noise_std_per_meter > 0.0:
        std = float(noise_std) + float(noise_std_per_meter) * torch.abs(depth)
        height = height + torch.randn_like(height) * std

    random_noise_min, random_noise_max = random_noise_range
    if random_noise_min != 0.0 or random_noise_max != 0.0:
        height = height + torch.empty_like(height).uniform_(float(random_noise_min), float(random_noise_max))

    if height_bias_by_index is not None:
        height_bias_items = (
            height_bias_by_index.items()
            if isinstance(height_bias_by_index, dict)
            else height_bias_by_index
        )
        height_bias_items = tuple(height_bias_items)
        height_bias = torch.zeros_like(height)
        indices = torch.as_tensor(
            tuple(index for index, _ in height_bias_items),
            device=height.device,
            dtype=torch.long,
        )
        biases = torch.as_tensor(
            tuple(bias for _, bias in height_bias_items),
            device=height.device,
            dtype=height.dtype,
        )
        height_bias[:, indices] = biases
        height = height + height_bias

    if dropout_prob > 0.0:
        dropout_mask = (torch.rand_like(height) < float(dropout_prob)) & valid_mask
        height = torch.where(dropout_mask, torch.full_like(height, float(base_height)), height)

    output = torch.where(valid_mask, height, torch.full_like(height, float(base_height)))
    if hold_steps > 1:
        cache[cache_key] = {
            "bucket": hold_bucket,
            "device": env.device,
            "value": output.detach().clone(),
        }
        masked_height_scan._sample_hold_cache = cache
    return output

def generated_partial_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, 0]


def generated_scaled_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, :3] *= torch.tensor(scale, device=env.device)
    return scaled_command

def generated_scaled_event_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, :2] *= torch.tensor(scale, device=env.device)
    return scaled_command

def joint_pos_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint positions for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return pos * gear_ratio

def joint_vel_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint velocities for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return vel * gear_ratio

def joint_pos_leg_gear_rel(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint positions relative to default for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return pos * gear_ratio

def joint_vel_leg_gear_rel(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint velocities relative to default for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    # print("joint vel:", vel * gear_ratio)  
    return vel * gear_ratio


def coupled_joint_pos_motor_space(
    env: ManagerBasedEnv,
    joint_names: Sequence[str],
    coupled_pairs: Sequence[tuple[str, str, float, float, bool]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint positions with coupled pitch/roll slots replaced by motor-space values."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return _replace_coupled_pairs_with_motor_values(pos, joint_names, coupled_pairs)


def coupled_joint_vel_motor_space(
    env: ManagerBasedEnv,
    joint_names: Sequence[str],
    coupled_pairs: Sequence[tuple[str, str, float, float, bool]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint velocities with coupled pitch/roll slots replaced by motor-space values."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return _replace_coupled_pairs_with_motor_values(vel, joint_names, coupled_pairs)


def _replace_coupled_pairs_with_motor_values(
    joint_value: torch.Tensor,
    joint_names: Sequence[str],
    coupled_pairs: Sequence[tuple[str, str, float, float, bool]],
) -> torch.Tensor:
    motor_value = joint_value.clone()
    joint_name_to_index = {name: index for index, name in enumerate(joint_names)}

    for pitch_name, roll_name, gear_ratio_1, gear_ratio_2, mirror in coupled_pairs:
        pitch_id = joint_name_to_index[pitch_name]
        roll_id = joint_name_to_index[roll_name]

        pitch = joint_value[:, pitch_id]
        roll = joint_value[:, roll_id]

        if mirror:
            motor_1 = float(gear_ratio_1) * (pitch - roll)
            motor_2 = float(gear_ratio_2) * (pitch + roll)
        else:
            motor_1 = float(gear_ratio_1) * (pitch + roll)
            motor_2 = float(gear_ratio_2) * (pitch - roll)

        motor_value[:, pitch_id] = motor_1
        motor_value[:, roll_id] = motor_2

    return motor_value
