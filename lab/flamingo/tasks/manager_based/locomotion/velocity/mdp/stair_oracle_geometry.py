"""Ground-truth geometry helpers for the square stair diagnostic collector.

These values are training diagnostics only.  They must not be added to the actor
observation because the real robot does not know the simulator terrain origin.
"""

from __future__ import annotations

import torch


def square_stair_oracle(
    robot_xy_w: torch.Tensor,
    robot_yaw_w: torch.Tensor,
    terrain_origin_xy_w: torch.Tensor,
    riser_radii: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the next outward riser hit by the robot heading.

    The inverted-pyramid terrain consists of concentric, axis-aligned square
    risers.  ``riser_radii`` contains their half widths measured from the terrain
    origin.  The returned values are, per environment:

    * distance along the robot heading to the riser [m],
    * perpendicular distance to the riser plane [m],
    * signed heading-to-ascent-normal angle [rad],
    * the selected ascent normal in world XY.

    A negative angle means the normal lies to the robot's right.  Environments
    with no higher riser ahead receive NaN distances/angles and a zero normal.
    """
    if robot_xy_w.ndim != 2 or robot_xy_w.shape[1] != 2:
        raise ValueError(f"robot_xy_w must have shape (N, 2), got {tuple(robot_xy_w.shape)}")
    if terrain_origin_xy_w.shape != robot_xy_w.shape:
        raise ValueError(
            "terrain_origin_xy_w must match robot_xy_w, got "
            f"{tuple(terrain_origin_xy_w.shape)} and {tuple(robot_xy_w.shape)}"
        )

    radii = torch.as_tensor(riser_radii, device=robot_xy_w.device, dtype=robot_xy_w.dtype)
    if radii.ndim == 1:
        radii = radii.unsqueeze(0).expand(robot_xy_w.shape[0], -1)
    elif radii.ndim != 2 or radii.shape[0] != robot_xy_w.shape[0]:
        raise ValueError(
            "riser_radii must have shape (K,) or (N, K), got "
            f"{tuple(radii.shape)} for N={robot_xy_w.shape[0]}"
        )
    if radii.shape[1] == 0:
        raise ValueError("riser_radii must contain at least one radius")

    local_xy = robot_xy_w - terrain_origin_xy_w
    x = local_xy[:, 0:1]
    y = local_xy[:, 1:2]
    fx = torch.cos(robot_yaw_w).unsqueeze(1)
    fy = torch.sin(robot_yaw_w).unsqueeze(1)
    radius = radii

    # Only boundaries outside the robot's current square ring are upward risers.
    inside = torch.isfinite(radius) & (radius > 0.0)
    inside &= radius > torch.maximum(x.abs(), y.abs()) + eps
    safe_fx = torch.where(fx.abs() > eps, fx, torch.ones_like(fx))
    safe_fy = torch.where(fy.abs() > eps, fy, torch.ones_like(fy))

    t_pos_x = (radius - x) / safe_fx
    t_neg_x = (-radius - x) / safe_fx
    t_pos_y = (radius - y) / safe_fy
    t_neg_y = (-radius - y) / safe_fy

    valid_pos_x = inside & (fx > eps) & (t_pos_x > eps) & ((y + t_pos_x * fy).abs() <= radius + eps)
    valid_neg_x = inside & (fx < -eps) & (t_neg_x > eps) & ((y + t_neg_x * fy).abs() <= radius + eps)
    valid_pos_y = inside & (fy > eps) & (t_pos_y > eps) & ((x + t_pos_y * fx).abs() <= radius + eps)
    valid_neg_y = inside & (fy < -eps) & (t_neg_y > eps) & ((x + t_neg_y * fx).abs() <= radius + eps)

    inf = torch.full_like(radius, torch.inf)
    candidates = torch.stack(
        (
            torch.where(valid_pos_x, t_pos_x, inf),
            torch.where(valid_neg_x, t_neg_x, inf),
            torch.where(valid_pos_y, t_pos_y, inf),
            torch.where(valid_neg_y, t_neg_y, inf),
        ),
        dim=-1,
    )  # (N, K, 4)
    flat_candidates = candidates.flatten(start_dim=1)
    distance_along, flat_index = torch.min(flat_candidates, dim=1)
    side_index = torch.remainder(flat_index, 4)
    has_hit = torch.isfinite(distance_along)

    normals = torch.tensor(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        device=robot_xy_w.device,
        dtype=robot_xy_w.dtype,
    )
    normal_xy_w = normals[side_index]
    normal_xy_w = torch.where(has_hit.unsqueeze(1), normal_xy_w, torch.zeros_like(normal_xy_w))
    forward_xy_w = torch.cat((fx, fy), dim=1)
    alignment_cos = torch.sum(forward_xy_w * normal_xy_w, dim=1).clamp(min=-1.0, max=1.0)
    cross_z = forward_xy_w[:, 0] * normal_xy_w[:, 1] - forward_xy_w[:, 1] * normal_xy_w[:, 0]
    alignment_error = torch.atan2(cross_z, alignment_cos)
    distance_normal = distance_along * alignment_cos.clamp(min=0.0)

    nan = torch.full_like(distance_along, torch.nan)
    distance_along = torch.where(has_hit, distance_along, nan)
    distance_normal = torch.where(has_hit, distance_normal, nan)
    alignment_error = torch.where(has_hit, alignment_error, nan)
    return distance_along, distance_normal, alignment_error, normal_xy_w


def infer_curriculum_stair_geometry(
    terrain_origin_z_w: torch.Tensor,
    step_height_range: tuple[float, float],
    step_width_range: tuple[float, float],
    tile_size: float,
    border_width: float,
    platform_width: float,
    max_steps: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover each generated tile's exact step height, width, and riser radii.

    Isaac Lab stores the inverted-pyramid origin at
    ``-(num_steps + 1) * step_height``. Height and width use the same curriculum
    difficulty, so testing the small set of possible integer step counts uniquely
    recovers both dimensions without exposing either value to the actor.
    """
    origin_z = torch.as_tensor(terrain_origin_z_w)
    height_start, height_target = step_height_range
    width_start, width_target = step_width_range
    if max_steps < 1 or height_target == height_start:
        raise ValueError("max_steps must be positive and the height curriculum must span a range")

    candidates = torch.arange(1, max_steps + 1, device=origin_z.device, dtype=origin_z.dtype)
    candidate_height = (-origin_z.unsqueeze(1)) / (candidates.unsqueeze(0) + 1.0)
    difficulty = (candidate_height - height_start) / (height_target - height_start)
    candidate_width = width_start + difficulty * (width_target - width_start)
    available_width = tile_size - 2.0 * border_width - platform_width
    recomputed_steps = torch.floor(available_width / (2.0 * candidate_width)) + 1.0
    valid = (
        (difficulty >= -1.0e-5)
        & (difficulty <= 1.0 + 1.0e-5)
        & (candidate_width > 0.0)
        & (recomputed_steps == candidates.unsqueeze(0))
    )
    # Invalid candidates receive infinity, leaving the first exact solution.
    candidate_score = torch.where(valid, candidates.unsqueeze(0), torch.full_like(candidate_height, torch.inf))
    selected_score, selected_index = torch.min(candidate_score, dim=1)
    has_solution = torch.isfinite(selected_score)
    selected_index = torch.where(has_solution, selected_index, torch.zeros_like(selected_index))
    row = torch.arange(origin_z.shape[0], device=origin_z.device)
    step_height = candidate_height[row, selected_index]
    step_width = candidate_width[row, selected_index]
    num_steps = candidates[selected_index].long()

    center_width = tile_size - 2.0 * border_width - 2.0 * num_steps * step_width
    step_index = torch.arange(max_steps, device=origin_z.device, dtype=origin_z.dtype).unsqueeze(0)
    radii = center_width.unsqueeze(1) / 2.0 + step_index * step_width.unsqueeze(1)
    radii = torch.where(step_index < num_steps.unsqueeze(1), radii, torch.full_like(radii, torch.nan))
    step_height = torch.where(has_solution, step_height, torch.full_like(step_height, torch.nan))
    step_width = torch.where(has_solution, step_width, torch.full_like(step_width, torch.nan))
    radii = torch.where(has_solution.unsqueeze(1), radii, torch.full_like(radii, torch.nan))
    return step_height, step_width, radii
