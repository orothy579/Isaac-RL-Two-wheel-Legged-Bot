"""UNWIRED proposal: estimate a riser from a measured, yaw-aligned height grid.

No simulator terrain origins or oracle geometry are inputs. Approval is required
before connecting this module to StairDetectEventCommand. Grid x/y coordinates
are relative to the base; height and ground_z must share the same vertical frame.
The default consumes the full Isaac Lab grid as requested on 2026-09-10.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ApproachConfig:
    min_rise_m: float = 0.03
    max_sample_gap_m: float = 0.12
    min_lateral_span_m: float = 0.15
    max_fit_error_m: float = 0.035
    max_angle_rad: float = math.radians(20)
    min_distance_m: float = 0.15
    max_distance_m: float = 0.40
    min_forward_speed_mps: float = 0.10
    min_command_mps: float = 0.05
    min_ttc_s: float = 0.18
    max_ttc_s: float = 0.55


class ScanApproach:
    """Adapter for this repo's centered yaw-aligned GridPattern ray scanner.

    By default reads all finite cells, without FOV masking. The optional mask
    mode is for separate sensor-visibility diagnostics. A real deployment needs
    a corresponding measured grid and ground estimate in the same vertical frame.
    """
    def __init__(self, sensor_cfg, device, use_fov_mask=False):
        if sensor_cfg.ray_alignment != 'yaw' or any(sensor_cfg.offset.pos[:2]):
            raise ValueError('proposal requires a centered yaw-aligned grid')
        if tuple(sensor_cfg.offset.rot) != (1.0, 0.0, 0.0, 0.0):
            raise ValueError('proposal requires identity scanner rotation')
        p = sensor_cfg.pattern_cfg
        if p.ordering not in ('xy', 'yx'):
            raise ValueError('unsupported grid ordering')
        self.ordering = p.ordering
        self.use_fov_mask = use_fov_mask
        self.x = torch.arange(-p.size[0]/2, p.size[0]/2+1e-9, p.resolution, device=device)
        self.y = torch.arange(-p.size[1]/2, p.size[1]/2+1e-9, p.resolution, device=device)

    def estimate(self, data, near_hits_z):
        height = data.ray_hits_w[..., 2]
        mask = torch.isfinite(data.ray_hits_w).all(-1)
        if self.use_fov_mask:
            fov = getattr(data, 'valid_mask', None)
            if fov is None:
                raise ValueError('visible-cell mask required in FOV diagnostic mode')
            mask &= fov > 0.5
        shape = (-1, len(self.y), len(self.x)) if self.ordering == 'xy' else (-1, len(self.x), len(self.y))
        height, mask = height.reshape(shape), mask.reshape(shape) > 0.5
        if self.ordering == 'yx':
            height, mask = height.transpose(1,2), mask.transpose(1,2)
        near_valid = torch.isfinite(near_hits_z)
        count = near_valid.sum(-1)
        ground = torch.where(near_valid, near_hits_z, 0.).sum(-1)/count.clamp(min=1)
        ground = torch.where(count > 0, ground, torch.nan)
        return estimate_approach(self.x, self.y, height, mask, ground)


def estimate_approach(x, y, height, visible, ground_z, cfg=ApproachConfig()):
    """Fit x = intercept + slope*y to the first observed upward edge in each row.

    x: (X,) strictly increasing; y: (Y,); height/visible: (N,Y,X);
    ground_z: (N,). Returns distance normal to riser, signed normal angle,
    fit RMS and validity, all (N,). Requires visible low AND high samples,
    enough lateral support and a coherent straight edge. Missing FOV, isolated
    hits and poor fits are rejected. A coarse grid can alias a steep ramp into
    a height discontinuity; this estimate is not a semantic stair classifier.
    """
    if height.ndim != 3 or visible.shape != height.shape:
        raise ValueError('height and visible must have matching (N,Y,X) shapes')
    n, ny, nx = height.shape
    if x.shape != (nx,) or y.shape != (ny,) or ground_z.shape != (n,):
        raise ValueError('x, y or ground_z shape mismatch')
    if nx < 2 or ny < 3 or not bool(torch.all(torch.diff(x) > 0)):
        raise ValueError('need >=2 sorted x samples and >=3 lateral rows')
    finite = visible.bool() & torch.isfinite(height)
    z = torch.where(finite, height, torch.zeros_like(height))
    near = ground_z[:, None, None]
    # Only the first rise from the currently supported floor qualifies. This
    # avoids fitting points from several different treads into one riser.
    edge = finite[..., :-1] & finite[..., 1:]
    edge &= (z[..., :-1] - near).abs() < cfg.min_rise_m * 0.5
    edge &= (z[..., 1:] - z[..., :-1]) >= cfg.min_rise_m
    edge &= (torch.diff(x) <= cfg.max_sample_gap_m)[None, None, :]
    midpoint = (x[:-1] + x[1:]) * 0.5
    edge &= midpoint[None, None, :] > 0
    candidates = torch.where(edge, midpoint[None, None, :], torch.inf)
    edge_x = candidates.min(dim=-1).values
    rows = torch.isfinite(edge_x)
    count = rows.sum(dim=-1).clamp(min=1)
    ex = torch.where(rows, edge_x, 0.0)
    ey = torch.where(rows, y[None, :], 0.0)
    mx, my = ex.sum(-1) / count, ey.sum(-1) / count
    dx = torch.where(rows, ex - mx[:, None], 0.0)
    dy = torch.where(rows, ey - my[:, None], 0.0)
    slope = (dx * dy).sum(-1) / dy.square().sum(-1).clamp(min=1e-8)
    intercept = mx - slope * my
    angle = -torch.atan(slope)
    distance = intercept / torch.sqrt(1 + slope.square())
    error = torch.where(rows, ex - intercept[:, None] - slope[:, None] * ey, 0.0)
    rms = torch.sqrt(error.square().sum(-1) / count)
    ymin = torch.where(rows, y[None, :], torch.inf).min(-1).values
    ymax = torch.where(rows, y[None, :], -torch.inf).max(-1).values
    valid = (rows.sum(-1) >= 3) & ((ymax - ymin) >= cfg.min_lateral_span_m)
    valid &= torch.isfinite(ground_z) & (distance > 0) & (rms <= cfg.max_fit_error_m)
    # Requiring support on both sides avoids extrapolating a distant edge to
    # the robot's travel line from a few points on only one side.
    valid &= (ymin < 0) & (ymax > 0)
    return {'distance_m': torch.where(valid, distance, torch.zeros_like(distance)),
            'angle_rad': torch.where(valid, angle, torch.zeros_like(angle)),
            'fit_rms_m': rms, 'valid': valid}


def timing_eligible(approach, velocity_xy, commanded_vx, cfg=ApproachConfig()):
    """Permission to START a hop; never cut short an already active hop.

    Time to contact uses measured speed toward the riser, including lateral
    drift. Distance is base-to-plane; future wheel-offset calibration is explicit.
    Thresholds are initial hypotheses, not tuned values or hardware validation.
    """
    angle = approach['angle_rad']
    closing = velocity_xy[:, 0] * torch.cos(angle) + velocity_xy[:, 1] * torch.sin(angle)
    ttc = approach['distance_m'] / closing.clamp(min=1e-6)
    valid = approach['valid'] & torch.isfinite(velocity_xy).all(-1) & torch.isfinite(commanded_vx)
    eligible = valid & (commanded_vx > cfg.min_command_mps)
    eligible &= (velocity_xy[:, 0] > cfg.min_forward_speed_mps) & (closing > 0)
    eligible &= angle.abs() <= cfg.max_angle_rad
    eligible &= (approach['distance_m'] >= cfg.min_distance_m) & (approach['distance_m'] <= cfg.max_distance_m)
    eligible &= (ttc >= cfg.min_ttc_s) & (ttc <= cfg.max_ttc_s)
    return eligible, torch.where(valid & (closing > 0), ttc, torch.zeros_like(ttc))
