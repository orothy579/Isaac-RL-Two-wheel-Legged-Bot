# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""ZMP-based terms for climbing stairs one wheel at a time (flamingo-light, alternating feet).

Theory (``docs/zmp.pdf``, ``docs/Biped_walking_pattern_generation_by_using_preview_control_of_zero.pdf``):

* Vukobratovic & Stepanenko 1972 - the zero-moment point (ZMP) is the point of the support
  surface through which the resultant ground reaction passes; about it the moments of the
  external and inertial forces vanish (d'Alembert). Dynamic equilibrium needs it inside the
  support area; in single support it sits at the centre of the support foot, in double
  support it moves gradually from the old to the new support foot. How far it is inside the
  support boundary is the stability margin. Multibody ZMP (their eqs. 12-13), with the rotational
  terms omitted::

      p_x = sum m_i [(z_i'' + g) x_i - x_i'' (z_i - z_0)] / sum m_i (z_i'' + g)

* Kajita et al. 2003 - cart-table model (eqs. 8-9) ``p = x - (z_c / g) x''``; pattern
  generation is a ZMP *tracking* servo (eq. 14: ``J = sum Q_e e^2 + R du^2`` with u = CoM
  jerk) whose reference steps from foot centre to foot centre (cubic in double support).

Mapping to flamingo-light v01_2_2. The wheels are the feet. Each leg has one actuated DoF (the
shoulder), and the 4-bar linkage moves the wheel almost straight down (|dx| < 5 mm), so the two
wheels always stay side by side:

* contact point = wheel centre - r z (exact on a tread), support polygon = the wheel contact
  patches (both wheels: the box spanning them; one wheel: that patch);
* a *step* = one wheel leaves the ground while the other carries the robot (clean single
  support) and lands on a higher tread. Both wheels airborne = a jump, which is penalized and
  never counted as a step;
* ZMP reference: centre of both feet while driving; when a riser is ahead of a wheel (privileged
  terrain scan), the reference moves toward the *other* (next stance) wheel before liftoff;
  during single support it is the stance wheel; a first-order filter (``ref_tau``) smooths the
  hand-over (Kajita's cubic double-support transition);
* rewards: ZMP tracking (Kajita servo error), ZMP stability margin (Vukobratovic), CoP weight
  transfer to the stance wheel before liftoff, CoM jerk (Kajita's R du^2), clean alternating
  step-ups, riser progress, swing clearance, flight.

Note on the margin: for physically consistent motion the dynamic ZMP *is* the centre of pressure,
so in simulation it never leaves the support polygon (up to the neglected link rotational
inertia and finite differences). A robot that tips has its ZMP *on* the boundary and rotates
about it (Vukobratovic's foot-edge angle zeta). The margin term therefore penalizes a margin
below a safety value instead of rewarding the ZMP for being "outside".

All terms share one :class:`ZmpGaitState` per env (``get_gait_state``), configured by the env
cfg's ``zmp_gait`` field and updated at most once per env step (lazily, by whichever term is
evaluated first).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GRAVITY = 9.81
LEFT, RIGHT = 0, 1


@configclass
class ZmpGaitCfg:
    """Settings of the shared ZMP / gait state (one per env, see :func:`get_gait_state`)."""

    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    feet_scanner_name: str = "feet_scanner"
    wheel_body_names: tuple[str, str] = ("left_wheel_link", "right_wheel_link")
    """(left, right) wheel links; the link origin is on the wheel axis."""
    wheel_radius: float = 0.076
    tire_offset: float = 0.015
    """Tire mid-plane offset [m] from the wheel link origin along its axle (local +z, outward on
    v01_2_2): the contact points sit at y = +-0.172 m, not at the link origins (+-0.157 m)."""
    tire_track_half: float = 0.172
    """Lateral position [m, base frame] of the tire mid-planes; selects the feet-scanner columns."""
    support_force_threshold: float = 5.0
    """Vertical wheel force [N] that counts as support (standing load is ~25 N per wheel)."""
    min_air_time: float = 0.04
    """A touch-down counts as a step only after this much air time [s] (filters contact chatter)."""
    step_up_height: float = 0.025
    """A touch-down this much higher than the liftoff support [m] is a step-up; a terrain rise this
    high ahead of a wheel is a riser that needs a step."""
    patch_half_x: float = 0.03
    """Half length [m] of a wheel contact patch along the heading (sagittal margin)."""
    patch_half_y: float = 0.023
    """Half width [m] of a wheel contact patch (half the wheel width)."""
    margin_axes: str = "lateral"
    """"lateral" (default) or "both". A wheel touches the ground along a line across the tread, so
    its CoP has no sagittal extent and a sagittal "margin" only measures estimation noise (the
    sagittal ZMP-to-wheel offset is still scored by ``zmp_tracking_exp``). The lateral margin is the
    real one: between the wheels in double support, within the tire width in single support."""
    zmp_model: str = "multibody"
    """"multibody" (Vukobratovic, every link as a point mass) or "cart_table" (Kajita, whole-body CoM)."""
    acc_clip: float = 30.0
    """Per-axis clip [m/s^2] of the finite-difference accelerations (contact impacts)."""
    ref_shift: float = 0.8
    """Fraction of the way from the feet centre to the stance wheel that the double-support
    reference moves before a step (1 = fully over the stance wheel)."""
    ref_tau: float = 0.1
    """Time constant [s] of the reference hand-over filter."""
    under_window: tuple[float, float] = (-0.06, 0.06)
    """Scanner x-window [m, base yaw frame] whose highest hit is the terrain under a wheel."""
    ahead_window: tuple[float, float] = (0.06, 0.21)
    """Scanner x-window [m] in front of a wheel that is checked for a riser."""
    column_tolerance: float = 0.03
    """Scanner rays within this lateral distance [m] of a wheel belong to that wheel's column."""
    flight_grace_time: float = 0.5
    """Flight right after spawn (the robot is dropped) is ignored for this long [s]."""
    progress_min_gain: float = 0.05
    """Horizontal distance [m] from the spawn point that counts as new progress (no-progress termination)."""
    step_rule: str = "air"
    """Which detector pays the step-up rewards.

    ``"air"`` (sweeps v1/v2): the wheel's *vertical* force stays below ``support_force_threshold``
    for ``min_air_time`` while the other wheel carries the robot, then it lands higher. A wheel
    rolled or bounced over a riser edge passes as well (the edge force is nearly horizontal), and
    the v1/v2 policies did exactly that: one shoulder pinned at its limit, that wheel carried over
    the edge by running at 1.0-1.3 m/s.

    ``"lift"``: the wheel must also be free of *any* contact (force magnitude) during the swing,
    its leg must be retracted at least ``lift_required`` relative to the stance leg, and it must
    land on a tread (vertical share of the landing force >= ``landing_vertical_share``). Both
    detectors always run; the ``lifted_*`` statistics come from the strict one."""
    lift_required: float = 0.04
    """Swing-leg retraction [m] relative to the stance leg (tire centre height in the base frame)
    that a strict step-up needs."""
    landing_vertical_share: float = 0.7
    """Minimum F_z / |F| of the landing contact of a strict step-up (lower = riser face or nose hit)."""
    yaw_rate_tau: float = 0.3
    """Time constant [s] of the low-pass filtered yaw rate (``yaw_rate_excess_l2``): sustained turning
    passes, the step-to-step jitter of the wheel-speed exploration noise does not."""


class ZmpGaitState:
    """Per-env contact, ZMP and step bookkeeping shared by every term of the stair_step task."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: ZmpGaitCfg):
        self.env = env
        self.cfg = cfg
        n, dev = env.num_envs, env.device
        self.num_envs, self.device = n, dev
        self.robot = env.scene[cfg.asset_name]
        self.contact_sensor = env.scene.sensors[cfg.contact_sensor_name]
        self.scanner = env.scene.sensors[cfg.feet_scanner_name]

        self.wheel_ids = torch.tensor(
            [self.robot.body_names.index(name) for name in cfg.wheel_body_names], device=dev
        )
        sensor_ids, _ = self.contact_sensor.find_bodies(list(cfg.wheel_body_names), preserve_order=True)
        self.sensor_wheel_ids = torch.tensor(sensor_ids, device=dev)

        # feet-scanner columns (rays under / ahead of each tire), from the pattern's local offsets
        starts = self.scanner.ray_starts[0]  # (R, 3) in the yaw-aligned sensor frame
        wheel_y = (cfg.tire_track_half, -cfg.tire_track_half)
        self.under_masks, self.ahead_masks = [], []
        for y in wheel_y:
            col = (starts[:, 1] - y).abs() <= (cfg.column_tolerance + 1.0e-6)
            if not bool(col.any()):  # fall back to the nearest column
                col = (starts[:, 1] - y).abs() <= (starts[:, 1] - y).abs().min() + 1.0e-6
            ux = (starts[:, 0] >= cfg.under_window[0] - 1.0e-6) & (starts[:, 0] <= cfg.under_window[1] + 1.0e-6)
            ax = (starts[:, 0] >= cfg.ahead_window[0] - 1.0e-6) & (starts[:, 0] <= cfg.ahead_window[1] + 1.0e-6)
            self.under_masks.append(col & ux)
            self.ahead_masks.append(col & ax)

        self._masses = None  # (N, B), read lazily after the startup mass randomization
        self._last_step = -1
        self._unit_z = torch.tensor([[0.0, 0.0, 1.0]], device=dev)

        z2 = lambda: torch.zeros(n, 2, device=dev)  # noqa: E731
        z1 = lambda: torch.zeros(n, device=dev)  # noqa: E731
        # contacts / swing bookkeeping (index 0 = left, 1 = right)
        self.contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.support_force = z2()
        self.air_time = z2()
        self.support_z = z2()
        self.support_known = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.liftoff_z = z2()
        self.foot_max_z = torch.full((n, 2), -1.0e6, device=dev)  # highest support of each wheel this episode
        self.clean_swing = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.last_step_foot = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.time_since_flight = torch.full((n,), 1.0e3, device=dev)
        self.floor_z = torch.full((n,), 1.0e6, device=dev)
        self.max_trailing = z1()
        # events of the current step
        self.touchdown = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.step_up = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.step_up_alternating = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.step_dz = z2()
        self.progress_gain = z1()
        self.flight = torch.zeros(n, dtype=torch.bool, device=dev)
        self.flight_counted = torch.zeros(n, dtype=torch.bool, device=dev)
        # ZMP quantities (heading frame of the base: x forward, y left, origin at the base)
        self.prev_vel = None  # (N, B, 3) or (N, 3)
        self.prev_acc = None
        self.has_prev = torch.zeros(n, dtype=torch.bool, device=dev)
        self.zmp = z2()
        self.zmp_ref = z2()
        self.cop = z2()
        self.margin = z1()
        self.jerk = z2()
        self.contact_h = torch.zeros(n, 2, 2, device=dev)  # (N, foot, xy) heading frame
        self.wheel_z = z2()
        self.wheel_clearance = z2()  # wheel bottom above the terrain under it [m]
        self.riser_height = z2()  # terrain rise ahead of each wheel w.r.t. its support [m]
        self.riser_top = z2()  # world z of the highest terrain ahead of each wheel
        self.blocked = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.swing_foot = torch.full((n,), -1, dtype=torch.long, device=dev)  # pending/actual swing wheel
        # strict ("lift") step detector: any contact force, leg retraction, landing on a tread
        self.touching = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.wheel_zb = z2()  # tire centre height in the base frame [m] (negative; higher = leg retracted)
        self.rel_lift = z2()  # wheel_zb minus the other wheel's: > 0 = this leg is shorter than the other
        self.lift_air_time = z2()
        self.lift_liftoff_z = z2()
        self.lift_clean = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.swing_lift_max = z2()
        self.lifted_step_up = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.lifted_step_up_alternating = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.last_lift_foot = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.yaw_rate_f = z1()  # low-pass filtered yaw rate [rad/s]
        # horizontal progress (for the no-progress termination)
        self.spawn_xy = self.robot.data.root_link_pos_w[:, :2].clone()
        self.best_dist = z1()
        self.last_progress_time = z1()
        # per-episode statistics (read by the log/curriculum terms in _reset_idx)
        self.ep_step_ups = z1()
        self.ep_alt_step_ups = z1()
        self.ep_flight_time = z1()
        self.ep_margin_sum = z1()
        self.ep_single_support_time = z1()
        self.ep_steps = z1()
        self.ep_lifted_step_ups = z1()
        self.ep_lifted_alt_step_ups = z1()

    # ------------------------------------------------------------------------------------------
    def _new_episode(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        for buf in (self.air_time, self.support_z, self.liftoff_z, self.step_dz, self.zmp, self.zmp_ref,
                    self.cop, self.jerk, self.wheel_clearance, self.riser_height, self.lift_air_time,
                    self.lift_liftoff_z, self.swing_lift_max, self.yaw_rate_f):
            buf[ids] = 0.0
        for buf in (self.contact, self.support_known, self.clean_swing, self.touchdown, self.step_up,
                    self.step_up_alternating, self.blocked, self.touching, self.lift_clean, self.lifted_step_up,
                    self.lifted_step_up_alternating):
            buf[ids] = False
        self.foot_max_z[ids] = -1.0e6
        self.last_step_foot[ids] = -1
        self.last_lift_foot[ids] = -1
        self.swing_foot[ids] = -1
        self.time_since_flight[ids] = 1.0e3
        self.floor_z[ids] = 1.0e6
        self.max_trailing[ids] = 0.0
        self.progress_gain[ids] = 0.0
        self.has_prev[ids] = False
        self.flight[ids] = False
        self.margin[ids] = 0.0
        self.spawn_xy[ids] = self.robot.data.root_link_pos_w[ids, :2]
        self.best_dist[ids] = 0.0
        self.last_progress_time[ids] = 0.0
        for buf in (self.ep_step_ups, self.ep_alt_step_ups, self.ep_flight_time, self.ep_margin_sum,
                    self.ep_single_support_time, self.ep_steps, self.ep_lifted_step_ups, self.ep_lifted_alt_step_ups):
            buf[ids] = 0.0

    def masses(self) -> torch.Tensor:
        if self._masses is None:
            self._masses = self.robot.root_physx_view.get_masses().to(self.device)
        return self._masses

    # ------------------------------------------------------------------------------------------
    def update(self) -> ZmpGaitState:
        env = self.env
        if self._last_step == env.common_step_counter:
            return self
        self._last_step = env.common_step_counter
        cfg = self.cfg
        dt = env.step_dt
        robot = self.robot

        # fresh episodes (episode_length_buf is 1 at the first reward of an episode, 0 at reset)
        self._new_episode(torch.nonzero(env.episode_length_buf <= 1).squeeze(-1))
        episode_time = env.episode_length_buf.float() * dt

        # ---------------- contacts: vertical support force on each wheel ----------------
        f_vec_hist = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_wheel_ids]  # (N, H, 2, 3)
        fz_max = f_vec_hist[..., 2].max(dim=1).values
        contact = fz_max > cfg.support_force_threshold
        # any contact (also a riser face or nose pushing the tire sideways): the strict detector
        touching = f_vec_hist.norm(dim=-1).max(dim=1).values > cfg.support_force_threshold
        f_vec = self.contact_sensor.data.net_forces_w[:, self.sensor_wheel_ids]  # (N, 2, 3)
        self.support_force = f_vec[..., 2].clamp(min=0.0)
        vertical_share = f_vec[..., 2] / f_vec.norm(dim=-1).clamp(min=1.0e-6)

        # tire mid-plane centre = link origin + tire_offset along the axle (local +z)
        link_pos = robot.data.body_link_pos_w[:, self.wheel_ids]  # (N, 2, 3)
        link_quat = robot.data.body_link_quat_w[:, self.wheel_ids]
        axle = math_utils.quat_apply(link_quat.reshape(-1, 4), self._unit_z.expand(link_quat.shape[0] * 2, 3))
        wheel_pos = link_pos + cfg.tire_offset * axle.reshape(-1, 2, 3)
        contact_z = wheel_pos[..., 2] - cfg.wheel_radius
        self.wheel_z = wheel_pos[..., 2]
        # leg length in the body frame: the tire centre's height below the base (body roll excluded)
        rel_w = (wheel_pos - robot.data.root_link_pos_w.unsqueeze(1)).reshape(-1, 3)
        base_q = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        self.wheel_zb = math_utils.quat_apply_inverse(base_q, rel_w)[:, 2].reshape(-1, 2)
        self.rel_lift = self.wheel_zb - self.wheel_zb.flip(-1)

        prev_contact = self.contact
        other_contact = contact.flip(-1)
        liftoff = prev_contact & ~contact
        landing = ~prev_contact & contact

        # liftoff: remember the support it left; the swing is clean if the other wheel carries
        self.liftoff_z = torch.where(liftoff, self.support_z, self.liftoff_z)
        self.clean_swing = torch.where(liftoff, other_contact, self.clean_swing)
        self.air_time = torch.where(liftoff, torch.zeros_like(self.air_time), self.air_time)
        airborne = ~contact
        self.air_time = torch.where(airborne, self.air_time + dt, self.air_time)
        # a swing stops being clean as soon as the other wheel is also off the ground
        self.clean_swing = self.clean_swing & ~(airborne & ~other_contact)

        # flight phase (both wheels airborne), ignored right after the spawn drop
        lift_rule = cfg.step_rule == "lift"
        flight = (~touching if lift_rule else airborne).all(dim=1) & (episode_time > cfg.flight_grace_time)
        self.flight = flight
        self.time_since_flight = torch.where(flight, torch.zeros_like(self.time_since_flight), self.time_since_flight + dt)

        # touch-down events
        touchdown = landing & (self.air_time >= cfg.min_air_time) & self.support_known
        self.touchdown = touchdown
        dz = contact_z - self.liftoff_z
        self.step_dz = torch.where(touchdown, dz, torch.zeros_like(dz))
        # only a NEW highest tread for that wheel counts (stepping down and up again does not pay)
        new_high = contact_z >= self.foot_max_z + 0.5 * cfg.step_up_height
        step_up = touchdown & self.clean_swing & (dz >= cfg.step_up_height) & new_high
        # alternation: this step-up uses the other wheel than the previous one (first one counts)
        foot_idx = torch.arange(2, device=self.device).unsqueeze(0)
        alternating = step_up & (self.last_step_foot.unsqueeze(1) != foot_idx)
        self.air_time = torch.where(landing, torch.zeros_like(self.air_time), self.air_time)

        # strict detector: free of any contact while the other wheel carries, the leg pulled up
        # (retracted relative to the stance leg; counted from before liftoff), landed on a tread
        lift_off = self.touching & ~touching
        lift_land = ~self.touching & touching
        self.lift_liftoff_z = torch.where(lift_off, self.support_z, self.lift_liftoff_z)
        self.lift_clean = torch.where(lift_off, other_contact, self.lift_clean)
        self.swing_lift_max = torch.where(lift_off, self.rel_lift, self.swing_lift_max)
        self.lift_air_time = torch.where(lift_off, torch.zeros_like(self.lift_air_time), self.lift_air_time)
        free = ~touching
        self.lift_air_time = torch.where(free, self.lift_air_time + dt, self.lift_air_time)
        self.lift_clean = self.lift_clean & ~(free & ~other_contact)
        self.swing_lift_max = torch.where(free, torch.maximum(self.swing_lift_max, self.rel_lift), self.swing_lift_max)
        lifted = (
            lift_land & (self.lift_air_time >= cfg.min_air_time) & self.support_known & self.lift_clean
            & (vertical_share >= cfg.landing_vertical_share) & new_high
            & (contact_z - self.lift_liftoff_z >= cfg.step_up_height)
            & (self.swing_lift_max >= cfg.lift_required)
        )
        lifted_alt = lifted & (self.last_lift_foot.unsqueeze(1) != foot_idx)
        self.lift_air_time = torch.where(lift_land, torch.zeros_like(self.lift_air_time), self.lift_air_time)
        self.lifted_step_up = lifted
        self.lifted_step_up_alternating = lifted_alt
        new_lift = torch.where(lifted[:, RIGHT], torch.full_like(self.last_lift_foot, RIGHT),
                               torch.full_like(self.last_lift_foot, LEFT))
        self.last_lift_foot = torch.where(lifted.any(dim=1), new_lift, self.last_lift_foot)
        self.touching = touching

        if lift_rule:
            step_up, alternating = lifted, lifted_alt
        self.step_up = step_up
        self.step_up_alternating = alternating
        stepped = step_up.any(dim=1)
        new_last = torch.where(step_up[:, RIGHT], torch.full_like(self.last_step_foot, RIGHT),
                               torch.full_like(self.last_step_foot, LEFT))
        self.last_step_foot = torch.where(stepped, new_last, self.last_step_foot)

        # support heights / levels
        self.support_z = torch.where(contact, contact_z, self.support_z)
        self.support_known = self.support_known | contact
        self.foot_max_z = torch.where(contact, torch.maximum(self.foot_max_z, contact_z), self.foot_max_z)
        floor_cand = torch.where(contact, contact_z, torch.full_like(contact_z, 1.0e6)).min(dim=1).values
        self.floor_z = torch.minimum(self.floor_z, floor_cand)
        level = self.support_z - self.floor_z.unsqueeze(1)
        trailing = torch.where(self.support_known.all(dim=1), level.min(dim=1).values,
                               torch.zeros_like(self.max_trailing))
        gain = (trailing - self.max_trailing).clamp(min=0.0)
        gain = torch.where(gain >= 0.5 * cfg.step_up_height, gain, torch.zeros_like(gain))
        self.progress_gain = gain
        self.max_trailing = self.max_trailing + gain
        self.contact = contact

        # ---------------- terrain around the wheels (privileged feet scanner) ----------------
        hits = self.scanner.data.ray_hits_w[..., 2]
        hits = torch.nan_to_num(hits, nan=-1.0e3, posinf=-1.0e3, neginf=-1.0e3)
        under = torch.stack([torch.where(m, hits, torch.full_like(hits, -1.0e3)).max(dim=1).values
                             for m in self.under_masks], dim=1)
        ahead = torch.stack([torch.where(m, hits, torch.full_like(hits, -1.0e3)).max(dim=1).values
                             for m in self.ahead_masks], dim=1)
        ref_z = torch.where(contact, contact_z, under)
        self.riser_height = (ahead - ref_z).clamp(-0.5, 0.5)
        self.riser_top = ahead
        self.blocked = self.riser_height > cfg.step_up_height
        self.wheel_clearance = (contact_z - under).clamp(-0.5, 0.5)

        # pending swing wheel: the blocked one; both blocked -> alternate (left first)
        alt_next = torch.where(self.last_step_foot >= 0, 1 - self.last_step_foot,
                               torch.full_like(self.last_step_foot, LEFT))
        pending = torch.full_like(self.last_step_foot, -1)
        pending = torch.where(self.blocked[:, LEFT] & ~self.blocked[:, RIGHT], torch.full_like(pending, LEFT), pending)
        pending = torch.where(self.blocked[:, RIGHT] & ~self.blocked[:, LEFT], torch.full_like(pending, RIGHT), pending)
        pending = torch.where(self.blocked.all(dim=1), alt_next, pending)
        # in single support the swing wheel is the airborne one
        single = contact.sum(dim=1) == 1
        swing_now = torch.where(contact[:, LEFT], torch.full_like(pending, RIGHT), torch.full_like(pending, LEFT))
        self.swing_foot = torch.where(single, swing_now, pending)

        # ---------------- ZMP (multibody / cart-table), heading frame ----------------
        masses = self.masses()  # (N, B)
        com_pos = robot.data.body_com_pos_w  # (N, B, 3)
        com_vel = robot.data.body_com_lin_vel_w
        total_mass = masses.sum(dim=1, keepdim=True)
        support_plane = torch.where(contact.any(dim=1),
                                    (contact_z * contact).sum(dim=1) / contact.sum(dim=1).clamp(min=1),
                                    contact_z.min(dim=1).values)
        if cfg.zmp_model == "cart_table":
            vel = (masses.unsqueeze(-1) * com_vel).sum(dim=1) / total_mass  # (N, 3)
            pos = (masses.unsqueeze(-1) * com_pos).sum(dim=1) / total_mass
        else:
            vel, pos = com_vel, com_pos
        if self.prev_vel is None:
            self.prev_vel = vel.clone()
            self.prev_acc = torch.zeros_like(vel)
        has_prev = self.has_prev.view(-1, *([1] * (vel.dim() - 1)))
        acc = torch.where(has_prev, (vel - self.prev_vel) / dt, torch.zeros_like(vel)).clamp(-cfg.acc_clip, cfg.acc_clip)
        jerk = torch.where(has_prev, (acc - self.prev_acc) / dt, torch.zeros_like(acc))
        self.prev_vel = vel.clone()
        self.prev_acc = acc
        self.has_prev[:] = True

        z0 = support_plane
        if cfg.zmp_model == "cart_table":
            den = (acc[:, 2] + GRAVITY).clamp(min=0.2 * GRAVITY)
            zc = (pos[:, 2] - z0).clamp(min=0.0)
            zmp_w = pos[:, :2] - zc.unsqueeze(1) * acc[:, :2] / den.unsqueeze(1)
            com_jerk = jerk[:, :2]
        else:
            m = masses.unsqueeze(-1)  # (N, B, 1)
            az_g = acc[..., 2:3] + GRAVITY  # (N, B, 1)
            den = (m * az_g).sum(dim=1).squeeze(-1).clamp(min=0.2 * GRAVITY * total_mass.squeeze(-1))
            num = (m * (az_g * pos[..., :2] - acc[..., :2] * (pos[..., 2:3] - z0.view(-1, 1, 1)))).sum(dim=1)
            zmp_w = num / den.unsqueeze(1)
            com_jerk = (m * jerk[..., :2]).sum(dim=1) / total_mass

        base_pos = robot.data.root_link_pos_w
        yaw_q = math_utils.yaw_quat(robot.data.root_link_quat_w)

        def to_heading(p_xy: torch.Tensor) -> torch.Tensor:
            shape = p_xy.shape
            flat = p_xy.reshape(shape[0], -1, 2)
            rel = torch.cat([flat - base_pos[:, None, :2], torch.zeros_like(flat[..., :1])], dim=-1)
            q = yaw_q[:, None, :].expand(-1, rel.shape[1], -1)
            out = math_utils.quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(shape[0], -1, 3)
            return out[..., :2].reshape(shape)

        self.zmp = to_heading(zmp_w)
        self.contact_h = to_heading(wheel_pos[..., :2])
        jerk3 = torch.cat([com_jerk, torch.zeros_like(com_jerk[:, :1])], dim=-1)
        self.jerk = math_utils.quat_apply_inverse(yaw_q, jerk3)[:, :2]

        # contact-force CoP (the physical ZMP: where the resultant reaction passes)
        f = self.support_force * contact
        f_sum = f.sum(dim=1, keepdim=True)
        cop = (f.unsqueeze(-1) * self.contact_h).sum(dim=1) / f_sum.clamp(min=1.0e-6)
        self.cop = torch.where(f_sum > 1.0e-6, cop, self.contact_h.mean(dim=1))

        # support polygon (axis-aligned box in the heading frame) and the ZMP stability margin
        big = torch.full_like(self.contact_h[..., 0], 1.0e3)
        cx_lo = torch.where(contact, self.contact_h[..., 0], big).min(dim=1).values - cfg.patch_half_x
        cx_hi = torch.where(contact, self.contact_h[..., 0], -big).max(dim=1).values + cfg.patch_half_x
        cy_lo = torch.where(contact, self.contact_h[..., 1], big).min(dim=1).values - cfg.patch_half_y
        cy_hi = torch.where(contact, self.contact_h[..., 1], -big).max(dim=1).values + cfg.patch_half_y
        sides = [self.zmp[:, 1] - cy_lo, cy_hi - self.zmp[:, 1]]
        if cfg.margin_axes == "both":
            sides += [self.zmp[:, 0] - cx_lo, cx_hi - self.zmp[:, 0]]
        margin = torch.stack(sides, dim=1).min(dim=1).values
        any_contact = contact.any(dim=1)
        self.margin = torch.where(any_contact, margin.clamp(-1.0, 1.0), torch.full_like(margin, -1.0))

        # ZMP reference: feet centre, shifted toward the next stance wheel before a step,
        # the stance wheel itself in single support; first-order hand-over filter
        mid = self.contact_h.mean(dim=1)
        stance = torch.where(self.swing_foot >= 0, 1 - self.swing_foot, torch.zeros_like(self.swing_foot))
        stance_pt = torch.gather(self.contact_h, 1, stance.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        target = torch.where((self.swing_foot >= 0).unsqueeze(1), mid + cfg.ref_shift * (stance_pt - mid), mid)
        target = torch.where(single.unsqueeze(1), stance_pt, target)
        alpha = min(1.0, dt / max(cfg.ref_tau, 1.0e-6))
        fresh = (env.episode_length_buf <= 1).unsqueeze(1)
        self.zmp_ref = torch.where(fresh, target, self.zmp_ref + alpha * (target - self.zmp_ref))
        self.zmp_ref = torch.where(any_contact.unsqueeze(1), self.zmp_ref, self.zmp)

        wz = robot.data.root_link_ang_vel_w[:, 2]
        self.yaw_rate_f = self.yaw_rate_f + min(1.0, dt / max(cfg.yaw_rate_tau, 1.0e-6)) * (wz - self.yaw_rate_f)

        # ---------------- horizontal progress away from the spawn point ----------------
        dist = torch.norm(robot.data.root_link_pos_w[:, :2] - self.spawn_xy, dim=1)
        improved = dist > self.best_dist + cfg.progress_min_gain
        self.best_dist = torch.where(improved, dist, self.best_dist)
        self.last_progress_time = torch.where(improved, episode_time, self.last_progress_time)

        # ---------------- per-episode statistics ----------------
        self.ep_step_ups += step_up.float().sum(dim=1)
        self.ep_alt_step_ups += alternating.float().sum(dim=1)
        self.ep_lifted_step_ups += lifted.float().sum(dim=1)
        self.ep_lifted_alt_step_ups += lifted_alt.float().sum(dim=1)
        self.ep_flight_time += flight.float() * dt
        self.ep_single_support_time += single.float() * dt
        self.ep_margin_sum += self.margin
        self.ep_steps += 1.0
        return self


def get_gait_state(env: ManagerBasedRLEnv) -> ZmpGaitState:
    """The env's shared :class:`ZmpGaitState`, created on first use and updated for this step."""
    state = getattr(env, "_zmp_gait_state", None)
    if state is None:
        cfg = getattr(env.cfg, "zmp_gait", None) or ZmpGaitCfg()
        state = ZmpGaitState(env, cfg)
        env._zmp_gait_state = state
    return state.update()


def _cmd_gate(env: ManagerBasedRLEnv, command_name: str | None, min_cmd_speed: float) -> torch.Tensor:
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    return (torch.norm(cmd_xy, dim=1) > min_cmd_speed).float()


# ============================================ rewards ============================================
def zmp_tracking_exp(
    env: ManagerBasedRLEnv, sigma_x: float = 0.05, sigma_y: float = 0.08, context: str = "stepping"
) -> torch.Tensor:
    """Kajita ZMP servo error: ``exp(-(ex/sx)^2 - (ey/sy)^2)`` between the dynamic ZMP and the gait reference.

    Zero while airborne. In single support the reference is the stance wheel, so this also
    rewards keeping the centroidal angular momentum constant (no tipping).

    ``context="stepping"`` (default) scores it only while a step is under way: a riser ahead of a
    wheel (reference shifted toward the next stance wheel) or single support. With ``"all"`` it
    also paid for balanced standing, and the first pilot settled for standing still in front of
    the riser (balanced = high tracking reward, no climbing).
    """
    s = get_gait_state(env)
    err = s.zmp - s.zmp_ref
    value = torch.exp(-torch.square(err[:, 0] / sigma_x) - torch.square(err[:, 1] / sigma_y))
    active = s.contact.any(dim=1)
    if context == "stepping":
        active = active & (s.swing_foot >= 0)
    return value * active.float()


def zmp_margin_penalty(env: ManagerBasedRLEnv, safe_margin: float = 0.01, scale: float = 0.05) -> torch.Tensor:
    """Vukobratovic stability margin: shortfall of the ZMP margin below ``safe_margin``.

    ``(safe_margin - margin) / scale`` clipped to [0, 1]; the margin is the distance of the ZMP
    inside the support polygon (negative = outside, numerically). Airborne = 1. Negative weight.
    """
    s = get_gait_state(env)
    return ((safe_margin - s.margin) / scale).clamp(min=0.0, max=1.0)


def cop_weight_transfer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pre-swing weight transfer (ZMP moves to the next support foot in double support).

    In double support with a riser ahead, ``2 * F_stance / (F_L + F_R) - 1`` clipped to [0, 1]:
    0 for an even load, 1 when the swing wheel is fully unloaded and ready to lift.
    """
    s = get_gait_state(env)
    double = s.contact.all(dim=1)
    pending = s.swing_foot >= 0
    stance = (1 - s.swing_foot).clamp(0, 1)
    f = s.support_force * s.contact
    f_stance = torch.gather(f, 1, stance.view(-1, 1)).squeeze(1)
    ratio = f_stance / f.sum(dim=1).clamp(min=1.0e-6)
    value = (2.0 * ratio - 1.0).clamp(0.0, 1.0)
    return value * (double & pending).float()


def com_jerk_l2(env: ManagerBasedRLEnv, jerk_scale: float = 100.0, max_value: float = 4.0) -> torch.Tensor:
    """Kajita's ``R du^2`` (u = CoM jerk): ``|jerk_xy / jerk_scale|^2`` clipped at ``max_value``. Negative weight."""
    s = get_gait_state(env)
    return torch.sum(torch.square(s.jerk / jerk_scale), dim=1).clamp(max=max_value)


def step_up_alternating(
    env: ManagerBasedRLEnv,
    alternation_bonus: float = 1.0,
    command_name: str | None = "base_velocity",
    min_cmd_speed: float = 0.05,
) -> torch.Tensor:
    """Event reward for a clean single-support step onto a higher tread (weight = reward per step).

    ``1 + alternation_bonus`` if the wheel differs from the previous stepping wheel (or it is the
    first step of the episode), else 1. Divided by ``step_dt`` so the weight is the per-event
    reward. Jumps (both wheels airborne during the swing) never count.
    """
    s = get_gait_state(env)
    value = s.step_up.float().sum(dim=1) + alternation_bonus * s.step_up_alternating.float().sum(dim=1)
    return value * _cmd_gate(env, command_name, min_cmd_speed) / env.step_dt


def riser_progress(
    env: ManagerBasedRLEnv,
    unit: float = 0.05,
    flight_cooldown: float = 1.0,
    command_name: str | None = "base_velocity",
    min_cmd_speed: float = 0.05,
) -> torch.Tensor:
    """Event reward for new height reached by the *trailing* wheel (the stair is climbed only once
    both wheels are up), per ``unit`` metres, per event. Withheld if the robot was airborne within
    ``flight_cooldown`` s (hopping up does not pay; the height is still recorded)."""
    s = get_gait_state(env)
    no_recent_flight = (s.time_since_flight > flight_cooldown).float()
    return (s.progress_gain / unit) * no_recent_flight * _cmd_gate(env, command_name, min_cmd_speed) / env.step_dt


def flight_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 while both wheels are off the ground (a jump), after the spawn grace time. Negative weight."""
    return get_gait_state(env).flight.float()


def swing_clearance(env: ManagerBasedRLEnv, clearance_target: float = 0.03) -> torch.Tensor:
    """Single support with a riser in front of the swing wheel: lift it above the riser top.

    Lift above the liftoff support as a fraction of what is needed (riser top + ``clearance_target``),
    clipped to [0, 1]: never negative, so trying a step is not punished; zero otherwise (no riser
    ahead, double support, flight), so lifting a wheel on flat ground earns nothing.
    """
    s = get_gait_state(env)
    single = s.contact.sum(dim=1) == 1
    swing = s.swing_foot.clamp(0, 1).view(-1, 1)
    swing_bottom = torch.gather(s.wheel_z, 1, swing).squeeze(1) - s.cfg.wheel_radius
    liftoff = torch.gather(s.liftoff_z, 1, swing).squeeze(1)
    riser_top = torch.gather(s.riser_top, 1, swing).squeeze(1)
    riser_h = torch.gather(s.riser_height, 1, swing).squeeze(1)
    need = (riser_top + clearance_target - liftoff).clamp(min=clearance_target)
    value = ((swing_bottom - liftoff) / need).clamp(0.0, 1.0)
    active = single & (riser_h > s.cfg.step_up_height)
    return value * active.float()


def swing_leg_retraction(env: ManagerBasedRLEnv, extra: float = 0.02) -> torch.Tensor:
    """Pick the foot up with the LEG: while a riser is ahead of the pending swing wheel, how far
    that leg is retracted relative to the stance leg (tire centre height in the base frame, so
    body roll does not count) as a fraction of the riser height + ``extra``, clipped to [0, 1].

    Also active in double support, where it pays for unloading the wheel by pulling it up. In
    sweeps v1/v2 one shoulder stayed pinned at its limit and that wheel was carried over the
    edges by speed; this term asks each leg to do its own lift.
    """
    s = get_gait_state(env)
    swing = s.swing_foot.clamp(0, 1).view(-1, 1)
    lift = torch.gather(s.rel_lift, 1, swing).squeeze(1)
    riser_h = torch.gather(s.riser_height, 1, swing).squeeze(1)
    active = (s.swing_foot >= 0) & (riser_h > s.cfg.step_up_height) & s.contact.any(dim=1)
    value = (lift / (riser_h + extra).clamp(min=extra)).clamp(0.0, 1.0)
    return value * active.float()


def swing_leg_retraction_gain(env: ManagerBasedRLEnv, extra: float = 0.02) -> torch.Tensor:
    """Potential-based form of :func:`swing_leg_retraction`: pays the INCREASE of the lift fraction
    (per second, so the episode sum is ``Phi(end) - Phi(start)``, times the weight).

    The level form paid every step the leg was held up: the Lift-Mirror policy stood square to a
    9.3 cm riser with the left leg lifted ~10 cm (0.88 of the maximum) until the 4 s stall
    termination, which earned more than that termination cost. Holding the leg up now earns
    nothing; lifting and lowering again nets zero, so the optimal policy is unchanged.
    """
    value = swing_leg_retraction(env, extra)
    prev = getattr(env, "_swing_retraction_prev", None)
    if prev is None:
        prev = value.clone()
    prev = torch.where(env.episode_length_buf <= 1, value, prev)
    env._swing_retraction_prev = value.clone()
    return (value - prev) / env.step_dt


def forward_speed_excess_l2(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", margin: float = 0.15
) -> torch.Tensor:
    """``max(0, v_x - cmd_x - margin)^2`` [m^2/s^2] in the base frame. Negative weight.

    The sweep v1/v2 policies ran into the risers at 1.0-1.3 m/s (command 0.2-0.5) and let the
    impact throw the wheel over the edge instead of lifting it."""
    v = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    return torch.square((v - cmd - margin).clamp(min=0.0))


def yaw_rate_excess_l2(env: ManagerBasedRLEnv, deadband: float = 0.5) -> torch.Tensor:
    """``max(0, |filtered yaw rate| - deadband)^2`` [rad^2/s^2] (low-pass, ``zmp_gait.yaw_rate_tau``).
    Negative weight.

    The wheels sit side by side, so one wheel gets ahead of the other only by yawing. A skewed
    stance (about 0.3-0.6 rad to the riser, held) is enough for a step-to climb; sweep-v2 trial 1
    instead pirouetted up the stairs, a half turn about the stance wheel per step (~5 rad/s),
    which the yaw-rate tracking reward (weight 0.5) did not stop."""
    return torch.square((get_gait_state(env).yaw_rate_f.abs() - deadband).clamp(min=0.0))


def base_height_over_support_l2(env: ManagerBasedRLEnv, target_height: float = 0.27) -> torch.Tensor:
    """``(base_z - higher support - target)^2`` while both wheels carry the robot on a level
    footing (supports within 2 cm); free during steps."""
    s = get_gait_state(env)
    base_z = s.robot.data.root_link_pos_w[:, 2]
    level = (s.support_z[:, 0] - s.support_z[:, 1]).abs() < 0.02
    active = s.contact.all(dim=1) & level
    top = s.support_z.max(dim=1).values
    return torch.square(base_z - top - target_height) * active.float()


def base_roll_pitch_l2(env: ManagerBasedRLEnv, roll_weight: float = 0.25, pitch_weight: float = 1.0) -> torch.Tensor:
    """Weighted ``roll^2`` / ``pitch^2`` from the projected gravity: stepping needs some roll
    (body over the stance wheel), the pitch should stay small. Negative weight."""
    s = get_gait_state(env)
    g = s.robot.data.projected_gravity_b
    return roll_weight * torch.square(g[:, 1]) + pitch_weight * torch.square(g[:, 0])


# ========================================= terminations =========================================
def no_progress_termination(
    env: ManagerBasedRLEnv,
    time_limit: float = 4.0,
    command_name: str | None = "base_velocity",
    min_cmd_speed: float = 0.05,
) -> torch.Tensor:
    """End the episode (as a failure) when a robot told to move has not got ``progress_min_gain``
    further from its spawn point for ``time_limit`` seconds.

    Removes the safe harbour found in the pilots: standing balanced in front of the first riser
    for the whole episode, which beat any attempt to step.
    """
    s = get_gait_state(env)
    stalled = (env.episode_length_buf.float() * env.step_dt - s.last_progress_time) > time_limit
    return stalled & (_cmd_gate(env, command_name, min_cmd_speed) > 0.5)


def nonfinite_state_termination(env: ManagerBasedRLEnv, asset_name: str = "robot") -> torch.Tensor:
    """Reset an env whose simulated robot state became NaN/inf (a rare PhysX blow-up).

    Without it a single exploded env feeds NaN observations to the policy and the whole training
    run dies (three of the four sweep-v1 trials ended this way).
    """
    robot = env.scene[asset_name]
    data = robot.data
    bad = ~torch.isfinite(data.root_state_w).all(dim=1)
    bad |= ~torch.isfinite(data.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(data.joint_vel).all(dim=1)
    return bad


# ========================================= observations =========================================
def zmp_gait_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged critic features (13): ZMP, ZMP reference, margin, contacts, riser heights,
    pending swing wheel (one-hot), last stepping wheel (one-hot)."""
    s = get_gait_state(env)
    pending = torch.stack([(s.swing_foot == LEFT), (s.swing_foot == RIGHT)], dim=1).float()
    last = torch.stack([(s.last_step_foot == LEFT), (s.last_step_foot == RIGHT)], dim=1).float()
    feats = torch.cat(
        [
            s.zmp.clamp(-0.5, 0.5),
            s.zmp_ref.clamp(-0.5, 0.5),
            s.margin.clamp(-0.3, 0.3).unsqueeze(1),
            s.contact.float(),
            s.riser_height.clamp(-0.2, 0.2),
            pending,
            last,
        ],
        dim=1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ========================================== curriculum ==========================================
def _row_step_height(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    """Expected riser height of each env's terrain row (curriculum interpolation of the stair cfg)."""
    terrain = env.scene.terrain
    tg = terrain.cfg.terrain_generator
    sub = next(iter(tg.sub_terrains.values()))
    h_lo, h_hi = sub.step_height_range
    rows = tg.num_rows
    scale = getattr(sub, "difficulty_scale", 1.0)
    level = terrain.terrain_levels[env_ids].float()
    difficulty = ((level + 0.5) / rows * scale).clamp(max=1.0)
    return h_lo + difficulty * (h_hi - h_lo)


def risers_climbed(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    s = env._zmp_gait_state
    return s.max_trailing[env_ids] / _row_step_height(env, env_ids)


def stair_step_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    promote_risers: float = 2.0,
    demote_risers: float = 0.5,
    min_step_ups: float = 2.0,
) -> torch.Tensor:
    """Riser-height curriculum: promote after ``promote_risers`` risers climbed (trailing wheel)
    *with* at least ``min_step_ups`` clean single-support step-ups in the episode (rolling both
    wheels over a low riser does not promote); demote below ``demote_risers``. Returns the mean
    terrain level."""
    terrain = env.scene.terrain
    if getattr(env, "_zmp_gait_state", None) is None:
        return torch.mean(terrain.terrain_levels.float())
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    climbed = risers_climbed(env, ids)
    stepped = env._zmp_gait_state.ep_step_ups[ids]
    move_up = (climbed >= promote_risers) & (stepped >= min_step_ups)
    move_down = (climbed < demote_risers) & ~move_up
    terrain.update_env_origins(ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def gait_episode_stats(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, torch.Tensor] | None:
    """Log-only curriculum term: per-episode step / ZMP statistics of the envs being reset."""
    s = getattr(env, "_zmp_gait_state", None)
    if s is None:
        return None
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return None
    steps = s.ep_steps[ids].clamp(min=1.0)
    ups = s.ep_step_ups[ids]
    lifted = s.ep_lifted_step_ups[ids]
    return {
        "step_ups": ups.mean(),
        "alternating_ratio": (s.ep_alt_step_ups[ids].sum() / ups.sum().clamp(min=1.0)),
        "risers_climbed": risers_climbed(env, ids).mean(),
        "climb_height": s.max_trailing[ids].mean(),
        "flight_frac": (s.ep_flight_time[ids] / (steps * env.step_dt)).mean(),
        "single_support_frac": (s.ep_single_support_time[ids] / (steps * env.step_dt)).mean(),
        "zmp_margin": (s.ep_margin_sum[ids] / steps).mean(),
        # strict detector (leg lifted, no edge contact, landed on a tread), whatever step_rule pays
        "lifted_step_ups": lifted.mean(),
        "lifted_alternating_ratio": (s.ep_lifted_alt_step_ups[ids].sum() / lifted.sum().clamp(min=1.0)),
        "lifted_share": (lifted.sum() / wheel_risers_climbed(env, ids).sum().clamp(min=1.0)).clamp(max=1.0),
    }


def wheel_risers_climbed(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    """Risers climbed by each wheel this episode, summed over both wheels (a step-to stair climb
    needs one step-up per wheel and riser): the denominator of ``lifted_share``."""
    s = env._zmp_gait_state
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    h = _row_step_height(env, ids).unsqueeze(1)
    gain = (s.foot_max_z[ids] - s.floor_z[ids].unsqueeze(1)).clamp(min=0.0, max=10.0)
    return torch.floor(gain / h + 0.3).sum(dim=1)
