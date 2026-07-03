# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reward terms for the flamingo-light stair-climbing task.

* ``StairClimbProgress`` — exponential reward per new highest step reached (climb driver).
* ``hop_up_event`` / ``foot_clearance_event`` — perception-triggered hop (stair_event).
* ``_cmd_gate`` — gate a reward by the velocity command (so a zero command stands still).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StairClimbProgress(ManagerTermBase):
    """Exponential reward for reaching a NEW greatest HEIGHT — anywhere on the stairs.

    Progress is the absolute height climbed above the pit floor, measured in FIXED nominal
    units of ``step_height`` (0.05 m) — deliberately NOT the terrain's current step height.
    So the exponent counts HEIGHT, not stair count: a tall riser advances it by more (a
    0.15 m step = +3 units -> ``growth**3``) and a shallow one by less. This gives (a)
    taller steps are worth much more per step, and (b) the reward keeps GROWING as the
    step-height curriculum raises the risers, instead of collapsing when a tall step covers
    the same height in fewer stairs.

    Non-farmable: only a NEW per-episode max height pays (the max resets each episode), so
    bobbing or going down-then-up earns nothing. The pit floor is the lowest ground seen
    this episode (robust to the terrain's origin-z convention). Command-gated so a zero
    velocity command earns nothing (the robot then holds still).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ground_sensor = env.scene.sensors[cfg.params["ground_sensor_cfg"].name]
        # max_step = greatest absolute height reached this episode, in nominal units (see
        # __call__). Also read by the ``stair_terrain_levels_climb`` curriculum term.
        self.max_step = torch.zeros(env.num_envs, device=env.device)
        self.base_ground = torch.full((env.num_envs,), 1.0e9, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.max_step[env_ids] = 0.0
        self.base_ground[env_ids] = 1.0e9

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ground_sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
        step_height: float = 0.05,
        growth: float = 2.0,
        coef: float = 1.0,
        vel_command_name: str = "base_velocity",
        min_cmd_speed: float = 0.05,
    ) -> torch.Tensor:
        hits = self.ground_sensor.data.ray_hits_w[:, :, 2]
        finite = torch.isfinite(hits)
        ground_z = (hits * finite).sum(dim=1) / finite.sum(dim=1).clamp(min=1)

        self.base_ground = torch.minimum(self.base_ground, ground_z)
        # progress = ABSOLUTE height climbed in FIXED nominal units (``step_height``=0.05),
        # NOT the terrain's current step height. A 0.15 m riser => +3 units (growth**3), a
        # 0.05 m one => +1 (growth**1): taller steps pay much more per step, and the reward
        # keeps growing as the curriculum raises the risers instead of collapsing.
        cur_step = torch.round((ground_z - self.base_ground) / step_height).clamp(min=0.0)

        new_max = cur_step > self.max_step
        reward = torch.where(new_max, coef * (growth ** cur_step), torch.zeros_like(cur_step))
        self.max_step = torch.maximum(self.max_step, cur_step)

        return reward * _cmd_gate(env, vel_command_name, min_cmd_speed)


class StandStillPosition(ManagerTermBase):
    """Penalize **xy position drift** while under a zero velocity command (hold your ground).

    Unlike a wheel-spin or base-velocity penalty (which only fights instantaneous motion)
    or :func:`stand_origin_base` (which pulls the robot back to its spawn/pit origin — wrong
    on stairs, it would drag the robot back down), this latches an **anchor** at the xy
    position where the zero command began and penalizes squared distance from that anchor
    for as long as the command stays zero. So the robot holds *wherever it currently is* —
    including partway up the stairs — rather than being pulled to a fixed point.

    The anchor is re-latched on every rising edge of "command became zero" and on reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.anchor_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self.was_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        # drop the standing latch so the anchor is re-captured next time the env stands
        self.was_standing[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        # standing = the velocity command (lin_vel_x, lin_vel_y, ang_vel_z) is exactly zero,
        # which is how the command generator marks its standing envs.
        command = env.command_manager.get_command(command_name)
        is_standing = torch.all(command[:, :3] == 0.0, dim=1)

        xy = asset.data.root_pos_w[:, :2]
        # rising edge (just started standing) -> latch the anchor to the current xy
        newly_standing = is_standing & ~self.was_standing
        self.anchor_xy[newly_standing] = xy[newly_standing]
        self.was_standing = is_standing

        drift = torch.sum(torch.square(xy - self.anchor_xy), dim=1)  # squared xy drift [m^2]
        return drift * is_standing.float()


def _cmd_gate(env: ManagerBasedRLEnv, vel_command_name: str, min_cmd_speed: float) -> torch.Tensor:
    """1.0 where the robot is commanded to move (|cmd xy| > min_cmd_speed), else 0.0.

    Used to gate the climb reward by the velocity command, so the robot only earns
    climbing reward WHEN told to move; a zero command earns nothing and it stands still.
    """
    cmd_xy = env.command_manager.get_command(vel_command_name)[:, :2]
    return (torch.norm(cmd_xy, dim=1) > min_cmd_speed).float()


def hop_up_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "stair_event",
    event_time_range: tuple = (0.25, 0.6),
    target_up_vel: float = 2.5,
    up_vel_coef: float = 20.0,
    temperature: float = 2.0,
    load_penalty_coef: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strong take-off reward for the hop window — flat-jump *strength*, but WITHOUT
    the vertical-alignment factor (stairs need up-AND-forward to land on the next step).

    Phases (``event_time`` since the window opened):
    * ``< event_time_range[0]`` (load): penalize dropping, so the robot loads its legs.
    * inside ``event_time_range`` (take-off): reward reaching ``target_up_vel`` upward,
      scaled by ``up_vel_coef`` (this is the impulse the weak ``hop_up`` was missing —
      run 1 climbed because it kept this coefficient).

    NOTE: this term is farmable in place (a vertical hop also earns it), so keep its
    weight modest and let the non-farmable ``StairClimbProgress`` dominate.
    """
    cmd = env.command_manager.get_command(event_command_name)
    flag = cmd[:, 0]
    event_time = cmd[:, 1]

    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_z = asset.data.root_lin_vel_w[:, 2]

    pre_jump = (event_time < event_time_range[0]).float()
    window = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()

    up_reward = torch.exp(-torch.abs((target_up_vel - lin_vel_z) / target_up_vel) * temperature)
    # discourage uncontrolled falling before take-off (load the legs instead)
    descent_vel = torch.clamp(-lin_vel_z, min=0.0)
    load_penalty = torch.clamp(descent_vel - 1.0, min=0.0) * load_penalty_coef

    return up_reward * up_vel_coef * flag * window - load_penalty * flag * pre_jump


def foot_clearance_event(
    env: ManagerBasedRLEnv,
    event_command_name: str = "stair_event",
    event_time_range: tuple = (0.25, 0.7),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_wheel_link"),
    ground_sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
) -> torch.Tensor:
    """Reward folding the legs up (wheel clearance) WHILE moving forward during a hop.

    Encourages the robot to retract its wheels above the ground so they clear the riser
    — the user's "legs fold to go higher". Scaled by forward speed so it pays only when
    hopping *forward over* the step, not when bobbing in place (anti-farming).
    """
    cmd = env.command_manager.get_command(event_command_name)
    flag = cmd[:, 0]
    event_time = cmd[:, 1]

    asset: RigidObject = env.scene[asset_cfg.name]
    wheel_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, n_wheels)

    sensor = env.scene.sensors[ground_sensor_cfg.name]
    ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)  # (N,1)
    clearance = torch.clamp(wheel_z - ground_z, min=0.0).mean(dim=1)  # avg wheel lift [m]

    fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)  # forward only
    window = torch.logical_and(
        event_time >= event_time_range[0], event_time <= event_time_range[1]
    ).float()
    return clearance * fwd_speed * flag * window
