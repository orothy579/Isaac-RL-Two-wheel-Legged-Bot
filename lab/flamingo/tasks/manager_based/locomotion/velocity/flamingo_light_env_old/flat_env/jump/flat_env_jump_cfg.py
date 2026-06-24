# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env_old.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    TerminationsCfg,
)

from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_LIGHT_CFG  # isort: skip


@configclass
class FlamingoJumpTerminationsCfg(TerminationsCfg):
    base_too_low = DoneTerm(
        func=mdp.base_height_too_low,
        params={"minimum_height": 0.13, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class FlamingoJumpRewardsCfg:
    # -- Jump: reward wheels leaving ground when high pos_z is commanded
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_z_cmd,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"]),
            "threshold": 0.1,
            "z_cmd_threshold": 0.38,
        },
    )
    # Proportional bonus for wheel height above standing position.
    # standing_wheel_height=0.035 is the wheel radius (center height when touching flat ground).
    # Rewards ACTUAL wheel clearance, not base height — more meaningful for stair climbing.
    # No contact-force gate needed: wheel_z > 0.035 already implies airborne.
    wheel_height_bonus = RewTerm(
        func=mdp.wheel_height_bonus,
        weight=10.0,
        params={
            "standing_wheel_height": 0.035,
            "command_name": "base_velocity",
            "z_cmd_threshold": 0.38,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"]),
        },
    )
    # Step-change bonus when BOTH wheels clear 0.15m (15cm stair target).
    # wheel center > 0.15m → wheel bottom > 0.115m → clears a ~11cm step edge.
    wheel_height_target_bonus = RewTerm(
        func=mdp.wheel_height_threshold_bonus,
        weight=15.0,
        params={
            "target_wheel_height": 0.15,
            "command_name": "base_velocity",
            "z_cmd_threshold": 0.38,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"]),
        },
    )

    # -- Stability: reward staying still (lin_vel commands are 0,0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Reward zero yaw velocity — gradient vanishes at high spin, so pair with L2 below
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_link_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # L2 yaw penalty: provides consistent gradient at any spin rate (exp alone vanishes)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_link_l2, weight=-0.05)

    # -- Stability penalties
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.005)
    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-1.0)

    # Pull to 0.31m. During airborne, base_height_bonus (+10.0 × delta) overrides this.
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-15.0,
        params={
            "target_height": 0.31,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # -- Joint constraints
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_caster_link", ".*_shoulder_link", ".*_leg_link", "base_link"]),
            "threshold": 1.0,
        },
    )
    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )

    # -- Smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    dof_torques_joints_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    dof_torques_wheels_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )
    dof_acc_joints_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    dof_acc_wheels_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )

    # -- Episode management
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)
    time_conditioned_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": "time_illegal_contact"},
    )
    is_alive = RewTerm(mdp.is_alive, weight=0.1)


@configclass
class FlamingoFlatJumpEnvCfg(LocomotionVelocityFlatEnvCfg):

    rewards: FlamingoJumpRewardsCfg = FlamingoJumpRewardsCfg()
    terminations: FlamingoJumpTerminationsCfg = FlamingoJumpTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Disable all height/terrain scanners (flat ground, no stair detection needed)
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        # ── Policy observations ──────────────────────────────────────────────
        # base_pos_z ENABLED (sensor_cfg=None → absolute z): policy must know its height
        self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        self.observations.none_stack_policy.lift_mask = None
        self.observations.none_stack_policy.height_scan = None
        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.event_commands = None
        # Enable base_pos_z with absolute z (no height sensor on flat terrain)
        self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None

        # ── Critic observations ───────────────────────────────────────────────
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.event_commands = None
        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_critic.base_height_scan = None
        self.observations.none_stack_critic.left_wheel_height_scan = None
        self.observations.none_stack_critic.right_wheel_height_scan = None
        self.observations.none_stack_critic.lift_mask = None
        self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        # ── Events ───────────────────────────────────────────────────────────
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot.interval_range_s = (7.0, 9.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3)},
        }
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (0.0, 0.0),
            },
        }

        # ── Terrain ──────────────────────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # ── Commands: in-place jump only ─────────────────────────────────────
        # pos_z command acts as jump trigger: > 0.33m → robot should jump
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.42, 0.58)

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "FL_caster_link", "FR_caster_link", "RL_caster_link", "RR_caster_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]


@configclass
class FlamingoFlatJumpEnvCfg_PLAY(FlamingoFlatJumpEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True

        self.observations.none_stack_policy.enable_corruption = False

        self.events.push_robot.interval_range_s = (10.0, 12.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2)},
        }
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 1.0)
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "FL_caster_link", "FR_caster_link", "RL_caster_link", "RR_caster_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]
