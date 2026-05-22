# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
)

from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_LIGHT_CFG  # isort: skip


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
    # -- Stability: reward staying still (lin_vel commands are 0,0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- Stability penalties
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.005)
    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-1.0)

    # Strong pull back to 0.31m after landing — counters shoulder extension on ground
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-20.0,
        params={
            "target_height": 0.31,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # -- Joint constraints
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_leg_link"]),
            "threshold": 1.0,
        },
    )
    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )

    # -- Smoothness (stricter than flat drive to prevent explosive actions)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
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
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    time_conditioned_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1000.0,
        params={"term_keys": "time_illegal_contact"},
    )
    is_alive = RewTerm(mdp.is_alive, weight=0.1)


@configclass
class FlamingoFlatJumpEnvCfg(LocomotionVelocityFlatEnvCfg):

    rewards: FlamingoJumpRewardsCfg = FlamingoJumpRewardsCfg()

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
        self.commands.base_velocity.ranges.pos_z = (0.35, 0.55)

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "left_leg_link",
            "right_leg_link",
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
            "left_leg_link",
            "right_leg_link",
        ]
