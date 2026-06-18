# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_LIGHT_CFG  # isort: skip


@configclass
class FlamingoRewardsCfg():
    # -- task
    # Zeroed on stairs: robot should prioritise height over speed when climbing
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_no_stair,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.05,
        },
    )
    # Positive height-tracking reward activated only when stairs are detected
    stair_height_tracking = RewTerm(
        func=mdp.track_base_height_exp_on_stair,
        weight=2.0,
        params={
            "target_height": 0.26, # origin : 0.31
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.05,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Reduced further: -0.5 directly penalized the vertical velocity a jump needs
    # (flat jump env has no such term). Keep a small value to damp bouncing on flat.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-0.1)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.02)

    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
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
    # Removed: DelayedPD actuator produces enormous applied-vs-computed torque
    # differences on stairs, causing value loss explosion and policy NaN.
    # joint_applied_torque_limits = RewTerm(...)
    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    shoulder_motion_no_stair = RewTerm(
        func=mdp.shoulder_motion_no_stair,
        weight=0.0,  # GAP task: disabled — it penalized the shoulder crouch needed to jump
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint"),
            "height_scan_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.05,
        },
    )

    # Reduced: stairs cause natural body pitch, so strict flat orientation is counterproductive
    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-0.3)
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-5.0,  # Reduced from -25.0: target height shifts as robot climbs stairs
        params={
            "target_height": 0.26, #origin 0.31
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

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

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    # Break the safe local optimum: a hop that risks a fall must not be a -50 gamble
    # for a tiny climbing bonus. Soften the fall penalty so the policy can experiment
    # with jumping, and drop the per-step survival reward so "drive around and stay
    # alive" is no longer a profitable do-nothing strategy.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    time_conditioned_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-20.0,
        params={"term_keys": "time_illegal_contact"},
    )
    is_alive = RewTerm(mdp.is_alive, weight=0.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_height_scan,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"]),
            "height_scan_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.05,
            "air_time_threshold": 0.12,
        },
    )
    # Proportional wheel clearance bonus on stairs — stair-gated counterpart to
    # feet_air_time_height_scan. Returns 0 on flat terrain so normal drive is unaffected.
    wheel_height_bonus = RewTerm(
        func=mdp.wheel_height_bonus_on_stair,
        weight=15.0,
        params={
            "standing_wheel_height": 0.035,
            "height_scan_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"]),
        },
    )
    # Fixed bonus when BOTH wheels clear 0.10m on stairs (10cm step target).
    # Flat env uses 0.15m; rough env uses 0.10m because step heights vary (curriculum).
    wheel_height_target_bonus = RewTerm(
        func=mdp.wheel_height_threshold_bonus_on_stair,
        weight=15.0,
        params={
            "target_wheel_height": 0.07,
            "height_scan_cfg": SceneEntityCfg("height_scanner"),
            "height_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel_link"]),
        },
    )
    # Ported from flat jump env (base_height_bonus_airborne): rewards the wheels
    # actually leaving the ground on a stair, proportional to how high the base
    # rises above the local terrain. This is the dense "lift-off" gradient that the
    # position-only wheel bonuses above don't provide — the missing ingredient that
    # makes the flat jump env jump.
    jump_airborne_bonus = RewTerm(
        func=mdp.base_height_bonus_airborne_on_stair,
        weight=0.0,  # GAP task: disabled — height-over-gap-floor would over-reward hovering
        params={
            "standing_height": 0.26,
            "height_scan_cfg": SceneEntityCfg("height_scanner"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"]),
            "height_threshold": 0.05,
            "force_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    # Kills the spurious jump at spawn: penalize being fully airborne when NO stair
    # is detected (privileged scan, reward-side only — policy stays blind). Forces
    # jumping to correlate with an actual step.
    airborne_no_stair_penalty = RewTerm(
        func=mdp.airborne_no_stair_penalty,
        weight=0.0,  # GAP task: disabled — it penalized the airborne takeoff from the flat platform
        params={
            "height_scan_cfg": SceneEntityCfg("height_scanner"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"]),
            "height_threshold": 0.05,
            "force_threshold": 1.0,
        },
    )
    # GAP task: reward forward speed while airborne = leaping across the gap. This is
    # the core "crossing pays" signal — the only way to make forward progress over a
    # gap (which the wheels can't roll across) is to jump it.
    airborne_forward_progress = RewTerm(
        func=mdp.airborne_forward_progress,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"]),
            "force_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # environment
        self.episode_length_s = 20.0
        # scene
        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Sensors commented out in velocity_env_cfg: disable them explicitly
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        # ── Policy observations (sim2real: only real-robot sensors) ──────────
        # Policy uses: joint_pos/vel (encoders), base_ang_vel/gravity (IMU),
        #              actions, velocity_commands — nothing else.
        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.event_commands = None
        self.observations.none_stack_policy.height_scan = None       # critic only
        self.observations.none_stack_policy.base_lin_vel = None      # no estimator on real robot
        self.observations.none_stack_policy.base_pos_z = None        # no base_height_scanner
        self.observations.none_stack_policy.current_reward = None    # sim only
        self.observations.none_stack_policy.is_contact = None        # no wheel force sensors
        self.observations.none_stack_policy.lift_mask = None         # no mask sensors

        # ── Critic observations (privileged, sim only) ────────────────────────
        # Critic additionally uses: height_scan (height_scanner), ground-truth
        # velocities, contact info, current reward.
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.event_commands = None
        self.observations.none_stack_critic.base_height_scan = None       # sensor disabled
        self.observations.none_stack_critic.left_wheel_height_scan = None  # sensor disabled
        self.observations.none_stack_critic.right_wheel_height_scan = None # sensor disabled
        self.observations.none_stack_critic.lift_mask = None               # sensor disabled
        # base_pos_z without sensor: returns raw z height (still useful as privilege)
        self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None
        self.observations.none_stack_critic.height_scan.func = mdp.height_scan_wheel

        # joint reset
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # push disturbance
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        }
        # mass randomization
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # friction: higher than flat env to help with stair contact
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.4, 0.4)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.0, 0.0),
            },
        }

        # Terrain curriculum: start at easy levels and let curriculum advance.
        # Narrow each sub-terrain so the robot meets stairs more often and from
        # closer to the spawn point (do not edit shared stair_config.py).
        self.scene.terrain.max_init_terrain_level = 2
        if self.scene.terrain.terrain_generator is not None:
            tg = self.scene.terrain.terrain_generator
            tg.num_rows = 10
            tg.num_cols = 10
            tg.curriculum = True
            tg.size = (10.0, 10.0)
            tg.border_width = 2.5
            # GAP terrain: spawn on the central platform, surrounded by a gap on all
            # sides. To go anywhere (and satisfy the velocity command) the robot MUST
            # jump the gap — wheels can't roll across (they fall in). This removes the
            # safe "drive around" basin and forces the jump. gap_width is the
            # curriculum difficulty: 0.1m (easy) -> 0.6m (hard).
            tg.difficulty_range = (0.0, 1.0)
            tg.sub_terrains = {
                "gap": terrain_gen.MeshGapTerrainCfg(
                    proportion=1.0,
                    gap_width_range=(0.1, 0.6),
                    platform_width=2.5,
                ),
            }

        # Enable pos_z command so robot learns to lift legs on stairs
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.ranges.pos_z = (0.1931942, 0.3531942)

        # terminations: only terminate on hard base impact (not light brushes)
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]
        self.terminations.base_contact.params["threshold"] = 20.0  


@configclass
class FlamingoRoughEnvCfg_PLAY(FlamingoRoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Disable noise for evaluation
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False

        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot.interval_range_s = (10.0, 12.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3)},
        }
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 1.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.5708, 1.5708)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.scene.terrain.max_init_terrain_level = 2
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True

        # terminations: relaxed for evaluation
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "left_leg_link",
            "right_leg_link",
            "base_link"
        ]
