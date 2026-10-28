# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flamingo-light v01_2_2 - climb stairs ONE WHEEL AT A TIME (alternating feet, ZMP-based rewards).

Replaces the jump approach of ``stair_jump`` (which is kept unchanged) with a stepping gait:
drive to the riser, shift the load onto one wheel (ZMP to the stance wheel), lift the other
wheel onto the tread while the stance wheel carries the robot, then bring the second wheel up.
Every reward of this task is in ``zmp_rewards.py`` (new) or a generic regularizer from ``mdp``;
nothing of the jump task is reused.

Observations:
* policy - exactly the v01_2_2 flat ``stand_drive`` layout (proprioception x 3 frames +
  velocity command = 52), so the stand-drive actor warm-starts unchanged. It is blind: the
  v01_2_2 robot has no camera. The ``...-Perceptive-...`` variant appends a wheel-area height
  scan (needs ``--warmstart_widen``).
* critic - the stand-drive critic (203) + 13 privileged ZMP / gait features appended at the
  end (``--warmstart_widen`` keeps the stand-drive critic weights for the first 203 inputs).

Warm start from the v01_2_2 flat stand-drive policy::

    ./run_gb10.sh scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-v1-ppo \\
        --algo ppo --headless --num_envs 4096 --max_iterations 5000 --batch_exit \\
        --warmstart_ckpt logs/co_rl/Flamingo_Light_v01_2_2_Flat_Stand_Drive/ppo/<run>/model_1499.pt --warmstart_widen
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.assets.flamingo.flamingo_light_v01_2_2 import (  # isort: skip
    FLAMINGO_LIGHT_V01_2_2_CFG,
    WHEEL_RADIUS_V01_2_2,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from ...default_height_map import use_default_height_map
from . import zmp_rewards as zmp
from .stair_step_terrain_cfg import STAIR_STEP_LIFT_TERRAINS_CFG, STAIR_STEP_TERRAINS_CFG

STAND_HEIGHT = 0.27  # v01_2_2 stand-drive base height (shoulder ~= -0.61 rad)


@configclass
class StairStepRewardsCfg:
    """Alternating-feet stair climbing. Event terms (step_up, riser_progress) are divided by
    step_dt inside the term, so their weight is the reward per event."""

    # -- task: follow the forward command (the deployable directive). std 0.3 (stand_drive: 0.5):
    #    with 0.5 a robot standing blocked at a riser still earned 0.6-0.85 of the reward, and the
    #    pilot settled for standing in front of the stairs
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.3}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- alternating-feet climbing
    step_up = RewTerm(func=zmp.step_up_alternating, weight=10.0, params={"alternation_bonus": 1.0})
    riser_progress = RewTerm(func=zmp.riser_progress, weight=1.0, params={"unit": 0.05, "flight_cooldown": 1.0})
    swing_clearance = RewTerm(func=zmp.swing_clearance, weight=4.0, params={"clearance_target": 0.03})
    flight = RewTerm(func=zmp.flight_penalty, weight=-5.0)
    # -- ZMP (Kajita tracking servo, Vukobratovic margin, pre-swing load transfer, CoM jerk)
    zmp_tracking = RewTerm(
        func=zmp.zmp_tracking_exp, weight=1.0, params={"sigma_x": 0.05, "sigma_y": 0.08, "context": "stepping"}
    )
    zmp_margin = RewTerm(func=zmp.zmp_margin_penalty, weight=-1.0, params={"safe_margin": 0.01, "scale": 0.05})
    cop_transfer = RewTerm(func=zmp.cop_weight_transfer, weight=1.0)
    com_jerk = RewTerm(func=zmp.com_jerk_l2, weight=-0.02, params={"jerk_scale": 100.0, "max_value": 4.0})
    # -- posture
    base_height = RewTerm(func=zmp.base_height_over_support_l2, weight=-10.0, params={"target_height": STAND_HEIGHT})
    roll_pitch = RewTerm(func=zmp.base_roll_pitch_l2, weight=-2.0, params={"roll_weight": 0.25, "pitch_weight": 1.0})
    # -- regularizers (generic mdp terms, stand-drive weights)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.02)
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_leg_link"]),
            "threshold": 1.0,
        },
    )
    dof_torques_joints_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-5.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])}
    )
    dof_torques_wheels_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-5.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])}
    )
    dof_acc_joints_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])}
    )
    dof_acc_wheels_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


def _proprioceptive_observations(cfg) -> None:
    """The v01_2_2 flat stand_drive observation layout (actor 52, critic 203), unchanged."""
    obs = cfg.observations
    # one SceneEntityCfg per term: the managers resolve (and write ids into) each instance
    for group in (obs.stack_policy, obs.stack_critic):
        group.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_shoulder_joint")
        group.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_joint", ".*_wheel_joint"]
        )
    obs.stack_policy.joint_vel.scale = 0.15
    obs.stack_critic.joint_vel.scale = 0.15
    for name in ("base_lin_vel", "base_pos_z", "current_reward", "is_contact", "lift_mask", "height_scan",
                 "roll_pitch_commands", "event_commands"):
        setattr(obs.none_stack_policy, name, None)
    for name in ("roll_pitch_commands", "event_commands", "base_height_scan", "left_wheel_height_scan",
                 "right_wheel_height_scan", "lift_mask"):
        setattr(obs.none_stack_critic, name, None)
    obs.none_stack_critic.base_pos_z.params["sensor_cfg"] = None
    # privileged ZMP / gait features, appended LAST so the first 203 critic inputs keep their meaning
    obs.none_stack_critic.zmp_gait = ObsTerm(func=zmp.zmp_gait_features)


def _setup_stair_step(cfg) -> None:
    cfg.episode_length_s = 20.0
    cfg.scene.robot = FLAMINGO_LIGHT_V01_2_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.zmp_gait.wheel_radius = WHEEL_RADIUS_V01_2_2

    # -- sensors: critic height map (default grid), privileged feet scanner, wheel contacts
    cfg.scene.base_height_scanner = None
    cfg.scene.left_wheel_height_scanner = None
    cfg.scene.right_wheel_height_scanner = None
    cfg.scene.left_mask_sensor = None
    cfg.scene.right_mask_sensor = None
    cfg.scene.feet_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        ray_alignment="yaw",
        # x -0.15..0.25 (9) x y -0.175..0.175 (8): the +-0.175 columns run under the tires (+-0.172)
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.40, 0.35]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    _proprioceptive_observations(cfg)
    use_default_height_map(cfg)
    step = cfg.decimation * cfg.sim.dt
    cfg.scene.height_scanner.update_period = step
    cfg.scene.feet_scanner.update_period = step

    # -- terrain: stair curriculum (all envs start on the 5 cm row)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = STAIR_STEP_TERRAINS_CFG.copy()
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = CurrTerm(
        func=zmp.stair_step_terrain_levels,
        params={"promote_risers": 2.0, "demote_risers": 0.5, "min_step_ups": 2.0},
    )
    cfg.curriculum.gait = CurrTerm(func=zmp.gait_episode_stats)

    # -- commands: walk forward into the stairs (straight, no yaw)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.02

    # -- events
    cfg.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
    cfg.events.push_robot.interval_range_s = (10.0, 15.0)
    cfg.events.push_robot.params = {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3)}}
    cfg.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
    cfg.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
    cfg.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
    # Heading: the legs move the wheels only vertically, so the two tires always sit side by side.
    # Square to a riser both tire centres are equally far from it: the stance wheel is blocked
    # too, and a lifted wheel can never be put down on the tread. One wheel can step up only if
    # the robot is skewed so that sin(yaw) * 0.344 m (track) > ~0.10 m, i.e. |yaw| > ~17 deg.
    # Spawn headings up to +-0.6 rad (34 deg) give oblique approaches; near-square ones must
    # turn first.
    cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.6, 0.6)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    }

    # -- v01_2_2's shoulder hard limit is -0.75 rad, exactly the inherited termination threshold, so
    #    a target past the limit pressed the joint a hair beyond it and ended the episode (57 % of
    #    the terminations in the first pilot). Keep the termination for real violations only.
    #    Do NOT clip the shoulder targets: the stand-drive policy commands targets past the limit
    #    for full PD torque when it stands up after the spawn drop; with targets clipped to
    #    [-0.745, 0] every warm-started robot fell over within 0.2 s.
    cfg.terminations.shoulder_lower_violation.params["threshold"] = -0.77
    # a dynamic step lets the body roll toward the swing wheel, which then lands on the higher
    # tread and catches it: allow 0.6 rad (34 deg) instead of the inherited 0.5 rad
    cfg.terminations.bad_orientation.params["limit_angle"] = 0.6

    # -- terminations: body, legs or casters touching anything; tipping (inherited 0.5 rad)
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = [
        "base_link", "left_shoulder_link", "right_shoulder_link", "left_leg_link", "right_leg_link", ".*_caster_link",
    ]
    # -- no progress for 4 s while commanded to move = failure. Without it the pilots settled for
    #    standing balanced in front of the first riser (97 % time-outs, zero steps).
    cfg.terminations.no_progress = DoneTerm(func=zmp.no_progress_termination, params={"time_limit": 4.0})
    # -- reset an env whose physics state became NaN (sweep v1 lost three of four runs to this)
    cfg.terminations.nonfinite_state = DoneTerm(func=zmp.nonfinite_state_termination)


def _add_feet_scan_to_policy(cfg) -> None:
    """Perceptive variant: wheel-area height scan (72 rays) appended to the policy input."""
    cfg.observations.none_stack_policy.feet_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("feet_scanner"), "offset": STAND_HEIGHT},
        clip=(-0.5, 0.5),
    )


@configclass
class FlamingoStairStepEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: StairStepRewardsCfg = StairStepRewardsCfg()
    zmp_gait: zmp.ZmpGaitCfg = zmp.ZmpGaitCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_step(self)


@configclass
class FlamingoStairStepEnvCfg_PLAY(FlamingoStairStepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.sim.render_interval = self.decimation
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        # spread the play envs over all riser heights (rows); the curriculum still moves them
        self.scene.terrain.max_init_terrain_level = 9


@configclass
class FlamingoStairStepPerceptiveEnvCfg(FlamingoStairStepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_feet_scan_to_policy(self)


@configclass
class FlamingoStairStepPerceptiveEnvCfg_PLAY(FlamingoStairStepEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _add_feet_scan_to_policy(self)


# ===================================== lift variant (sweep v3) =====================================
# Sweeps v1/v2 found loopholes: on the 5-7 cm rows (below the 7.6 cm wheel radius) a wheel pushed
# into the riser rolls or bounces over the edge, the old step detector (vertical force only) took
# that for a swing, and the perceptive policies kept one shoulder pinned at its limit and ran at
# 0.8-1.3 m/s; the blind v2 trial 1 lifted both legs but pirouetted up the stairs (~5 rad/s).
# Here every riser is above the axle (9-15 cm), a step-up only pays when the leg itself lifted the
# wheel (zmp_gait.step_rule = "lift"), and running into the risers and spinning are penalized.
@configclass
class StairStepLiftRewardsCfg(StairStepRewardsCfg):
    """v1/v2 rewards; ``step_up``, its alternation bonus and the curriculum's step count come from
    the strict detector. Plus: each leg pulls its own wheel up, no running into the risers."""

    swing_retraction = RewTerm(func=zmp.swing_leg_retraction, weight=2.0, params={"extra": 0.02})
    speed_excess = RewTerm(
        func=zmp.forward_speed_excess_l2, weight=-5.0, params={"command_name": "base_velocity", "margin": 0.15}
    )
    # no pirouetting up the stairs (sweep-v2 trial 1: ~5 rad/s, a half turn per step)
    yaw_rate_excess = RewTerm(func=zmp.yaw_rate_excess_l2, weight=-2.0, params={"deadband": 0.5})


def _setup_lift(cfg) -> None:
    cfg.zmp_gait.step_rule = "lift"
    cfg.scene.terrain.terrain_generator = STAIR_STEP_LIFT_TERRAINS_CFG.copy()


@configclass
class FlamingoStairStepLiftEnvCfg(FlamingoStairStepEnvCfg):
    rewards: StairStepLiftRewardsCfg = StairStepLiftRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_lift(self)


@configclass
class FlamingoStairStepLiftEnvCfg_PLAY(FlamingoStairStepEnvCfg_PLAY):
    rewards: StairStepLiftRewardsCfg = StairStepLiftRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_lift(self)


@configclass
class FlamingoStairStepLiftPerceptiveEnvCfg(FlamingoStairStepLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_feet_scan_to_policy(self)


@configclass
class FlamingoStairStepLiftPerceptiveEnvCfg_PLAY(FlamingoStairStepLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _add_feet_scan_to_policy(self)
