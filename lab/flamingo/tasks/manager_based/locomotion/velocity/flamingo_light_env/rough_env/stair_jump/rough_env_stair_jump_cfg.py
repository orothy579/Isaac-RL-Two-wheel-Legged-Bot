# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flamingo-light stair-climbing — Phase 2 (perception-triggered hop).

Extends the Phase 1 stair/coin task with a **stair-detection event**: when the
forward height-scan sees an upward step of >= ``step_threshold`` (default 3 cm), a
hop window opens and the jump rewards (reused from the flat jump task) reward a
vertical take-off so the robot hops its wheels onto the step. Everything else
(forward velocity command, coin shaping in the critic, curriculum) is inherited
from Phase 1.

Deployability: the hop trigger is a deterministic function of the real height
scanner, so the event flag is a legitimate policy observation — the real robot
computes the same signal. (Coins remain training-only / critic-only.)

Warm-start from the Phase 1 stair-drive policy::

    python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo \\
        --algo ppo --headless \\
        --warmstart_ckpt logs/co_rl/Flamingo_Light_Rough_Stair/ppo/<run>/<model_*.pt>
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.stair_rewards as mdp_stair
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.flat_env.jump.jump_rewards as mdp_jump
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.rough_env_stair_drive_cfg import (
    FlamingoRoughEnvCfg as StairDriveRoughEnvCfg,
    FlamingoRoughEnvCfg_PLAY as StairDriveRoughEnvCfg_PLAY,
    FlamingoStairCommandsCfg,
    FlamingoStairRewardsCfg,
)


@configclass
class FlamingoStairJumpCommandsCfg(FlamingoStairCommandsCfg):
    """Phase 1 commands + a perception-triggered hop event."""

    stair_event = mdp.StairDetectEventCommandCfg(
        asset_name="robot",
        sensor_name="height_scanner",
        resampling_time_range=(1.0e9, 1.0e9),  # state machine, not time-resampled
        step_threshold=0.03,  # hop when a >= 3 cm step is detected ahead
        forward_band=(0.15, 0.45),
        y_halfwidth=0.2,
        event_during_time=0.5,  # known-good jump window (bfbca92)
        cooldown=0.3,
        debug_vis=True,
    )


@configclass
class FlamingoStairJumpRewardsCfg(FlamingoStairRewardsCfg):
    """Phase 1 rewards + hop rewards (gated on the stair-detect event).

    Coin guidance is removed (coins sat at the stair center and pinned the robot there);
    the main climb driver is now ``stair_climb`` — an exponential per-step reward for
    reaching a new highest step ANYWHERE on the stairs. All coin rewards + the plain
    base-height are disabled in ``__post_init__``.
    """

    # MAIN CLIMB DRIVER: exponential reward per new highest step reached (any direction).
    stair_climb = RewTerm(
        func=mdp_stair.StairClimbProgress,
        weight=1.0,
        params={
            "ground_sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "step_height": 0.05,
            "growth": 2.0,   # step1->2, step2->4, step3->8, ... (per new step)
            "coef": 1.0,
        },
    )
    # base height held only while NOT hopping (so the body can rise during a hop)
    base_height_jump = RewTerm(
        func=mdp_jump.base_height_when_not_jumping,
        weight=-25.0,
        params={
            "target_height": 0.31,
            "event_command_name": "stair_event",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )
    # -- hop reward: run-1's jump_lin_vel_z, ROLLED BACK but with use_alignment=False
    # (drops the |vz|/|v| factor that penalized forward motion). Keeps the pre-jump
    # fall/load penalty. hop_up was removed: it was a duplicate of this.
    # hop motor reward: known-good timing (bfbca92: take-off at 0.05-0.25 of the window),
    # but use_alignment=False so the take-off may go up-AND-forward (not pure vertical).
    jump_lin_vel_z = RewTerm(
        func=mdp_jump.lin_vel_z_event,
        weight=5,
        params={
            "event_command_name": "stair_event",
            "event_time_range": (0.05, 0.25),
            "max_up_vel": 2.0,
            "up_vel_coef": 10.0,
            "down_vel_coef": 0.0,
            "temperature": 2.0,
            "use_alignment": False,
        },
    )
    # legs fold up (wheel clearance) WHILE moving forward -> clears the riser.
    foot_clearance = RewTerm(
        func=mdp_stair.foot_clearance_event,
        weight=3.0,
        params={
            "event_command_name": "stair_event",
            "event_time_range": (0.1, 0.4),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "ground_sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )
    # push-off and air-phase: small weights ONLY (both are farmable in place).
    jump_push_ground = RewTerm(
        func=mdp_jump.push_ground_event,
        weight=0.05,
        params={
            "event_command_name": "stair_event",
            "event_time_range": (0.05, 0.25),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )
    jump_feet_off = RewTerm(
        func=mdp_jump.feet_off_ground_event,
        weight=10.0,
        params={
            "event_command_name": "stair_event",
            "event_time_range": (0.05, 0.25),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )


def _setup_stair_jump(cfg) -> None:
    """Phase-2 additions on top of the Phase 1 stair setup."""
    # expose the (deployable) hop trigger to policy + critic
    cfg.observations.none_stack_policy.stair_event_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "stair_event"}
    )
    cfg.observations.none_stack_critic.stair_event_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "stair_event"}
    )
    # remove ALL coin guidance (coins pinned the robot to the stair center); climbing is
    # now driven by stair_climb. Also disable the plain (always-on) base-height term.
    cfg.rewards.track_coin_xy = None
    cfg.rewards.coin_collected = None
    cfg.rewards.reach_top = None
    cfg.rewards.heading_to_coin = None
    cfg.rewards.base_height = None
    # coin command/obs/curriculum are now inert (no coin reward references them); drop
    # the coin critic obs and coin curriculum so nothing depends on coins.
    cfg.observations.none_stack_critic.coin_commands = None
    if getattr(cfg.curriculum, "coin_levels", None) is not None:
        cfg.curriculum.coin_levels = None
    # a run-up take-off needs the body to PITCH; the inherited -10 flat-orientation
    # penalty forbids that (flat-jump used only -1). Relax it for the jump task.
    if cfg.rewards.flat_orientation_l2 is not None:
        cfg.rewards.flat_orientation_l2.weight = -2.0
    # stand firmly on a zero command: give the policy much more standing practice
    # (with the command-gated coin rewards, standing envs get no forward pull, so the
    # velocity-tracking reward teaches them to hold still).
    cfg.commands.base_velocity.rel_standing_envs = 0.2
    # responsive step detection / fresher height map for hopping
    if cfg.scene.height_scanner is not None:
        cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt


@configclass
class FlamingoRoughEnvCfg(StairDriveRoughEnvCfg):
    commands: FlamingoStairJumpCommandsCfg = FlamingoStairJumpCommandsCfg()
    rewards: FlamingoStairJumpRewardsCfg = FlamingoStairJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_jump(self)


@configclass
class FlamingoRoughEnvCfg_PLAY(StairDriveRoughEnvCfg_PLAY):
    commands: FlamingoStairJumpCommandsCfg = FlamingoStairJumpCommandsCfg()
    rewards: FlamingoStairJumpRewardsCfg = FlamingoStairJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_jump(self)
