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
        event_during_time=0.5,
        cooldown=0.3,
        debug_vis=True,
    )


@configclass
class FlamingoStairJumpRewardsCfg(FlamingoStairRewardsCfg):
    """Phase 1 rewards + hop rewards (gated on the stair-detect event).

    ``track_coin_xy`` and the plain ``base_height`` are disabled in ``__post_init__``
    in favor of the 3D coin reward and a jump-gated base-height term.
    """

    # 3D coin distance (height gap counted) so leaning can't game the dense reward
    track_coin_xyz = RewTerm(
        func=mdp_stair.track_coin_xyz_exp,
        weight=3.0,
        params={"command_name": "coin", "temperature": 1.0, "scaler": 1.0},
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
    # -- hop rewards (small-step tuned; iterate from here)
    # NOTE: stair-specific upward reward WITHOUT the flat-jump vertical-alignment
    # penalty, so the robot hops up-AND-forward onto the step (not in place).
    hop_up = RewTerm(
        func=mdp_stair.hop_up_event,
        weight=1.0,
        params={
            "event_command_name": "stair_event",
            "event_time_range": (0.05, 0.25),
            "target_up_vel": 1.5,
            "temperature": 2.0,
        },
    )
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
            "event_time_range": (0.1, 0.4),
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
    # swap dense coin reward (xy -> 3D) and base-height (plain -> jump-gated)
    cfg.rewards.track_coin_xy = None
    cfg.rewards.base_height = None
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
