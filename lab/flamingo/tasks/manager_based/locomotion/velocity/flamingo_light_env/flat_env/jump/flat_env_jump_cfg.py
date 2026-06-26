# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flamingo-light flat *jump* task.

Built on top of the flat ``stand_drive`` task so a stand_drive policy can be
continued here (warm-start). The only functional additions are:

* an ``event`` command (``mdp.EventCommandCfg``) that opens a jump window a few
  seconds into each episode (``resampling_time_range`` kept at the original
  ``(3.0, 5.0)``),
* the ``event`` flag/elapsed-time exposed to both policy and critic observations,
* jump-specific rewards from ``jump_rewards`` (vertical take-off, push-off,
  air phase), with the stand/drive terms that fight the jump removed or gated.

Continuing a stand_drive policy
-------------------------------
The policy observation gains the 2-dim ``event`` channel, so its input layer
differs from stand_drive by 2 dims. Use the transfer (warm-start) path, which
copies every shape-matching tensor (all the locomotion hidden layers) and
re-initializes only the changed input layer::

    python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-Light-Jump-v1-ppo \\
        --algo ppo --warmstart_ckpt <path-to-stand_drive checkpoint .pt>

(Use ``--resume`` only when continuing a run of *this* same task.)
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.flat_env.jump.jump_rewards as mdp_jump
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    CommandsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.flat_env.stand_drive.flat_env_stand_drive_cfg import (
    FlamingoFlatEnvCfg as StandDriveFlatEnvCfg,
    FlamingoFlatEnvCfg_PLAY as StandDriveFlatEnvCfg_PLAY,
    FlamingoRewardsCfg as StandDriveRewardsCfg,
)


@configclass
class FlamingoJumpCommandsCfg(CommandsCfg):
    """stand_drive commands + a periodic jump trigger."""

    event = mdp.EventCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),  # original timing (jump fires ~3-5 s in)
        rel_standing_envs=0.1,
        event_during_time=1.2,
        debug_vis=True,
    )


@configclass
class FlamingoJumpRewardsCfg(StandDriveRewardsCfg):
    """stand_drive locomotion rewards + jump terms.

    ``lin_vel_z_l2`` from stand_drive directly penalizes vertical velocity, so it
    is disabled here; the base-height term is swapped for an event-gated version
    that only holds the nominal height *between* jumps.
    """

    # NOTE: ``lin_vel_z_l2`` (vertical-velocity penalty) is disabled in
    # ``__post_init__`` below because it directly fights the jump.

    # replace the always-on base-height penalty with one gated off during jumps
    base_height = RewTerm(
        func=mdp_jump.base_height_when_not_jumping,
        weight=-25.0,
        params={
            "target_height": 0.33,
            "event_command_name": "event",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # -- jump task rewards
    lin_vel_z_event = RewTerm(
        func=mdp_jump.lin_vel_z_event,
        weight=2.5,
        params={
            "event_command_name": "event",
            "event_time_range": (0.3, 0.8),
            "max_up_vel": 4.0,
            "up_vel_coef": 20.0,
            "down_vel_coef": 0.0,
            "temperature": 2.0,
        },
    )
    push_ground_event = RewTerm(
        func=mdp_jump.push_ground_event,
        weight=0.1,
        params={
            "event_command_name": "event",
            "event_time_range": (0.3, 0.8),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )
    feet_off_ground_event = RewTerm(
        func=mdp_jump.feet_off_ground_event,
        weight=1.0,
        params={
            "event_command_name": "event",
            "event_time_range": (0.3, 0.8),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )


def _enable_event_observations(cfg) -> None:
    """Re-add the ``event`` command to policy & critic obs (stand_drive sets them None)."""
    event_obs = ObsTerm(func=mdp.generated_commands, params={"command_name": "event"})
    cfg.observations.none_stack_policy.event_commands = event_obs
    cfg.observations.none_stack_critic.event_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "event"}
    )


@configclass
class FlamingoFlatEnvCfg(StandDriveFlatEnvCfg):
    commands: FlamingoJumpCommandsCfg = FlamingoJumpCommandsCfg()
    rewards: FlamingoJumpRewardsCfg = FlamingoJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _enable_event_observations(self)
        # vertical-velocity penalty fights the jump -> disable it
        self.rewards.lin_vel_z_l2 = None


@configclass
class FlamingoFlatEnvCfg_PLAY(StandDriveFlatEnvCfg_PLAY):
    commands: FlamingoJumpCommandsCfg = FlamingoJumpCommandsCfg()
    rewards: FlamingoJumpRewardsCfg = FlamingoJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _enable_event_observations(self)
        # vertical-velocity penalty fights the jump -> disable it
        self.rewards.lin_vel_z_l2 = None
