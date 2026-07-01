# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flamingo-light stair-climbing task — Phase 1 (driving, no jump).

Builds on the light **rough** stand_drive env (which keeps the forward height-scan in
the policy obs). The robot is driven by a **forward velocity command** (the real
deployment directive) and climbs the stairs it sees via the height-scan.

Climbing is rewarded by ``StairClimbProgress``: an exponential reward each time the
terrain *under the robot* reaches a NEW highest step — anywhere on the stairs (the
robot is free to climb out any side; no coins). ``stand_still`` makes it hold still on
a zero command. Deployed policy needs only height_scan + proprio + a velocity command.
Warm-start from the stable stand_drive policy.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.stair_rewards as mdp_stair
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    CommandsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stand_drive.rough_env_stand_drive_cfg import (
    FlamingoRoughEnvCfg as StandDriveRoughEnvCfg,
    FlamingoRoughEnvCfg_PLAY as StandDriveRoughEnvCfg_PLAY,
    FlamingoRewardsCfg as StandDriveRewardsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.stair_terrain_cfg import (
    STAIR_TERRAINS_CFG,
    STEP_HEIGHT,
)


@configclass
class FlamingoStairCommandsCfg(CommandsCfg):
    """stand_drive commands (no coins)."""

    pass


@configclass
class FlamingoStairRewardsCfg(StandDriveRewardsCfg):
    """stand_drive locomotion + velocity tracking + exponential stair-climb reward."""

    # MAIN CLIMB DRIVER: exponential reward per new highest step reached (any direction).
    stair_climb = RewTerm(
        func=mdp_stair.StairClimbProgress,
        weight=10.0,
        params={
            "ground_sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "step_height": STEP_HEIGHT,
            "growth": 2.0,  # step1->2, step2->4, step3->8, ... (per new step)
            "coef": 1.0,
        },
    )
    # Stop on a zero velocity command: penalize wheel spin when the command is zero
    # (forward command -> inactive). With track_lin_vel_xy_exp -> "no command: stop,
    # command: go forward".
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.5,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
        },
    )


def _setup_stair_task(cfg) -> None:
    """Shared stair-task setup applied after the stand_drive __post_init__."""
    # fixed-height stair terrain, no terrain-level curriculum
    cfg.scene.terrain.terrain_generator = STAIR_TERRAINS_CFG
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.curriculum.terrain_levels = None

    # keep velocity-command tracking (deployable forward directive); drop the
    # integral-position term (a competing position objective).
    cfg.rewards.error_track_pos_integral = None

    # forward velocity command = the real deployment directive (operator/joystick).
    # Modest forward speed; straight climb (no commanded yaw). Tunable.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

    # spawn at the pit center facing +x (so the stairs are straight ahead)
    cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.1, 0.1)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
    }


@configclass
class FlamingoRoughEnvCfg(StandDriveRoughEnvCfg):
    commands: FlamingoStairCommandsCfg = FlamingoStairCommandsCfg()
    rewards: FlamingoStairRewardsCfg = FlamingoStairRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_task(self)


@configclass
class FlamingoRoughEnvCfg_PLAY(StandDriveRoughEnvCfg_PLAY):
    commands: FlamingoStairCommandsCfg = FlamingoStairCommandsCfg()
    rewards: FlamingoStairRewardsCfg = FlamingoStairRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_task(self)
        # deterministic spawn for inspection
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
