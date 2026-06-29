# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flamingo-light stair-climbing ("coin") task — Phase 1 (driving, no jump).

Builds on the light **rough** stand_drive env (which keeps the forward height-scan in
the policy obs). The robot is driven by a **forward velocity command** (the real
deployment directive) and climbs the stairs it sees via the height-scan.

Coins are TRAINING-ONLY scaffolding — there are no coins on the real robot:
  * coin REWARDS shape clean per-step climbing (training only),
  * the coin OBS is fed to the CRITIC only (privileged), never the policy,
  * the curriculum grows the number of required steps/coins one at a time.

So the deployed policy needs only height_scan + proprioception + a forward velocity
command. Warm-start from the stable stand_drive policy.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.stair_rewards as mdp_stair
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.curriculums as stair_curr
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stand_drive.rough_env_stand_drive_cfg import (
    FlamingoRoughEnvCfg as StandDriveRoughEnvCfg,
    FlamingoRoughEnvCfg_PLAY as StandDriveRoughEnvCfg_PLAY,
    FlamingoRewardsCfg as StandDriveRewardsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_drive.stair_terrain_cfg import (
    STAIR_TERRAINS_CFG,
    TILE_SIZE,
    SUBTERRAIN_BORDER_WIDTH,
    PLATFORM_WIDTH,
    STEP_WIDTH,
    STEP_HEIGHT,
)


@configclass
class FlamingoStairCommandsCfg(CommandsCfg):
    """stand_drive commands + the coin ladder (geometry MUST match STAIR_TERRAINS_CFG)."""

    coin = mdp.CoinSequenceCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),  # only (re)sample on reset
        tile_size=TILE_SIZE,
        border_width=SUBTERRAIN_BORDER_WIDTH,
        platform_width=PLATFORM_WIDTH,
        step_width=STEP_WIDTH,
        step_height=STEP_HEIGHT,
        start_level=1,
        collect_radius=0.35,
        debug_vis=True,
    )


@configclass
class FlamingoStairRewardsCfg(StandDriveRewardsCfg):
    """stand_drive locomotion regularizers + velocity tracking + coin shaping.

    Velocity tracking is kept (forward command = deployment directive). Only the
    integral-position term is disabled in ``__post_init__`` (redundant with the
    coin position shaping). Coin rewards are training-only shaping.
    """

    # -- coin task rewards
    track_coin_xy = RewTerm(
        func=mdp_stair.track_coin_xy_exp,
        weight=3.0,
        params={"command_name": "coin", "temperature": 1.0, "scaler": 1.0},
    )
    heading_to_coin = RewTerm(
        func=mdp_stair.heading_to_coin_exp,
        weight=0.5,
        params={"command_name": "coin", "temperature": 2.0},
    )
    coin_collected = RewTerm(
        func=mdp_stair.coin_collected_bonus,
        weight=25.0,
        params={"command_name": "coin"},
    )
    reach_top = RewTerm(
        func=mdp_stair.reach_top_bonus,
        weight=50.0,
        params={"command_name": "coin"},
    )


@configclass
class FlamingoStairCurriculumCfg(CurriculumCfg):
    """Replace the terrain-level curriculum with the coin-count curriculum."""

    coin_levels = CurrTerm(func=stair_curr.coin_count_levels, params={"command_name": "coin"})


def _setup_stair_task(cfg) -> None:
    """Shared stair-task setup applied after the stand_drive __post_init__."""
    # fixed-height stair terrain, no terrain-level curriculum
    cfg.scene.terrain.terrain_generator = STAIR_TERRAINS_CFG
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.curriculum.terrain_levels = None

    # keep velocity-command tracking (the deployable forward directive); only drop
    # the integral-position term (redundant with the coin position shaping below)
    cfg.rewards.error_track_pos_integral = None

    # Coins are TRAINING-ONLY scaffolding (no coins exist on the real robot):
    #   * coin REWARDS shape clean per-step climbing (training only, vanish at deploy)
    #   * coin OBS goes to the CRITIC only (privileged / asymmetric actor-critic)
    # The POLICY must stay deployable, so it does NOT see the coin — it climbs from
    # height_scan + proprioception + the forward velocity command.
    cfg.observations.none_stack_critic.coin_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "coin"}
    )

    # forward velocity command = the real deployment directive (operator/joystick).
    # Modest forward speed; straight climb (no commanded yaw). Tunable.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

    # spawn at the pit center facing +x (so the stairs/coins are straight ahead)
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
    curriculum: FlamingoStairCurriculumCfg = FlamingoStairCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_task(self)


@configclass
class FlamingoRoughEnvCfg_PLAY(StandDriveRoughEnvCfg_PLAY):
    commands: FlamingoStairCommandsCfg = FlamingoStairCommandsCfg()
    rewards: FlamingoStairRewardsCfg = FlamingoStairRewardsCfg()
    curriculum: FlamingoStairCurriculumCfg = FlamingoStairCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_task(self)
        # deterministic spawn for inspection
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
