# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flamingo-light **Phase 2** — climb the stairs (drive up + perception-triggered hop).

Phase split (kept deliberately clean for debugging):

* **Phase 1 = rough ``stand_drive``** — robust locomotion only: hold still on a zero
  command, drive in the commanded direction (x + yaw), and keep the base height fixed at
  0.33. No stairs, no climbing. Train flat ``stand_drive`` first, then warm-start into the
  rough one.
* **Phase 2 = this task** — warm-start from the rough ``stand_drive`` policy and learn to
  climb: the stair terrain + step-height curriculum, the ``StairClimbProgress`` climb
  reward, a position-hold ``stand_still``, and the perception-triggered hop. Everything
  stair/jump-specific lives HERE (Phase 1 has none of it).

Climbing: a forward velocity command drives the robot at the stairs it sees via the
height-scan; ``StairClimbProgress`` pays exponentially per new highest step reached. When
the forward scan sees an upward step >= ``step_threshold`` (3 cm) a hop window opens and
the jump rewards (reused from the flat jump task) reward an up-and-forward take-off so the
robot hops its wheels onto the step. The step height ramps up over the terrain rows
(curriculum) as the robot proves it can climb.

Deployability: the hop trigger is a deterministic function of the real height scanner, so
the event flag is a legitimate policy observation — the real robot computes the same signal.

Warm-start from the Phase 1 rough stand_drive policy::

    python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo \\
        --algo ppo --headless \\
        --warmstart_ckpt logs/co_rl/Flamingo_Light_Rough_Stand_Drive/ppo/<run>/<model_*.pt>
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_jump.stair_rewards as mdp_stair
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.flat_env.jump.jump_rewards as mdp_jump
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    CommandsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stand_drive.rough_env_stand_drive_cfg import (
    FlamingoRoughEnvCfg as StandDriveRoughEnvCfg,
    FlamingoRoughEnvCfg_PLAY as StandDriveRoughEnvCfg_PLAY,
    FlamingoRewardsCfg as StandDriveRewardsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_jump.stair_terrain_cfg import (
    STAIR_TERRAINS_CFG,
    STEP_HEIGHT,
)


@configclass
class FlamingoStairJumpCommandsCfg(CommandsCfg):
    """Base velocity command + a perception-triggered hop event (no flat-jump event)."""

    stair_event = mdp.StairDetectEventCommandCfg(
        asset_name="robot",
        sensor_name="height_scanner",
        resampling_time_range=(1.0e9, 1.0e9),  # state machine, not time-resampled
        step_threshold=0.03,  # hop when a >= 3 cm step is detected ahead
        forward_band=(0.25, 0.35),
        y_halfwidth=0.2,
        event_during_time=0.5,  # known-good jump window (bfbca92)
        cooldown=0.25,
        debug_vis=True,
    )


@configclass
class FlamingoStairJumpRewardsCfg(StandDriveRewardsCfg):
    """Phase 1 locomotion rewards (inherited from stand_drive) + the Phase-2 climb & hop.

    Everything stair/jump-specific is defined here (not in stand_drive):
    ``stair_climb`` (exponential climb driver), ``stand_still`` (xy position hold on a zero
    command), the perception-triggered hop rewards, and a jump-gated base height (swapped
    in ``_setup_stair_jump``).
    """

    # MAIN CLIMB DRIVER: exponential reward per new highest step reached (any direction).
    # step_height is recovered per-env from the terrain-level curriculum inside the term.
    stair_climb = RewTerm(
        func=mdp_stair.StairClimbProgress,
        weight=20.0,
        params={
            "ground_sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "step_height": STEP_HEIGHT,
            "growth": 2.0,  # step1->2, step2->4, step3->8, ... (per new step)
            "coef": 1.0,
        },
    )
    # Stop on a zero velocity command: penalize xy POSITION drift from where the zero
    # command began (hold your ground), so it holds wherever it is — even partway up.
    stand_still = RewTerm(
        func=mdp_stair.StandStillPosition,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
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
    # hop motor reward: known-good timing (bfbca92: take-off at 0.05-0.25 of the window),
    # but use_alignment=False so the take-off may go up-AND-forward (not pure vertical).
    jump_lin_vel_z = RewTerm(
        func=mdp_jump.lin_vel_z_event,
        weight=5.0,
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
        weight=2.0,
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
    # calm touch-down right after each hop: low roll/pitch rate + both wheels back in
    # contact, inside a short post-hop window. Take-off has three rewards; this is the
    # one that teaches the LANDING (directly targets bad_orientation terminations).
    landing_stability = RewTerm(
        func=mdp_stair.LandingStability,
        weight=5.0,
        params={
            "event_command_name": "stair_event",
            "landing_window": 0.4,
            "ang_vel_std": 2.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )
    jump_feet_off = RewTerm(
        func=mdp_jump.feet_off_ground_event,
        weight=8.0,
        params={
            "event_command_name": "stair_event",
            "event_time_range": (0.05, 0.25),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        },
    )


def _setup_stair_jump(cfg) -> None:
    """Turn the inherited Phase-1 stand_drive env into the Phase-2 climb+hop env."""
    # --- stair terrain + step-height curriculum ---
    # all envs start on the shallowest row and are promoted to taller-step rows once
    # StairClimbProgress reports enough stairs climbed.
    cfg.scene.terrain.terrain_generator = STAIR_TERRAINS_CFG
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 0  # everyone begins at the shallowest step
    cfg.curriculum.terrain_levels = CurrTerm(
        func=mdp.stair_terrain_levels_climb,
        params={"reward_term_name": "stair_climb", "promote_steps": 5.0, "demote_steps": 1.0},
    )
    # log-only: mean height climbed per episode in nominal 5 cm units (Curriculum/climb_units).
    # Meaningful even when terrain_levels is disabled (e.g. fixed-height capability probes).
    cfg.curriculum.climb_units = CurrTerm(
        func=mdp.stair_climb_units, params={"reward_term_name": "stair_climb"}
    )

    # --- online reward balancer: ROGER-style penalty-gain adaptation (curated penalties) ---
    # Relax the motion penalties that suppress the run-up hop / climb while that skill is still
    # weak (so the robot dares to attempt it), then tighten them back toward nominal as the task
    # reward grows. Drives off the per-episode reward trend; logs Curriculum/penalty_gain.
    cfg.curriculum.adaptive_reward = CurrTerm(
        func=mdp.AdaptiveRewardBalancer,
        params={
            "positive_terms": ["stair_climb", "jump_lin_vel_z", "jump_feet_off", "foot_clearance"],
            "penalty_terms": ["flat_orientation_l2", "stand_still", "base_height_jump"],
            "penalty_budget": 0.5,   # target: summed penalty ~= 0.5 x summed task reward
            "g_min": 0.1,            # never fully disable a penalty
            "ema": 0.98,             # smoothing of the reward trend across resets
            "adapt_rate": 0.05,      # max +-5% weight move per update
            "warmup_batches": 50,    # collect stats before adapting
            "update_interval": 200,  # move the controller at most every 200 control steps
        },
    )

    # keep velocity-command tracking (deployable forward directive); drop the
    # integral-position term (a competing position objective).
    cfg.rewards.error_track_pos_integral = None

    # forward velocity command = the real deployment directive (operator/joystick).
    # Straight climb (no commanded yaw); Phase 1 already learned x+yaw driving.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.8)
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

    # --- perception-triggered hop additions ---
    # expose the (deployable) hop trigger to policy + critic
    cfg.observations.none_stack_policy.stair_event_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "stair_event"}
    )
    cfg.observations.none_stack_critic.stair_event_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "stair_event"}
    )
    # swap the plain (always-on) base-height for the jump-gated version.
    cfg.rewards.base_height = None
    # a run-up take-off needs the body to PITCH; the inherited -10 flat-orientation
    # penalty forbids that (flat-jump used only -1). Relax it for the jump task.
    if cfg.rewards.flat_orientation_l2 is not None:
        cfg.rewards.flat_orientation_l2.weight = -2.0
    # stand firmly on a zero command: more standing practice (stand_still + the
    # command-gated climb reward mean standing envs get no forward pull).
    cfg.commands.base_velocity.rel_standing_envs = 0.05
    # responsive step detection / fresher height map for hopping
    if cfg.scene.height_scanner is not None:
        cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt


@configclass
class FlamingoRoughEnvCfg(StandDriveRoughEnvCfg):
    commands: FlamingoStairJumpCommandsCfg = FlamingoStairJumpCommandsCfg()
    rewards: FlamingoStairJumpRewardsCfg = FlamingoStairJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_jump(self)


@configclass
class FlamingoRoughEnvCfg_PLAY(StandDriveRoughEnvCfg_PLAY):
    commands: FlamingoStairJumpCommandsCfg = FlamingoStairJumpCommandsCfg()
    rewards: FlamingoStairJumpRewardsCfg = FlamingoStairJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_stair_jump(self)
        # deterministic pit-center spawn for inspection
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
