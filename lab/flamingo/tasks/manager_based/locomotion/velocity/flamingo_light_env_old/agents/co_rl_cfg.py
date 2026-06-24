# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
    CoRlSrmPpoAlgorithmCfg,
)

######################################## [ PPO CONFIG] ########################################


@configclass
class FlamingoPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "FlamingoLightStand-v0"
    experiment_description = "test"
    empirical_normalization = False
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FlamingoLightFlatPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Light_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

@configclass
class FlamingoLightFlatJumpPPORunnerCfg(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "Flamingo_Light_Flat_Jump"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.learning_rate = 1e-4

        # ── MOO: Lagrangian Reward Autotuning ────────────────────────────────
        # Ref: JJUNE0 et al., "isaaclab_add_reward_autotuning" (POSTECH, 2024)
        #      https://github.com/JJUNE0/isaaclab_add_reward_autotuning
        # Ref: Tessler et al., "Reward Constrained Policy Optimization" ICLR 2019
        #
        # Each term's λ adjusts automatically so that the mean per-step
        # contribution converges to "target".  Terms NOT listed here keep
        # their fixed weights unchanged (termination_penalty, is_alive, etc.).
        #
        # Targets are in the same units as Episode_Reward/<term> in tensorboard
        # (mean per-step contribution, already scaled by weight × step_dt=0.02s).
        # Tune them by observing the tensorboard values after a few hundred iters.
        #
        # Update rule:  log λ ← log λ + lr · sign(init_weight) · (target − actual)
        self.moo = {
            # ── Jump objective ──────────────────────────────────────────────
            # Want wheels to be meaningfully airborne each episode.
            # Target ≈ 0.002 means avg ~0.1 m above standing over 10s episode.
            # (wheel_height_bonus weight×dt = 10×0.02 = 0.2; 0.002/0.2 = 0.01 m lift)
            "wheel_height_bonus": {
                "init_weight": 10.0,
                "target": 0.002,
                "lr": 5e-3,
                "min_abs_weight": 1.0,
                "max_abs_weight": 50.0,
            },
            # ── Stability objectives ────────────────────────────────────────
            # Upright posture: allow small tilt (euler angle ~0.1 rad)
            # flat_orientation returns euler² ≥ 0; weight=-1.0, step_dt=0.02
            # target=-0.002 ≈ euler²<0.1 mean
            "flat_orientation": {
                "init_weight": -1.0,
                "target": -0.002,
                "lr": 2e-3,
                "min_abs_weight": 0.1,
                "max_abs_weight": 5.0,
            },
            # Height regulation: allow deviation during jump but pull back on land
            # base_height returns (z-0.31)²; weight=-15.0, step_dt=0.02
            # target=-0.06 ≈ mean deviation ~0.14 m (reasonable during jumps)
            "base_height": {
                "init_weight": -15.0,
                "target": -0.06,
                "lr": 2e-3,
                "min_abs_weight": 2.0,
                "max_abs_weight": 50.0,
            },
            # ── Spin suppression ────────────────────────────────────────────
            # ang_vel_z_l2 returns ω²; weight=-0.05, step_dt=0.02
            # target=-0.001 ≈ ω_z < 0.3 rad/s mean
            "ang_vel_z_l2": {
                "init_weight": -0.05,
                "target": -0.001,
                "lr": 2e-3,
                "min_abs_weight": 0.005,
                "max_abs_weight": 1.0,
            },
        }


@configclass
class FlamingoLightRoughPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "Flamingo_Light_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        # ── MOO: Lagrangian Reward Autotuning ────────────────────────────────
        # Ref: JJUNE0 et al., "isaaclab_add_reward_autotuning" (POSTECH, 2024)
        # Ref: Tessler et al., "Reward Constrained Policy Optimization" ICLR 2019
        #
        # Only terms that need auto-tuning are listed. Fixed terms like
        # termination_penalty and is_alive are left out intentionally.
        # Targets come from observing Episode_Reward/<term> in tensorboard
        # after ~300 iters; adjust if the policy saturates or collapses.
        self.moo = {
            # Wheel clearance on stairs — want consistent lift when stairs appear
            # weight=5.0, step_dt=0.02 → target≈0.001 means avg ~0.01m clearance
            "wheel_height_bonus": {
                "init_weight": 5.0,
                "target": 0.001,
                "lr": 3e-3,
                "min_abs_weight": 0.5,
                "max_abs_weight": 20.0,
            },
            # Velocity tracking: allow lower tracking when climbing stairs
            # weight=2.0, exp returns ≤1.0 → target≈0.04 means ~73% tracking quality
            "track_lin_vel_xy_exp": {
                "init_weight": 2.0,
                "target": 0.04,
                "lr": 2e-3,
                "min_abs_weight": 0.5,
                "max_abs_weight": 5.0,
            },
            # Stair height tracking: positive reward only on stairs
            # weight=2.0, exp returns ≤1.0 → target≈0.02 means decent height keeping
            "stair_height_tracking": {
                "init_weight": 2.0,
                "target": 0.02,
                "lr": 2e-3,
                "min_abs_weight": 0.5,
                "max_abs_weight": 5.0,
            },
            # Orientation: stairs cause pitch — allow more tilt than flat env
            "flat_orientation": {
                "init_weight": -0.3,
                "target": -0.006,
                "lr": 1e-3,
                "min_abs_weight": 0.05,
                "max_abs_weight": 2.0,
            },
        }
