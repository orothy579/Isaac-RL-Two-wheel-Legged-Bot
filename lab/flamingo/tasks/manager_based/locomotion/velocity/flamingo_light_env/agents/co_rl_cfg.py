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
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
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
class FlamingoLightV0122FlatPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "Flamingo_Light_v01_2_2_Flat_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoLightFlatPPORunnerCfg_Jump(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Light_Flat_Jump"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoLightRoughPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Light_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoLightRoughPPORunnerCfg_Stair(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Light_Rough_Stair"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class FlamingoLightRoughPPORunnerCfg_StairStep(FlamingoPPORunnerCfg):
    """Alternating-feet (ZMP) stair climbing on v01_2_2, warm-started from its flat stand-drive."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Light_Rough_Stair_Step"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        # the warm start keeps the actor but not its (converged, ~0.06) action std: start with
        # enough leg exploration to find a step without wrecking the inherited balance
        # (0.4 in the first pilot: 11 % flight time and 32 % tip-overs from noise alone)
        self.policy.init_noise_std = 0.25
        # sweep v1 used 0.01 (pilot 2: 0.005 let the std collapse 0.25 -> 0.13 before any step was
        # found, but that was before the no-progress termination). With 0.01 every v1 trial pushed
        # the LEFT shoulder std to the 3.0 clamp of ppo.py (random +-3 rad targets: the leg flails at
        # its joint limits) and three of four runs died of a NaN physics blow-up.
        # v2: 0.005, and the std is capped at 0.4 (shoulder range is 0.75 rad, wheel action 1 = 40 rad/s).
        self.algorithm.entropy_coef = 0.005
        self.algorithm.max_std = 0.4
        # actions past +-1 have no physical meaning (shoulder range [-0.75, 0] rad, wheel 40 rad/s):
        # bound them (and NaN-sanitize) at the env boundary; PPO keeps the unclipped samples
        self.action_clip = 1.0
        # warm start: the first PPO update at 1e-3 broke the transferred policy (KL so large that the
        # adaptive schedule fell to its 1e-5 floor at once; the perceptive pilot never recovered).
        # Start low and let the adaptive schedule raise it.
        self.algorithm.learning_rate = 1.0e-4


@configclass
class FlamingoLightRoughPPORunnerCfg_StairStepMirror(FlamingoLightRoughPPORunnerCfg_StairStep):
    """Sweep-v2 stair_step (loose step-up rule: a wheel rolled or bounced over a riser also counts) +
    left-right mirror loss: stage 1 of the two-stage symmetric run. A strict-rule run from the flat
    stand-drive never lifted a wheel in 10000 it; the loose rule is what bootstrapped stepping in v2."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Flamingo_Light_Rough_Stair_Step_Mirror"
        self.mirror_fn = (
            "lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_step."
            "symmetry:stair_step_policy_mirror"
        )
        self.algorithm.mirror_loss_coef = 1.0


@configclass
class FlamingoLightRoughPPORunnerCfg_StairStepLift(FlamingoLightRoughPPORunnerCfg_StairStep):
    """Sweep v3 (risers above the wheel radius, a step-up only pays when the leg lifted the wheel):
    the v2 PPO settings, logged under their own experiment."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Flamingo_Light_Rough_Stair_Step_Lift"


@configclass
class FlamingoLightRoughPPORunnerCfg_StairStepLiftMirror(FlamingoLightRoughPPORunnerCfg_StairStepLift):
    """Stair-Step-Lift + left-right mirror loss. Sweep v3 trial 3 started EVERY staircase with the
    right wheel (256 of 256 first steps); approaching with the left wheel ahead it stalled. The
    rewards are symmetric, so the mirror loss transfers the right-first first step to the left."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Flamingo_Light_Rough_Stair_Step_Lift_Mirror"
        self.mirror_fn = (
            "lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_step."
            "symmetry:stair_step_policy_mirror"
        )
        self.algorithm.mirror_loss_coef = 1.0


@configclass
class FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGain(FlamingoLightRoughPPORunnerCfg_StairStepLiftMirror):
    """Lift-Mirror with the potential-based swing retraction reward (``...-Lift-Mirror-Gain-...``)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain"


@configclass
class FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGainLowStart(FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGain):
    """Fresh symmetric run from the v01_2_2 flat stand-drive: mirror loss from iteration 0, potential
    retraction reward, risers 5 -> 15 cm with the strict lift rule (``...-Lift-Mirror-Gain-LowStart-...``)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain_LowStart"
        self.max_iterations = 10000


@configclass
class FlamingoLightRoughPPORunnerCfg_StairJump(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "Flamingo_Light_Rough_Stair_Jump"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
