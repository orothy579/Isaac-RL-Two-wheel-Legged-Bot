# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env,
)

##
# Register Gym environments.
##


#########################################CoRL##########################################################
#######################################################################################################

###########################################Track Velocity##############################################
gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-v01-2-2-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_v01_2_2_cfg.FlamingoV0122FlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightV0122FlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-v01-2-2-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_v01_2_2_cfg.FlamingoV0122FlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightV0122FlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Sit-Flamingo-Light-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_sit_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Sit-Flamingo-Light-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_sit_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-Jump-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_jump_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatPPORunnerCfg_Jump,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-Jump-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_jump_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatPPORunnerCfg_Jump,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_jump_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairJump,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_jump_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairJump,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStep,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStep,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Perceptive-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepPerceptiveEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStep,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Perceptive-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepPerceptiveEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStep,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLift,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLift,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Perceptive-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftPerceptiveEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLift,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Perceptive-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftPerceptiveEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLift,
    },
)

# Stair-Step-Lift + left-right mirror loss (same envs, PPO with the mirror loss)
gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirror,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirror,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Perceptive-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftPerceptiveEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirror,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Perceptive-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftPerceptiveEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirror,
    },
)

# Lift-Mirror + potential-based swing retraction reward (no pay for holding a leg up)
gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGain,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGain,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainPerceptiveEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGain,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainPerceptiveEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGain,
    },
)

# fresh symmetric start from the flat stand-drive: Lift-Mirror-Gain on risers 5 -> 15 cm
gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-LowStart-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainLowStartEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGainLowStart,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-LowStart-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainLowStartEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGainLowStart,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-LowStart-Perceptive-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainLowStartPerceptiveEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGainLowStart,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-LowStart-Perceptive-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stair_step_cfg.FlamingoStairStepLiftGainLowStartPerceptiveEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightRoughPPORunnerCfg_StairStepLiftMirrorGainLowStart,
    },
)

###########################################Track Velocity##############################################


######################################################################################################
############################################CoRL######################################################
