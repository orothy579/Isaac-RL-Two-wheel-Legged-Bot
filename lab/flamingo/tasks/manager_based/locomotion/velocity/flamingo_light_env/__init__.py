# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env
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
    id="Isaac-Flat-Jump-Flamingo-Light-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_jump_cfg.FlamingoFlatJumpEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatJumpPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Flat-Jump-Flamingo-Light-Play-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_jump_cfg.FlamingoFlatJumpEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoLightFlatJumpPPORunnerCfg,
    },
)


###########################################Track Velocity##############################################


######################################################################################################
############################################CoRL######################################################