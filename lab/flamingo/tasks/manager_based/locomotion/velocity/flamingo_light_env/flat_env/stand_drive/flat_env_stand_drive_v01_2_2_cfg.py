# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flat stand_drive on the flamingo_light **v01_2_2** robot (newest camera-less model).

Identical to ``flat_env_stand_drive_cfg`` (same rewards, weights, observations, events,
terminations) except for the robot and the one reward parameter that depends on its
geometry: the base-height target.

v03_0_1 stands 2.2 cm taller than v01_2_2 at the same shoulder angle, and its 0.33 m target
is beyond v01_2_2's full leg extension (0.307 m at the -0.75 rad hard limit). 0.27 m puts
v01_2_2 at shoulder ~= -0.61 rad: inside the inherited ``joint_target_deviation_range``
window [-0.70, -0.55] and inside the soft joint limit (-0.675 rad), so the height and joint
terms no longer pull against each other.

The policy observation layout is unchanged (52-dim actor input with 2 stacks), so this
policy warm-starts the v01_2_2 stair tasks.
"""

from isaaclab.utils import configclass

from lab.flamingo.assets.flamingo.flamingo_light_v01_2_2 import FLAMINGO_LIGHT_V01_2_2_CFG  # isort: skip

from .flat_env_stand_drive_cfg import FlamingoFlatEnvCfg, FlamingoFlatEnvCfg_PLAY

STAND_HEIGHT_V01_2_2 = 0.27
"""Base-height target [m] for v01_2_2 (shoulder ~= -0.61 rad)."""


def _use_v01_2_2(cfg) -> None:
    cfg.scene.robot = FLAMINGO_LIGHT_V01_2_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.rewards.base_height.params["target_height"] = STAND_HEIGHT_V01_2_2


@configclass
class FlamingoV0122FlatEnvCfg(FlamingoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_v01_2_2(self)


@configclass
class FlamingoV0122FlatEnvCfg_PLAY(FlamingoFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_v01_2_2(self)
