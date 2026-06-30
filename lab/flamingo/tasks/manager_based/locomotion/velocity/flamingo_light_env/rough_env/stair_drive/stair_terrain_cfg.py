# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Dedicated stair terrain for the coin task.

Inverted pyramid stairs (robot spawns at the low pit center and climbs outward/up)
with a **fixed, drivable** step height — difficulty is driven by the coin-count
curriculum, not by step height, so the geometry stays constant for the analytic
coin placement in ``mdp.CoinSequenceCommand``.

These params (``tile_size`` = ``size[0]``, sub-terrain ``border_width``,
``platform_width``, ``step_width``, ``step_height``) MUST match the
``CoinSequenceCommandCfg`` in the env cfg.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

STEP_HEIGHT = 0.05
STEP_WIDTH = 0.55
PLATFORM_WIDTH = 2.5
SUBTERRAIN_BORDER_WIDTH = 1.0
TILE_SIZE = 10.0

STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(TILE_SIZE, TILE_SIZE),
    border_width=7.5,
    num_rows=10,
    num_cols=10,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.0, 1.0),  # no effect: step height range is a single value
    use_cache=True,
    curriculum=False,
    sub_terrains={
        "stair_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            holes=False,
            proportion=1.0,
            step_height_range=(STEP_HEIGHT, STEP_HEIGHT),  # fixed, drivable
            step_width=STEP_WIDTH,
            platform_width=PLATFORM_WIDTH,
            border_width=SUBTERRAIN_BORDER_WIDTH,
        ),
    },
)
"""Inverted pyramid stairs with a fixed drivable step height."""
