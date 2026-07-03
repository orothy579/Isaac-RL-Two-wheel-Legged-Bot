# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Dedicated stair terrain for the stair-climbing task.

Inverted pyramid stairs (robot spawns at the low pit center and climbs outward/up).
Step height follows a **terrain-level curriculum**: with ``curriculum=True`` the step
height is interpolated over the difficulty rows as
``step_height = STEP_HEIGHT_MIN + difficulty * (STEP_HEIGHT_MAX - STEP_HEIGHT_MIN)``,
where ``difficulty = (row_id + noise) / num_rows``. So row 0 is the shallowest step and
the last row the tallest; robots are promoted/demoted across rows by the curriculum
term. ``STEP_HEIGHT_MIN/MAX`` and ``NUM_LEVELS`` are imported by ``StairClimbProgress``
to recover each env's actual step height from its current terrain level.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# step-height curriculum range (row 0 -> MIN, last row -> ~MAX)
STEP_HEIGHT_MIN = 0.05
STEP_HEIGHT_MAX = 0.15
STEP_HEIGHT = STEP_HEIGHT_MIN  # nominal / fallback (e.g. non-curriculum terrains)
NUM_LEVELS = 10  # difficulty rows = curriculum levels
STEP_WIDTH = 0.4
PLATFORM_WIDTH = 5
SUBTERRAIN_BORDER_WIDTH = 1.0
TILE_SIZE = 10.0

STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(TILE_SIZE, TILE_SIZE),
    border_width=7.5,
    num_rows=NUM_LEVELS,
    num_cols=10,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.0, 1.0),  # maps rows -> step height across the range below
    use_cache=True,  # regenerate from the current cfg (avoid stale cached geometry)
    curriculum=True,
    sub_terrains={
        "stair_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            holes=False,
            proportion=1.0,
            step_height_range=(STEP_HEIGHT_MIN, STEP_HEIGHT_MAX),  # curriculum over rows
            step_width=STEP_WIDTH,
            platform_width=PLATFORM_WIDTH,
            border_width=SUBTERRAIN_BORDER_WIDTH,
        ),
    },
)
"""Inverted pyramid stairs whose step height ramps up over the difficulty rows."""
