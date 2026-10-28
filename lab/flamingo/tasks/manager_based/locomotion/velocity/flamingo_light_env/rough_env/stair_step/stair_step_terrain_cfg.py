# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Stair terrain for the alternating-feet task.

Same inverted-pyramid stairs and riser/tread curriculum as the stair_jump task
(``stair_jump/stair_terrain_cfg.py``: risers 5 -> 15 cm, treads 0.50 -> 0.30 m over 10 rows),
but with a 2 m requested centre platform instead of 5 m. The generator rounds the step count
up, so the real pit is 1.0-1.4 m wide and the first riser is 0.5-0.7 m from the spawn point
(instead of ~2 m): more risers per 20 s episode at walking-step speeds (6-10 per side).
"""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from ..stair_jump.stair_terrain_cfg import (
    NUM_LEVELS,
    STEP_HEIGHT_MAX,
    STEP_HEIGHT_MIN,
    STEP_WIDTH_RANGE,
    STEP_WIDTH_START,
    SUBTERRAIN_BORDER_WIDTH,
    TILE_SIZE,
    CurriculumInvertedPyramidStairsTerrainCfg,
)

STEP_PLATFORM_WIDTH = 2.0


def _stair_terrains(step_height_range: tuple[float, float]) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=42,
        size=(TILE_SIZE, TILE_SIZE),
        border_width=7.5,
        num_rows=NUM_LEVELS,
        num_cols=10,
        color_scheme="random",
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.5,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "stair_up": CurriculumInvertedPyramidStairsTerrainCfg(
                holes=False,
                proportion=1.0,
                step_height_range=step_height_range,
                step_width=STEP_WIDTH_START,
                step_width_range=STEP_WIDTH_RANGE,
                platform_width=STEP_PLATFORM_WIDTH,
                border_width=SUBTERRAIN_BORDER_WIDTH,
            ),
        },
    )


STAIR_STEP_TERRAINS_CFG = _stair_terrains((STEP_HEIGHT_MIN, STEP_HEIGHT_MAX))

# Lift variant (sweep v3): every riser is taller than the wheel radius (7.6 cm). Below it a wheel
# pushed into the riser by the other wheel (and by speed) rolls over the edge, and the v1/v2
# policies climbed the 5-7 cm rows that way instead of lifting a leg. At 9 cm the edge is above
# the axle, the edge reaction pushes the tire down and back, and the wheel has to be lifted.
LIFT_STEP_HEIGHT_MIN = 0.09
STAIR_STEP_LIFT_TERRAINS_CFG = _stair_terrains((LIFT_STEP_HEIGHT_MIN, STEP_HEIGHT_MAX))
