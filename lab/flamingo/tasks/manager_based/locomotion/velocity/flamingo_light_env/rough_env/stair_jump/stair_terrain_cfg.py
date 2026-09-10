# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Dedicated stair terrain for the stair-climbing task.

Inverted pyramid stairs (robot spawns at the low pit center and climbs outward/up).
Step height follows a **terrain-level curriculum**: with ``curriculum=True`` the step
height is interpolated over the difficulty rows as
``step_height = 0.05 + difficulty * 0.10`` while the tread width follows
``step_width = 0.50 - difficulty * 0.20``. Thus promotion makes risers taller and
treads narrower together. The last level is clamped to the requested 15 cm by
30 cm target.
"""

import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


# Curriculum endpoints requested for sim-to-real stair training.
STEP_HEIGHT_MIN = 0.05
STEP_HEIGHT_MAX = 0.15
STEP_HEIGHT = STEP_HEIGHT_MIN  # nominal / fallback (e.g. non-curriculum terrains)
NUM_LEVELS = 10
STEP_WIDTH_START = 0.50
STEP_WIDTH_TARGET = 0.30
STEP_WIDTH_RANGE = (STEP_WIDTH_START, STEP_WIDTH_TARGET)
PLATFORM_WIDTH = 5
SUBTERRAIN_BORDER_WIDTH = 1.0
TILE_SIZE = 10.0


def curriculum_inverted_pyramid_stairs_terrain(difficulty: float, cfg):
    """Generate inverted stairs with height and width coupled to difficulty.

    Isaac Lab samples row ``r`` in ``[r / N, (r + 1) / N)``.  Scaling by
    ``N / (N - 1)`` makes the final row exactly the requested endpoint instead
    of merely approaching it.  All final-row tiles are consequently 15 cm high
    and 30 cm wide, while lower rows remain monotonic.
    """
    resolved_cfg = cfg.copy()
    resolved_difficulty = min(difficulty * cfg.difficulty_scale, 1.0)
    width_start, width_target = cfg.step_width_range
    resolved_cfg.step_width = width_start + resolved_difficulty * (width_target - width_start)
    return mesh_terrains.inverted_pyramid_stairs_terrain(resolved_difficulty, resolved_cfg)


@configclass
class CurriculumInvertedPyramidStairsTerrainCfg(terrain_gen.MeshInvertedPyramidStairsTerrainCfg):
    """Inverted stairs whose tread width shrinks along the curriculum."""

    function = curriculum_inverted_pyramid_stairs_terrain
    step_width_range: tuple[float, float] = (0.50, 0.30)
    difficulty_scale: float = NUM_LEVELS / (NUM_LEVELS - 1)

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
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    curriculum=True,
    sub_terrains={
        "stair_up": CurriculumInvertedPyramidStairsTerrainCfg(
            holes=False,
            proportion=1.0,
            step_height_range=(STEP_HEIGHT_MIN, STEP_HEIGHT_MAX),  # curriculum over rows
            step_width=STEP_WIDTH_START,  # resolved per tile by the custom function
            step_width_range=STEP_WIDTH_RANGE,
            platform_width=PLATFORM_WIDTH,
            border_width=SUBTERRAIN_BORDER_WIDTH,
        ),
    },
)
"""Inverted pyramid stairs whose risers grow while their treads shrink."""
