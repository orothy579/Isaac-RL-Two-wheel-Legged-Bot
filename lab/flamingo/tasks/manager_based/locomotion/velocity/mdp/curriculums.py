# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def stair_terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "stair_climb",
    promote_steps: float = 4.0,
    demote_steps: float = 1.0,
) -> torch.Tensor:
    """Step-height curriculum driven by how many stairs the robot climbed this episode.

    Reuses the per-episode ``max_step`` tracked by the :class:`StairClimbProgress` reward
    term (steps counted at the env's *own* current step height). An env that reached at
    least ``promote_steps`` new steps moves up to a taller-step row; one that reached
    fewer than ``demote_steps`` moves down to a shallower row.

    This runs inside ``_reset_idx`` *before* the reward manager resets, so ``max_step``
    still holds the finished episode's value. Only usable with a ``generator`` terrain
    running ``curriculum=True`` (step height interpolated over the rows).

    Returns:
        The mean terrain level over all envs.
    """
    terrain: TerrainImporter = env.scene.terrain
    # class-based reward terms store their instance on ``term_cfg.func``
    reward_term = env.reward_manager.get_term_cfg(reward_term_name).func
    steps = reward_term.max_step[env_ids]
    # robots that climbed enough progress to taller steps; those that barely climbed drop
    move_up = steps >= promote_steps
    move_down = steps < demote_steps
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def stair_climb_units(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "stair_climb",
) -> torch.Tensor:
    """Log-only curriculum term: mean height climbed per finished episode, in the nominal
    5 cm units tracked by :class:`StairClimbProgress` (``max_step``).

    Unlike ``Curriculum/terrain_levels`` (which conflates climbing with promotion policy and
    is meaningless on fixed-height terrain), this directly answers "how much did the robots
    actually climb": e.g. on a fixed 0.15 m probe terrain, 3.0 = every resetting env got onto
    the first tread, 0.0 = nobody ever did. Reads the finished episode's ``max_step`` (this
    runs in ``_reset_idx`` before the reward manager resets it) and changes nothing.
    """
    reward_term = env.reward_manager.get_term_cfg(reward_term_name).func
    return torch.mean(reward_term.max_step[env_ids])


class AdaptiveRewardBalancer(ManagerTermBase):
    """Online reward balancer — keeps the positive/penalty budget in check as the policy learns.

    Runs as a curriculum term (fires in ``_reset_idx`` *before* the reward manager zeroes its
    episode sums, so it sees each finished episode's totals). It drives two feedback
    controllers off every managed term's *actual* per-episode contribution — read from
    ``reward_manager._episode_sums`` and EMA-smoothed across resets, so both react to the
    reward **trend**, not one noisy episode:

    * **Penalty-gain adaptation (ROGER-style).** One global gain ``g_neg`` in ``[g_min, 1]``
      multiplies every ``penalty_terms`` weight. It is nudged so the summed penalty magnitude
      tracks ``penalty_budget`` times the summed positive (task) reward. Early on the new skill
      earns little, so penalties are automatically *relaxed* (the robot is free to attempt the
      risky climb/hop instead of freezing); as the task reward grows they are *tightened* back
      toward nominal (cleaner motion). This automates hand-tuning like ``flat_orientation`` -10 -> -2.

    * **Per-term normalization (opt-in).** For each ``name: target`` in ``normalize_terms`` a
      per-term multiplier rescales that term's weight so its running |contribution| tracks
      ``target`` — so no term dominates purely because its raw numbers are large. Empty by
      default (the exponential ``stair_climb`` is deliberately *not* normalized: its growth is
      intentional and drives the terrain curriculum).

    The effective weight written back each update is
    ``base_weight * norm_mult * (g_neg if penalty else 1)`` — one composed write per term.
    Base weights are captured once (lazily), so this term is the sole owner of them while
    active. Controller state (``penalty_gain``, ``penalty_over_positive``) is logged under
    ``Curriculum/``.

    Cadence is decoupled from the reset rate: the EMA updates every call, but the controller
    only *moves* every ``update_interval`` control steps (so it can't slam to the bounds when
    resets fire almost every step). Non-invasive: no IsaacLab/runner changes — only reward-term
    weights are mutated at runtime, which ``RewardManager.compute`` reads live each step.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._initialized = False
        self._base_w: dict[str, float] = {}          # nominal weight, captured once
        self._is_penalty: dict[str, bool] = {}       # gets the g_neg multiplier
        self._norm_mult: dict[str, float] = {}        # per-term normalization multiplier
        self._ema_mag: dict[str, float] = {}          # EMA of |effective contribution rate|
        self._g_neg = 1.0
        self._batches = 0
        self._last_update_step = -1_000_000_000

    def reset(self, env_ids=None):
        # controller state is global (not per-env); nothing to reset on episode boundaries.
        return None

    def _lazy_init(self, env, positive_terms, penalty_terms, normalize_terms):
        available = set(env.reward_manager.active_terms)
        for name in list(positive_terms) + list(penalty_terms) + list(normalize_terms.keys()):
            if name not in available:
                # e.g. a term disabled to None (flat_orientation_l2) — skip it gracefully.
                print(f"[AdaptiveRewardBalancer] skipping unknown reward term '{name}'")
                continue
            self._base_w[name] = float(env.reward_manager.get_term_cfg(name).weight)
        for name in penalty_terms:
            if name in self._base_w and name not in normalize_terms:
                self._is_penalty[name] = True
        for name in normalize_terms:
            if name in self._base_w:
                self._norm_mult[name] = 1.0
        self._initialized = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        positive_terms=("stair_climb",),
        penalty_terms=("flat_orientation_l2", "stand_still", "base_height_jump"),
        normalize_terms: dict | None = None,
        penalty_budget: float = 0.5,
        g_min: float = 0.1,
        ema: float = 0.98,
        adapt_rate: float = 0.05,
        norm_mult_range: tuple = (0.25, 4.0),
        warmup_batches: int = 50,
        update_interval: int = 200,
    ):
        normalize_terms = normalize_terms or {}
        if not self._initialized:
            self._lazy_init(env, positive_terms, penalty_terms, normalize_terms)

        rm = env.reward_manager
        ids = torch.arange(env.num_envs, device=env.device) if isinstance(env_ids, slice) else env_ids
        if len(ids) == 0:
            return self._log(positive_terms, penalty_terms)

        # -- per-term |effective contribution rate| over the just-finished envs, EMA-smoothed.
        inv_T = 1.0 / env.max_episode_length_s
        for name in self._base_w:
            mag = abs(float(torch.mean(rm._episode_sums[name][ids])) * inv_T)
            prev = self._ema_mag.get(name)
            self._ema_mag[name] = mag if prev is None else ema * prev + (1.0 - ema) * mag
        self._batches += 1

        # -- move the controllers at most once per ``update_interval`` control steps.
        due = env.common_step_counter - self._last_update_step >= update_interval
        if self._batches > warmup_batches and due:
            self._last_update_step = env.common_step_counter
            # ROGER penalty gain: pull summed penalty toward penalty_budget * summed positive.
            P = sum(self._ema_mag.get(n, 0.0) for n in positive_terms)
            N = sum(self._ema_mag.get(n, 0.0) for n in penalty_terms)
            if P > 1e-8 and N > 1e-8:
                ratio = (penalty_budget * P) / N  # >1 -> penalties too small -> raise g_neg
                step = min(max(ratio, 1.0 - adapt_rate), 1.0 + adapt_rate)
                self._g_neg = float(min(max(self._g_neg * step, g_min), 1.0))
            # per-term normalization toward a target magnitude.
            for name, target in normalize_terms.items():
                mag = self._ema_mag.get(name, 0.0)
                if name in self._norm_mult and mag > 1e-8:
                    step = min(max(target / mag, 1.0 - adapt_rate), 1.0 + adapt_rate)
                    m = self._norm_mult[name] * step
                    self._norm_mult[name] = float(min(max(m, norm_mult_range[0]), norm_mult_range[1]))

        # -- apply composed weights: base * norm_mult * (g_neg if penalty).
        for name, w0 in self._base_w.items():
            m = self._norm_mult.get(name, 1.0)
            g = self._g_neg if self._is_penalty.get(name, False) else 1.0
            rm.get_term_cfg(name).weight = w0 * m * g

        return self._log(positive_terms, penalty_terms)

    def _log(self, positive_terms, penalty_terms):
        P = sum(self._ema_mag.get(n, 0.0) for n in positive_terms)
        N = sum(self._ema_mag.get(n, 0.0) for n in penalty_terms)
        out = {"penalty_gain": self._g_neg, "penalty_over_positive": (N / P) if P > 1e-8 else 0.0}
        for name, m in self._norm_mult.items():
            out[f"norm_mult/{name}"] = m
        return out


def modify_base_velocity_range(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, mod_range: dict, num_steps: int
):
    """
    Modifies the range of a command term (e.g., base_velocity) in the environment after a specific number of steps.

    Args:
        env: The environment instance.
        term_name: The name of the command term to modify (e.g., "base_velocity").
        end_range: The target range for the term (e.g., {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.5, 1.5)}).
        activation_step: The step count after which the range modification is applied.
    """
    # Check if the curriculum step exceeds the activation step
    if env.common_step_counter >= num_steps:
        # Get the term object
        command_term = env.command_manager.get_term(term_name)

        # Update the ranges directly
        for key, target_range in mod_range.items():
            if hasattr(command_term.cfg.ranges, key):
                setattr(command_term.cfg.ranges, key, target_range)

def modify_action_scale_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_name: str,
    start_scale: float,
    end_scale: float,
    num_steps: int,
    start_step: int = 0,
):
    """Linearly modifies the scale of an action term.

    Example:
        wheel action scale: 0.0 -> 1.0 over 5000 steps.

    Args:
        env: The environment instance.
        env_ids: Environment ids. Not used here, but required by curriculum manager.
        action_name: Name of the action term, e.g. "wheel_vel".
        start_scale: Initial action scale.
        end_scale: Final action scale.
        num_steps: Number of env steps required to reach end_scale.
        start_step: Step at which curriculum starts.
    """
    del env_ids  # unused

    # before curriculum starts
    if env.common_step_counter < start_step:
        target_scale = start_scale
    else:
        progress = (env.common_step_counter - start_step) / float(num_steps)
        progress = max(0.0, min(1.0, progress))
        target_scale = start_scale + progress * (end_scale - start_scale)

    # get action term
    action_term = env.action_manager.get_term(action_name)

    # update config value
    if hasattr(action_term, "cfg") and hasattr(action_term.cfg, "scale"):
        action_term.cfg.scale = target_scale

    # update runtime cached scale
    # Isaac Lab action terms often cache cfg.scale internally,
    # so changing cfg.scale alone may not be enough.
    if hasattr(action_term, "_scale"):
        if isinstance(action_term._scale, torch.Tensor):
            action_term._scale[:] = target_scale
        else:
            action_term._scale = target_scale

    if hasattr(action_term, "scale"):
        if isinstance(action_term.scale, torch.Tensor):
            action_term.scale[:] = target_scale
        else:
            try:
                action_term.scale = target_scale
            except AttributeError:
                pass

    return torch.tensor(target_scale, device=env.device)