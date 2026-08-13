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


class RogerThresholdBalancer(ManagerTermBase):
    """ROGER's actual threshold/:math:`\\Delta_t`-based penalty-gain mechanism, as opposed to
    :class:`AdaptiveRewardBalancer`'s simplified ratio-target approximation.

    ``AdaptiveRewardBalancer`` targets an arbitrary ratio (``penalty ≈ penalty_budget × task
    reward``) that has no basis in the ROGER paper (arXiv:2510.10759) — that value (0.5) was an
    unjustified default we picked, and our own sweep of it actually favored *lower* values.
    ROGER's real mechanism instead targets **proximity to a physical safety threshold**
    :math:`\\tau_j` per constraint, not a reward ratio::

        Delta_t   = min( sum_j (mag_j / tau_j)^2, 1.0 )     # how close to violating ANY constraint
        lambda_pos = 1 - Delta_t                            # task reward is SUPPRESSED near violation
        lambda_j   = r_j * Delta_t                          # each penalty is AMPLIFIED near violation

    where ``mag_j`` is the (EMA-smoothed) per-episode magnitude of penalty term ``j``. Unlike
    the ratio balancer, this also scales the **positive/task** terms down as the population
    approaches its safety limits — the paper's actual design, not something we added.

    ``tau`` should be a *real* physical/safety limit wherever one exists, not a free knob:
    e.g. ``flat_orientation_l2 = sin(tilt)^2`` and the ``bad_orientation`` termination fires at
    ``tilt > limit_angle`` (0.5 rad here), so ``tau = sin(limit_angle)**2 ≈ 0.230`` is the exact
    boundary the termination itself uses — not a guess. Terms with no natural hard limit
    (``stand_still``, ``base_height_jump``) fall back to a user-supplied soft target.

    Because ``lambda`` is a direct function of the current EMA (not an integrator with a
    step-size like the ratio balancer), there is no drift/adapt_rate to tune — it reacts
    immediately to the smoothed signal, clamped only by the EMA's own smoothing.

    ⚠️ **Deviation from the pure paper formula**: at ``Delta_t = 0`` (far from every threshold,
    which is most of training early on) the paper's ``lambda_j = r_j * Delta_t`` sends EVERY
    penalty weight to exactly **zero**. We already have direct evidence this is dangerous for
    this task — ``AdaptiveRewardBalancer`` with ``g_min=0.1`` (penalties merely *reduced*, never
    zeroed) still collapsed into flat-ground hop farming once the orientation/height penalties
    were relaxed enough. Zeroing them entirely would very likely reproduce or worsen that
    collapse. So this implementation adds ``delta_floor`` (default 0.05, **not** in the paper):
    penalty gain is computed from ``max(Delta_t, delta_floor)`` instead of ``Delta_t`` directly,
    so no penalty term is *ever* fully disabled. Set ``delta_floor=0`` to match the paper exactly.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._initialized = False
        self._base_w: dict[str, float] = {}
        self._ema_mag: dict[str, float] = {}
        self._delta_t = 0.0
        self._batches = 0
        self._last_update_step = -1_000_000_000
        self._lambda_pos = 1.0

    def reset(self, env_ids=None):
        return None  # controller state is global, nothing per-env to reset

    def _lazy_init(self, env, positive_terms, penalty_terms):
        available = set(env.reward_manager.active_terms)
        for name in list(positive_terms) + list(penalty_terms):
            if name not in available:
                print(f"[RogerThresholdBalancer] skipping unknown reward term '{name}'")
                continue
            self._base_w[name] = float(env.reward_manager.get_term_cfg(name).weight)
        self._initialized = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        positive_terms=("stair_climb", "jump_lin_vel_z", "jump_feet_off", "foot_clearance"),
        penalty_terms=("flat_orientation_l2", "stand_still", "base_height_jump"),
        tau: dict | None = None,
        r: dict | None = None,
        delta_floor: float = 0.05,
        ema: float = 0.98,
        warmup_batches: int = 50,
        update_interval: int = 200,
    ):
        tau = tau or {}
        r = r or {}
        if not self._initialized:
            self._lazy_init(env, positive_terms, penalty_terms)

        rm = env.reward_manager
        ids = torch.arange(env.num_envs, device=env.device) if isinstance(env_ids, slice) else env_ids
        if len(ids) == 0:
            return self._log(penalty_terms)

        # -- per-term |effective contribution rate|, EMA-smoothed (same signal as the ratio balancer).
        inv_T = 1.0 / env.max_episode_length_s
        for name in self._base_w:
            mag = abs(float(torch.mean(rm._episode_sums[name][ids])) * inv_T)
            prev = self._ema_mag.get(name)
            self._ema_mag[name] = mag if prev is None else ema * prev + (1.0 - ema) * mag
        self._batches += 1

        due = env.common_step_counter - self._last_update_step >= update_interval
        if self._batches > warmup_batches and due:
            self._last_update_step = env.common_step_counter
            # Delta_t = how close (in normalized units) the penalties are to their tau limits.
            s = 0.0
            for name in penalty_terms:
                t = tau.get(name)
                if t is None or t <= 0.0:
                    continue  # no threshold supplied for this term -> excluded from Delta_t
                mag = self._ema_mag.get(name, 0.0)
                s += (mag / t) ** 2
            self._delta_t = float(min(s, 1.0))
            self._lambda_pos = 1.0 - self._delta_t

        # -- apply: positive terms suppressed by lambda_pos, penalties amplified by r_j * gain.
        # gain uses max(Delta_t, delta_floor) so no penalty is EVER fully zeroed (see docstring).
        gain = max(self._delta_t, delta_floor)
        for name in positive_terms:
            if name in self._base_w:
                rm.get_term_cfg(name).weight = self._base_w[name] * self._lambda_pos
        for name in penalty_terms:
            if name in self._base_w:
                rj = float(r.get(name, 1.0))
                rm.get_term_cfg(name).weight = self._base_w[name] * rj * gain

        return self._log(penalty_terms)

    def _log(self, penalty_terms):
        out = {"delta_t": self._delta_t, "lambda_pos": self._lambda_pos}
        for name in penalty_terms:
            out[f"ema_mag/{name}"] = self._ema_mag.get(name, 0.0)
        return out


class CurriculumWeightSchedule(ManagerTermBase):
    """Curriculum-coupled reward scheduling (approach D) — reward weights/params follow the
    terrain difficulty instead of staying fixed for the whole run.

    A weight set that is optimal on the shallow rows is not optimal on the tall ones (e.g. a
    5 cm step needs exploration-friendly gentle hops; a 15 cm step needs a much stronger
    impulse and a rewarded landing). This term makes selected reward knobs a **piecewise-linear
    function of the mean terrain level**: ``stages`` is a list of knots sorted by ``level``,
    each with a ``set`` mapping of targets to values, and every target is interpolated between
    the two surrounding knots at the population's current mean level (clamped at the ends).

    Target syntax in ``set``:

    * ``"jump_lin_vel_z"``        -> that reward term's ``weight``
    * ``"stair_climb/growth"``    -> that reward term's ``params["growth"]``

    Plays nicely with :class:`AdaptiveRewardBalancer` (bi-level inner loop): for weight targets
    the balancer manages, the schedule rewrites the balancer's captured *base* weight, so the
    penalty-gain ``g_neg`` still composes on top instead of the two fighting over
    ``term_cfg.weight``. Unknown terms are skipped with a warning (e.g. disabled to ``None``).

    Logged under ``Curriculum/``: the driving ``schedule_level`` and every scheduled value.
    Opt-in at train time via ``--weight_schedule`` (mirrors ``--adaptive_reward``).
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        stages = cfg.params["stages"]
        if len(stages) < 2:
            raise ValueError("CurriculumWeightSchedule needs >= 2 stages to interpolate")
        levels = [float(s["level"]) for s in stages]
        if levels != sorted(levels):
            raise ValueError(f"stage levels must be ascending, got {levels}")
        keys = set(stages[0]["set"].keys())
        for s in stages[1:]:
            if set(s["set"].keys()) != keys:
                raise ValueError("every stage must schedule the same set of targets")
        self._skip: set[str] = set()

    def reset(self, env_ids=None):
        return None  # global controller — nothing per-env to reset

    @staticmethod
    def _interp(level: float, stages: list) -> dict[str, float]:
        """Piecewise-linear value of every target at ``level`` (clamped to the knot range)."""
        if level <= float(stages[0]["level"]):
            return {k: float(v) for k, v in stages[0]["set"].items()}
        if level >= float(stages[-1]["level"]):
            return {k: float(v) for k, v in stages[-1]["set"].items()}
        for lo, hi in zip(stages[:-1], stages[1:]):
            l0, l1 = float(lo["level"]), float(hi["level"])
            if l0 <= level <= l1:
                a = (level - l0) / (l1 - l0) if l1 > l0 else 1.0
                return {
                    k: (1.0 - a) * float(lo["set"][k]) + a * float(hi["set"][k])
                    for k in lo["set"]
                }
        return {k: float(v) for k, v in stages[-1]["set"].items()}  # unreachable

    def _write(self, env: ManagerBasedRLEnv, target: str, value: float) -> None:
        rm = env.reward_manager
        name, _, param = target.partition("/")
        if name in self._skip:
            return
        if name not in rm.active_terms:
            print(f"[CurriculumWeightSchedule] skipping unknown reward term '{name}'")
            self._skip.add(name)
            return
        if param:
            rm.get_term_cfg(name).params[param] = value
            return
        # weight target: route through the balancer's base weight when it owns this term,
        # so g_neg / norm_mult still compose on top of the scheduled value.
        # (CurriculumManager has no get_term_cfg — index its parallel name/cfg lists.)
        balancer = None
        cm = getattr(env, "curriculum_manager", None)
        if cm is not None and "adaptive_reward" in getattr(cm, "active_terms", []):
            term_cfg = cm._term_cfgs[cm._term_names.index("adaptive_reward")]
            balancer = term_cfg.func
        if balancer is not None and getattr(balancer, "_base_w", None) and name in balancer._base_w:
            balancer._base_w[name] = value
        else:
            rm.get_term_cfg(name).weight = value

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: list | None = None,
        metric: str = "mean_terrain_level",
    ):
        if metric == "mean_terrain_level":
            terrain: TerrainImporter = env.scene.terrain
            level = float(terrain.terrain_levels.float().mean())
        else:
            raise ValueError(f"unknown schedule metric '{metric}'")

        values = self._interp(level, stages)
        for target, value in values.items():
            self._write(env, target, value)

        log = {"schedule_level": level}
        for target, value in values.items():
            log[f"sched/{target}"] = value
        return log


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