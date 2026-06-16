"""Lagrangian multiplier autotuner for reward weight balancing.

References
----------
[1] JJUNE0 et al., "isaaclab_add_reward_autotuning" (POSTECH Flamingo Lab, 2024)
    https://github.com/JJUNE0/isaaclab_add_reward_autotuning
    — LagrangianTuner design and log-space parameterisation

[2] Tessler, Chen, Mankowitz, Mann, Mannor, Shalit, Tennenholtz,
    "Reward Constrained Policy Optimization", ICLR 2019
    https://arxiv.org/abs/1805.11074
    — Lagrangian dual ascent for constrained RL

[3] Altman, "Constrained Markov Decision Processes", CRC Press 1999
    — CMDP theoretical foundation
"""

from __future__ import annotations

import math


class LagrangianTuner:
    """Per-term Lagrange multiplier autotuner.

    Each reward term i has:
      - λᵢ > 0  stored as log λᵢ  (numerical stability)
      - sᵢ = sign(init_weight)    (+1 objective, -1 penalty)
      - cᵢ : target contribution  (desired mean per-step reward, signed)

    Update rule (sign-free, works for both objectives and penalties):
        log λᵢ  ←  log λᵢ  +  lr · (cᵢ − actualᵢ)

    Derivation
    ----------
    Primal constraint: actual_i ≥ c_i  (for both objectives and penalties)
    Dual update (gradient ascent on Lagrangian w.r.t. λ):
        λ += lr · (c_i − actual_i)

    For objective terms (w > 0, ∂actual/∂λ = +feature > 0):
        actual < target → increase λ → reward increases → actual rises toward target ✓

    For penalty terms (w = −λ, ∂actual/∂λ = −feature < 0):
        actual < target → increase λ → more penalty → policy avoids bad behavior
                       → feature decreases → actual rises toward target ✓
        actual > target → decrease λ → less penalty → policy relaxes → stable ✓

    The sign factor s = sign(init_weight) is NOT included in the update;
    it is only used to reconstruct the effective weight: w = s · exp(log λ).

    Effective weight applied to the env reward manager:
        wᵢ = sᵢ · clamp(exp(log λᵢ), min_abs, max_abs)
    """

    EP_REWARD_PREFIX = "Episode_Reward/"

    def __init__(self, term_configs: dict[str, dict]):
        """
        Args:
            term_configs: mapping from reward term name to config dict with keys:
                - init_weight (float): initial reward weight (non-zero; sign sets objective/penalty)
                - target (float): desired mean per-step contribution (signed, same sign as init_weight)
                - lr (float): Lagrangian learning rate (default 1e-3)
                - min_abs_weight (float): minimum |weight| (default 0.01)
                - max_abs_weight (float): maximum |weight| (default 1e3)
        """
        self.term_configs = term_configs
        self.log_lambdas: dict[str, float] = {}
        self.signs: dict[str, float] = {}

        for name, cfg in term_configs.items():
            w0 = float(cfg["init_weight"])
            if w0 == 0.0:
                raise ValueError(f"init_weight for '{name}' must be non-zero")
            self.signs[name] = 1.0 if w0 > 0 else -1.0
            self.log_lambdas[name] = math.log(abs(w0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weight(self, term_name: str) -> float:
        """Return current effective weight (signed) for a term."""
        cfg = self.term_configs[term_name]
        abs_w = math.exp(self.log_lambdas[term_name])
        abs_w = max(cfg.get("min_abs_weight", 0.01), min(cfg.get("max_abs_weight", 1e3), abs_w))
        return self.signs[term_name] * abs_w

    def update(self, term_name: str, actual: float) -> float:
        """Update multiplier for one term and return the new weight."""
        cfg = self.term_configs[term_name]
        target = float(cfg["target"])
        lr = float(cfg.get("lr", 1e-3))

        self.log_lambdas[term_name] += lr * (target - actual)

        min_abs = cfg.get("min_abs_weight", 0.01)
        max_abs = cfg.get("max_abs_weight", 1e3)
        self.log_lambdas[term_name] = max(
            math.log(min_abs), min(math.log(max_abs), self.log_lambdas[term_name])
        )
        return self.get_weight(term_name)

    def update_from_ep_stats(self, ep_stats: dict[str, float]) -> dict[str, float]:
        """Update all configured terms from an episode-stats dict.

        Args:
            ep_stats: {key: mean_value} where key is the full tensorboard key,
                      e.g. "Episode_Reward/wheel_height_bonus".

        Returns:
            Dict of updated weights {term_name: new_weight}.
        """
        updated: dict[str, float] = {}
        for name in self.term_configs:
            key = self.EP_REWARD_PREFIX + name
            if key in ep_stats:
                updated[name] = self.update(name, ep_stats[key])
        return updated

    def get_all_weights(self) -> dict[str, float]:
        return {name: self.get_weight(name) for name in self.term_configs}

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"log_lambdas": dict(self.log_lambdas)}

    def load_state_dict(self, state: dict) -> None:
        self.log_lambdas.update(state["log_lambdas"])
