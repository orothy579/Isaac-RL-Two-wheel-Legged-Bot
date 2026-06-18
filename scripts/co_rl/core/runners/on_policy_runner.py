#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from scripts import co_rl
from scripts.co_rl.core.algorithms import PPO
from scripts.co_rl.core.env import VecEnv
from scripts.co_rl.core.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization, LagrangianTuner
from scripts.co_rl.core.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]

        assert self.alg_cfg["class_name"] == "PPO"

        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.cfg,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [co_rl.__file__]

        # MOO: Lagrangian autotuner (optional)
        moo_cfg = self.cfg.get("moo", None)
        self.lagrangian_tuner: LagrangianTuner | None = (
            LagrangianTuner(moo_cfg) if moo_cfg else None
        )


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from scripts.co_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from scripts.co_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # move to the right device
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        if not self.cfg["use_constraint_rl"]:
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                        else:
                            new_ids = (dones == 1.0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            # MOO: update Lagrangian multipliers and apply to reward manager
            if self.lagrangian_tuner is not None and ep_infos:
                self._update_lagrangian(ep_infos)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
                # write params.json once at the start of training
                if self.log_dir is not None:
                    self._save_params(self.log_dir)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def _get_reward_weights(self) -> dict:
        try:
            rm = self.env.unwrapped.reward_manager
            return {name: float(cfg.weight) for name, cfg in zip(rm._term_names, rm._term_cfgs)}
        except AttributeError:
            return {}

    def _get_terrain_params(self) -> dict:
        """Extract terrain configuration from the env config, if present."""
        try:
            terrain_cfg = self.env.unwrapped.cfg.scene.terrain
        except AttributeError:
            return {}

        result: dict = {
            "terrain_type": getattr(terrain_cfg, "terrain_type", None),
            "max_init_terrain_level": getattr(terrain_cfg, "max_init_terrain_level", None),
        }

        tg = getattr(terrain_cfg, "terrain_generator", None)
        if tg is None:
            return result  # flat / plane terrain — nothing more to record

        result["terrain_generator"] = {
            "num_rows": getattr(tg, "num_rows", None),
            "num_cols": getattr(tg, "num_cols", None),
            "curriculum": getattr(tg, "curriculum", None),
            "difficulty_range": list(getattr(tg, "difficulty_range", [])),
            "size": list(getattr(tg, "size", [])),
            "border_width": getattr(tg, "border_width", None),
        }

        sub = getattr(tg, "sub_terrains", {})
        if sub:
            sub_out = {}
            for name, scfg in sub.items():
                sub_out[name] = {
                    k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                    for k, v in vars(scfg).items()
                    if not k.startswith("_") and not callable(v)
                }
            result["terrain_generator"]["sub_terrains"] = sub_out

        return result

    def _save_params(self, log_dir: str):
        """Write params.json once at training start — reward weights, PPO hyperparams, terrain."""
        params = {
            "reward_weights": self._get_reward_weights(),
            "moo": self.cfg.get("moo") or {},
            "algorithm": {k: v for k, v in self.cfg.get("algorithm", {}).items() if not callable(v)},
            "num_steps_per_env": self.cfg.get("num_steps_per_env"),
            "experiment_name": self.cfg.get("experiment_name"),
            "terrain": self._get_terrain_params(),
        }
        with open(os.path.join(log_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

    def _update_lagrangian(self, ep_infos: list) -> None:
        """Compute per-term episode reward means, update λ, apply to reward manager."""
        # Aggregate per-term means across all episodes in this iteration
        ep_stats: dict[str, float] = {}
        if ep_infos:
            all_keys = set().union(*(e.keys() for e in ep_infos))
            for key in all_keys:
                vals = []
                for ep_info in ep_infos:
                    if key not in ep_info:
                        continue
                    v = ep_info[key]
                    if isinstance(v, torch.Tensor):
                        vals.extend(v.cpu().flatten().tolist())
                    else:
                        vals.append(float(v))
                if vals:
                    ep_stats[key] = sum(vals) / len(vals)

        updated = self.lagrangian_tuner.update_from_ep_stats(ep_stats)

        # Apply updated weights to the reward manager
        try:
            rm = self.env.unwrapped.reward_manager
            for term_name, new_weight in updated.items():
                rm.get_term_cfg(term_name).weight = new_weight
        except AttributeError:
            pass

        # Log each multiplier to tensorboard
        if self.writer is not None:
            for term_name, new_weight in updated.items():
                self.writer.add_scalar(
                    f"MOO/weight/{term_name}", new_weight, self.current_learning_iteration
                )

    def save(self, path, infos=None):
        reward_weights = self._get_reward_weights()
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "reward_weights": reward_weights,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        if self.lagrangian_tuner is not None:
            saved_dict["lagrangian_state"] = self.lagrangian_tuner.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if self.lagrangian_tuner is not None and "lagrangian_state" in loaded_dict:
            self.lagrangian_tuner.load_state_dict(loaded_dict["lagrangian_state"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def load_transfer(self, path, load_optimizer=False):
        """Warm-start (transfer) load that tolerates obs-dim differences.

        Unlike :meth:`load` (strict), this copies only the tensors whose name AND
        shape match the current model — e.g. the full actor and every
        shape-compatible critic layer — and re-initializes the rest (such as the
        critic input layer when the critic obs dim changed). Each observation
        normalizer is loaded only if its shape matches; the optimizer and the
        learning-iteration counter are reset so training starts fresh on the new
        task. Use this for cross-task transfer (e.g. flat-drive -> rough) where
        the policy matches but the critic obs space differs.
        """
        loaded_dict = torch.load(path, map_location=self.device)
        ckpt_model = loaded_dict["model_state_dict"]
        model_sd = self.alg.actor_critic.state_dict()

        matched, skipped = [], []
        for k, v in ckpt_model.items():
            # Do not transfer the action-noise std: keep the fresh init_noise_std
            # so the warm-started policy explores the new task instead of inheriting
            # the source model's converged (near-zero) exploration.
            if k == "std":
                skipped.append(k)
                continue
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                matched.append(k)
            else:
                skipped.append(k)
        self.alg.actor_critic.load_state_dict(model_sd)

        # Load each obs normalizer only when its shape matches (policy normally
        # matches; critic is skipped when the critic obs dim differs).
        if self.empirical_normalization:
            for normalizer, key in (
                (self.obs_normalizer, "obs_norm_state_dict"),
                (self.critic_obs_normalizer, "critic_obs_norm_state_dict"),
            ):
                sd = loaded_dict.get(key)
                if sd is None:
                    continue
                try:
                    normalizer.load_state_dict(sd)
                    matched.append(key)
                except RuntimeError:
                    skipped.append(key)

        if load_optimizer:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except (ValueError, RuntimeError):
                print("[WARM-START] optimizer state incompatible; using a fresh optimizer.")

        print(f"[WARM-START] transferred {len(matched)} tensors from: {path}")
        print(f"[WARM-START] re-initialized (name/shape mismatch): {skipped}")
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
