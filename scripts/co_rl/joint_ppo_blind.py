"""Joint PPO of the BLIND stair policy in Isaac (GPU) and the MuJoCo cosim (CPU) at the same time.

The deployable policy has to climb in both simulators. Isaac-only training learns approaches that PhysX lets
through and the cosim does not (0 % there); cosim-only training (``mujoco_ppo_blind.py``) can drift away from Isaac.
Here one actor learns from both every iteration:

* MuJoCo: the workers of ``mujoco_ppo_blind.py`` (evaluator tiles and physics, Isaac's speed limits, its rewards and
  curriculum) roll out their envs with the current weights on the CPU ...
* ... while this process steps ``--isaac_envs`` envs of the Isaac task (``Lift-Mirror-Gain`` by default, its own
  rewards, terminations and terrain curriculum, co_rl's time-out bootstrap and action clip) on the GPU;
* one PPO update: the clipped surrogate of each simulator (advantages normalized per simulator) weighted
  ``--isaac_weight`` / ``--mujoco_weight``, a value loss per simulator (one critic each: Isaac's privileged 216
  inputs, warm-started from the co_rl checkpoint, and the MuJoCo one), the left-right mirror loss on both, and
  co_rl's adaptive learning rate.

Checkpoints: ``model_<it>.pt`` (co_rl format: actor, std, the Isaac critic) for ``eval_stair_step.py`` and co_rl,
``actor_<it>.npz`` for the cosim evaluator, ``latest.pt`` to resume (``--init`` accepts it).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \\
        <venv python> -u scripts/co_rl/joint_ppo_blind.py --init <model.pt | latest.pt> --out <dir>
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_ppo_blind as M  # noqa: E402  (sets the blind cosim layout before importing the evaluator)

import numpy as np  # noqa: E402

GAIN_TASK = "Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-v1-ppo"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="co_rl blind model_*.pt, or latest.pt of this or mujoco_ppo_blind.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--isaac_task", default=GAIN_TASK)
    ap.add_argument("--isaac_envs", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--envs_per_worker", type=int, default=72)
    ap.add_argument("--horizon", type=int, default=96, help="MuJoCo steps per env and iteration")
    ap.add_argument("--isaac_horizon", type=int, default=48, help="Isaac steps per env and iteration (the workers' "
                    "MuJoCo rollout runs meanwhile: pick it so both take about as long)")
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--isaac_weight", type=float, default=1.0)
    ap.add_argument("--mujoco_weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl_target", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--mirror", type=float, default=1.0)
    ap.add_argument("--mirror_samples", type=int, default=32768, help="per simulator and minibatch")
    ap.add_argument("--std_init", type=float, default=0.15)
    ap.add_argument("--std_range", type=float, nargs=2, default=[0.03, 0.4])
    ap.add_argument("--critic_warmup", type=int, default=30, help="iterations of value learning only (new critics)")
    ap.add_argument("--start_level", type=int, default=0)
    ap.add_argument("--promote", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--isaac_like_eval", action="store_true",
                    help="Isaac as it is evaluated (and as MuJoCo always is): clean observations and the history of a "
                         "reset env refilled with its first frame. Without it the policy told the simulators apart by "
                         "the stale co_rl history and the noise: 4.9 risers per episode in training, 0.8 in evaluation")
    ap.add_argument("--mujoco_obs_noise", action="store_true",
                    help="Isaac's training observation noise on the MuJoCo frames too: with clean MuJoCo and noisy "
                         "Isaac frames the policy can tell the simulators apart by the noise alone")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    a = ap.parse_args()
    a.headless = True
    import torch
    from torch import nn

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda:0")
    n_mj = a.workers * a.envs_per_worker

    def mlp(inp, out):
        return nn.Sequential(nn.Linear(inp, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128),
                             nn.ELU(), nn.Linear(128, out))

    # ---------------- networks ----------------
    X = M.ext_dim()
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    actor, critic_mj = mlp(M.OBS_DIM, 4).to(dev), mlp(M.OBS_DIM + X, 1).to(dev)
    log_std = nn.Parameter(torch.full((4,), float(np.log(a.std_init)), device=dev))
    critic_is = None  # built once the Isaac critic width is known
    start_it, level_max, warm_mj, warm_is = 0, a.start_level, a.critic_warmup, a.critic_warmup
    if "model_state_dict" in ck:  # co_rl: actor, std (floored), and the Isaac critic
        sd = ck["model_state_dict"]
        base_ckpt = a.init
        actor.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("actor.")})
        log_std.data = torch.log(torch.maximum(sd["std"].to(dev), torch.tensor(a.std_init, device=dev)))
        is_sd = {k[7:]: v for k, v in sd.items() if k.startswith("critic.")}
        warm_is = 0
    else:  # latest.pt of mujoco_ppo_blind.py (MuJoCo critic) or of this script (both critics)
        actor.load_state_dict(ck["actor"])
        critic_mj.load_state_dict(ck["critic"] if "critic" in ck else ck["critic_mj"])
        log_std.data = ck["log_std"].to(dev)
        base_ckpt, level_max = ck["base_ckpt"], ck["level_max"]
        start_it = ck["iteration"] + 1
        warm_mj = 0
        if "critic_is" in ck:
            is_sd, warm_is = ck["critic_is"], 0
        else:
            is_sd = {k[7:]: v for k, v in torch.load(base_ckpt, map_location="cpu", weights_only=False)[
                "model_state_dict"].items() if k.startswith("critic.")}
            warm_is = 0
    warmup = max(warm_mj, warm_is)

    # ---------------- MuJoCo workers (before Isaac Sim starts) ----------------
    if a.mujoco_obs_noise:  # read by cosim_eval_policy when the spawned workers import it
        os.environ["COSIM_EVAL_OBS_NOISE"] = "1"
    params_w = [("actor." + k, p) for k, p in actor.named_parameters()] + \
               [("critic." + k, p) for k, p in critic_mj.named_parameters()] + [("log_std", log_std)]
    layout = [(k, tuple(p.shape)) for k, p in params_w]
    ctx = mp.get_context("spawn")
    shared = M.Shared(ctx, n_mj, a.horizon, X, sum(int(np.prod(s)) for _, s in layout))
    sv = shared.views()
    E = a.envs_per_worker
    conns, procs = [], []
    for w in range(a.workers):
        p, c = ctx.Pipe()
        pr = ctx.Process(target=M.worker, args=(c, shared, w * E, E, a.seed * 1000 + w * 104729, a.seconds, layout),
                         daemon=True)
        pr.start()
        conns.append(p)
        procs.append(pr)

    def push_weights():
        with torch.no_grad():
            sv["weights"][:] = torch.cat([p.detach().float().reshape(-1) for _, p in params_w]).cpu().numpy()

    # ---------------- Isaac ----------------
    app = AppLauncher(a).app  # noqa: F841
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import lab.flamingo.tasks  # noqa: F401
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper

    env_cfg = parse_env_cfg(a.isaac_task, device="cuda:0", num_envs=a.isaac_envs)
    n_is = a.isaac_envs
    factor = -(-n_is // 4096)  # the capacity formula of scripts/co_rl/h200/scale_train.py
    env_cfg.sim.physx.gpu_max_rigid_patch_count = max(5 * 2 ** 15, n_is * 32)
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = max(2 ** 21, n_is * 64)
    env_cfg.sim.physx.gpu_heap_capacity = 2 ** 26 * min(factor, 8)
    env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 26 * min(factor, 4)
    if a.isaac_like_eval:
        env_cfg.observations.stack_policy.enable_corruption = False
    agent_cfg = load_cfg_from_registry(a.isaac_task, "co_rl_cfg_entry_point")
    agent_cfg.num_policy_stacks = 49
    agent_cfg.num_critic_stacks = 2
    env = CoRlVecEnvWrapper(gym.make(a.isaac_task, cfg=env_cfg), agent_cfg)
    obs_i, ex = env.get_observations()
    crit_i = ex["observations"]["critic"]
    critic_is = mlp(crit_i.shape[1], 1).to(dev)
    critic_is.load_state_dict(is_sd)
    if obs_i.shape[1] != M.OBS_DIM:
        sys.exit(f"Isaac policy input {obs_i.shape[1]} != {M.OBS_DIM}")

    all_params = list(actor.parameters()) + list(critic_mj.parameters()) + list(critic_is.parameters()) + [log_std]
    opt = torch.optim.Adam(all_params, lr=a.lr)
    if "optimizer_joint" in ck:
        opt.load_state_dict(ck["optimizer_joint"])
    lr = opt.param_groups[0]["lr"]
    midx, msign = (torch.as_tensor(x, device=dev) for x in M.mirror_maps())
    act_perm = torch.tensor([1, 0, 3, 2], device=dev)
    lo, hi = np.log(a.std_range[0]), np.log(a.std_range[1])
    base_sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    stats = {lv: deque(maxlen=300) for lv in range(M.LEVELS)}
    is_eps = deque(maxlen=50)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, list, bool, type(None)))}
                  | {"extra_dim": X, "isaac_critic_dim": int(crit_i.shape[1])}, f, indent=1)

    def save(it: int, tag: str):
        torch.save({"actor": actor.state_dict(), "critic_mj": critic_mj.state_dict(), "critic_is": critic_is.state_dict(),
                    "log_std": log_std.detach().cpu(), "optimizer_joint": opt.state_dict(), "iteration": it,
                    "level_max": level_max, "base_ckpt": base_ckpt}, os.path.join(a.out, f"{tag}.pt"))
        sd = {k: v.clone() for k, v in base_sd.items()}
        for k, v in actor.state_dict().items():
            sd["actor." + k] = v.detach().cpu()
        for k, v in critic_is.state_dict().items():
            sd["critic." + k] = v.detach().cpu()
        sd["std"] = log_std.detach().exp().cpu()
        torch.save({"model_state_dict": sd, "iter": it, "infos": {"joint_ppo_blind": a.out}},
                   os.path.join(a.out, f"model_{tag}.pt"))
        w = {}
        for i, j in enumerate([0, 2, 4, 6]):
            w[f"W{i}"] = actor[j].weight.detach().cpu().numpy().astype(np.float64)
            w[f"b{i}"] = actor[j].bias.detach().cpu().numpy().astype(np.float64)
        np.savez(os.path.join(a.out, f"actor_{tag}.npz"), **w)

    def gae(rew, val, done, last_v, extra_next=None):
        adv = torch.zeros_like(rew)
        g = torch.zeros_like(last_v)
        for t in reversed(range(rew.shape[0])):
            nv = last_v if t == rew.shape[0] - 1 else val[t + 1]
            nxt = (1 - done[t]) * nv + (extra_next[t] if extra_next is not None else 0.0)
            delta = rew[t] + a.gamma * nxt - val[t]
            g = delta + a.gamma * a.lam * (1 - done[t]) * g
            adv[t] = g
        return adv, adv + val

    T = a.isaac_horizon
    t_start = time.time()
    for it in range(start_it, start_it + a.iterations):
        t0 = time.time()
        push_weights()
        for c in conns:
            c.send(("rollout", level_max))
        # ---- Isaac rollout while the MuJoCo workers run ----
        bi = {"obs": torch.zeros(T, n_is, M.OBS_DIM, device=dev), "crit": torch.zeros(T, n_is, crit_i.shape[1], device=dev),
              "act": torch.zeros(T, n_is, 4, device=dev), "logp": torch.zeros(T, n_is, device=dev),
              "val": torch.zeros(T, n_is, device=dev), "rew": torch.zeros(T, n_is, device=dev),
              "done": torch.zeros(T, n_is, device=dev)}
        with torch.no_grad():
            std = log_std.exp()
            for t in range(T):
                mu = actor(obs_i)
                act = mu + std * torch.randn_like(mu)
                v = critic_is(crit_i).squeeze(-1)
                bi["obs"][t], bi["crit"][t], bi["act"][t], bi["val"][t] = obs_i, crit_i, act, v
                bi["logp"][t] = torch.distributions.Normal(mu, std).log_prob(act).sum(-1)
                env_act = torch.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                obs_i, rew, dones, extras = env.step(env_act)
                rew = rew.to(dev).float()
                if "time_outs" in extras:  # co_rl: bootstrap a time-out with the value of the state it left
                    rew = rew + a.gamma * v * extras["time_outs"].to(dev).float()
                bi["rew"][t], bi["done"][t] = rew, dones.to(dev).float()
                if a.isaac_like_eval and dones.any():  # a new episode starts with its first frame repeated
                    ids = dones.bool().nonzero().squeeze(-1)
                    for handler in (env.policy_state_handler, env.critic_state_handler):
                        for i in range(1, len(handler.stack_buffer)):
                            handler.stack_buffer[i][ids] = handler.stack_buffer[0][ids]
                    ob_d = extras["observations"]
                    obs_i = torch.cat([env.policy_state_handler.get_stacked(), ob_d["none_stack_policy"]], -1)
                    ob_d["critic"] = torch.cat([env.critic_state_handler.get_stacked(), ob_d["none_stack_critic"]], -1)
                crit_i = extras["observations"]["critic"]
                log = extras.get("log", {})
                if "Curriculum/terrain_levels" in log:
                    is_eps.append({k: float(v) for k, v in log.items() if isinstance(v, (int, float)) or
                                   (hasattr(v, "numel") and v.numel() == 1)})
            last_vi = critic_is(crit_i).squeeze(-1)
        t_isaac = time.time() - t0
        for c in conns:
            for info in c.recv():
                stats[info["level"]].append(info)
        t_roll = time.time() - t0
        bm = {k: torch.from_numpy(sv[k].copy()).to(dev)
              for k in ("obs", "ext", "act", "logp", "val", "rew", "term", "tout", "fval")}
        with torch.no_grad():
            done_m = torch.clamp(bm["term"] + bm["tout"], max=1.0)
            # a MuJoCo time-out bootstraps from the state it ended in, a failure from nothing
            adv_m, ret_m = gae(bm["rew"], bm["val"], done_m, torch.from_numpy(sv["last_val"].copy()).to(dev),
                               bm["tout"] * bm["fval"])
            adv_i, ret_i = gae(bi["rew"], bi["val"], bi["done"], last_vi)
        fm = {"obs": bm["obs"].reshape(-1, M.OBS_DIM), "crit": torch.cat([bm["obs"], bm["ext"]], 2).reshape(-1, M.OBS_DIM + X),
              "act": bm["act"].reshape(-1, 4), "logp": bm["logp"].reshape(-1), "ret": ret_m.reshape(-1),
              "adv": adv_m.reshape(-1)}
        fi = {"obs": bi["obs"].reshape(-1, M.OBS_DIM), "crit": bi["crit"].reshape(-1, crit_i.shape[1]),
              "act": bi["act"].reshape(-1, 4), "logp": bi["logp"].reshape(-1), "ret": ret_i.reshape(-1),
              "adv": adv_i.reshape(-1)}
        for f_ in (fm, fi):
            f_["adv"] = (f_["adv"] - f_["adv"].mean()) / (f_["adv"].std() + 1e-8)
        del bm, bi
        train_actor = it - start_it >= warmup
        kls, vls, mls = [], [], []

        def surrogate(f_, ix, critic):
            ob = f_["obs"][ix]
            vloss = (critic(f_["crit"][ix]).squeeze(-1) - f_["ret"][ix]).square().mean()
            if not train_actor:
                return vloss, None, None, ob
            mu = actor(ob)
            dist = torch.distributions.Normal(mu, log_std.exp())
            lp = dist.log_prob(f_["act"][ix]).sum(-1)
            ratio = (lp - f_["logp"][ix]).exp()
            A = f_["adv"][ix]
            pg = -torch.minimum(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
            with torch.no_grad():
                kls.append(float((f_["logp"][ix] - lp).mean()))
            return vloss, pg, mu, ob

        for _ in range(a.epochs):
            pm = torch.randperm(fm["obs"].shape[0], device=dev).chunk(a.minibatches)
            pi = torch.randperm(fi["obs"].shape[0], device=dev).chunk(a.minibatches)
            for ixm, ixi in zip(pm, pi):
                vl_m, pg_m, mu_m, ob_m = surrogate(fm, ixm, critic_mj)
                vl_i, pg_i, mu_i, ob_i = surrogate(fi, ixi, critic_is)
                loss = vl_m + vl_i
                if train_actor:
                    loss = loss + a.mujoco_weight * pg_m + a.isaac_weight * pg_i
                    if a.mirror > 0:
                        ml = 0.0
                        for ob, mu in ((ob_m, mu_m), (ob_i, mu_i)):
                            k = min(a.mirror_samples, ob.shape[0])
                            mm = actor(ob[:k][:, midx] * msign)
                            ml = ml + (mm - mu[:k].detach()[:, act_perm]).square().mean() + \
                                (mu[:k] - mm.detach()[:, act_perm]).square().mean()
                        loss = loss + a.mirror * 0.25 * ml
                        mls.append(float(ml))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
                with torch.no_grad():
                    log_std.clamp_(lo, hi)
                vls.append((float(vl_m), float(vl_i)))
            if train_actor and kls and abs(np.mean(kls[-2 * a.minibatches:])) > 2.5 * a.kl_target:
                break
        if train_actor and kls:
            kl_m = abs(float(np.mean(kls[-2 * a.minibatches:])))
            lr = max(lr / 1.5, 1e-6) if kl_m > 2 * a.kl_target else (min(lr * 1.5, 1e-3) if kl_m < a.kl_target / 2 else lr)
            for g in opt.param_groups:
                g["lr"] = lr
        top = stats[level_max]
        if level_max < M.LEVELS - 1 and len(top) >= 150 and np.mean([s["risers"] >= 2.9 for s in top]) >= a.promote:
            level_max += 1
        row = {"it": it, "wall_s": round(time.time() - t_start, 1), "iter_s": round(time.time() - t0, 2),
               "isaac_s": round(t_isaac, 2), "roll_s": round(t_roll, 2), "level_max": level_max, "lr": lr,
               "kl": float(np.mean(kls)) if kls else 0.0, "vloss_mj": float(np.mean([v[0] for v in vls])),
               "vloss_is": float(np.mean([v[1] for v in vls])), "mirror": float(np.mean(mls)) if mls else 0.0,
               "std": [round(float(x), 4) for x in log_std.detach().exp().cpu()]}
        if is_eps:
            for key in ("Curriculum/terrain_levels", "Curriculum/gait/risers_climbed", "Curriculum/gait/lifted_step_ups",
                        "Episode_Termination/time_out", "Episode_Termination/base_contact"):
                vals = [e[key] for e in is_eps if key in e]
                if vals:
                    row["is_" + key.split("/")[-1]] = round(float(np.mean(vals)), 3)
        for lv in range(M.LEVELS):
            if len(stats[lv]) >= 20:
                r_ = np.array([s["risers"] for s in stats[lv]])
                row[f"L{lv}"] = [round(float((r_ >= 2.9).mean()), 3), round(float(r_.mean()), 2), len(r_)]
        with open(os.path.join(a.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        if it % 10 == 0:
            print(json.dumps(row), flush=True)
        if it % a.save_every == 0 or stop[0]:
            save(it, f"{it}")
        if it % 10 == 0:
            save(it, "latest")
        if stop[0]:
            break
    save(it, "latest")
    for c in conns:
        c.send(("close", 0))
    for p in procs:
        p.join(5)
    env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
