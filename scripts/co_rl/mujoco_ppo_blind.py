"""PPO of the BLIND stair policy (50 proprioceptive frames + command = 804 inputs) directly in the MuJoCo cosim.

Every Isaac-trained policy climbs ~0 risers in the cosim (fast, low-foot approaches that PhysX lets through).
Codex's ``cosim-anaylze/tools/finetune_stair_step_mujoco.py`` showed for the perceptive model_2749 that PPO in
the cosim itself fixes that (15 cm, one riser 0 -> 70 %). This is the blind counterpart, sized for every CPU core:

* the robot, stairs, actuators, randomisation, terminations and riser count are the evaluator's own
  (``eval_stair_step_v01_2_2.py`` via ``cosim_eval_policy.py``, which also adds Isaac's joint / link speed limits);
  tiles are the evaluator's rows 0-9 (9.3-15 cm, its columns) after four low warm-up heights (4-8 cm);
* rewards follow the evaluator's metric: the trailing wheel's climb (its own support / floor bookkeeping), the
  leading wheel's highest tread, milestones at 3 and 5 risers, forward progress; failures cost; jumping is free;
* the actor (and its std) starts from a co_rl blind checkpoint; the critic is new (privileged MuJoCo state) and is
  trained alone for ``--critic_warmup`` iterations;
* checkpoints are written as the cosim actor NPZ and as a co_rl ``model_<it>.pt`` (actor + std, the critic of the
  start checkpoint) that ``eval_stair_step.py`` / ``export`` accept.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \\
        <venv python> -u scripts/co_rl/mujoco_ppo_blind.py --init <model_1400.pt> --out <dir>
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import deque

os.environ.setdefault("COSIM_EVAL_FRAMES", "50")
os.environ.setdefault("COSIM_EVAL_SCAN", "0")
os.environ.setdefault("COSIM_EVAL_SPEED_LIMITS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/work/postech/flamingo_light/cosim_venv/lib/python3.11/site-packages")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import cosim_eval_policy as cep  # noqa: E402  (blind observation + Isaac speed limits)

ev = cep.ev
FRAMES = int(os.environ["COSIM_EVAL_FRAMES"])
OBS_DIM = 16 * FRAMES + 4
# curriculum levels: four low warm-up stairs, then the evaluator's terrain rows 0..9
WARMUP = [0.04, 0.055, 0.07, 0.08]
LEVELS = len(WARMUP) + ev.ROWS
AHEAD = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])  # terrain probes in front of each wheel [m] (critic only)


def level_tile(level: int, rng) -> tuple[float, float, int, int]:
    if level < len(WARMUP):
        return WARMUP[level], 0.5, -1, -1
    row, col = level - len(WARMUP), int(rng.integers(ev.COLS))
    h, w = ev.tile_geometry(row, col)
    return h, w, row, col


class BlindEnv:
    """One robot on one stair tile, episode protocol and riser count of ``ev.run_trial``."""

    def __init__(self, seed: int, seconds: float):
        self.rng = np.random.default_rng(seed)
        self.seconds = seconds
        self.sims = {}
        self.level = 0

    SIM_CACHE = 4  # tiles kept per env: an unbounded cache (~100 tiles x every env) ran a worker to 12 GiB, OOM

    def _sim(self, h: float, w: float):
        key = (round(h, 5), round(w, 5))
        if key in self.sims:
            self.sims[key] = self.sims.pop(key)  # most recently used last
        else:
            while len(self.sims) >= self.SIM_CACHE:
                self.sims.pop(next(iter(self.sims)))
            s = ev.Sim(h, w)  # ~34 ms
            s.mass0, s.ipos0 = float(s.m.body_mass[s.base]), s.m.body_ipos[s.base].copy()
            self.sims[key] = s
        return self.sims[key]

    def stair_heights(self, ahead: np.ndarray) -> np.ndarray:
        """Analytic tread height under points ``ahead`` m in front of each tire (x of the world, stairs ring the pit)."""
        s = self.sim
        p = np.repeat(s.d.geom_xpos[s.wheel_geoms, None, :2], len(ahead), axis=1)  # (2 wheels, probes, xy)
        p[..., 0] += ahead
        nn = int((ev.TILE - 2 * ev.BORDER - ev.PLATFORM) // (2 * s.w) + 1)
        k = np.floor((ev.TILE / 2 - ev.BORDER - np.abs(p).max(axis=2)) / s.w).astype(int)
        return np.where((k >= 0) & (k < nn), (nn - k) * s.h, 0.0)  # (2, len(ahead))

    def reset(self, level: int):
        self.level = level
        h, w, self.row, self.col = level_tile(level, self.rng)
        s = self.sim = self._sim(h, w)
        m, rng = s.m, self.rng
        # the evaluator's randomisation (run_trial), plus a little heading spread
        m.body_ipos[s.base] = s.ipos0 + rng.uniform(-0.05, 0.05, 3)
        m.body_mass[s.base] = s.mass0 + rng.uniform(-1.0, 1.0)
        mu = rng.uniform(0.5, 0.8)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0 and m.geom_contype[g]:
                m.geom_friction[g] = [mu, 0.02, 0.01]
        lag = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
        yaw = rng.uniform(-0.15, 0.15) if rng.random() < 0.5 else 0.0
        s.reset(yaw, -0.05 + rng.uniform(-0.1, 0.1, 2), lag)
        self.t = 0.0
        self.floor_z, self.support, self.best = np.inf, np.full(2, np.nan), 0.0
        self.lead = 0.0
        self.spawn = s.d.qpos[:2].copy()
        self.best_dist, self.last_prog = 0.0, 0.0
        self.prev_a = np.zeros(4)
        self.milestones = 0
        self.ret = 0.0
        return self.observe()

    def observe(self):
        s, d = self.sim, self.sim.d
        obs = s.observe(0.35).astype(np.float32)
        R = ev.quat_rot(d.qpos[3:7])
        floor = self.floor_z if np.isfinite(self.floor_z) else d.qpos[2] - 0.27
        sup = np.nan_to_num(self.support - floor, nan=0.0) / s.h
        hz = self.stair_heights(AHEAD) / s.h
        extra = np.r_[R.T @ d.qvel[:3], d.qpos[:2] - self.spawn, d.qpos[2] - floor, sup, s.h * 10, s.w,
                      self.best / s.h, self.t / self.seconds, float(self.milestones), float(self.level) / LEVELS,
                      hz.reshape(-1), np.array(s.lag) / 4.0]
        return obs, extra.astype(np.float32)

    def step(self, action: np.ndarray):
        s, d = self.sim, self.sim.d
        a = np.clip(action, -1.0, 1.0)
        x0 = d.qpos[0]
        s.step(a)
        self.t += ev.DT * ev.DECIMATION
        # --- the evaluator's bookkeeping (run_trial) ---
        fz, illegal = s.contacts()
        wz = s.wheel_bottom_z()
        on = fz > 5.0
        self.support = np.where(on, wz, self.support)
        if on.any():
            self.floor_z = min(self.floor_z, wz[on].min())
        climbed = 0.0
        if not np.isnan(self.support).any():
            gain = self.support.min() - self.floor_z - self.best
            if gain >= 0.0125:
                self.best += gain
                climbed = gain / s.h
        lead = float(np.nanmax(self.support) - self.floor_z) / s.h if np.isfinite(self.floor_z) else 0.0
        lead_gain = max(0.0, lead - self.lead)
        self.lead = max(self.lead, lead)
        dist = np.linalg.norm(d.qpos[:2] - self.spawn)
        if dist > self.best_dist + 0.05:
            self.best_dist, self.last_prog = dist, self.t
        R = ev.quat_rot(d.qpos[3:7])
        tilt = float(np.arccos(np.clip(-(R.T @ np.array([0, 0, -1.0]))[2], -1, 1)))
        vx = float((R.T @ d.qvel[:3])[0])
        fail = bool(illegal or tilt > 0.6 or min(d.qpos[s.sh_q]) < -0.77 or self.t - self.last_prog > 4.0
                    or not np.isfinite(d.qpos).all())
        timeout = self.t >= self.seconds - 1e-9
        # --- reward ---
        risers = self.best / s.h
        r = 6.0 * climbed + 1.0 * min(lead_gain, 2.0) + 2.0 * float(np.clip(d.qpos[0] - x0, -0.04, 0.04))
        for k, mark in enumerate((2.9, 4.9)):
            if self.milestones <= k and risers >= mark:
                self.milestones = k + 1
                r += 10.0
        r += 0.02 * float(np.exp(-((vx - 0.35) / 0.3) ** 2)) - 0.02 * tilt ** 2
        r -= 0.05 * max(0.0, (vx - 0.6) / 0.4) ** 2
        r -= 0.01 * float(np.square(np.maximum(-0.745 - d.qpos[s.sh_q], 0.0) / 0.025).sum())
        r -= 0.005 * float(np.square(a - self.prev_a).mean())
        self.prev_a = a
        if fail:
            r -= 8.0
        self.ret += r
        info = None
        if fail or timeout:
            info = {"level": self.level, "row": self.row, "risers": risers, "time": self.t, "fail": fail,
                    "ret": self.ret}
        return r, fail, timeout, info


class Shared:
    """RAM shared by the learner and the workers: one iteration of trajectories (horizon x envs) and the current
    weights. The workers run the policy themselves, so the learner (and the busy GPU) is not in the step loop."""

    def __init__(self, ctx, n: int, horizon: int, ext_dim: int, n_weights: int):
        self.n, self.T, self.ext_dim, self.n_weights = n, horizon, ext_dim, n_weights
        self.shapes = {"obs": (horizon, n, OBS_DIM), "ext": (horizon, n, ext_dim), "act": (horizon, n, 4),
                       "logp": (horizon, n), "val": (horizon, n), "rew": (horizon, n), "term": (horizon, n),
                       "tout": (horizon, n), "fval": (horizon, n), "last_val": (n,), "weights": (n_weights,)}
        self.raw = {k: ctx.RawArray("f", int(np.prod(s))) for k, s in self.shapes.items()}

    def views(self):
        return {k: np.frombuffer(self.raw[k], np.float32).reshape(s) for k, s in self.shapes.items()}


def mlp_np(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))  # ELU
    return x


def unpack(flat, layout):
    """[(name, shape)] -> actor layers, critic layers, log_std (views into the shared weights)."""
    out, k = {}, 0
    for name, shape in layout:
        size = int(np.prod(shape))
        out[name] = flat[k:k + size].reshape(shape)
        k += size
    pairs = lambda pre: [(out[f"{pre}.{i}.weight"], out[f"{pre}.{i}.bias"]) for i in (0, 2, 4, 6)]  # noqa: E731
    return pairs("actor"), pairs("critic"), out["log_std"]


def worker(conn, shared: Shared, lo: int, count: int, seed: int, seconds: float, layout):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    v = shared.views()
    sl = slice(lo, lo + count)
    rng = np.random.default_rng(seed + 1)
    envs = [BlindEnv(seed + 7919 * i, seconds) for i in range(count)]
    obs = np.zeros((count, OBS_DIM), np.float32)
    ext = np.zeros((count, shared.ext_dim), np.float32)
    started = False
    while True:
        cmd, level_max = conn.recv()
        if cmd == "close":
            break

        def draw() -> int:  # half at the top unlocked level, the rest spread below it
            return level_max if rng.random() < 0.5 else int(rng.integers(0, level_max + 1))

        actor, critic, log_std = unpack(v["weights"], layout)
        std = np.exp(log_std)
        if not started:
            for i, e in enumerate(envs):
                obs[i], ext[i] = e.reset(draw())
            started = True
        infos = []
        rew = np.zeros(count, np.float32)
        term = np.zeros(count, np.float32)
        tout = np.zeros(count, np.float32)
        fval = np.zeros(count, np.float32)
        const = float(np.log(std).sum() + 2.0 * np.log(2.0 * np.pi))
        for t in range(shared.T):
            mu = mlp_np(actor, obs)
            act = (mu + std * rng.standard_normal(mu.shape)).astype(np.float32)
            v["obs"][t, sl], v["ext"][t, sl], v["act"][t, sl] = obs, ext, act
            v["logp"][t, sl] = -0.5 * (((act - mu) / std) ** 2).sum(1) - const
            v["val"][t, sl] = mlp_np(critic, np.concatenate([obs, ext], 1))[:, 0]
            fval[:] = 0.0
            for i, e in enumerate(envs):
                r, fail, timeout, info = e.step(act[i])
                rew[i], term[i], tout[i] = r, float(fail), float(timeout and not fail)
                if fail or timeout:
                    if not fail:  # a time-out bootstraps from the state it ended in
                        fo, fx = e.observe()
                        fval[i] = mlp_np(critic, np.concatenate([fo, fx])[None])[0, 0]
                    infos.append(info)
                    obs[i], ext[i] = e.reset(draw())
                else:
                    obs[i], ext[i] = e.observe()
            v["rew"][t, sl], v["term"][t, sl], v["tout"][t, sl], v["fval"][t, sl] = rew, term, tout, fval
        v["last_val"][sl] = mlp_np(critic, np.concatenate([obs, ext], 1))[:, 0]
        conn.send(infos)


def ext_dim() -> int:
    e = BlindEnv(0, 15.0)
    return int(e.reset(0)[1].shape[0])


def mirror_maps():
    frame = np.array([1, 0, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11, 13, 12, 15, 14])
    fsign = np.ones(16, np.float32)
    fsign[[6, 8, 10]] = -1.0  # roll rate, yaw rate, lateral gravity
    idx = np.concatenate([k * 16 + frame for k in range(FRAMES)] + [16 * FRAMES + np.arange(4)])
    sign = np.concatenate([fsign] * FRAMES + [np.array([1.0, -1.0, -1.0, 1.0], np.float32)])
    return idx, sign


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="co_rl blind checkpoint (model_*.pt) or a latest.pt of this trainer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--envs_per_worker", type=int, default=72)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl_target", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--mirror", type=float, default=1.0)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--std_init", type=float, default=0.15)
    ap.add_argument("--std_range", type=float, nargs=2, default=[0.03, 0.4])
    ap.add_argument("--critic_warmup", type=int, default=30)
    ap.add_argument("--start_level", type=int, default=0)
    ap.add_argument("--promote", type=float, default=0.5, help="3-riser share at the top level that promotes it")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import torch
    from torch import nn

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    n_env = a.workers * a.envs_per_worker

    def mlp(inp, out):
        return nn.Sequential(nn.Linear(inp, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128),
                             nn.ELU(), nn.Linear(128, out))

    X = ext_dim()
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    level_max = a.start_level if "model_state_dict" in ck else ck["level_max"]
    actor, critic = mlp(OBS_DIM, 4).to(dev), mlp(OBS_DIM + X, 1).to(dev)
    log_std = nn.Parameter(torch.full((4,), float(np.log(a.std_init)), device=dev))
    start_it, base_ckpt = 0, a.init
    if "model_state_dict" in ck:  # co_rl checkpoint: the actor (and its std, floored at std_init) only
        sd = ck["model_state_dict"]
        actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
        log_std.data = torch.log(torch.maximum(sd["std"].to(dev), torch.tensor(a.std_init, device=dev)))
    else:
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        log_std.data = ck["log_std"].to(dev)
        start_it, base_ckpt = ck["iteration"] + 1, ck["base_ckpt"]
        a.critic_warmup = 0
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=a.lr)
    if "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    lr = a.lr
    midx, msign = mirror_maps()
    midx, msign = torch.as_tensor(midx, device=dev), torch.as_tensor(msign, device=dev)
    act_perm = torch.tensor([1, 0, 3, 2], device=dev)
    lo, hi = np.log(a.std_range[0]), np.log(a.std_range[1])
    base_sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    stats = {lv: deque(maxlen=300) for lv in range(LEVELS)}

    # workers: shared RAM for the trajectories and the weights, each runs its envs and the policy for a horizon
    params = [("actor." + k, p) for k, p in actor.named_parameters()] + \
             [("critic." + k, p) for k, p in critic.named_parameters()] + [("log_std", log_std)]
    layout = [(k, tuple(p.shape)) for k, p in params]
    ctx = mp.get_context("spawn")
    shared = Shared(ctx, n_env, a.horizon, X, sum(int(np.prod(s)) for _, s in layout))
    sv = shared.views()
    E = a.envs_per_worker
    conns, procs = [], []
    for w in range(a.workers):
        p, c = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(c, shared, w * E, E, a.seed * 1000 + w * 104729, a.seconds, layout),
                         daemon=True)
        pr.start()
        conns.append(p)
        procs.append(pr)

    def push_weights():
        with torch.no_grad():
            sv["weights"][:] = torch.cat([p.detach().float().reshape(-1) for _, p in params]).cpu().numpy()
    log_path = os.path.join(a.out, "metrics.jsonl")
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a) | {"obs_dim": OBS_DIM, "extra_dim": X, "levels": LEVELS, "warmup": WARMUP}, f, indent=1)

    def save(it: int, tag: str):
        state = {"actor": actor.state_dict(), "critic": critic.state_dict(), "log_std": log_std.detach().cpu(),
                 "optimizer": opt.state_dict(), "iteration": it, "level_max": level_max, "base_ckpt": base_ckpt}
        torch.save(state, os.path.join(a.out, f"{tag}.pt"))
        sd = {k: v.clone() for k, v in base_sd.items()}
        for k, v in actor.state_dict().items():
            sd["actor." + k] = v.detach().cpu()
        sd["std"] = log_std.detach().exp().cpu()
        torch.save({"model_state_dict": sd, "iter": it, "infos": {"mujoco_ppo_blind": a.out}},
                   os.path.join(a.out, f"model_{tag}.pt"))
        w = {}
        for i, j in enumerate([0, 2, 4, 6]):
            w[f"W{i}"] = actor[j].weight.detach().cpu().numpy().astype(np.float64)
            w[f"b{i}"] = actor[j].bias.detach().cpu().numpy().astype(np.float64)
        np.savez(os.path.join(a.out, f"actor_{tag}.npz"), **w)

    t_start = time.time()
    for it in range(start_it, start_it + a.iterations):
        t0 = time.time()
        push_weights()
        for c in conns:
            c.send(("rollout", level_max))
        for c in conns:
            for info in c.recv():
                stats[info["level"]].append(info)
        t_roll = time.time() - t0
        B = {k: torch.from_numpy(sv[k].copy()).to(dev)
             for k in ("obs", "ext", "act", "logp", "val", "rew", "term", "tout", "fval")}
        with torch.no_grad():
            last_v = torch.from_numpy(sv["last_val"].copy()).to(dev)
            adv = torch.zeros_like(B["rew"])
            gae = torch.zeros(n_env, device=dev)
            for t in reversed(range(a.horizon)):
                nv = last_v if t == a.horizon - 1 else B["val"][t + 1]
                done = torch.clamp(B["term"][t] + B["tout"][t], max=1.0)
                # a time-out bootstraps from the state it ended in, a failure from nothing
                target_next = (1 - done) * nv + B["tout"][t] * B["fval"][t]
                delta = B["rew"][t] + a.gamma * target_next - B["val"][t]
                gae = delta + a.gamma * a.lam * (1 - done) * gae
                adv[t] = gae
            ret = adv + B["val"]
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in B.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        N = flat["obs"].shape[0]
        train_actor = it - start_it >= a.critic_warmup
        kls, vls, mls = [], [], []
        for _ in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for ix in perm.chunk(a.minibatches):
                ob = flat["obs"][ix]
                vpred = critic(torch.cat([ob, flat["ext"][ix]], 1)).squeeze(-1)
                vloss = (vpred - flat["ret"][ix]).square().mean()
                loss = vloss
                if train_actor:
                    mu = actor(ob)
                    dist = torch.distributions.Normal(mu, log_std.exp())
                    lp = dist.log_prob(flat["act"][ix]).sum(-1)
                    ratio = (lp - flat["logp"][ix]).exp()
                    A = flat["adv"][ix]
                    pg = -torch.minimum(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
                    loss = loss + pg - a.entropy * dist.entropy().sum(-1).mean()
                    if a.mirror > 0:
                        mm = actor(ob[:, midx] * msign)
                        ml = (mm - mu.detach()[:, act_perm]).square().mean() + (mu - mm.detach()[:, act_perm]).square().mean()
                        loss = loss + a.mirror * 0.5 * ml
                        mls.append(float(ml))
                    with torch.no_grad():
                        kl = (flat["logp"][ix] - lp).mean()
                        kls.append(float(kl))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [log_std], 1.0)
                opt.step()
                with torch.no_grad():
                    log_std.clamp_(lo, hi)
                vls.append(float(vloss))
            if train_actor and kls and abs(np.mean(kls[-a.minibatches:])) > 2.5 * a.kl_target:
                break
        if train_actor and kls:  # co_rl's adaptive schedule
            kl_m = abs(float(np.mean(kls[-a.minibatches:])))
            if kl_m > 2 * a.kl_target:
                lr = max(lr / 1.5, 1e-6)
            elif kl_m < a.kl_target / 2:
                lr = min(lr * 1.5, 1e-3)
            for g in opt.param_groups:
                g["lr"] = lr
        # curriculum: unlock the next level when the top one is climbed (3 risers) often enough
        top = stats[level_max]
        if level_max < LEVELS - 1 and len(top) >= 150:
            share = float(np.mean([s["risers"] >= 2.9 for s in top]))
            if share >= a.promote:
                level_max += 1
        row = {"it": it, "wall_s": round(time.time() - t_start, 1), "iter_s": round(time.time() - t0, 2),
               "roll_s": round(t_roll, 2), "level_max": level_max, "lr": lr, "kl": float(np.mean(kls)) if kls else 0.0,
               "vloss": float(np.mean(vls)), "mirror": float(np.mean(mls)) if mls else 0.0,
               "std": [round(float(x), 4) for x in log_std.detach().exp().cpu()],
               "rew_step": float(B["rew"].mean()), "samples": N}
        for lv in range(LEVELS):
            if len(stats[lv]) >= 20:
                rs_ = np.array([s["risers"] for s in stats[lv]])
                row[f"L{lv}"] = [round(float((rs_ >= 2.9).mean()), 3), round(float(rs_.mean()), 2), len(rs_)]
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if it % 10 == 0:
            print(json.dumps(row), flush=True)
        if it % a.save_every == 0 or stop[0]:
            save(it, f"{it}")
        save(it, "latest") if it % 10 == 0 else None
        if stop[0]:
            break
    save(it, "latest")
    for c in conns:
        c.send(("close", 0))
    for p in procs:
        p.join(5)


if __name__ == "__main__":
    main()
