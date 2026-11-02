"""Record the MuJoCo cosim's own trials of a blind actor in the format of ``record_states_stair_step.py``, so that
``render_states_mujoco.py`` films them like the Isaac rollouts.

The trials are the evaluator's (``cosim_eval_policy.py``: ``ev.run_trial`` with the blind 50-frame observation and
Isaac's speed limits, the same seeds and tile columns for ``--trials`` per level); ``Sim.reset`` / ``Sim.step`` are
wrapped only to copy the root pose and the joint positions after every policy step, plus the evaluator's
trailing-wheel climb, recomputed by its own rule from the wheel contacts.

    <cosim_venv python> scripts/co_rl/record_states_cosim.py --actor <actor.npz> --checkpoint <model.pt> \\
        --level 9 --trials 64 --terrain_json_from logs/co_rl/_reset_bank/stair_bank_v1.npz --out <dir>/states_L9.npz
"""

import argparse
import os
import sys

os.environ.setdefault("COSIM_EVAL_FRAMES", "50")
os.environ.setdefault("COSIM_EVAL_SCAN", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import cosim_eval_policy as cep  # noqa: E402  (blind observation + Isaac speed limits)

ev = cep.ev
NAMES = ["FL_caster_joint", "FR_caster_joint", "RL_caster_joint", "RR_caster_joint", "left_shoulder_joint",
         "left_upper_joint", "right_shoulder_joint", "right_upper_joint", "left_leg_joint", "right_leg_joint",
         "left_wheel_joint", "right_wheel_joint"]  # Isaac's joint order
REC = []
_reset, _step = ev.Sim.reset, ev.Sim.step


def _snap(sim):
    d, m = sim.d, sim.m
    fz, _ = sim.contacts()
    REC.append((d.qpos[:3].copy(), d.qpos[3:7].copy(),
                np.array([d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in NAMES]), fz.copy(), sim.wheel_bottom_z().copy()))


def reset(self, *a, **k):
    _reset(self, *a, **k)
    REC.clear()
    _snap(self)


def step(self, a):
    _step(self, a)
    _snap(self)


ev.Sim.reset, ev.Sim.step = reset, step


def climb_track(h: float) -> np.ndarray:
    """The evaluator's riser count after every step (run_trial: support > 5 N, floor, 1.25 cm gain rule)."""
    floor, support, best, out = np.inf, np.full(2, np.nan), 0.0, []
    for _, _, _, fz, wz in REC:
        on = fz > 5.0
        support = np.where(on, wz, support)
        if on.any():
            floor = min(floor, wz[on].min())
        if not np.isnan(support).any():
            gain = support.min() - floor - best
            if gain >= 0.0125:
                best += gain
        out.append(best / h)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", type=int, default=9)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain_json_from", required=True, help="an npz holding the Isaac terrain generator's json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lv, n = a.level, a.trials
    runs = []
    for i in range(n):
        res = ev.run_trial((lv, int(i // (n / ev.COLS)), i, a.seed * 100003 + lv * 10007 + i, a.actor, a.seconds,
                            0.35, 0.0, False))
        runs.append((res, list(REC), climb_track(res["riser_m"])))
        print(f"[record] trial {i}: col {res['col']} risers {res['risers']:.2f} end {res['end']} at {res['t_end']} s",
              flush=True)
    T = max(len(r[1]) for r in runs)

    def stack(k):
        return np.stack([np.stack([r[1][min(t, len(r[1]) - 1)][k] for t in range(T)]) for r in runs], axis=1)

    climb = np.stack([np.array([r[2][min(t, len(r[2]) - 1)] for t in range(T)]) for r in runs], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, pos=stack(0).astype(np.float32), quat=stack(1).astype(np.float32), joint_pos=stack(2).astype(np.float32),
        climb=climb.astype(np.float32), joint_names=np.array(NAMES), row=np.full(n, lv), col=np.array([r[0]["col"] for r in runs]),
        yaw=np.zeros(n), risers=np.array([r[0]["risers"] for r in runs]), end=np.array([r[0]["end"] for r in runs]),
        end_step=np.array([len(r[1]) - 1 for r in runs]), step_dt=ev.DT * ev.DECIMATION, riser_nominal=ev.nominal_riser(lv),
        checkpoint=a.checkpoint, terrain_json=str(np.load(a.terrain_json_from)["terrain_json"]))
    r = np.array([x[0]["risers"] for x in runs])
    print(f"[record] {n} trials -> {a.out}: 3 risers {np.mean(r >= 2.9) * 100:.1f} %, mean {r.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
