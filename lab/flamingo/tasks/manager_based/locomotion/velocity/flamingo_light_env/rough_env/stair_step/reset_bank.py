# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Diverse resets for the blind stair task, after OmniReset ("Emergent dexterity via diverse resets and large-scale
reinforcement learning", docs/): a share of the episodes starts from a state recorded on the same stair tile
instead of the standing spawn in the pit - approaching a riser, a wheel against its face, one wheel up, on the
stairs, and the last second before a failure. Every episode still runs from there under the same rewards and
terminations, so the policy meets the hard moments of a climb many times per episode budget, not once.

The bank is written by ``scripts/co_rl/build_reset_bank.py``: per state the root link pose relative to the
tile origin, the root COM velocity (world), every joint position / velocity, and the tile (row, column) - a
state only fits the tile it came from, so the env is moved to that column of its own curriculum row.
"""

from __future__ import annotations

import json
import os
import torch
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.managers import SceneEntityCfg

from . import symmetry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

LABELS = ("approach", "face_contact", "one_wheel_up", "on_stairs", "pre_failure")


def terrain_signature(gen) -> dict:
    """What fixes every tile of the stair terrain (the generator draws one difficulty per tile from its seed)."""
    sub = next(iter(gen.sub_terrains.values()))
    return {
        "seed": gen.seed, "num_rows": gen.num_rows, "num_cols": gen.num_cols, "size": list(gen.size),
        "difficulty_range": list(gen.difficulty_range), "step_height_range": list(sub.step_height_range),
        "step_width_range": list(getattr(sub, "step_width_range", (sub.step_width, sub.step_width))),
        "difficulty_scale": getattr(sub, "difficulty_scale", 1.0), "platform_width": sub.platform_width,
        "border_width": sub.border_width,
    }


def _mirror_name(name: str) -> str:
    for a, b in (("left_", "right_"), ("FL_", "FR_"), ("RL_", "RR_")):
        if name.startswith(a):
            return b + name[len(a):]
        if name.startswith(b):
            return a + name[len(b):]
    return name


class StateBank:
    """The bank on the env's device, indexed by terrain row, with joints in the robot's order."""

    def __init__(self, path: str, robot, terrain_gen, device: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"reset state bank {path} not found (scripts/co_rl/build_reset_bank.py)")
        d = np.load(path)
        recorded, here = json.loads(str(d["terrain_json"])), terrain_signature(terrain_gen)
        if recorded != here:
            raise ValueError(f"bank {path} was recorded on other stairs: {recorded} != {here}")
        names = [str(n) for n in d["joint_names"]]
        if sorted(names) != sorted(robot.joint_names):
            raise ValueError(f"bank joints {names} != robot joints {robot.joint_names}")
        order = [names.index(n) for n in robot.joint_names]

        def t(key, dtype=torch.float32):
            return torch.as_tensor(d[key], dtype=dtype, device=device)

        self.rel_pos, self.quat, self.vel = t("rel_pos"), t("quat"), t("vel")
        self.joint_pos, self.joint_vel = t("joint_pos")[:, order], t("joint_vel")[:, order]
        self.row, self.col, self.label = t("row", torch.long), t("col", torch.long), t("label", torch.long)
        # the policy's stack_policy frames up to the state (M, frames, dim), newest first
        self.history = t("history")
        self.by_row = [torch.nonzero(self.row == r).squeeze(-1) for r in range(int(self.row.max()) + 1)]
        # left <-> right: shoulder angles and wheel speeds keep their sign (see symmetry.py)
        self.mirror = torch.tensor([robot.joint_names.index(_mirror_name(n)) for n in robot.joint_names], device=device)


def reset_from_state_bank(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    bank_path: str,
    probability: float = 0.7,
    mirror: bool = True,
    lin_vel_noise: float = 0.1,
    ang_vel_noise: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset event (after reset_base / reset_robot_joints): with ``probability`` an env starts from a random bank
    state of its curriculum row (moved to that state's column), mirrored left-right half of the time, with
    uniform root velocity noise (+-``lin_vel_noise`` m/s, +-``ang_vel_noise`` rad/s)."""
    robot = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    bank = getattr(env, "_state_bank", None)
    if bank is None:
        bank = env._state_bank = StateBank(bank_path, robot, terrain.cfg.terrain_generator, env.device)
        perm, sign = symmetry._group_map(env, "stack_policy", {})
        if len(perm) != bank.history.shape[-1]:
            raise ValueError(f"bank frames have {bank.history.shape[-1]} entries, stack_policy has {len(perm)}")
        bank.frame_perm = torch.tensor(perm, device=env.device)
        bank.frame_sign = torch.tensor(sign, device=env.device)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not hasattr(env, "_bank_episode"):  # which envs run an episode started from the bank, and its label
        env._bank_episode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._bank_label = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    env._bank_episode[ids] = False
    env._bank_label[ids] = -1
    # frames still waiting for the wrapper (several resets before one observation) of envs reset again now
    pending = getattr(env, "_bank_history", None)
    if pending is not None:
        keep = ~torch.isin(pending[0], ids)
        env._bank_history = (pending[0][keep], pending[1][keep]) if keep.any() else None
    ids = ids[torch.rand(len(ids), device=env.device) < probability]
    rows = terrain.terrain_levels[ids]
    idx = torch.full_like(ids, -1)
    for r in torch.unique(rows).tolist():
        if r >= len(bank.by_row) or bank.by_row[r].numel() == 0:
            continue
        sel = rows == r
        pool = bank.by_row[r]
        idx[sel] = pool[torch.randint(len(pool), (int(sel.sum()),), device=env.device)]
    ok = idx >= 0
    ids, idx, rows = ids[ok], idx[ok], rows[ok]
    env._bank_reset_count = getattr(env, "_bank_reset_count", 0) + len(ids)
    if len(ids) == 0:
        return
    env._bank_episode[ids] = True
    env._bank_label[ids] = bank.label[idx]
    col = bank.col[idx]
    terrain.terrain_types[ids] = col
    terrain.env_origins[ids] = terrain.terrain_origins[rows, col]

    rel, quat, vel = bank.rel_pos[idx].clone(), bank.quat[idx].clone(), bank.vel[idx].clone()
    q, qd = bank.joint_pos[idx].clone(), bank.joint_vel[idx].clone()
    frames = bank.history[idx].clone()
    if mirror:
        # reflection y -> -y about the tile centre (the inverted pyramid is symmetric about it)
        m = torch.rand(len(ids), device=env.device) < 0.5
        flip = torch.tensor([1.0, -1.0, 1.0], device=env.device)
        rel[m] = rel[m] * flip
        quat[m] = quat[m] * torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device)
        vel[m, :3] = vel[m, :3] * flip
        vel[m, 3:] = vel[m, 3:] * -flip
        q[m] = q[m][:, bank.mirror]
        qd[m] = qd[m][:, bank.mirror]
        frames[m] = frames[m][..., bank.frame_perm] * bank.frame_sign
    # the co_rl wrapper puts these frames into the policy's observation history (a blind policy that sees the
    # previous episode's frames after the jump to a moving state falls within 0.2 s, even on flat approach states)
    pending = getattr(env, "_bank_history", None)
    if pending is not None:
        env._bank_history = (torch.cat([pending[0], ids]), torch.cat([pending[1], frames]))
    else:
        env._bank_history = (ids, frames)
    noise = torch.rand(len(ids), 6, device=env.device) * 2.0 - 1.0
    vel[:, :3] += noise[:, :3] * lin_vel_noise
    vel[:, 3:] += noise[:, 3:] * ang_vel_noise

    pos = terrain.env_origins[ids] + rel
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    robot.write_root_com_velocity_to_sim(vel, env_ids=ids)
    robot.write_joint_state_to_sim(q, qd, env_ids=ids)
