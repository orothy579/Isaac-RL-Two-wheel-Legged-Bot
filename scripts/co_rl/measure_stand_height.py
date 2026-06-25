# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the v03 flamingo_light base_link world-Z height as a function of shoulder angle.

This launches Isaac Sim headless, spawns FLAMINGO_CFG on a ground plane, holds the
shoulder joints at a series of commanded angles, lets the robot settle on its wheels,
and prints the settled base_link world-Z. The maximum is the full-extension standing
height = the value to use for base_height `target_height`.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure flamingo_light standing height.")
parser.add_argument("--usd", type=str, default=None, help="override USD path (default: FLAMINGO_CFG)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below runs after the app is up ----
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_CFG


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim.set_camera_view(eye=(2.0, 2.0, 1.0), target=(0.0, 0.0, 0.3))

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1000.0))

    cfg = FLAMINGO_CFG.replace(prim_path="/World/Robot")
    if args_cli.usd is not None:
        cfg.spawn.usd_path = args_cli.usd
        print(f"  [override] USD = {args_cli.usd}", flush=True)
    robot = Articulation(cfg)

    sim.reset()

    jn = robot.data.joint_names
    lim = robot.data.joint_pos_limits[0]  # (num_joints, 2)
    soft = robot.data.soft_joint_pos_limits[0]
    base_idx = robot.data.body_names.index("base_link")
    sh_ids = [i for i, n in enumerate(jn) if "shoulder_joint" in n]
    wh_ids = [i for i, n in enumerate(jn) if "wheel_joint" in n]

    print("\n================ JOINT INFO ================")
    for i, n in enumerate(jn):
        print(f"  [{i}] {n:24s} hard=({lim[i,0]:+.4f},{lim[i,1]:+.4f})  soft=({soft[i,0]:+.4f},{soft[i,1]:+.4f})")
    print(f"  shoulder ids={sh_ids}  wheel ids={wh_ids}  base body id={base_idx}")
    print(f"  spawn base_link world Z (init) = {robot.data.body_pos_w[0, base_idx, 2].item():.4f}")

    # sweep shoulder angle across the usable (soft) range
    sh_lo = float(soft[sh_ids[0], 0])
    sh_hi = float(soft[sh_ids[0], 1])
    angles = torch.linspace(sh_lo, sh_hi, 13, device=sim.device)

    print("\n========== STANDING HEIGHT vs SHOULDER ANGLE ==========", flush=True)
    print("  (settled base_link world-Z on flat ground; wheels held still)", flush=True)
    print("  also reporting passive leg/upper joint angles after settle\n", flush=True)
    passive_ids = [i for i, n in enumerate(jn) if ("leg_joint" in n or "upper_joint" in n)]
    results = []
    default_jp = robot.data.default_joint_pos.clone()
    for a in angles:
        # reset to spawn, set shoulder target = a
        root = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        jp = default_jp.clone()
        for sid in sh_ids:
            jp[0, sid] = a
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))

        tgt = default_jp.clone()
        for sid in sh_ids:
            tgt[0, sid] = a
        # settle
        for _ in range(150):
            robot.set_joint_position_target(tgt[:, sh_ids], joint_ids=sh_ids)
            robot.set_joint_velocity_target(torch.zeros((1, len(wh_ids)), device=sim.device), joint_ids=wh_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        z = robot.data.body_pos_w[0, base_idx, 2].item()
        sh_act = robot.data.joint_pos[0, sh_ids[0]].item()
        passive = {jn[i]: round(robot.data.joint_pos[0, i].item(), 3) for i in passive_ids}
        results.append((a.item(), sh_act, z))
        print(f"  shoulder_cmd={a.item():+.4f}  actual={sh_act:+.4f}  ->  base_link Z = {z:.4f}  passive={passive}", flush=True)

    zmax = max(results, key=lambda r: r[2])
    print("\n================ RESULT ================")
    print(f"  spawn (shoulder=0) base_link Z would be ~ first row above")
    print(f"  MAX standing base_link Z = {zmax[2]:.4f}  at shoulder cmd={zmax[0]:+.4f}")
    print(f"  -> use base_height target_height ~ {zmax[2]:.3f} (full extension) for straight legs")
    print("========================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
