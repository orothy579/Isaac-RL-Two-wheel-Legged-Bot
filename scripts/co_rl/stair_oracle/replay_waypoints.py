#!/usr/bin/env python3
"""Replay human joint waypoints through physics, without an RL policy or training."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.co_rl.stair_oracle.waypoint_profile import JOINTS, load_profile, sample

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", default=str(Path(__file__).with_name("waypoints") / "example.json"))
parser.add_argument("--output-root", default=str(Path(__file__).with_name("waypoint_runs")))
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--validate-only", action="store_true", help="Validate timing and motor command limits without Isaac Sim")
# Validate-only intentionally does not require simulator imports.
known, _ = parser.parse_known_args()
profile = load_profile(known.profile)
if known.validate_only:
    print("Valid profile; actual USD joint limits are additionally checked at replay startup.")
    sys.exit(0)

from scripts.co_rl.stair_oracle.replay_guard import acquire_replay_lock
# Same lock regardless of --output-root; retained throughout simulator shutdown.
_replay_lock = acquire_replay_lock(Path(__file__).with_name("waypoint_runs") / ".replay.lock")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video
launcher = AppLauncher(args)
app = launcher.app

import csv
from datetime import datetime
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.utils.math import euler_xyz_from_quat
from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_CFG
from scripts.co_rl.stair_oracle.waypoint_report import write_report

_output_dir = None


def main():
    global _output_dir
    dt, fps = 0.005, 20
    stride = round(1 / (dt * fps))
    torch.manual_seed(42)
    np.random.seed(42)
    s = profile["scene"]
    out = Path(args.output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    _output_dir = out
    (out / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"[WAYPOINT] Output: {out}", flush=True)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=s["friction"], dynamic_friction=s["friction"], restitution=0.)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics_material=material))
    plane = sim_utils.GroundPlaneCfg(physics_material=material)
    plane.func("/World/Ground", plane)
    # One finite tread: x=0 is the front riser, x=0.30 is the back edge.
    step = sim_utils.CuboidCfg(size=(s["tread_m"], s["width_m"], s["height_m"]),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              physics_material=material,
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.45, 0.65)))
    step.func("/World/Step", step, translation=(s["tread_m"]/2, 0, s["height_m"]/2))
    light = sim_utils.DomeLightCfg(intensity=2500.)
    light.func("/World/Light", light)
    cfg = FLAMINGO_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.init_state.pos = (-s["distance_m"], 0., s["initial_base_z_m"])
    cfg.init_state.lin_vel = (s["initial_vx_mps"], 0., 0.)
    cfg.init_state.joint_pos = {JOINTS[i]: profile["waypoints"][0]["shoulder_rad"][i] for i in range(2)}
    cfg.init_state.joint_pos.update({name: 0. for name in JOINTS[2:]})
    # Reproducible replay, same existing PD gains, no random delay on this first bench.
    # Explicitly enforce the verified speed/torque limits in PhysX as well as the actuator.
    for actuator in cfg.actuators.values():
        actuator.min_delay = actuator.max_delay = 0
        actuator.effort_limit = actuator.effort_limit_sim = 17.
        actuator.velocity_limit = actuator.velocity_limit_sim = 40.
    robot = Articulation(cfg)
    contacts = [ContactSensor(ContactSensorCfg(
        prim_path=f"/World/Robot/{side}_wheel_link", update_period=0.,
        filter_prim_paths_expr=["/World/Step"], track_air_time=True)) for side in ("left", "right")]
    camera = None
    if not args.no_video:
        camera = Camera(CameraCfg(prim_path="/World/Camera", height=480, width=640, data_types=["rgb"],
                                 spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=25., clipping_range=(0.02, 100.))))
    sim.set_camera_view((0.9, -2.1, 1.1), (-0.2, 0., 0.25))
    print("[WAYPOINT] Initializing physics and sensors...", flush=True)
    sim.reset()
    print("[WAYPOINT] Physics ready.", flush=True)
    ids, names = robot.find_joints(list(JOINTS), preserve_order=True)
    if tuple(names) != JOINTS:
        raise RuntimeError(f"Unexpected motor ordering: {names}")
    wheel_ids, _ = robot.find_bodies(["left_wheel_link", "right_wheel_link"], preserve_order=True)
    limits = robot.data.joint_pos_limits[0, ids].cpu().numpy()
    print(f"[WAYPOINT] USD shoulder limits (rad): {limits[:2].tolist()}", flush=True)
    for row in profile["waypoints"]:
        for i, q in enumerate(row["shoulder_rad"]):
            if not limits[i, 0] <= q <= limits[i, 1]:
                raise ValueError(f"{names[i]} target {q} at t={row['t']} outside USD limits {limits[i].tolist()}")
    if camera:
        camera.set_world_poses_from_view(torch.tensor([[0.9, -2.1, 1.1]], device=sim.device),
                                        torch.tensor([[-0.2, 0., 0.25]], device=sim.device))
        for _ in range(4):
            sim.render()
    robot.reset()
    # sim.reset()/camera warm-up can advance physics. Establish the requested
    # initial condition once, AFTER warm-up, before logging or motor playback.
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    for c in contacts:
        c.reset()
    rows, states, controls, next_states = [], [], [], []
    writer = imageio.get_writer(str(out / "replay.mp4"), fps=fps, codec="libx264", quality=8) if camera else None
    supported_before_flight = False
    air_duration = 0.
    airborne = False
    hold = 0.
    stop_reason = "duration_elapsed"
    try:
        for k in range(round(profile["duration_s"] / dt)):
            if not app.is_running():
                stop_reason = "app_closed"
                break
            t = k*dt
            target = sample(profile, t)
            qtarget = torch.tensor([target["shoulder_rad"]], device=sim.device)
            vtarget = torch.tensor([target["shoulder_rad_s"] + target["wheel_rad_s"]], device=sim.device)
            fftarget = torch.tensor([target["feedforward_nm"]], device=sim.device)
            # Dataset aligns pre-step state, requested command, and post-step state.
            pre = torch.cat((robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel), dim=1)[0].cpu().numpy().copy()
            robot.set_joint_position_target(qtarget, joint_ids=ids[:2])
            robot.set_joint_velocity_target(vtarget, joint_ids=ids)
            robot.set_joint_effort_target(fftarget, joint_ids=ids)
            robot.write_data_to_sim()
            requested = robot.data.computed_torque[0, ids].cpu().numpy().copy()
            applied = robot.data.applied_torque[0, ids].cpu().numpy().copy()
            sim.step(render=False)
            robot.update(dt)
            for c in contacts:
                c.update(dt)
            post = torch.cat((robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel), dim=1)[0].cpu().numpy().copy()
            if not np.isfinite(post).all():
                stop_reason = "nonfinite_physics"
                break
            q = robot.data.joint_pos[0, ids].cpu().numpy()
            v = robot.data.joint_vel[0, ids].cpu().numpy()
            base = robot.data.root_pos_w[0].cpu().numpy()
            roll, pitch, yaw = [float(z[0]) for z in euler_xyz_from_quat(robot.data.root_quat_w)]
            wheel_pos = robot.data.body_pos_w[0, wheel_ids].cpu().numpy()
            total_contact = [float(c.data.net_forces_w.norm(dim=-1).sum()) for c in contacts]
            step_contact = [float(c.data.force_matrix_w.norm(dim=-1).sum()) for c in contacts]
            on_ground = all(f > 1. for f in total_contact)
            if on_ground:
                supported_before_flight = True
            air_duration = air_duration + dt if supported_before_flight and max(total_contact) < 1. else 0.
            airborne = airborne or air_duration >= 0.03
            # Require both wheels on the finite top surface, not side contact or historical height gain.
            on_top = (all(f > 1. for f in step_contact)
                      and bool(np.all((wheel_pos[:, 0] > 0.) & (wheel_pos[:, 0] < s["tread_m"])))
                      and bool(np.all(np.abs(wheel_pos[:, 1]) < s["width_m"]/2))
                      and bool(np.all(wheel_pos[:, 2] > s["height_m"]))
                      and abs(roll) < 0.25 and abs(pitch) < 0.35
                      and float(robot.data.root_lin_vel_w.norm()) < 0.25
                      and float(robot.data.root_ang_vel_w.norm()) < 0.5)
            hold = hold + dt if on_top else 0.
            next_target = sample(profile, t+dt)
            row = {"t": t+dt, "command_t": t, "phase": target["phase"],
                   "base_x": float(base[0]), "base_z": float(base[2]), "pitch": pitch,
                   "base_x_ref": next_target["base_reference"][0], "base_z_ref": next_target["base_reference"][1],
                   "pitch_ref": next_target["base_reference"][2],
                   "distance_to_riser_m": -float(base[0]), "yaw_error_deg": math.degrees(yaw),
                   "airborne_observed": int(airborne), "top_hold_s": hold}
            for i, name in enumerate(names):
                row.update({f"{name}_q": float(q[i]), f"{name}_v": float(v[i]),
                            f"{name}_v_target": float(vtarget[0, i]),
                            f"{name}_torque_requested": float(requested[i]), f"{name}_torque_applied": float(applied[i]),
                            f"{name}_saturated": int(abs(requested[i]) > 17.+1e-5),
                            f"{name}_speed_violation": int(abs(v[i]) > 40.+1e-3),
                            f"{name}_position_violation": int(q[i] < limits[i, 0]-1e-3 or q[i] > limits[i, 1]+1e-3)})
                if i < 2:
                    row[f"{name}_q_target"] = target["shoulder_rad"][i]
                    row[f"{name}_q_error"] = float(target["shoulder_rad"][i]-q[i])
            for i, side in enumerate(("left", "right")):
                row[f"{side}_contact_n"] = total_contact[i]
                row[f"{side}_step_contact_n"] = step_contact[i]
            rows.append(row)
            states.append(pre)
            next_states.append(post)
            controls.append(target["shoulder_rad"] + target["shoulder_rad_s"] + target["wheel_rad_s"] + target["feedforward_nm"])
            if camera and (k+1) % stride == 0:
                sim.render()
                camera.update(dt*stride)
                frame = Image.fromarray(camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 83), fill=(20, 24, 30))
                draw.text((8, 4), f"t={t+dt:.2f}s | {target['phase']} | d={-base[0]:.3f}m | yaw={math.degrees(yaw):.1f}deg", fill="white")
                draw.text((8, 22), "Torque L/R shoulder,wheel [Nm]: " + " ".join(f"{v:.1f}" for v in applied), fill="white")
                draw.text((8, 40), f"Shoulder target/actual: {target['shoulder_rad'][0]:.2f}/{q[0]:.2f}, {target['shoulder_rad'][1]:.2f}/{q[1]:.2f} rad", fill="white")
                draw.text((8, 58), f"Air phase observed={airborne} | current stable top hold={hold:.2f}s", fill="white")
                writer.append_data(np.asarray(frame))
            if (k+1) % 200 == 0:
                print(f"[WAYPOINT] t={t+dt:.1f}s base_x={base[0]:.3f} z={base[2]:.3f} hold={hold:.2f}", flush=True)
    finally:
        if writer:
            writer.close()
        if rows:
            with (out / "trace.csv").open("w", newline="") as f:
                csv_writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                csv_writer.writeheader()
                csv_writer.writerows(rows)
            np.savez_compressed(out / "transitions.npz", state=np.asarray(states), command=np.asarray(controls),
                                next_state=np.asarray(next_states), time=np.asarray([r["command_t"] for r in rows]))
    if not rows:
        raise RuntimeError("No physics samples collected")
    summary = {
        "source": "human_waypoints_no_policy", "profile_validated_by_user": False,
        "stop_reason": stop_reason, "samples": len(rows), "physics_dt_s": dt,
        "stable_top_at_end": hold >= 0.5, "airborne_observed": airborne,
        "jump_success": airborne and hold >= 0.5 and stop_reason == "duration_elapsed",
        "bc_expert_approved": False,
        "success_definition": "prior wheel support, >=30 ms flight, and both wheels contacting within finite tread with small body velocity/tilt continuously for final >=0.5 s",
        "hardware_limits": {"torque_nm": 17., "speed_rad_s": 40., "provenance": "user confirmed torque/current limits validated"},
        "controller": "existing shoulder position PD and wheel velocity PD plus waypoint feedforward; no base forces or state overrides during replay",
        "replay_delay_steps": 0, "randomization": False,
        "torque_note": "requested torque is this PD controller's demand, not a unique inverse-dynamics minimum; applied is explicit actuator output to PhysX",
        "current_note": "current not inferred: no motor torque constant/gearing/efficiency conversion supplied",
        "state_layout": ["root_state_w(13: xyz,wxyz,world_linear_velocity,world_angular_velocity)", "all_joint_positions", "all_joint_velocities"],
        "all_joint_names": robot.joint_names,
        "command_layout": ["left_shoulder_q", "right_shoulder_q", "left_shoulder_v", "right_shoulder_v", "left_wheel_v", "right_wheel_v", "left_shoulder_ff", "right_shoulder_ff", "left_wheel_ff", "right_wheel_ff"],
        "usd": cfg.spawn.usd_path,
        "total_mass_kg": float(robot.root_physx_view.get_masses().sum()),
        "robot_material_properties": robot.root_physx_view.get_material_properties().tolist(),
        "pd_gains": {key: {"stiffness": value.stiffness.cpu().tolist(), "damping": value.damping.cpu().tolist()}
                     for key, value in robot.actuators.items()},
        "joint_limits_rad": {name: [float(x) if math.isfinite(x) else None for x in limits[i]] for i, name in enumerate(names)},
        "motors": {}
    }
    for name in names:
        summary["motors"][name] = {
            "peak_requested_nm": max(abs(r[f"{name}_torque_requested"]) for r in rows),
            "peak_applied_nm": max(abs(r[f"{name}_torque_applied"]) for r in rows),
            "saturated_time_s": sum(r[f"{name}_saturated"] for r in rows)*dt,
            "peak_speed_rad_s": max(abs(r[f"{name}_v"]) for r in rows),
            "position_violation_samples": sum(r[f"{name}_position_violation"] for r in rows),
            "speed_violation_samples": sum(r[f"{name}_speed_violation"] for r in rows),
        }
    summary["limits_passed"] = all(m["speed_violation_samples"] == 0 and m["position_violation_samples"] == 0 and m["peak_applied_nm"] <= 17.001 for m in summary["motors"].values())
    summary["jump_success"] = summary["jump_success"] and summary["limits_passed"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_report(out, rows, summary, profile)
    print(f"[WAYPOINT] Completed. jump_success={summary['jump_success']}; BC approval=False; report={out/'report.html'}", flush=True)


if __name__ == "__main__":
    exit_code = 1
    try:
        main()
        exit_code = 0
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Video/report/trace writers are flushed synchronously in main().  Native
        # Kit shutdown can retain the GIL while hanging, which also prevents a
        # Python-thread watchdog from firing.  This is a dedicated CLI process,
        # so clear Python-side callbacks and let the OS reclaim Kit/GPU resources
        # instead of entering SimulationApp.close().
        context = sim_utils.SimulationContext.instance()
        if context is not None:
            context.clear_all_callbacks()
            context.clear_instance()
        if _output_dir is not None:
            (_output_dir / "shutdown.json").write_text(json.dumps({
                "direct_process_exit": True,
                "exit_code": exit_code,
                "note": "All result writers were closed; OS reclaims Kit and GPU resources"
            }, indent=2))
        print(f"[WAYPOINT] Exit {exit_code}; direct_process_exit=True", flush=True)
        sys.stderr.flush()
        os._exit(exit_code)
