import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=args_cli.usd, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        actuators={},
    )
    robot = Articulation(cfg.replace(prim_path="/World/Robot"))
    sim.reset()
    jn = robot.data.joint_names
    lim = robot.data.joint_pos_limits[0]
    print("\n================ JOINT INFO ================", flush=True)
    print(f"  USD: {args_cli.usd}", flush=True)
    print(f"  bodies ({len(robot.data.body_names)}): {robot.data.body_names}", flush=True)
    print(f"  total joints: {len(jn)}", flush=True)
    for i, n in enumerate(jn):
        print(f"   [{i}] {n:24s} range=({lim[i,0]:+.3f},{lim[i,1]:+.3f})", flush=True)
    print("============================================", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
