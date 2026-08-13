# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from scripts.co_rl.core.runners import OffPolicyRunner

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CO-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--experiment_description", type=str, default=None, help="Description of the experiment.")
parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")

# append CO-RL cli arguments
cli_args.add_co_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pickle
import torch
from datetime import datetime

from core.runners import OnPolicyRunner, SRMOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from lab.flamingo.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv, ManagerBasedConstraintRLEnvCfg

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

"""Use CO-RL Wrapper."""
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg, CoRlVecEnvWrapper


# Import extensions to set up environment tasks
import lab.flamingo.tasks  # noqa: F401  TODO: import orbit.<your_extension_name>

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data: object) -> None:
    """Save object to a pickle file; creates parent directories if needed.

    `isaaclab.utils.io.dump_pickle` was removed upstream; keep behavior for training logs.
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def apply_param_overrides(env_cfg, agent_cfg, spec: str | None) -> None:
    """Apply ``dotpath -> value`` overrides (JSON string or file) to the built cfgs.

    Runs after the cfgs are constructed (so it wins over ``__post_init__``). Keys must start
    with ``env.`` (env cfg) or ``agent.`` (agent cfg); intermediate ``params`` dicts are
    indexed as dicts, everything else via attributes. Fails fast on a bad path so a typo
    aborts the trial in the first seconds instead of wasting a full training run.
    """
    import json

    if not spec:
        return
    text = spec
    if os.path.isfile(spec):
        with open(spec) as f:
            text = f.read()
    overrides = json.loads(text)
    for key, value in overrides.items():
        if key.startswith("env."):
            root, sub = env_cfg, key[len("env.") :]
        elif key.startswith("agent."):
            root, sub = agent_cfg, key[len("agent.") :]
        else:
            raise ValueError(f"param override '{key}' must start with 'env.' or 'agent.'")
        parts = sub.split(".")
        obj = root
        for p in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj[p]
            elif isinstance(obj, (list, tuple)) and p.lstrip("-").isdigit():
                obj = obj[int(p)]  # list index, e.g. curriculum.weight_schedule.params.stages.1
            else:
                obj = getattr(obj, p)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        elif isinstance(obj, list) and last.lstrip("-").isdigit():
            obj[int(last)] = value
        else:
            setattr(obj, last, value)
        print(f"[INFO] param override: {key} = {value}")


@hydra_task_config(args_cli.task, "co_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | ManagerBasedConstraintRLEnvCfg, agent_cfg: CoRlPolicyRunnerCfg):
    """Train with CO-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_co_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg.experiment_description = (
        args_cli.experiment_description
        if args_cli.experiment_description is not None
        else agent_cfg.experiment_description
    )
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks

    # online reward balancer is opt-in: only tasks that define it (e.g. stair_jump) carry the
    # `adaptive_reward` curriculum term. Disable it unless --adaptive_reward is passed; warn if the
    # flag is set on a task that has no such term.
    if hasattr(env_cfg, "curriculum") and getattr(env_cfg.curriculum, "adaptive_reward", None) is not None:
        if args_cli.adaptive_reward:
            print("[INFO]: Adaptive reward balancer ENABLED (ROGER-style penalty-gain adaptation).")
        else:
            env_cfg.curriculum.adaptive_reward = None
            print("[INFO]: Adaptive reward balancer DISABLED (pass --adaptive_reward to enable).")
    elif args_cli.adaptive_reward:
        print("[WARN]: --adaptive_reward set but this task defines no 'adaptive_reward' curriculum term; ignoring.")

    # ROGER-faithful threshold/Delta_t balancer: alternative to --adaptive_reward, same reward
    # terms -> mutually exclusive (adaptive_reward wins if both are passed, since it's the
    # longer-validated one; warn so the conflict isn't silent).
    if hasattr(env_cfg, "curriculum") and getattr(env_cfg.curriculum, "roger_threshold", None) is not None:
        if args_cli.roger_threshold and not args_cli.adaptive_reward:
            print("[INFO]: ROGER threshold/Delta_t balancer ENABLED (paper-faithful, tau-based).")
        else:
            if args_cli.roger_threshold and args_cli.adaptive_reward:
                print("[WARN]: both --roger_threshold and --adaptive_reward passed; they would fight "
                      "over the same reward-term weights. Disabling --roger_threshold (adaptive_reward wins).")
            env_cfg.curriculum.roger_threshold = None
            if not args_cli.roger_threshold:
                print("[INFO]: ROGER threshold/Delta_t balancer DISABLED (pass --roger_threshold to enable).")
    elif args_cli.roger_threshold:
        print("[WARN]: --roger_threshold set but this task defines no 'roger_threshold' curriculum term; ignoring.")

    # curriculum-coupled reward scheduling (approach D) is opt-in the same way.
    if hasattr(env_cfg, "curriculum") and getattr(env_cfg.curriculum, "weight_schedule", None) is not None:
        if args_cli.weight_schedule:
            print("[INFO]: Curriculum weight schedule ENABLED (reward knobs follow the terrain level).")
        else:
            env_cfg.curriculum.weight_schedule = None
            print("[INFO]: Curriculum weight schedule DISABLED (pass --weight_schedule to enable).")
    elif args_cli.weight_schedule:
        print("[WARN]: --weight_schedule set but this task defines no 'weight_schedule' curriculum term; ignoring.")

    # forward-progress gate (anti-farming): opt-in. Turn on the hop-reward gate when requested
    # so an in-place vertical bob earns ~0 (only relevant to tasks with the jump hop terms).
    if args_cli.forward_gate is not None and hasattr(env_cfg, "rewards"):
        gated = []
        for term_name in ("jump_lin_vel_z", "jump_feet_off"):
            term = getattr(env_cfg.rewards, term_name, None)
            if term is not None:
                term.params["forward_gate_ref"] = args_cli.forward_gate
                gated.append(term_name)
        if gated:
            print(f"[INFO]: forward-progress gate ENABLED (ref={args_cli.forward_gate} m/s) on {gated}.")
        else:
            print("[WARN]: --forward_gate set but no gateable hop terms (jump_lin_vel_z/jump_feet_off) found; ignoring.")

    # sweep hook: inject this trial's reward weights / hyperparameters (dotpath -> value).
    apply_param_overrides(env_cfg, agent_cfg, args_cli.param_overrides)

    is_off_policy = False if agent_cfg.to_dict()["algorithm"]["class_name"] in ["PPO", "SRMPPO"] else True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    if isinstance(env.unwrapped, ManagerBasedConstraintRLEnv):
        agent_cfg.use_constraint_rl = True

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for co-rl
    env = CoRlVecEnvWrapper(env, agent_cfg)
    # create runner from co-rl
    if is_off_policy:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        if args_cli.algo == "srmppo":
            runner = SRMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif args_cli.algo == "ppo":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif args_cli.warmstart_ckpt:
        # transfer-load from a (possibly cross-task) checkpoint, then train fresh
        print(f"[INFO]: Warm-starting (transfer) from: {args_cli.warmstart_ckpt}")
        runner.load_transfer(args_cli.warmstart_ckpt)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
