# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Distil a perceptive stair-step policy (teacher) into a BLIND student with a long proprioceptive history.

The v01_2_2 robot has no camera, but only policies that see the wheel-area height scan learned to
lift a wheel onto risers above the axle. The student sees what the deployable policy sees - the
``stack_policy`` proprioception and the velocity command - over ``--student_frames`` frames
(1 s at 50 Hz by default), so it can feel a riser through the blocked wheel and remember the last
step. DAgger: the env runs the perceptive task; each iteration rolls out ``--steps`` steps with the
teacher's action in a fraction ``beta`` of the envs and the student's in the rest (``beta`` falls
1 -> 0 over ``--beta_iters``), labels every visited state with the teacher's deterministic action,
and regresses the student on them (+ a left-right mirror loss on the student).

The checkpoint is an ordinary co_rl ``ActorCritic`` for the blind task with
``--num_policy_stacks = student_frames - 1``: the critic (same input in the blind and perceptive
tasks) is copied from the teacher, so PPO can fine-tune it directly::

    ./run_gb10.sh -u scripts/co_rl/distill_stair_step.py --teacher_ckpt <perceptive model.pt> \\
        --num_envs 4096 --iterations 1500 --out_dir logs/co_rl/Flamingo_Light_Rough_Stair_Step_Distill/<run>
    ./run_gb10.sh scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-v1-ppo \\
        --num_policy_stacks 49 --warmstart_ckpt <out_dir>/student_1499.pt --warmstart_std_scale 1.0 ...
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger: perceptive stair-step teacher -> blind student.")
parser.add_argument("--task", default="Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Perceptive-v1-ppo",
                    help="perceptive training task (the teacher's observation and the training terrain)")
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--student_frames", type=int, default=50, help="proprioceptive frames the student sees")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=1500)
parser.add_argument("--steps", type=int, default=24, help="env steps per iteration")
parser.add_argument("--beta_iters", type=int, default=300, help="teacher-executed share falls 1 -> 0 over these")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=5.0e-4)
parser.add_argument("--mirror_coef", type=float, default=1.0)
parser.add_argument("--student_std", type=float, default=0.15, help="action std stored for PPO fine-tuning")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import lab.flamingo.tasks  # noqa: E402, F401
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.rough_env.stair_step.symmetry import (  # noqa: E402
    policy_mirror_maps,
)
from scripts.co_rl.core.modules import ActorCritic  # noqa: E402
from scripts.co_rl.core.runners import OnPolicyRunner  # noqa: E402
from scripts.co_rl.core.utils.state_handler import StateHandler  # noqa: E402
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper  # noqa: E402


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "distill_args.json"), "w"), indent=1)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "co_rl_cfg_entry_point")
    agent_cfg.num_policy_stacks = 2  # the teacher's input
    agent_cfg.num_critic_stacks = 2
    agent_cfg.mirror_fn = None
    env = CoRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), agent_cfg)
    base = env.unwrapped
    dev = base.device

    # teacher: the perceptive policy (actor only is used)
    teacher_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    teacher_runner.load_transfer(args.teacher_ckpt, std_scale=1.0)
    teacher = teacher_runner.alg.actor_critic
    teacher.eval()

    # student input: student_frames x stack_policy, then none_stack_policy without the trailing feet scan
    om = base.observation_manager
    stack_dim = om.group_obs_dim["stack_policy"][0]
    terms = om.active_terms["none_stack_policy"]
    rest_dim = om.group_obs_dim["none_stack_policy"][0]
    if terms[-1] == "feet_scan":
        rest_dim -= om.group_obs_term_dim["none_stack_policy"][-1][0]
    history = StateHandler(args.student_frames, stack_dim, rest_dim)
    num_student_obs = args.student_frames * stack_dim + rest_dim
    num_critic_obs = teacher.critic[0].in_features
    pc = agent_cfg.policy
    student = ActorCritic(num_student_obs, num_critic_obs, base.action_manager.total_action_dim,
                          actor_hidden_dims=pc.actor_hidden_dims, critic_hidden_dims=pc.critic_hidden_dims,
                          activation=pc.activation, init_noise_std=args.student_std).to(dev)
    student.critic.load_state_dict(teacher.critic.state_dict())  # same critic input in the blind task
    opt = torch.optim.Adam(student.actor.parameters(), lr=args.lr)
    mirror_obs, mirror_actions = policy_mirror_maps(base, args.student_frames, feet_scan=False)
    print(f"[distill] student obs {num_student_obs} = {args.student_frames} x {stack_dim} + {rest_dim}; "
          f"critic {num_critic_obs} (teacher's); envs {base.num_envs}", flush=True)

    def student_obs(obs_dict):
        return history.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"][:, :rest_dim])

    writer = SummaryWriter(args.out_dir)
    teacher_obs, extras = env.get_observations()
    s_obs = history.reset(extras["observations"]["stack_policy"], extras["observations"]["none_stack_policy"][:, :rest_dim])
    n, T = base.num_envs, args.steps
    buf_obs = torch.zeros(T, n, num_student_obs, device=dev)
    buf_act = torch.zeros(T, n, base.action_manager.total_action_dim, device=dev)
    ep_stats = {}
    t_start = time.time()
    for it in range(args.iterations):
        beta = max(0.0, 1.0 - it / max(args.beta_iters, 1))
        use_teacher = (torch.rand(n, device=dev) < beta).unsqueeze(1)
        t0 = time.time()
        with torch.no_grad():
            for t in range(T):
                a_t = teacher.act_inference(teacher_obs).clamp(-1.0, 1.0)
                a_s = student.act_inference(s_obs).clamp(-1.0, 1.0)
                buf_obs[t] = s_obs
                buf_act[t] = a_t
                teacher_obs, _, dones, infos = env.step(torch.where(use_teacher, a_t, a_s))
                s_obs = student_obs(infos["observations"])
                for k, v in infos.get("log", {}).items():
                    if k.startswith("Curriculum/") or k.startswith("Episode_Termination/"):
                        ep_stats[k] = float(v)
        collect = time.time() - t0
        # supervised update on this rollout (DAgger labels)
        X, Y = buf_obs.reshape(-1, num_student_obs).clone(), buf_act.reshape(-1, buf_act.shape[-1]).clone()
        mb = X.shape[0] // args.minibatches
        tot_bc, tot_mir, k = 0.0, 0.0, 0
        for _ in range(args.epochs):
            perm = torch.randperm(X.shape[0], device=dev)
            for m in range(args.minibatches):
                idx = perm[m * mb:(m + 1) * mb]
                mu = student.actor(X[idx])
                bc = (mu - Y[idx]).pow(2).mean()
                mir = (student.actor(mirror_obs(X[idx])) - mirror_actions(mu)).pow(2).mean()
                loss = bc + args.mirror_coef * mir
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
                opt.step()
                tot_bc += bc.item(); tot_mir += mir.item(); k += 1
        writer.add_scalar("Distill/bc_loss", tot_bc / k, it)
        writer.add_scalar("Distill/mirror_loss", tot_mir / k, it)
        writer.add_scalar("Distill/beta", beta, it)
        for key, v in ep_stats.items():
            writer.add_scalar(key, v, it)
        if it % 10 == 0 or it == args.iterations - 1:
            lvl = ep_stats.get("Curriculum/terrain_levels", float("nan"))
            ris = ep_stats.get("Curriculum/gait/risers_climbed", float("nan"))
            lift = ep_stats.get("Curriculum/gait/lifted_step_ups", float("nan"))
            print(f"[distill] it {it:5d} beta {beta:.2f} bc {tot_bc / k:.4f} mirror {tot_mir / k:.4f} | level {lvl:.2f} "
                  f"risers {ris:.2f} lifted {lift:.2f} | collect {collect:.1f}s total {time.time() - t_start:.0f}s", flush=True)
        if (it + 1) % args.save_interval == 0 or it == args.iterations - 1:
            with torch.no_grad():
                student.std.fill_(args.student_std)
            path = os.path.join(args.out_dir, f"student_{it}.pt")
            torch.save({"model_state_dict": student.state_dict(), "optimizer_state_dict": opt.state_dict(),
                        "iter": it, "infos": {"teacher": args.teacher_ckpt, "student_frames": args.student_frames,
                                              "num_policy_stacks": args.student_frames - 1}}, path)
    writer.close()
    print(f"[distill] done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # Isaac Sim 5.1 can hang in app.close()
