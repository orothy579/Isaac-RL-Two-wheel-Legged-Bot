# H200 작업 인계 (Claude → Codex, 2026-10-29)

H200 이전(`scripts/co_rl/h200/`, unit `flamingo-h200-after-current`)에 더해야 할 작업과, 이전 전에 바뀐 설정이다.
이 폴더 전체가 복사되므로 아래 파일과 checkpoint는 H200에도 그대로 있다.

## 1. blind student PPO 이어 학습 (로컬 예약은 취소함)

- 목적: 스캔 없이 50 frame(1 s) 기록만 보는 DAgger student를 blind task에서 PPO로 이어 학습한다.
  증류만으로는 11.3 cm 이상을 거의 못 오른다 (1단 이상 오른 env 2 / 0 / 0 %, 대부분 몸통이 계단에 닿아 종료).
- task: `Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-v1-ppo` (blind, mirror loss, potential retraction)
- 시작점: `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Distill/2026-10-29_09-52-52_gain2999_h50/student_1499.pt`
- 필수 인자: `--num_policy_stacks 49` (policy 입력 50 × 16 + 4 = 804), `--warmstart_ckpt <위 파일> --warmstart_std_scale 1.0`.
  입력 크기가 같아서 `--warmstart_widen`은 필요 없다. critic(216)은 teacher 것을 복사해 둔 상태다.
- 로컬에서 계획한 설정: 4096 env, 3000 it, `--run_name student_ft`. H200에서는 env 수를 늘려도 된다.
- 로컬 smoke test(256 env, 3 it) 통과: mirror loss 연결, 17개 tensor 전이, 재초기화 없음.
- 평가:
  `scripts/co_rl/eval_stair_step.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Play-v1-ppo --num_policy_stacks 49 --checkpoint <model_N.pt> --levels 0 3 6 9`
- 순서: perceptive Gain 학습과 GPU를 나눠 쓰면 둘 다 느려진다 (GB10에서 측정). GPU가 여러 장이면 다른 GPU에서 동시에,
  한 장이면 Gain 학습 뒤에 돌린다.

## 2. 학습 설정 수정 (이미 반영됨)

- `lab/.../rough_env/stair_step/rough_env_stair_step_cfg.py`: `cfg.commands.base_velocity.initial_phase_time = 0.0`
- 이유: 공용 속도 명령(`mdp/velocity_command.py`)은 reset 뒤 0.5 s 초기 구간에 명령을 0으로 만들고, 다음 resample(5–10 s 뒤)까지
  되돌리지 않는다. 계단 보상과 정체 종료는 명령이 있을 때만 켜지므로, 지금까지 모든 stair-step 학습에서 에피소드(20 s)의 25–50 %가
  명령 0이었다. 정책은 그동안 계단 앞까지 가서 기다리는 법을 배웠다.
- 로컬 2단계 학습(`flamingo-two-stage`)은 stage 1이 수정 전, stage 2부터 수정 후 설정으로 돈다. H200 학습은 수정된 설정을 쓴다.
- 다른(이전) task의 명령 설정은 바꾸지 않았다.

## 3. 평가 스크립트 수정 (이미 반영됨)

- `eval_stair_step.py`, `analyze_first_step.py`, `record_stair_step.py`가 `initial_phase_time = 0.0`으로 평가한다.
- 수정 전에는 한 프로세스에서 처음 평가한 레벨(항상 level 0 = 9.3 cm)만 명령 0으로 평가됐다. 그래서 "9.3 cm 정체"는 평가 버그였다
  (`docs/stair_step_zmp.md` §8.1). 9.3 cm도 명령이 있으면 11.3 cm와 같은 수준(256 env 중 156)으로 첫 단을 오른다.
