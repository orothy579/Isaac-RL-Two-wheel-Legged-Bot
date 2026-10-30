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

## 4. H200 큐 반영 (Codex, 2026-10-29)

- `bootstrap.sh`가 설치 후 `scripts/co_rl/h200/dispatch.py`를 실행한다.
- 실제 실행 시 GPU 수를 다시 조회한다. 1장이면 Gain 5000 it → student 3000 it,
  2장 이상이면 Gain과 student를 서로 다른 GPU에서 동시에 실행한다.
  현재 접속 서버에서는 H200 1장(index 0)이 확인되어 순차 실행된다.
- student는 1번의 `student_1499.pt`에서 시작한다. blind task, policy stacks 49,
  critic stacks 2, std scale 1.0, run name `student_ft_<고유번호>`를 사용하며
  `--warmstart_widen`은 넣지 않는다. 로컬에서 actor 입력 804, critic 입력 216,
  유한한 17개 tensor를 CPU로 확인했다.
- 각 작업은 별도로 smoke test와 4096~65536 환경 벤치마크를 수행한다.
  student의 긴 이력 입력 때문에 Gain의 환경 수를 그대로 복사하지 않는다.
- `--device`, `env.sim.device`, `agent.device`를 같은 GPU로 지정하고 renderer의
  multi-GPU 자동 사용을 끈다. GPU별 잠금으로 같은 GPU에서 중복 실행되지 않게 한다.
- Gain 실패로 독립적인 student 예약이 사라지지 않도록, 한 GPU에서는 Gain 프로세스가
  종료된 후 student도 시도한다. 실패 코드는 전체 큐 상태에 남긴다.
- 이전 시 Gain 최종 checkpoint와 student checkpoint의 SHA-256을 모두 확인한다.
- 현재 로컬 2단계 학습 종료 대기는 유지한다. 명령 0 수정 파일은 다시 변경하지 않았으며,
  수정된 소스를 포함한 폴더 전체가 기존 이전 절차로 복사된다.

상태 파일 (원격 `/home/work/postech/flamingo_h200/`):

- `job_queue_status.json`: GPU 배정, 순차/동시 실행, 두 작업의 종료 코드.
- `training_status.json`, `gain_worker.log`: Gain 상태와 worker 로그.
- `student_training_status.json`, `student_worker.log`: student 상태와 worker 로그.
- `student_benchmark_results.json`: student 환경 수별 측정과 선택 결과.

CPU 큐 검증:

```bash
python3 -m unittest discover -s scripts/co_rl/h200 -p 'test_*.py' -v
```

가상 worker의 실제 프로세스로 1 GPU 순서, 2 GPU 동시 실행과 장치 분리,
Gain 실패 뒤 student 실행, 필수 인자와 GPU 없는 경우를 검증했다.
실제 H200 물리 검증과 학습은 현재 로컬 훈련이 모두 끝난 뒤 수행한다.

## 5. 새 세션으로 이전 재개 (2026-10-30)

- 새 접속: `work@proxy2.ainexus.ktcloud.com:10537`. 비밀번호는 파일에 저장하지 않는다.
- 로컬 2단계는 02:04:35 KST에 정상 완료(`model_4999.pt`). 이전 세션 연결이 끊겨
  첫 자동 복사는 시작 전에 실패했으며, 새 접속을 확인한 뒤 이전을 재개했다.
- `after_current.py --completed-checkpoint <최종 checkpoint>`로 이미 완료된 훈련의
  완료 마커와 체크포인트를 재검증하고 복사할 수 있다. 실행 중인 로컬 훈련이 있으면 거부한다.
- `/home/work/postech/flamingo_h200/Isaac-RL-Two-wheel-Legged-Bot`로
  2,837개 파일, 9,892,340,640바이트를 복사했고 **전 파일 SHA-256 검증을 통과**했다.
- 검증 기록: 원격 `flamingo_h200/transfer_manifest.json`, `transfer_verification.json`.
  원본 프로젝트와 체크포인트도 로컬에 보존했다. ARM venv는 옮겨 실행하지 않고
  같은 영구 저장소 안에서 x86_64 환경을 새로 설치한다.
- 새 서버도 H200 1장이다. 설치·물리 검증 후 Gain → student 순서를 유지한다.
- Python 3.11 / Torch 2.7.0+cu128 / Isaac Sim 5.1 / Isaac Lab 2.3.2.post1 설치 완료.
  실제 H200에서 기본 headless 64환경·3 iteration smoke test를 통과했고,
  4096/8192/16384환경의 각 30 iteration 검증도 통과했다.
  처리량은 각각 약 42104/68499/89713 steps/s이다. 현재 큐는 더 큰 환경 수를 측정하며,
  측정 후 Gain 5000 iteration → student 3000 iteration을 자동 실행한다.
  저장된 실행 설정에서 `initial_phase_time: 0.0`과 `cuda:0`도 확인했다.
