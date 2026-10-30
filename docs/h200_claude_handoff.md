# Flamingo 한 바퀴씩 계단 오르기 — H200 서버 Claude 인계 (2026-10-30)

로컬 GB10(DGX Spark)에서 Claude가 2026-10-23부터 해 온 작업을 H200 서버의 Claude가 이어받기 위한 문서다.
처음 읽는 사람을 기준으로 썼다. 자세한 실험 기록은 `docs/stair_step_zmp.md`, H200 이전·큐는 `docs/h200_handoff.md`,
애니메이션(motion prior) 조사는 `docs/h200_motion_priors.md`에 있다.

## 0. 작업 규칙

- 사용자와는 **한국어**로 소통한다. 최종 보고뿐 아니라 도구 호출 사이의 짧은 진행 메모도 한국어로 쓴다.
- 커밋과 push는 사용자가 요청할 때만 한다. force push는 매번 확인을 받는다.
- 비밀번호를 파일, 명령, 로그에 남기지 않는다. 사용자가 대화에 붙여 넣은 비밀번호도 다시 적지 않는다.
- H200 큐(`scripts/co_rl/h200/`)와 그 상태 파일은 **Codex**가 만들고 관리한다. 바꾸기 전에 사용자에게 묻는다.
- 서버의 다른 폴더(`/home/work/postech`의 `dataset`, `lerobot`, `training`, `outputs` 등)는 건드리지 않는다.
- **H200 복사본에는 `.git`이 없다** (tar로 복사할 때 뺐다). 코드 변경은 로컬 저장소
  (`/home/midbar/ws/Isaac-RL-Two-wheel-Legged-Bot`, branch `alternating_feet`)로 되돌려야 커밋할 수 있다.
  무엇을 바꿨는지 기록해 두고 사용자와 동기화 방법을 정한다.
- GPU 하나를 여러 학습이 나눠 쓰면 각 학습이 느려진다 (GPU가 프로세스를 시분할한다). 병렬보다 순차 실행이 낫다.

## 1. 목표

- 로봇: flamingo_light **v01_2_2**. 카메라가 없는 2륜 다리 로봇이다. 능동 관절은 어깨 2개와 바퀴 2개다.
  다리는 4절 링크라 어깨를 돌리면 바퀴가 거의 수직으로 오르내린다. 바퀴 반지름은 7.6 cm다.
- 과제: 계단을 **한 바퀴씩 번갈아** 오른다 (점프 금지). 보상은 ZMP 기반이다 (`docs/zmp.pdf`,
  Kajita 2003 PDF). 기존 task(stair_jump 등)는 건드리지 않는다.
- 사용자가 정한 것:
  - 어느 바퀴부터 올라도 상관없다. 첫 단을 오르는 것이 중요하다.
  - 실제 로봇에는 카메라가 없으므로 최종 목표는 스캔 없이 오르는 정책(blind)이다.
  - 센서를 추가하는 방안은 하지 않기로 했다.
  - 다음 방향으로 애니메이션 궤적을 정답(GT)으로 쓰는 접근을 생각하고 있다 (두 논문, §8).

## 2. H200 환경

- 프로젝트: `/home/work/postech/flamingo_light/Isaac-RL-Two-wheel-Legged-Bot`. venv: `/home/work/postech/flamingo_light/venv`.
- Python 3.11, Torch 2.7.0+cu128, Isaac Sim 5.1, Isaac Lab 2.3.2.post1. H200 1장(메모리 약 140 GiB)이다.
- 실행은 `bash scripts/co_rl/h200/run.sh <script.py> <인자>`로 한다. 로컬의 `./run_gb10.sh`에 해당하며 PYTHONPATH와 캐시 경로를 맞춘다.
- Codex 측정 처리량: 4096 / 8192 / 16384 env에서 약 42k / 68k / 90k steps/s. 큐가 32768, 65536 env를 이어서 측정한다.
- H200에는 RT core가 없다. **영상 렌더링은 로컬 GB10에서** 한다. H200에서 만든 checkpoint를 로컬로 가져와 찍는다.
- Isaac Sim 5.1 주의:
  - `train.py`에는 `--batch_exit`를 준다. 직접 만든 스크립트는 끝에 `os._exit(0)`를 부른다 (`app.close()`가 멈춘다).
  - "No space left on device" carb 경고는 inotify 한도 경고이며 무시해도 된다.

## 3. H200 큐 (Codex, 지금 진행 중일 수 있음)

`bootstrap.sh` → `dispatch.py`가 GPU 수를 보고 실행한다. GPU가 1장이라 아래 두 작업을 순서대로 돈다:

1. **Gain 5000 it**: task `Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-v1-ppo`.
   로컬 2단계 학습의 최종 checkpoint
   `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain/ppo/2026-10-29_21-53-24_two_stage_s2/model_4999.pt`에서
   시작하고, env 수는 벤치마크로 고른다 (최대 65536).
2. **student 3000 it**: blind task `...-Lift-Mirror-Gain-v1-ppo`, `--num_policy_stacks 49`.
   시작점은 `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Distill/2026-10-29_09-52-52_gain2999_h50/student_1499.pt`다.

상태 파일은 `/home/work/postech/flamingo_light/`에 있다:
- `job_queue_status.json`: 전체 큐
- `training_status.json`, `gain_worker.log`: Gain
- `student_training_status.json`, `student_worker.log`: student
- `benchmark_results.json`, `student_benchmark_results.json`: env 수 측정

자세한 동작은 `docs/h200_handoff.md` §4–5.

## 4. 코드 지도

- task 정의: `lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_light_env/rough_env/stair_step/`
  - `rough_env_stair_step_cfg.py`: 모든 stair-step 변형의 환경 설정
  - `zmp_rewards.py`: `ZmpGaitState`(접촉, 막힘, step-up 판정, ZMP/CoP)와 보상·종료 항목
  - `symmetry.py`: 좌우 미러(관측·행동 교환, mirror loss용)
  - `stair_step_terrain_cfg.py`: 역피라미드 계단. 행이 높아질수록 단이 높아지고 디딤판이 좁아진다
- 등록: `.../flamingo_light_env/__init__.py`. runner 설정: `.../flamingo_light_env/agents/co_rl_cfg.py`.
- PPO: `scripts/co_rl/core/algorithms/ppo.py` (`mirror_loss_coef`). runner: `scripts/co_rl/core/runners/on_policy_runner.py`.

| task (`Isaac-Velocity-Rough-Flamingo-Light-...-v1-ppo`) | 내용 |
|---|---|
| `Stair-Step[-Perceptive]` | v1/v2: 느슨한 step-up 판정(굴러 넘기도 인정), 단 5–15 cm |
| `Stair-Step-Mirror[-Perceptive]` | v2 + mirror loss (2단계 학습의 stage 1) |
| `Stair-Step-Lift[-Perceptive]` | v3: 엄격 판정(다리로 들어 올려야 인정), 단 9–15 cm |
| `Stair-Step-Lift-Mirror[-Perceptive]` | v3 + mirror loss |
| `Stair-Step-Lift-Mirror-Gain[-Perceptive]` | + retraction 보상을 potential 형태로. **현재 주 task** |
| `Stair-Step-Lift-Mirror-Gain-LowStart[-Perceptive]` | 엄격 판정을 5 cm부터. 평지 정책에서 시작해 실패했다 |

모든 task에 평가용 `-Play` 변형이 있다 (예: `...-Lift-Mirror-Gain-Perceptive-Play-v1-ppo`).
warm start 원본(평지 주행)은 `Isaac-Velocity-Flat-Flamingo-Light-v01-2-2-v1-ppo`의
`logs/co_rl/Flamingo_Light_v01_2_2_Flat_Stand_Drive/ppo/2026-10-23_11-56-03_v0122_stand_drive/model_1499.pt`다.

**관측과 행동**
- policy 입력은 `stack_policy` 16개 × (num_policy_stacks + 1) frame에 `none_stack_policy`를 붙인 것이다.
  - 한 frame: 어깨 위치 2, 관절 속도 4, 각속도 3, 중력 3, 이전 행동 4.
  - `none_stack_policy`: 속도 명령 4, perceptive 변형이면 발밑 스캔 72(9 × 8 ray, 5 cm 간격, 로봇 앞 25 cm까지).
  - 기본은 stack 2 (3 frame)이고, blind student는 49 (50 frame = 1 s, 입력 804)다.
- 행동: [왼 어깨, 오른 어깨, 왼 바퀴, 오른 바퀴]. 전진할 때 두 바퀴 행동이 모두 양수라 미러는 부호 없이 교환만 한다.
- critic 입력은 216이며 특권 정보(height map)를 포함한다. student의 critic은 teacher 것을 복사했다.

**스크립트** (`scripts/co_rl/`)
- `train.py`: `--warmstart_ckpt`, `--warmstart_widen`(입력이 늘어난 첫 층을 0으로 채워 이식), `--warmstart_std_scale`,
  `--param_overrides <json>`, `--num_policy_stacks`, `--resume True --load_run <dir> --checkpoint model_N.pt`.
  `learn(n)`은 불러온 iteration에서 n번 더 돈다.
- `eval_stair_step.py`: 레벨별(0, 3, 6, 9 = 9.3, 11.3, 13.3, 15 cm) 1단 이상 오른 env 비율, 오른 단 수, 좌우 들기 횟수, 교대 비율.
- `analyze_first_step.py`: 첫 단에서 어느 바퀴가 먼저 막히고 먼저 오르는지, 막힌 뒤 3 s의 동작, 종료 원인, 명령 기록.
- `record_stair_step.py`: 한 로봇을 따라가는 영상. `--pick_best`(가장 많이 오른 env), `--pick_fail`(첫 단 실패 env),
  `--yaw_spread`, `--solo`, `--trace`(시계열 npz/png). GPU 물리는 매번 같지 않아서 `--retries`로 다시 찍는다.
- `distill_stair_step.py`: perceptive teacher → blind student DAgger 증류.
- `sweeps/stair_step_two_stage_mirror.sh`: 2단계 대칭 학습 (§6).

## 5. 꼭 알아야 할 함정

1. **명령 0 버그** (2026-10-29에 찾음).
   - 공용 속도 명령(`.../velocity/mdp/velocity_command.py`의 `UniformVelocityWithZCommand`)은 reset 뒤 0.5 s 초기 구간에
     전진 명령을 0으로 만들고, 다음 resample(5–10 s 뒤)까지 되돌리지 않는다.
   - 계단 보상과 정체 종료는 명령이 있을 때만 켜진다. 그래서 그동안 학습한 에피소드마다 5–10 s를 계단 앞에서 기다렸다.
   - 평가 스크립트는 `_reset_idx`를 직접 불러 이 시계를 다시 시작하지 않았다. 그래서 프로세스에서 처음 평가한 레벨만 명령 0이었다.
     "9.3 cm 정체"는 이것이었다.
   - 수정: stair-step 설정과 평가 스크립트 3개에 `initial_phase_time = 0.0`. **다른 task에는 이 버그가 남아 있다.**
   - H200 학습은 수정된 설정으로 돈다 (Codex가 저장된 설정에서 `initial_phase_time: 0.0`을 확인했다).
2. **평가 편차**. 환경 설정에 seed가 없어서 프로세스마다 domain randomization 값이 달라진다.
   64 env는 표준오차가 약 6 %p이고, 프로세스 사이 차이가 더 클 때도 있다 (mirror 1500의 11.3 cm가 72 % → 53 %).
   모델 비교는 256 env 이상으로 한다.
3. `randomize_com_positions`(startup 이벤트)는 env마다 base_link CoM을 축별 ±5 cm 옮긴다.
   로봇 모델을 조사할 때는 끄고 본다. 로봇 자체는 사실상 좌우 대칭이다 (전체 CoM y −0.3 mm, 차이는 leg_link 13 g뿐).
4. co_rl `StateHandler`는 env별 reset 때 쌓인 frame을 비우지 않는다. 평가 스크립트는 레벨마다 `stack_buffer = None`으로 비운다.
5. 정면으로 나란히 막히면 한 바퀴를 디딤판에 올리기 어렵다. 약 17° 이상 비스듬히 접근해야 한다 (`docs/stair_step_zmp.md` §2).
   학습된 정책은 막힌 뒤 ~50° 돌아 한 바퀴를 내밀고 든다.

## 6. 지금까지의 결과

**수정된 평가** (모든 레벨에서 처음부터 명령 0.35 m/s, 64 env, 1단 이상 오른 env 비율 / 평균 오른 단 수):

| checkpoint | 9.3 cm | 11.3 cm | 13.3 cm | 15 cm |
|---|---|---|---|---|
| v3 trial 3 model_4999 | 28 % / 2.23 | 23 % / 1.57 | 2 % / 0.09 | 0 % |
| mirror model_1500 | 61 % / 4.67 | 53 % / 4.22 | 28 % / 1.58 | 3 % / 0.08 |
| Gain model_2999 | 64 % / 5.06 | 58 % / 4.82 | 42 % / 2.76 | 5 % / 0.26 |
| student_1499 (blind) | 53 % / 2.91 | 5 % / 0.10 | 0 % | 0 % |

**2단계 대칭 학습의 stage 2가 가장 잘한다** (2026-10-30 평가, **256 env**, 같은 조건):

| checkpoint | 9.3 cm | 11.3 cm | 13.3 cm | 15 cm |
|---|---|---|---|---|
| Gain model_2999 | 63.7 % / 5.09 | 59.8 % / 4.95 | 47.3 % / 3.75 | 3.5 % / 0.13 |
| two-stage stage 2 model_1000 | 80.9 % / 5.03 | 72.7 % / 4.58 | 82.8 % / 4.50 | 48.4 % / 1.66 |
| two-stage stage 2 model_2000 | 79.3 % / 5.03 | 85.9 % / 5.80 | 80.1 % / 5.09 | 54.7 % / 2.83 |
| two-stage stage 2 model_3000 | 89.8 % / 5.93 | 88.7 % / 6.11 | 84.4 % / 5.85 | 71.1 % / 4.09 |
| two-stage stage 2 model_4000 | 90.6 % / 5.80 | 90.6 % / 6.22 | 91.4 % / 6.18 | 79.7 % / 4.27 |
| **two-stage stage 2 model_4999** | **87.9 % / 6.09** | **94.1 % / 6.96** | **93.4 % / 6.68** | **83.6 % / 5.04** |

- 최고 모델: `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain/ppo/2026-10-29_21-53-24_two_stage_s2/model_4999.pt`.
  H200 큐의 Gain 학습도 이 checkpoint에서 시작한다.
- model_4999는 좌우를 고르게 들고 (15 cm에서 4.87 / 4.72), 교대 비율은 0.94–0.96, 들어 올린 step-up 비율은 0.90–0.94다.
- 15 cm에서 Gain 2999(3.5 %)와 차이가 가장 크다.
- 원인은 둘 중 무엇인지 아직 나누지 못했다.
  1. stage 2부터 명령 버그를 고쳤다 (에피소드 전체에서 오르기 보상).
  2. 계보가 다르다: 평지 주행에서 mirror loss를 켠 채 느슨한 판정 → 엄격 판정.
- 이전 계보: 평지 주행 → v2 trial 2(느슨한 판정) → v3 trial 3(엄격 판정) → mirror(좌우 대칭 loss) → Gain(potential retraction).
  checkpoint:
  - v3 trial 3: `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift/ppo/2026-10-27_17-55-26_sweep003/model_4999.pt`
  - mirror: `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror/ppo/2026-10-28_11-41-33_t3_mirror/model_1500.pt`
  - Gain: `logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain/ppo/2026-10-28_20-17-29_mirror1500_gain/model_2999.pt`
- 2단계 대칭 학습(`two_stage`, 2026-10-29 18:57 – 10-30 02:04):
  - stage 1: v2 판정 + mirror loss, 평지 주행 model_1499에서 3000 it. 마지막 200 it 평균 1.42단/에피소드.
  - stage 2: 엄격 판정 + Gain, 5000 it. 학습 중 평균 terrain level 5.4–5.6, 들어 올린 step-up 비율 0.91.
  - stage 1은 명령 버그가 있는 설정이었고, stage 2부터 수정된 설정이다.
  - 엄격 판정만으로 평지 정책에서 시작한 학습(LowStart, 10000 it)은 한 번도 들어 올리지 못했다.
    느슨한 판정으로 먼저 오르기를 익히는 단계가 필요하다.
- 남은 문제 (최고 모델 model_4999 기준):
  1. **첫 단은 늘 오른 바퀴로 시작한다.** `analyze_first_step.py` 256 env에서 레벨 0/3/6/9 모두 오른 바퀴 238–254, 왼 바퀴 0이다.
     다가가면서 오른쪽으로 조금 돌아(막힐 때 yaw +0.13–0.21 rad) 오른 바퀴가 먼저 막히게 한다 (190–239 / 256).
     왼 바퀴가 먼저 막혀도(6–41 env) 오른 바퀴를 먼저 올린다.
     처음부터 mirror loss를 켠 학습인데도 그렇다 (mirror loss 0.003). 로봇 모델 탓은 아니다 (§5-3).
     먼저 볼 곳: `symmetry.py`의 미러 규칙이 실제 관측 순서·부호와 맞는지, `zmp_rewards.py`에서 동률일 때 한쪽을 고르는 코드
     (예: 두 바퀴가 같은 step에 step-up이면 RIGHT로 기록)가 보상을 한쪽으로 기울이는지.
  2. **가장 흔한 실패는 돌진 충돌이다.** 첫 라이저에 ~1.0 m/s(명령 0.35 m/s의 3배)로 다가가, 1–2 s 만에 몸통이 계단에 닿아 끝난다.
     촬영용 64 env(spawn yaw ±0.6)에서 첫 단 실패 중 base 접촉 비율: 9.3 cm 13/16, 13.3 cm 8/11, 15 cm 15/23.
     11.3 cm 실패는 정체 종료가 가장 많았다 (5/8). 찍은 env는 한 바퀴만 올린 채 멈췄다.
  3. **blind student**는 11.3 cm 이상을 못 오른다. 스캔이 없으면 닿기 전에는 단 높이를 알 수 없다
     (바퀴 반지름보다 높은 단은 높이와 상관없이 바퀴 축 높이에서 라이저 면에 닿는다).
     teacher는 높이에 따라 다리를 다르게 든다 (Gain 2999, 9.3 / 11.3 cm에서 12.8 / 15.5 cm). DAgger는 그 평균을 배운다.
     H200 큐가 PPO로 이어 학습한다.
     teacher를 최고 모델(model_4999)로 바꿔 다시 증류하는 것도 후보다.

**최고 모델 영상** (로컬 GB10, `logs/co_rl/_videos/two_stage_s2_4999/`, 1280×720, 50 fps, spawn yaw ±0.6 중에서 고름):

| 단 | 성공 | 실패 |
|---|---|---|
| 9.3 cm | `L0_9.3cm_success.mp4` (8.1단, 15 s) | `L0_9.3cm_fail.mp4` (1.3 s, base 접촉; 4배 느린 `_slow4x.mp4`) |
| 11.3 cm | `L3_11.3cm_success.mp4` (9.2단, 11.3 s) | `L3_11.3cm_fail.mp4` (한 바퀴만 올리고 8.9 s 정체 종료) |
| 13.3 cm | `L6_13.3cm_success.mp4` (10.1단, 15 s) | `L6_13.3cm_fail.mp4` (1.8 s, base 접촉; `_slow4x.mp4`) |
| 15 cm | `L9_15.0cm_success.mp4` (6.9단, 10 s) | `L9_15.0cm_fail.mp4` (1.1 s, base 접촉; `_slow4x.mp4`) |

각 영상 옆의 `.npz`/`.png`는 그 로봇의 시계열이다 (바퀴 높이, 어깨 각, 접촉, ZMP, 속도).
실패 영상은 `record_stair_step.py --pick_fail`로 찍었다. 첫 단을 못 오른 env 중 가장 흔한 종료 원인의 env를 고른다.

## 7. 다음에 할 일

1. H200 큐 결과(Gain, student)를 고친 평가 스크립트로 256 env, 레벨 0 3 6 9에서 평가한다. 기준은 §6의 model_4999다.
   - perceptive: `eval_stair_step.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-Play-v1-ppo --checkpoint <model> --levels 0 3 6 9 --num_envs 256 --seconds 15 --out <json>`
   - blind student: 같은 명령에 `--task ...-Lift-Mirror-Gain-Play-v1-ppo --num_policy_stacks 49`
   - 첫 바퀴 편향: `analyze_first_step.py --task <Play task> --checkpoint <model> --levels 0 3 6 9 --num_envs 256 --out <json>`
2. 영상이 필요하면 checkpoint를 로컬 GB10으로 보내 `record_stair_step.py`로 찍는다 (H200은 렌더링 불가).
3. §6의 남은 문제 1–3을 다룬다.
   - 이미 효과가 없었던 것: mirror loss만으로 첫 바퀴 편향 고치기, 엄격 판정만으로 처음부터 학습.
   - 확률적 정책 평가는 명령 버그 아래에서 한 실험이라 다시 볼 여지가 있다.
4. 명령 버그 수정과 계보 중 무엇이 성능을 올렸는지 나누는 실험 (예: Gain 2999에서 수정된 설정으로 이어 학습).
5. 애니메이션/motion prior 방향 (사용자 관심, §8). 시작 전에 사용자와 방향을 정한다.

## 8. 애니메이션 궤적을 정답으로 쓰는 방향 (사용자 아이디어)

- 사용자가 참고하라고 한 논문 두 편이 `docs/`에 있다:
  - "humanoid mobility via attention-based contacts": Mixamo 동작을 로봇에 리타게팅해 접촉 라벨을 만들고, 접촉 tracker →
    attention 접촉 planner → RL fine-tuning 순서로 학습한다.
  - "emergent dexerity via diverse resets and largescale reinforcement learning": 다양한 유효 초기 상태(reset bank)와 대규모 PPO
    (최대 65536 env)를 함께 쓴다.
- Codex 조사 결과(`docs/h200_motion_priors.md`):
  - Mixamo, Kimodo, MotionBricks는 사람형이나 G1 골격용이다. Flamingo URDF/USD를 넣어 바로 애니메이션을 만들 수는 없다.
  - Isaac Sim의 Physics Inspector는 관절 슬라이더로 자세를 확인하는 정도다.
- 제안된 길 (아직 구현 안 됨):
  - 한 단을 6–8개 주요 접촉 상태로 나눈다: 사선 접근 → 하중 이동 → 선행 바퀴 상승 → 위 디딤판 접촉 → 하중 이동 → 나머지 바퀴.
  - 중간은 보간과 물리 rollout으로 채운다.
  - 지금 성공하는 정책의 상태·접촉 기록(`record_stair_step.py --trace`)을 초기 후보로 쓴다.
  - 좌우 반전(왼 바퀴 먼저 오르는 궤적을 만들 수 있다)과 yaw·단 높이 변형으로 reset bank를 늘린다.
- 사용자에게 방향을 확인한 뒤 진행한다.

## 9. 로컬 GB10에만 있는 것

- 영상: `logs/co_rl/_videos/` (H200 복사 뒤에 찍은 것은 H200에 없다). 최고 모델 영상 8개는 §6의 표에 있다.
- `record_stair_step.py`의 `--pick_fail`과 이 문서는 H200 복사 뒤에 생겼다. H200 복사본에는 없을 수 있다.
- 로컬 git 상태: branch `alternating_feet`.
  - 로컬 커밋은 be55033, 3ade6e7, e90338c다.
  - `origin`에는 사용자의 원래 커밋 2844f06이 있다. 로컬에서 그 커밋을 고쳐서 브랜치가 갈라졌으므로, 올리려면 force push가 필요하다.
  - 2026-10-30 기준 사용자 확인을 기다린다.
