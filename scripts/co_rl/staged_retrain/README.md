# 단계별 재학습 (2026-09-09 시작)

보상함수와 보상 가중치를 유지한다. 전진·후진과 좌우 회전을 학습한다.
height map은 Isaac Lab 기본 `RayCasterCfg`와 `isaaclab.envs.mdp.height_scan`을 사용한다.
카메라/경험적 마스킹, 커스텀 sample-hold/dropout/바이어스를 사용하지 않는다.
기존 1.1 m × 1.1 m, 10 cm grid와 센서 갱신 주기는 유지한다.
표준 height_scan의 기본 offset 0.5 m와 기존 observation clip을 사용한다.
`flat_baseline.json`과 `flat_clip01.json`은 PPO clip 값(0.2 / 0.1)만 다르다.
각 모델은 scratch, seed 42, 1,024 환경, 24 rollout steps, 최대 1,000 updates다.
두 모델의 경험을 이어 붙이지 않는다. 모델당 24,576,000 transitions다.
설정의 `env.seed=42`는 환경과 네트워크를 만들기 전에 난수를 고정한다.
9월 9일 완료된 최초 기준 모델은 `env.seed=null`이었으므로 통제 비교에서 제외하고 보존한다.
rough는 flat 검증 이후 별도로 진행하며 모델당 누적 rough updates를 10,000 이하로 제한한다.
iteration 수 제한만으로 sim-to-real 성능을 보장하지 않는다.

## 변경 범위

- 실사용 코드 수정: `mdp/terminations.py`에서 plane의 종료조건 반환을 단일 bool에서
  환경별 bool tensor로 수정. 무한 평지에서 종료하지 않는 의미는 동일하다.
- 실험 설정, 평가 코드, 실행 도구는 이 디렉터리에 분리했다.
- `flamingo_light_env/default_height_map.py`를 flat stand-drive와 rough stair-jump에
  적용했다. flat actor에는 원래 없던 height scan을 추가하지 않고 critic만 유지한다.
- 기존 작업 중이던 stair reward/terrain/diagnostic/runner 변경은 덮어쓰지 않았다.
- `approach_proposal.py`와 `jump_timing.patch`는 **승인 대기**다. 패치는 적용하지 않았다.

## 재현

저장된 결과 디렉터리와 실행 서비스 이름은 재사용하지 않고 새 이름을 사용한다.
GPU 작업은 한 번에 하나만 실행한다. `launch.sh`는 7 GiB RAM, 256 MiB swap,
기본 1시간 실행 제한과 기존 replay lock을 사용한다. `FLAMINGO_JOB_SECONDS`로
최대 8시간까지 지정할 수 있다. GPU 공유로 iteration당 약 10–12초가 측정되어
현재 두 모델 비교에는 28,800초를 지정했다. iteration 상한은 모델당 1,000으로 동일하다.
외부 GPU 작업은 종료하지 않는다.

2026-09-10 자원 큐가 시작한 두 모델 비교는 정상 완료했다. clip 0.15 후속 실험은
`outputs/retrain_20260909/controlled_flat_clip015_20260910/status.json`에서 확인한다.
다음은 완료된 최초 큐의 자원 대기 조건이다.
`outputs/retrain_20260909/resource_queue_status.json`이 최신 상태다.
RAM 8 GiB 이상, GPU 여유 6,500 MiB 이상, GPU 사용률 20% 미만을 30초 간격으로
3번 연속 확인하면 두 모델 비교를 한 번 시작한다. 다른 작업을 종료하지 않는다.
대기는 최대 24시간이며, 작업 중 RAM이 지속적으로 부족하면 자체 학습만 중단한다.
학습 완료와 자원 대기를 구분해야 하며, 실패한 비교를 최적 모델로 선택하지 않는다.

```bash
cat outputs/retrain_20260909/resource_queue_status.json
# 대기만 취소 (이미 시작된 학습은 별도 서비스)
systemctl --user stop flamingo-default-map-resource-queue-20260910.service
```

두 실험과 평가를 순서대로 완료하고 한 값만 다른지 검증하려면 다음을 사용한다.
출력 디렉터리는 새 이름이어야 하며, 어느 단계든 실패하면 큐를 중단한다.

```bash
FLAMINGO_JOB_SECONDS=28800 bash scripts/co_rl/staged_retrain/launch.sh flat-controlled-new \
  python -u scripts/co_rl/staged_retrain/compare_flat.py \
  --output outputs/retrain_20260909/controlled_new
```

```bash
bash scripts/co_rl/staged_retrain/launch.sh flat-new-baseline \
  python -u scripts/co_rl/train.py \
  --task Isaac-Velocity-Flat-Flamingo-Light-v1-ppo --algo ppo \
  --headless --batch_exit --num_envs 1024 --seed 42 --max_iterations 1000 \
  --run_name scratch_commands_baseline \
  --param_overrides scripts/co_rl/staged_retrain/flat_baseline.json
```

clip 비교는 run_name과 파일을 `flat_clip01.json`으로 바꾼다.
`--warmstart_ckpt`, `--resume`, `--adaptive_reward`, `--weight_schedule`,
`--roger_threshold`, `--forward_gate`를 추가하지 않는다.

```bash
bash scripts/co_rl/staged_retrain/launch.sh flat-new-benchmark \
  python -u scripts/co_rl/staged_retrain/benchmark.py --headless --mode flat \
  --checkpoint /absolute/path/to/model_999.pt --output outputs/new_flat_benchmark

bash scripts/co_rl/staged_retrain/runtime.sh \
  python -m unittest scripts.co_rl.staged_retrain.test_approach -v

git apply --check scripts/co_rl/staged_retrain/jump_timing.patch
```

`benchmark.py`는 checkpoint와 함께 저장된 env/agent 설정을 읽고 strict load한다.
현재 Isaac Lab으로 읽기 위한 옛 액추에이터 config 모듈 경로 호환 처리가 있다.
기록기는 비활성화한다. 관측·액션·동역학을 현재 task 기본값으로 교체하지 않는다.
평가는 학습하지 않으며 성공 사례만 골라내지 않는다. 세 평가 seed는
101/202/303이고, startup dynamics randomization은 환경을 만들 때 seed 101로 고정된다.
이는 세 번의 독립 학습 seed 검증을 의미하지 않는다.

flat은 전진 0.7 m/s, 후진 -0.7 m/s, 좌/우 회전 ±0.5 rad/s, 정지를 각각 평가한다.
추종오차는 최초 1초 이후와 첫 종료 이전만 집계하므로 반드시 실패율과 함께 읽는다.
stairs는 5/10/15 cm 높이, 50/40/30 cm 발판, -30/-15/0/15/30도 초기 yaw를 사용한다.
성공은 0.5초 동안 양쪽 바퀴 접촉, 계단 높이 85% 이상 지면 상승,
roll/pitch 절댓값 0.35 rad 미만을 유지하는 것이다. 공중에 뜬 점프인지는 별도 판정하지 않는다.

## 승인 대기: 점프 시작 조건

현재 전방 높이 평균만 보는 감지에 다음 조건을 추가하는 작은 패치를 제안한다.
관측 차원, 보상, 액션을 변경하지 않는다. `approach_mode` 기본값은 `off`다.
먼저 승인받을 `shadow`는 추정값과 허용 여부를 **기록만** 한다.
별도 `gate` 모드만 새 점프 시작을 제한한다. 진행 중인 점프와 재감지 상태 로직은 유지한다.

1. Isaac 기본 grid의 유한한 높이 셀에서 낮은 지면과 높은 발판의 경계를 찾는다.
2. 좌우 여러 줄의 경계에 직선을 맞춰 계단 면까지 수직거리와 면의 법선 방향을 추정한다.
3. 명령과 실제 속도가 전진이고, 거리·각도·도달시간이 범위 안일 때만 시작을 허용한다.

초기 가설값: base-to-plane 거리 0.15–0.40 m, 진입각 ±20도,
계단 법선 방향 실측 속도로 계산한 도달시간 0.18–0.55초.
0.10 m/s 이하, 후진/정지 명령, 유효하지 않은 관측에서는 시작하지 않는다.
이 숫자는 최적화 결과가 아니다. 실제 10 cm grid에서는 거리 약 반 셀,
각도 수 도의 양자화 오차가 생길 수 있다. 바퀴 위치 기준 보정도 별도 검증 대상이다.

기본 모드에서는 `valid_mask` 없이 전체 grid를 사용한다. NaN/Inf는 제외한다.
별도 FOV 진단 옵션과 `synthetic_fov_check.json`은 이전 시야 검사 기록이며
현재 학습/제안의 기본 조건에 적용하지 않는다. 첫 승인 요청은 `shadow` 연결이다.
추정 거리와 각도를 검증한 후 gate를 별도 활성화한다. 시뮬레이터의 전체 높이 grid가
실제 센서와 같다는 의미는 아니며, 현재 변경은 사용자가 요청한 기본 map 학습 조건이다.

flat actor 입력 52차원과 과거 rough actor 입력 198차원은 다르다.
현재 `load_transfer`는 모양이 다른 첫 레이어를 재초기화하므로 flat 정책을 완전히
보존하는 전이라고 볼 수 없다. rough 단계 전에 관측 항목 대응을 확인해야 한다.
이 문제를 점프 타이밍 변경과 동시에 고치지 않는다.
