# 🦿 Flamingo-light v01_2_2 — 한 바퀴씩 계단 오르기 (Alternating feet, ZMP 기반 보상)

> 브랜치 `alternating_feet` (2026-10-23 작성). 기존 점프 접근(`rough_env/stair_jump/`)과 그 보상
> (`stair_rewards.py`, `jump_rewards.py`, `mdp/rewards.py`)은 **수정하지 않았다.** 새 태스크는 전부
> `rough_env/stair_step/` 아래에 있다.

## 1. 무엇이 바뀌었나

| | stair_jump (기존) | stair_step (신규) |
|---|---|---|
| 로봇 | flamingo_light v03_0_1 | **flamingo_light v01_2_2** (카메라 없는 로봇 중 가장 최신) |
| 오르는 방법 | 두 바퀴 동시 hop (perception trigger) | **한 바퀴씩**: 하중을 한쪽 바퀴로 옮기고 → 반대 바퀴를 들어 tread에 올리고 → 나머지 바퀴를 올림 |
| 보상 | StairClimbProgress + hop/landing/feet-off | **ZMP 기반** (아래 §3) + 교대 step-up 이벤트 |
| 점프 | 보상 대상 | **패널티** (두 바퀴 모두 공중 = flight) |
| warm start | v03 flat stand_drive | **v01_2_2 flat stand_drive** (1500 it, 4096 env, 이번에 새로 학습) |
| policy 관측 | height scan 포함 | **proprioception only (52)** — v01_2_2에는 카메라가 없음. `-Perceptive-` 변형은 바퀴 주변 72-ray scan 추가 |

## 2. 로봇에서 측정한 사실 (설계 근거)

`flamingo_light_v01_2_2` USD를 Isaac Sim에서 직접 측정:

- 바퀴 반지름 0.076 m, 타이어 중심면 y = ±0.172 m (링크 원점 ±0.157 m에서 축 방향 +0.015 m), 총질량 5.23 kg.
- 다리 = 어깨 1-DoF + 4절 링크. **바퀴는 몸체 아래로 거의 수직으로만 움직인다** (어깨 0 → −0.75 rad 동안 |dx| < 5 mm).
  → 두 바퀴는 항상 나란히 있다. 사람처럼 발을 앞뒤로 벌릴 수 없으므로 계단은 **step-to** 패턴
  (L 올림 → R 올림 → 다음 단)으로만 오를 수 있다.
- 다리 행정 0.178 m: 서 있는 높이 0.129 m(접음) ~ 0.307 m(완전 신전). 기존 stand_drive 목표 0.33 m는 v01_2_2에서
  도달 불가 → v01_2_2 stand_drive 목표를 **0.27 m**(어깨 ≈ −0.61 rad)로 맞춤.
- 어깨 hard limit = −0.75 rad = 기존 종료 임계값 → 탐색 노이즈만으로 에피소드가 끝나는 문제가 있어
  stair_step에서는 종료 임계값을 −0.77로 둔다 (실제 limit 위반만 종료).
- ~~라이저를 굴러 넘을 수 없다~~ **(틀림, §6.3)**: 바퀴 반지름 7.6 cm에 라이저 5.6 cm(레벨 0)이면 모서리 접촉 법선이
  수직에서 75° → 바퀴 자신의 토크만으로 오르려면 μ ≥ 3.7. 그러나 반대쪽 바퀴가 몸체를 밀어 주면 μ ≈ 1로도 넘어가고
  (모서리 반력의 수직 성분 + 바퀴 토크의 마찰), 1 m/s로 부딪히면 관성으로 튀어 넘는다. 라이저가 차축보다 높을 때
  (h > r)만 반력이 바퀴를 아래·뒤로 밀어 굴러 넘기가 불가능하다.
- **정면 접근으로는 한 바퀴씩 오를 수 없다**: 두 타이어 중심이 라이저에서 같은 거리에 있으므로 한쪽 바퀴를 들면
  다른 쪽(지지) 바퀴도 라이저에 막혀 앞으로 갈 수 없고, 든 바퀴를 tread에 내려놓을 수 없다.
  트랙 0.344 m에서 든 바퀴 중심이 모서리를 넘어가려면 두 바퀴의 라이저 방향 거리 차가 ≥ ~0.10 m,
  즉 **라이저에 대해 |yaw| ≳ 17°로 비스듬히** 서야 한다. → spawn yaw ±0.6 rad (±34°).

## 3. 논문 → 보상 매핑

### Vukobratović & Stepanenko 1972 (`docs/zmp.pdf`)
* ZMP = 지면 반력의 합력이 지나는 점, 그 점에 대한 외력+관성력 모멘트 = 0.
* 단일 지지: ZMP는 지지발 중심. 이중 지지: 이전 지지발 → 새 지지발로 점진 이동.
* 안정 여유(stability margin) = 지지 영역 경계까지의 거리 (발 모서리 회전각 ζ 분석).
* 다물체 ZMP (식 12–13, 회전 관성 항 생략):
  `p_x = Σ m_i[(z̈_i+g)x_i − ẍ_i(z_i−z_0)] / Σ m_i(z̈_i+g)` → `zmp_model: multibody`

### Kajita et al. 2003 (`docs/Biped_walking_pattern_generation_...pdf`)
* cart-table 모델 (식 8–9): `p = x − (z_c/g)ẍ` → `zmp_model: cart_table` (전신 CoM 사용, (g+z̈) 보정)
* 보행 생성 = ZMP 추종 서보, 성능지수 (식 14) `J = Σ Q_e e² + R Δu²` (u = CoM jerk)
* ZMP reference: 단일 지지 중 지지발 중심, 이중 지지에서 다음 지지발로 (cubic) 이동

### 이 로봇에 적용 (`zmp_rewards.py`)

| 보상 항 | 식 / 의미 | 논문 근거 |
|---|---|---|
| `zmp_tracking` (+) | `exp(−(e_x/σ_x)² − (e_y/σ_y)²)`, e = ZMP − ZMP_ref. **스텝 중에만** (라이저 감지 또는 단일 지지) | Kajita 식 14의 Q_e e² |
| ZMP reference | 주행 중 두 바퀴 중앙 → 바퀴 앞에 라이저가 감지되면 **다음 지지 바퀴 쪽으로 80%** 이동 → 단일 지지 중 지지 바퀴 → 1차 필터(τ=0.1 s)로 hand-over | Vukobratović (단일 지지=지지발 중심, 이중 지지=점진 이동), Kajita (cubic 전이) |
| `zmp_margin` (−) | 지지 다각형 안쪽 여유가 `safe_margin` 미만으로 줄어든 양 (lateral) | Vukobratović stability margin |
| `cop_transfer` (+) | 이중 지지에서 스텝 예정 시 접촉력 CoP가 지지 바퀴로 이동한 정도 (= 들 바퀴의 unloading) | "ZMP translates into the other foot" |
| `com_jerk` (−) | ‖CoM jerk‖² | Kajita 식 14의 R Δu² |
| `step_up` (+, 이벤트) | 한 바퀴만 공중(깨끗한 단일 지지)이었다가 **더 높은 새 tread**에 착지 = 1회 스텝. 직전 스텝과 다른 바퀴면 보너스 | "한 발씩" |
| `riser_progress` (+, 이벤트) | 뒤쪽(낮은) 바퀴가 새 높이에 도달 (계단 1단 완료). 1 s 이내 flight가 있으면 지급 안 함 | 진행 |
| `swing_clearance` (+) | 단일 지지 + 라이저 앞일 때 swing 바퀴를 라이저 위로 들어 올린 비율 [0,1] | 발 들기 shaping |
| `flight` (−) | 두 바퀴 모두 공중 (점프) | 점프 금지 |

**시뮬레이션에서 ZMP margin에 대한 주의**: 물리적으로 일관된 운동이면 동역학 ZMP = CoP이므로 ZMP는 지지 다각형
밖으로 나가지 않는다 (점질량 근사에서 정확히 일치). 넘어지는 로봇은 ZMP가 **경계 위에** 있고 그 경계를 축으로 회전한다
(Vukobratović의 발 모서리 회전). 그래서 margin 항은 "밖으로 나간 양"이 아니라 **안전 여유의 부족분**에 패널티를 준다.
또 바퀴는 앞뒤 방향으로 선 접촉이라 sagittal margin은 의미가 없어 lateral만 쓴다 (`margin_axes`).
검증: model_300 stand-drive 정책으로 이중 지지에서 동역학 ZMP와 접촉력 CoP 차이 평균 1.4 cm (x) / 2.2 cm (y).
Kajita가 보고한 multibody vs cart-table 오차(2.3 cm)와 같은 수준이다.

그 외: 속도 추종(std 0.3), yaw rate 추종, 자세(roll 약하게/pitch 강하게), 지지면 기준 몸체 높이(평지 이중 지지에서만),
관절·토크·action rate 정규화, 종료 패널티 — 모두 generic `mdp` 함수.

### 파일럿에서 고친 것 (2048 env, 각 수십~300 it)

| 파일럿 | 관찰 | 수정 |
|---|---|---|
| 1 | 종료의 57%가 `shoulder_lower_violation` (v01_2_2 hard limit −0.75 = 종료 임계값, 노이즈 std 0.4), flight 11% | 종료 임계 −0.77 (실제 limit 위반만), init std 0.25 |
| 2 | 90 it에 time-out 95%·오른 단 0: **계단 앞에 서서 버티는** 국소해 (속도 std 0.5면 막혀도 보상 0.6–0.85, `zmp_tracking`이 정지 균형에 보상), std 0.25→0.13 | `zmp_tracking`은 스텝 중에만, 속도 std 0.3, entropy 0.01, `swing_clearance` 4, `cop_transfer` 1.0, swing 보상 음수 제거 |
| 3 | spawn yaw ±0.15 rad = 사실상 정면 → 물리적으로 한 바퀴 스텝 불가 (§2) | spawn yaw ±0.6 rad |
| 4 | 처음 60–80 it 동안 시작 0.1–0.3 s 만에 넘어짐: warm start가 std를 0.25로 새로 만들어 어깨 노이즈가 stand-drive 정책(0.055)의 4.5배 | `--warmstart_std_scale 1.5` (체크포인트 std × 1.5로 시작), 넘어짐 종료 0.5 → 0.6 rad |
| 5 | blind: 다시 "하중만 옮기고 서 있기"로 수렴 (time-out 97%). perceptive: iteration 1에 value loss 45로 붕괴, 이후 300 it 내내 6 step 만에 몸체 접촉 종료 — 첫 PPO 업데이트(lr 1e-3)가 이식된 정책을 망가뜨림 | **정체 종료** (4 s 동안 spawn 지점에서 5 cm 더 못 가면 실패), warm start lr 1e-4 |
| 6 | 두 변형 모두 안정, 정체 종료 86–96%, std 0.18→0.30 (탐색 증가), 단일 지지 4%→10%, step-up은 아직 드묾 (165 it) | — (sweep 설정으로 채택) |

**실현 가능성 시험** (`scratchpad` probe): v01_2_2 stand-drive 정책 + 스크립트(선행 바퀴가 막히면 그 다리를 완전히 접고
지지 바퀴로 전진)로, 레벨 0 (5.6 cm), yaw 0.5 rad에서 32회 중 **3회 선행 바퀴가 위 단에 올라감, 1회 깨끗한 단일 지지
step-up** (단일 지지 최대 0.12 s). → 한 바퀴씩 오르기는 물리적으로 가능하지만 타이밍이 매우 빡빡한 동작이다.
(어깨 목표 clip [−0.745, 0]을 한때 넣었다가 뺐다: warm-start 정책은 clip 없이 학습되어 action 의미가 달라짐.)

## 4. 커리큘럼

같은 inverted-pyramid 계단(라이저 5→15 cm, tread 0.50→0.30 m, 10 rows), 중앙 플랫폼만 5 m → 2 m
(첫 라이저까지 0.5–0.7 m). 승급 = 에피소드에서 뒤쪽 바퀴 기준 **2단 이상** 오르고 **깨끗한 step-up ≥ 2회**
(낮은 라이저를 두 바퀴로 굴러 넘는 것만으로는 승급하지 않음). 강등 = 0.5단 미만.

## 5. 실행

```bash
# 1) v01_2_2 flat stand-drive (warm start 원본)
./run_gb10.sh scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-Light-v01-2-2-v1-ppo \
    --algo ppo --headless --num_envs 4096 --max_iterations 1500 --batch_exit
# 2) 한 바퀴씩 계단 (blind)
./run_gb10.sh scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-v1-ppo \
    --algo ppo --headless --num_envs 4096 --max_iterations 5000 --batch_exit \
    --warmstart_ckpt <stand_drive>/model_1499.pt --warmstart_widen --warmstart_std_scale 1.5
# 3) sweep (L4 직교배열, 4 × 5000 it; v1 = stair_step_zmp_l4.yaml, v2 = stair_step_zmp_l4_v2.yaml)
./run_gb10.sh scripts/co_rl/sweep.py --config scripts/co_rl/sweeps/stair_step_zmp_l4_v2.yaml --python ./run_gb10.sh
./run_gb10.sh scripts/co_rl/analyze_stair_step_sweep.py logs/co_rl/Flamingo_Light_Rough_Stair_Step/ppo/_sweeps/<sweep>
# 4) 평가 (레벨별 오른 계단 수, step-up, 교대 비율, flight, ZMP margin; --stochastic = 학습 때의 std로 샘플)
./run_gb10.sh scripts/co_rl/eval_stair_step.py --checkpoint <run>/model_4999.pt --levels 0 3 6 9
# 5) 영상 + 시계열: 32 env를 yaw ±0.6으로 한 번 돌려 가장 많이 오른 env를 다시 돌리며 촬영
./run_gb10.sh scripts/co_rl/record_stair_step.py --checkpoint <run>/model_4999.pt --level 3 --num_envs 32 \
    --yaw_spread 0.6 --pick_best --view side --solo --out logs/co_rl/_videos/x.mp4 --trace logs/co_rl/_videos/x
```

`run_gb10.sh`: 이 DGX Spark(GB10)용 런처 (Isaac Lab 2.3.2 pip 환경, isaaclab_tasks 소스 경로, TorchScript GPU fuser off).
`--warmstart_widen`: 관측을 **뒤에 추가**해서 입력층이 커졌을 때 기존 열은 복사하고 새 열은 0으로 초기화 → 시작 시점의
네트워크 출력이 체크포인트와 동일.

## 6. 결과

### 6.1 sweep v1 (2026-10-23 ~ 24, 4 × 5000 it, entropy 0.01, std 상한 3.0)

L4 직교배열: 관측(blind / perceptive) × ZMP 모델(multibody / cart_table) × ZMP shaping(low = `zmp_tracking`·`cop_transfer`
1.0 / high = 3.0). 학습 지표는 마지막 10 % iteration 평균 (확률적 정책, 커리큘럼 레벨에서).

| trial | 관측 | ZMP | shaping | 도달 it | terrain level | 오른 단/에피소드 | step-up/에피소드 | 교대 비율 | 단일 지지 | flight | 종료: 정체 / 몸체 접촉 / 기울어짐 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | blind | multibody | low | 5000 | 0.15 | 0.70 | 0.85 | 0.74 | 22 % | 1.8 % | 55 / 13 / 32 % |
| 1 | blind | cart_table | high | 2890 ✗ | **1.12** | **2.53** | **4.05** | 0.78 | 29 % | 2.1 % | 62 / 19 / 16 % |
| 2 | perceptive | multibody | high | 4502 ✗ | 0.78 | 1.86 | 3.13 | 0.72 | 25 % | 2.0 % | 56 / 18 / 14 % |
| 3 | perceptive | cart_table | low | 4005 ✗ | 1.05 | 2.04 | 3.92 | 0.72 | 24 % | 3.1 % | 61 / 18 / 15 % |

✗ = NaN 물리 폭주로 학습 프로세스 종료 (마지막 저장 checkpoint까지 사용).

**병리 현상**: 네 trial 모두 **왼쪽 어깨 action std가 3.0 상한**까지 커졌다 (다른 세 차원은 0.1–0.5).
평가 trace를 보면 결정적 정책의 왼쪽 어깨 action은 **항상 −1**(clip 값, hard limit −0.75 rad 너머)이다. 포화된 action
차원은 노이즈가 결과를 거의 바꾸지 않으므로 entropy 보너스(0.01)가 std를 계속 키운다. 계단 오르기는 이 std 증가와
함께 1000–1500 it 사이에 나타났다 (trial 1: step-up 0.36 → 1.95/에피소드).

**레벨별 평가** (`eval_stair_step.py`, 64 env, 15 s, 첫 에피소드, 명령 0.35 m/s, spawn yaw ±0.6).
오른 단 평균 / 1단 이상 오른 env 비율 / step-up 수 / 교대 비율:

| checkpoint | 레벨 0 (5.6 cm) | 레벨 3 (8.9 cm) | 레벨 6 (12.2 cm) |
|---|---|---|---|
| trial 0 model_4999 | 0.48 / 17 % / 0.5 / 0.73 | 0 / 0 % / 0 | 0 / 0 % / 0.0 |
| trial 1 model_2800 | 1.31 / 17 % / 1.7 / 0.86 | 0.41 / 14 % / 1.1 / 0.79 | 0.02 / 2 % / 0.2 |
| trial 2 model_4500 | **2.01 / 27 %** / 1.9 / 0.55 | **0.70 / 14 %** / 2.0 / 0.76 | 0.03 / 0 % / 0.1 |
| trial 3 model_4000 | 1.58 / 39 % / 0.9 / 0.72 | 0.61 / 11 % / 1.5 / 0.69 | 0.01 / 0 % / 0.1 |
| trial 1, `--stochastic` | 2.77 / 39 % / 3.3 / 0.79 | 2.12 / 38 % / 4.4 / 0.75 | — |
| trial 2, `--stochastic` | 3.05 / 41 % / 4.5 / 0.76 | 3.54 / 58 % / 7.2 / 0.73 | — |

결과가 **양극화**되어 있다: 성공한 env는 타일 끝(8–9단, 레벨 3에서 0.82 m)까지 오르고 실패한 env는 0단이다.
실패의 대부분(레벨 0: 23개 중 21개, 레벨 3: 25개 중 21개)은 첫 라이저 앞에서 4 s 정체 종료로 끝났다. 즉 병목은 정지 상태에서
**첫 스텝을 시작하는 것**이다. 학습된 std로 샘플링하면(왼쪽 다리가 무작위로 움직임) 성공률이 2–4배가 된다.
정책은 노이즈에 기대어 막힌 상태를 빠져나온다.

**학습된 보행** (trial 2, 레벨 3에서 가장 많이 오른 env의 trace, `logs/co_rl/_videos/stair_step_v1_trial2_best_L3.png`):
- **step-to 패턴**: 라이저마다 오른쪽 바퀴가 먼저 올라가고 왼쪽이 따라 올라간다. 9단을 4 s에 올랐고 깨끗한
  단일 지지 step-up은 23회(R 14, L 9; 교대 비율 0.59, 레벨 0에서는 0.78)였다.
- **한쪽 다리만 능동**: 왼쪽 어깨는 98 % 시간 동안 limit에 고정되어 있고, 오른쪽 어깨만 −0.75 → −0.40 rad로 펄스한다.
  왼쪽 바퀴는 몸체 yaw 진동(−0.37 ~ +0.24 rad)과 오른쪽 다리 신전으로 들어 올리는 것으로 보인다 (hip-hike에 가까움).
- 단일 지지 구간은 평균 0.07 s, 최대 0.34 s, flight는 5–7 %. 정적으로 안정한 단일 지지는 불가능하며(§2), 실제로
  측정된 lateral ZMP margin 평균은 ≈ 0 (p10 −0.05 ~ −0.16 m)이다. 동적으로 넘어지는 동안 반대 바퀴를 내려놓는다.
- **속도 명령을 무시**한다: 오르는 동안 0.9–1.2 m/s (명령 0.35 m/s). step-up 이벤트 보상(10/회)과 정체 종료가
  속도 추종(1.0, std 0.3)보다 훨씬 크다.
- 타일 끝까지 오른 뒤에는 이웃 타일 구덩이로 **계단을 내려가다가**(학습한 적 없음) 몸체 접촉 / 기울어짐으로 끝난다.

**L4 주효과 (v1)는 해석할 수 없다**: 끝까지 살아남은 trial 0이 혼자 나쁘고(blind·multibody·low), 나머지 셋은 서로
다른 iteration에서 멈췄다. 4개 run의 L4에서는 한 run이 이상치면 세 요인 모두 같은 크기의 "효과"를 받는다.

→ **v2 수정**: entropy 0.01 → 0.005, `max_std` 0.4 (어깨 범위 0.75 rad, 바퀴 action 1 = 40 rad/s), env 경계에서
action ±1 clip + NaN 정리, 비유한 상태 종료(`nonfinite_state`), 관측 `nan_to_num`.

### 6.2 sweep v2 (2026-10-26 ~ 27, 4 × 5000 it, entropy 0.005, std 상한 0.4, action clip 1)

같은 L4 배치. 학습 지표는 마지막 10 % iteration 평균 (`analyze_stair_step_sweep.py --tail 0.1`).
v1과의 곡선 비교: `_sweeps/stair_step_zmp_l4_v2_2026-10-26_13-16-41/compare_v1_v2.png`.

| trial | 관측 | ZMP | shaping | 도달 it | terrain level | 오른 단/에피소드 | step-up/에피소드 | 교대 비율 | 단일 지지 | flight | 종료: 정체 / 몸체 접촉 / 기울어짐 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | blind | multibody | low | 5000 | 0.15 | 0.75 | 1.17 | 0.76 | 13 % | 0.5 % | 69 / 16 / 15 % |
| 1 | blind | cart_table | high | 5000 | 0.19 | 0.80 | 1.56 | 0.78 | 14 % | 0.4 % | 77 / 9 / 7 % |
| 2 | perceptive | multibody | high | 5000 | 1.03 | 1.91 | 4.96 | 0.77 | 19 % | 0.7 % | 63 / 16 / 9 % |
| 3 | perceptive | cart_table | low | 5000 | **1.08** | **2.33** | **5.56** | 0.80 | 16 % | 1.6 % | 58 / 14 / 6 % |

- **학습 안정성은 해결**: 네 run 모두 5000 it 완주 (NaN 없음), 평균 std 0.1–0.3 유지. perceptive trial은 std가 작은
  상태(0.11–0.19)에서 1750–1950 it에 계단을 오르기 시작했다 → v1의 std 폭주가 계단 오르기를 만든 것은 아니다.
- **L4 주효과** (terrain level / 오른 단 / step-up): 관측 perceptive − blind = **+0.88 / +1.34 / +3.90**,
  ZMP multibody − cart_table = −0.05 / −0.23 / −0.50, shaping low − high = +0.01 / +0.19 / +0.10.
  지형을 보는 것이 지배적이고, ZMP 모델과 shaping 세기는 이 실험에서 차이가 없다.

**레벨별 평가** (model_4999, 결정적 정책, 64 env, 15 s, 첫 에피소드). 엄격 판정(§7)을 같이 기록했다.
오른 단 평균 / 1단 이상 오른 env 비율, step-up/env는 좌 / 우:

| checkpoint | 레벨 | 오른 단 | 기존 판정 step-up | **엄격 판정 (다리로 들어 올림)** | lifted_share |
|---|---|---|---|---|---|
| trial 0 | 0 (5.6 cm) | 0.80 / 11 % | 0.55 / 0.52 | 0.06 / 0.44 | 0.28 |
| trial 0 | 3 (8.9 cm) | 0.09 / 3 % | 0.02 / 0.50 | 0.00 / 0.09 | 0.21 |
| trial 1 | 0 | 0.78 / 12 % | 0.72 / 0.73 | 0.14 / 0.45 | 0.37 |
| trial 1 | 3 | 0.35 / 12 % | 0.77 / 0.58 | 0.14 / 0.19 | 0.25 |
| trial 2 | 0 | 2.20 / 33 % | 2.08 / 2.48 | 0.02 / 1.14 | 0.27 |
| trial 2 | 3 | **3.52 / 58 %** | 4.09 / 5.84 | 0.72 / 2.67 | 0.47 |
| trial 3 | 0 | **3.63 / 50 %** | 3.06 / 2.73 | 2.41 / 0.23 | 0.37 |
| trial 3 | 3 | 0.79 / 11 % | 0.84 / 1.22 | 0.47 / 0.33 | 0.49 |

레벨 6 (12.2 cm)은 네 정책 모두 0.07단 이하.

v2 정책도 **한 다리로 오른다**. 다리로 들어 올려 오른 단은 27–49 %뿐이고, 들어 올린 쪽이 한 다리에 몰려 있다
(trial 2는 오른쪽, trial 3은 왼쪽). 나머지 단은 굴러 넘기거나 부딪혀 넘긴다 (§6.3).

### 6.3 영상 검증 (2026-10-27): 한 다리 + 튕겨 넘기, 그리고 회전 보행

영상에서 "발을 번갈아 움직이지 않고, 바퀴 반지름보다 낮은 계단은 그냥 올라간다"는 지적이 나왔다. 녹화 trace
(`record_stair_step.py --trace`)에서 step-up마다 직전 0.4 s 동안 디딘 다리의 어깨가 반대 다리보다 0.15 rad
(≈ 3.5 cm) 이상 접혔는지 확인했다:

| 정책 (가장 많이 오른 env) | 레벨 | step-up L (다리 접음) | step-up R (다리 접음) | 어깨가 limit에 고정 | 평균 속도 | 패턴 |
|---|---|---|---|---|---|---|
| v1 trial 2 model_4500 | 0 | 6 (0) | 4 (4) | L 100 % | 1.29 m/s | 오른 다리만 들고, 왼 바퀴는 모서리에 부딪혀 튕겨 넘김 |
| v1 trial 2 model_4500 | 3 | 8 (0) | 15 (14) | L 100 % | 1.13 m/s | 〃 |
| v2 trial 2 model_4300 | 0 | 6 (0) | 12 (12) | L 40 % | 1.02 m/s | 〃 (yaw ≈ +0.5 rad 유지) |
| v2 trial 1 model_4300 | 0 | 8 (8) | 5 (5) | — | 0.61 m/s | 양 다리 교대 (L R L R …), 그러나 **제자리 회전** |

v2 trial 1(blind)은 실제로 양쪽 다리를 번갈아 들지만, yaw가 1.2 s마다 한 바퀴씩 돈다(~5 rad/s). 지지 바퀴를
축으로 반 바퀴 돌면서 반대 바퀴를 다음 단에 올린다. 두 바퀴가 나란히 있고 다리가 수직으로만 움직이므로, 한
바퀴를 앞에 두는 방법은 yaw뿐이다(§2). 그래서 계속 도는 것이 가장 싼 방법이 됐다.

원인:
1. **step-up 판정이 수직 접촉력만 봤다**: 모서리를 타고 넘는 바퀴는 반력이 거의 수평이라 "공중"으로 보였다.
   다리를 들었는지도 보지 않았다.
2. **교대 보너스는 바퀴 좌/우만 봤다**: 비스듬히 굴러 올라가도 L, R이 번갈아 나온다.
3. **커리큘럼이 머무는 레벨 0–1(5–6.7 cm)은 바퀴 반지름보다 낮다** (§2 정정).
4. 속도 추종(가중 1.0)과 yaw rate 추종(0.5)이 step-up 이벤트(10 + 교대 10)보다 훨씬 약하다. 그래서
   부딪혀 넘기와 회전 보행이 이득이었다.

→ v1/v2의 "오른 단 / step-up / 교대 비율"은 **한 발씩 오르기의 지표로 쓸 수 없다**.

v3의 엄격한 판정기(§7)를 같은 정책에 돌려 검증했다 (`eval_stair_step.py`, 64 env, 15 s, 결정적 정책).
step-up/env는 좌 / 우:

| 정책 | 레벨 | 기존 판정 step-up | **엄격 판정 (다리로 들어 올림)** | lifted_share | 전진 속도 |
|---|---|---|---|---|---|
| v1 trial 2 model_4500 | 0 (5.6 cm) | 0.72 / 1.38 | **0.00** / 0.75 | 0.18 | 0.32 m/s |
| v1 trial 2 model_4500 | 3 (8.9 cm) | 1.84 / 2.95 | **0.00** / 0.92 | 0.20 | 0.73 m/s |
| v2 trial 2 model_4300 | 0 | 0.97 / 1.30 | **0.00** / 0.53 | 0.22 | 0.19 m/s |
| v2 trial 2 model_4300 | 3 | 2.80 / 3.95 | **1.28 / 1.69** | **0.60** | 0.43 m/s |
| v2 trial 1 model_4300 | 0 | 1.05 / 1.05 | 0.02 / 0.89 | 0.36 | 0.06 m/s (회전) |

(전진 속도는 평가 env 전체의 평균이고, 대부분 첫 라이저 앞에서 멈춘 env다. 오르는 env 자체는 0.8–1.3 m/s.)
- 왼쪽 다리를 고정한 정책은 왼쪽 step-up을 하나도 인정받지 못한다.
- v2 trial 2는 **8.9 cm(차축보다 높은) 라이저에서는 양쪽 다리를 실제로 든다** (바퀴가 오른 단의 60 %).
  5.6 cm에서는 굴러 넘긴다. 즉 라이저가 바퀴 반지름보다 높으면 다리 들기가 강제된다 → v3 지형의 근거.
- v2 trial 1의 왼쪽 step-up은 스윙 중 접촉은 없지만 다리 들기가 0.4–2.8 cm뿐이다 (몸체를 기울이고 회전해서
  올림). 4 cm 기준에 걸린다. 오른쪽은 4.3–12 cm로 들어 올린 4회만 인정됐다.

## 7. v3: 다리로 들어 올려야만 보상 (`Stair-Step-Lift`)

태스크 `Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift[-Perceptive][-Play]-v1-ppo`, 로그
`logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift/`, sweep `scripts/co_rl/sweeps/stair_step_lift_v3.yaml`.
v1/v2 설정(`Stair-Step`)은 바꾸지 않았고, 기존 판정기도 그대로 돈다(새 지표는 v1/v2에서도 함께 기록).

| 변경 | 내용 |
|---|---|
| 지형 | 라이저 9 → 15 cm (`STAIR_STEP_LIFT_TERRAINS_CFG`): 모든 단이 차축보다 높다 |
| step-up 판정 (`zmp_gait.step_rule = "lift"`) | 스윙 동안 **어떤 접촉도 없음**(힘의 크기 기준), 스윙 다리가 지지 다리보다 **4 cm 이상 접힘**(몸체 좌표계의 바퀴 높이 차, 들기 직전 포함, 몸체 roll은 제외), **단 위에 착지**(착지 힘의 70 % 이상이 수직), 그 바퀴의 새 최고 단. 교대 보너스와 커리큘럼 승급도 이 판정을 쓴다 |
| `swing_retraction` (+2) | 앞에 라이저가 있을 때 들 바퀴의 다리를 (라이저 높이 + 2 cm) 대비 얼마나 접었나, [0, 1] |
| `speed_excess` (−5) | max(0, v_x − 명령 − 0.15 m/s)² |
| `yaw_rate_excess` (−2) | max(0, \|저역통과 yaw rate\| − 0.5 rad/s)², τ = 0.3 s (바퀴 노이즈의 떨림은 거르고 지속 회전만) |
| 지표 | `lifted_step_ups`, `lifted_alternating_ratio`, `lifted_share` = 바퀴가 오른 단 중 다리로 들어 올려 오른 비율 |

ZMP 항(`zmp_tracking`, `zmp_margin`, `cop_transfer`, `com_jerk`)과 나머지 보상은 v2와 같다.

### 7.1 sweep v3 (2026-10-27 13:41 시작, 4 × 5000 it)

2 × 2 요인 실험: 관측 (blind / perceptive) × warm start (v01_2_2 flat stand-drive / 같은 관측의 v2 정책).
ZMP 모델은 cart_table, shaping은 low로 고정했다 (v2에서 효과 없음).

v2 warm start 후보를 v3 지형에서 엄격 판정으로 평가했다 (model_4999, 결정적 정책, 64 env, 15 s):

| 정책 | 레벨 0 (9.3 cm): 오른 단 / 들어 올림 L / R / lifted_share | 레벨 3 (11.3 cm) |
|---|---|---|
| v2 trial 1 (blind) | 0.01 / 0 % — 0.00 / 0.03 | 0 |
| v2 trial 2 (perceptive) | 0.71 / 17 % — 0.36 / 0.67 — **0.67** | 0.23 / 5 % — 0.14 / 0.09 — 0.37 |
| v2 trial 3 (perceptive) | 1.34 / 22 % — 0.50 / 0.28 — 0.27 | 0.27 / 6 % — 0.06 / 0.06 — 0.20 |

라이저가 모두 차축보다 높아지면 v2 정책은 대부분 첫 단 앞에서 멈춘다.
- perceptive 행은 **trial 2**에서 시작한다: 오른 단 중 다리로 들어 올린 비율이 가장 높다.
- blind 행은 **trial 1**에서 시작한다: v3 지형에서는 거의 오르지 못하지만, 양 다리를 번갈아 드는 유일한
  blind 정책이다 (§6.3).

| trial | 관측 | warm start |
|---|---|---|
| 0 | blind | stand-drive `model_1499` |
| 1 | blind | v2 trial 1 `model_4999` |
| 2 | perceptive | stand-drive `model_1499` |
| 3 | perceptive | v2 trial 2 `model_4999` |

1100 it까지는 4개를 병렬로 돌렸다 (trial당 12.8 s/it). GB10은 프로세스를 시분할하므로 병렬로 돌려도 총 처리량은
약 3만 env-step/s로 같다. 그래서 1100 it부터 한 trial씩 이어서 학습했다
(`scripts/co_rl/sweeps/stair_step_lift_v3_sequential.sh`, 2.8 s/it, trial 3 → 1 → 2 → 0, 10-28 06:23 종료).
step 시간의 15 %만 물리 연산이고, 나머지는 sim 쓰기·리셋·종료 판정·ZMP 상태 계산의 호출 오버헤드다.
env 4096 → 8192에서 처리량이 1.66배 늘었다.

**학습 지표** (마지막 50 it 평균; 괄호는 최고점 2400–2600 it):

| trial | 관측 / warm start | 들어 올린 step-up/에피소드 | 교대 비율 | lifted_share | 오른 단 | terrain level |
|---|---|---|---|---|---|---|
| 0 | blind / stand-drive | 0 | – | – | 0 | 0 |
| 1 | blind / v2 trial 1 | 0.55 | 0.99 | 0.92 | 0.00 | 0 |
| 2 | perceptive / stand-drive | 0 | – | – | 0 | 0 |
| 3 | perceptive / v2 trial 2 | **1.76 (2.0)** | 0.80 | 0.78 | **0.92 (1.04)** | 0.29 (0.46) |

**레벨별 평가** (결정적 정책, 64 env, 15 s). 오른 단 평균 / 1단 이상 오른 env, 들어 올린 step-up/env 좌 / 우,
lifted_share:

| checkpoint | 9.3 cm | 11.3 cm | 13.3 cm | 15.0 cm |
|---|---|---|---|---|
| trial 3 model_700 | 0.88 / 17 % — 0.83 / 0.75 — 0.86 | 1.86 / **39 %** — 1.84 / 1.91 — 0.91 | **0.36 / 17 %** — 0.34 / 0.39 — 0.63 | — |
| trial 3 model_2500 | **1.08 / 17 %** — 1.08 / 0.98 — 0.92 | **1.91** / 28 % — 1.77 / 1.91 — 0.89 | 0.18 / 6 % — 0.17 / 0.28 — 0.49 | 0 / 0 % |
| trial 3 model_4999 | 0.62 / 8 % — 0.59 / 0.56 — 0.93 | 1.87 / 31 % — 1.77 / 1.70 — 0.88 | 0.14 / 3 % — 0.08 / 0.19 — 0.57 | 0 / 0 % |
| trial 1 model_4999 | 0 — 0.00 / 0.44 | 0 — 0.00 / 0.20 | 0 | 0 |
| trial 0, 2 model_4999 | 0 | 0 | 0 | 0 |
| (비교) v2 trial 2 model_4999 | 0.71 / 17 % — 0.36 / 0.67 — 0.67 | 0.23 / 5 % — 0.14 / 0.09 — 0.37 | — | — |

- **trial 3은 양 다리를 번갈아 들어 오른다**: 들어 올린 step-up이 좌우 거의 같고(11.3 cm에서 1.77 / 1.70),
  교대 비율 0.94, 바퀴가 오른 단의 88–93 %를 다리로 들어 올렸다. 같은 지형의 v2 trial 2보다 11.3 cm에서 8배 더 오른다.
- 다만 **성공률은 낮고 700 it 이후 거의 늘지 않았다**: 11.3 cm에서 env의 28–39 %, 13.3 cm 이상은 거의 0.
  커리큘럼 레벨도 0.2–0.5에 머물렀다. 학습 중 에피소드의 약 76 %가 정체(4 s 진전 없음)로 끝났다.
- **blind(trial 1)는 한 바퀴만 첫 단에 올리고 멈춘다** (오른쪽만, 몸체는 0단). 차축보다 높은 라이저는 보지 않고는
  오르지 못했다.
- **stand-drive에서 시작한 두 trial은 5000 it 동안 다리 들기를 한 번도 발견하지 못했다.** v2 정책의 warm start가 필수였다.
- 과속(`speed_excess` −0.03/에피소드)과 회전(`yaw_rate_excess` −0.04)은 전 구간에서 작았다. 부딪혀 넘기와 회전 보행은
  다시 나타나지 않았다.

**영상 검증** (`logs/co_rl/_videos/stair_step_v3_trial3_it4999_L3_side.mp4/.png`, 11.3 cm, 64 env 중 가장 많이 오른 env).
GPU 물리가 재현되지 않으므로 3단 이상 오를 때까지 다시 찍었다 (`record_stair_step.py --retry_until 3`):
- 12 s 동안 6단을 올랐다. 들어 올린 step-up 11회는 **L R L R … 순서로 10/10 교대**했다 (L 6, R 5).
  매번 스윙 다리를 지지 다리보다 9–16 cm 접었다.
- 단마다 오른 바퀴가 먼저 오르고 왼 바퀴가 따라 오른다. 양 어깨가 번갈아 접힌다.
- yaw는 +0.25 ~ +1.0 rad에서 진동한다. 비스듬히 접근한 상태로 한 단마다 흔들 뿐, 회전 보행은 없다.
  평균 전진 속도는 0.19 m/s(명령 0.35)이다.
- 13.3 cm(model_700): 2단, 5회, L R L R R 순서로 4/4 교대 (`stair_step_v3_trial3_it700_L6_side.*`).
