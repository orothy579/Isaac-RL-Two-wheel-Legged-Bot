# 🔬 Flamingo Stair-Jump RL: 연구 방향 & RAL 논문 전략

## 1. 현재 프로젝트 분석

코드를 전체적으로 분석했습니다. 현재 시스템의 구성:

| 구성요소 | 현황 | 파일 |
|---------|------|------|
| **Phase 1** | Stand & Drive (flat → rough) | [stand_drive](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_light_env/rough_env/stand_drive) |
| **Phase 2** | Stair Jump (warm-start from Phase 1) | [stair_jump_cfg](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_light_env/rough_env/stair_jump/rough_env_stair_jump_cfg.py) |
| **Reward 설계** | StairClimbProgress (지수적) + hop/clearance | [stair_rewards.py](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_light_env/rough_env/stair_jump/stair_rewards.py) |
| **Optuna Sweep** | TPE 기반 reward weight 탐색 (이미 구현됨) | [sweep.py](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/scripts/co_rl/sweep.py) |
| **Adaptive Balancer** | ROGER-style penalty-gain 자동 조절 | [curriculums.py](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L85-L216) |
| **SRM-PPO** | State Representation Model + PPO | [srmppo.py](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/scripts/co_rl/core/algorithms/srmppo.py) |
| **ACAPS** | Action smoothness regularization | [srmppo.py L222-L243](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/scripts/co_rl/core/algorithms/srmppo.py#L222-L243) |

> [!IMPORTANT]
> 이미 상당한 인프라가 갖춰져 있습니다: Optuna sweep, adaptive reward balancer (ROGER-style), SRM, ACAPS, 
> curriculum terrain levels. 이것들을 **체계적으로 통합하고 검증**하는 것 자체가 contribution이 될 수 있습니다.

---

## 2. 시도할 수 있는 접근법 총정리

### 📋 접근법 A: Optuna + Adaptive Balancer 통합 ✅ 완료 (2026-07-15 판정)

> ✅ **결과**: bi-level(Optuna best weight + adaptive balancer) 검증 완료. 밴드 비교(n=6 vs 4)에서
> adaptive가 auc 유의(t≈3.3)·final_mean 우세(0.88±0.10 vs 0.79±0.14). best 노브 = penalty_budget
> 0.572 / g_min 0.595. ⚠️ g_min 0.1까지 열면 페널티 과완화 → 제자리 hop farming 붕괴(g_min ≥ 0.4 필수).
> 탐색 공간 축소(7→3개)도 sweep_analyze.py 상관/중요도 분석으로 완료 — 아래 로드맵 Phase 1 참고.

현재 sweep.py에서 Optuna TPE로 reward weight를 탐색하고 있고, `AdaptiveRewardBalancer`가 온라인으로 
penalty gain을 조절합니다. 이 둘을 **2단계(bi-level) 최적화**로 프레이밍할 수 있습니다:

```
Outer loop (Optuna): reward weight 초기값 탐색 (base weights)
  Inner loop (Adaptive Balancer): 학습 중 penalty-gain 동적 조절
```

**구체적 개선 방향:**
1. **탐색 공간 줄이기**: 현재 sweep config에 7개 파라미터가 있는데, 20 trial로는 부족합니다. 
   상관관계 분석으로 중요 파라미터를 3-4개로 줄이세요.
2. **Multi-Objective Optuna**: `stair_climb` metric 외에 sim-to-real transferability 관련 
   지표(action smoothness, energy consumption)도 목적함수에 추가
3. **Warm-start 체인**: Optuna best → fine-tune with Adaptive Balancer → 최종 정책

---

### 📋 접근법 B: Population-Based Training (PBT) ✅ 구현 / 🔄 라운드2 실행 중

> ✅ **구현 완료** (`scripts/co_rl/pbt.py`, 2026-07-17): population 4, 순차 1-GPU, gen마다
> final_mean 랭킹 → 하위가 상위 ckpt+params exploit 후 5개 knob(growth/jlvz/entropy/pb/g_min)
> ×1.2 perturb. 라운드1 교훈 = **1500it/gen은 이륙 전**(전멤버 ~0.03) + growth 상한 클립 승리
> → 라운드2(2026-07-19~): **5000it×2gen**(생존 lineage ≤10k iter 상한 준수), growth 경계 3.4로
> 확장. 완료 후 Optuna 챔피언(1.344)과 동일 프로토콜 비교 예정.

**PBT vs Optuna의 핵심 차이:**
- Optuna: 각 trial이 **독립적**으로 처음부터 끝까지 학습 (비효율적)
- PBT: 학습 **도중에** 성적이 나쁜 agent의 weight를 좋은 agent에서 복사(exploit) + 
  하이퍼파라미터를 변이(explore). 동일 GPU 시간 대비 훨씬 효율적
  ⚠️ 이 "효율적"이라는 말은 **원 논문(Jaderberg+2017)의 멀티-워커 비동기 설계**를 가리킨다.
  우리 1-GPU 순차 구현은 이 이점을 온전히 못 받는다 — 아래에서 정정.

**2026-07-20 작성, 2026-07-20 재정정 — 원 논문 확인 후 "1-GPU에서도 효율적인가?"의 정확한 답:**

지난 정리에서 "체크포인트 덮어쓰기는 1-GPU에서도 완벽히 동작하니 온라인 학습의 이점이 그대로
산다"고 썼는데, **이건 메커니즘과 효율성 이점을 섞어 과장한 것**이었다. 원 논문
([arXiv:1711.09846](https://arxiv.org/abs/1711.09846))을 다시 확인해 바로잡는다.

**원 논문이 실제로 설계한 것 — 진짜 비동기(asynchronous):**
- *"PBT is decentralised and asynchronous, requiring minimal overhead and infrastructure."*
  워커들이 **동시에** 돌면서 공유 저장소(key-value store/파일시스템)를 각자 독립적으로 체크한다.
- 점검 주기(t_ready)는 도메인별로 **기계번역 2000 step마다, GAN 5000 step마다** 등 —
  전체 학습 길이에 비하면 상당히 **잦은** 빈도.
- 핵심 효율성 주장: *"wall-clock run time that is no greater than that of a single
  optimisation process"* — **N개 워커를 동시에 돌리면 실제 걸리는 시간이 모델 하나 학습
  시간과 맞먹는다.** "효율적"이란 말은 정확히 이 **wall-clock 단축**을 가리키는 것이지,
  "PBT가 Optuna보다 총 GPU-시간(계산량)을 덜 쓴다"는 뜻이 아니다.
- 논문은 "semi-serial"(준순차) 변형이 이론상 가능하다고만 언급 — **핵심 설계는 아니다.**

**우리 1-GPU 순차 `pbt.py`는 바로 그 "semi-serial" 근사판이다:**
```
세대 N:  멤버0 학습(5000it, GPU) → 체크포인트 저장 → 멤버1 학습(5000it) → ... (하나씩 순서대로)
        └ 전 멤버 다 끝나야만 성적 비교 → 하위 멤버가 상위 멤버의 체크포인트로 교체
세대 N+1: 그 체크포인트를 --warmstart_ckpt로 이어 학습
```
**"gen0의 모든 멤버가 끝나야만 exploit이 가능하다"는 관찰이 정확히 맞다** — 이게 논문의
비동기·잦은 체크(2000~5000 *step*마다) 설계와 정반대인 이유다. Isaac Sim 서브프로세스를
매번 새로 띄우는 오버헤드 때문에, 논문처럼 "몇 천 step마다 가볍게 체크"가 불가능해 우리는
"한 세대=5000 *iteration* 전체"라는 훨씬 굵은 단위로만 체크한다.

**그래서 우리가 실제로 얻는 것 / 잃는 것:**
- ✅ **얻는 것**: exploit의 **메커니즘**(체크포인트 파일을 다음 학습의 warmstart로 넘기는 것)은
  GPU 개수와 무관하게 동작 — 나쁜 lineage를 좋은 lineage의 가중치로 교체하는 것 자체는 1-GPU
  에서도 실제로 일어난다(라운드1/2에서 확인).
- ❌ **잃는 것 1 — wall-clock 이점 전부**: 논문의 "N배 빨라짐"은 동시 실행을 전제라, 1-GPU
  순차는 오히려 population배만큼 **더** 걸린다(우리 4×2세대×5000it ≈ 22시간, 4-GPU면 ≈5.5시간).
- ❌ **잃는 것 2 — 잗은 체크의 이점**: 논문은 2000~5000 *step*마다 나쁜 워커를 조기에 갈아치워
  "나쁜 설정에 계산을 낭비하지 않는다"(Hyperband의 조기중단과 같은 발상). 우리는 5000
  *iteration* 전체를 다 태워야 비교하므로 — 실제로 gen0에서 m0/m1/m3가 5000it를 **전부 다
  쓰고 나서야** 평범한 성적임이 드러났다. 즉 **"계산 낭비를 막는다"는 효율성도 우리 구현에서는
  거의 실현되지 않는다.**
- 📦 **population 크기도 축소**: 원조는 16~80워커, 우리는 wall-clock 때문에 4개로 제한 —
  탐색 다양성이 훨씬 좁다.

**정정된 결론**: 사용자 지적이 맞다 — "학습 도중 덮어쓰기 = 효율적"이라는 원 논문의 주장은
**동시 다중 워커를 전제**로 하며, 우리 1-GPU 순차 구현은 그 효율성(wall-clock 단축, 조기
낭비 방지)을 **거의 못 받는다.** 우리에게 남는 PBT의 실질적 가치는 딱 하나 — **"나쁜 초기
설정에서 처음부터 재도전하는 대신, 좋은 lineage의 가중치를 이어받아 계속 다듬을 수 있다"**는
가중치 재사용(compounding) 뿐이다(Optuna는 이것조차 없음 — trial마다 매번 처음부터). 이
좁아진 장점 하나만으로 PBT를 쓸 가치가 있는지는 (a) Optuna로 이미 찾은 챔피언을 얼마나
더 다듬을 수 있는지, (b) 그 대가로 치르는 wall-clock 비용, 두 가지를 저울질해서 판단해야
한다 — 무조건 "PBT가 Optuna보다 효율적"이라 단정할 근거는 우리 환경에서는 없다.

**Optuna+PBT 결합에 대한 남은 주의**: 그럼에도 결합 자체(Optuna로 좁힌 뒤 PBT로 가중치를
이어 다듬기)는 여전히 일리 있다 — Optuna의 "제안이 똑똑함" + PBT의 "가중치 재사용"이
서로 다른 축의 장점이라서다. 단 2026-07-20 실험에서 PBT의 center를 Optuna의 최종 챔피언이
아니라 그보다 먼저 알려진 "그때까지의 최고"로 잘못 앵커링한 사고가 있었음(§ 진행 로그 참고)
— 결합이 제대로 작동하려면 PBT 시작 전 **최신 Optuna 결과로 반드시 재확인**해야 한다.

**구현 방향:**
```python
# PBT 루프 핵심 (sweep.py 확장)
population = [Agent(random_reward_weights) for _ in range(N)]
for generation in range(max_gen):
    # 1. 각 agent를 K steps 학습
    for agent in population:
        agent.train(K_steps)
    
    # 2. Exploit: 하위 20% agent → 상위 20% agent의 policy weight 복사
    for bad_agent in bottom_20:
        good_agent = random.choice(top_20)
        bad_agent.load_weights(good_agent)
    
    # 3. Explore: reward weight를 perturb
    for bad_agent in bottom_20:
        bad_agent.perturb_reward_weights(noise_scale=0.2)
```

**이걸 논문으로 가져가면:**
- "PBT for Reward Weight Discovery in Agile Wheeled-Legged Locomotion"
- Contribution: 기존에 quadruped에 적용하던 PBT를 **two-wheeled balancing robot의 동적 도약**에 
  최초 적용, Optuna baseline 대비 X% 빠른 수렴, sim-to-real 성공

---

### 📋 접근법 D: Curriculum-Aware Bi-Level Optimization (⭐ 추천)

> [!IMPORTANT]
> **RAL contribution으로 가장 유망한 방향입니다.**

> ✅ **구현 완료 (2026-07-19)** — `mdp.CurriculumWeightSchedule`
> ([curriculums.py](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/curriculums.py)):
> 선택한 reward weight/param들이 **평균 terrain level의 piecewise-linear 함수**로 stage knot
> (level 0/4/8) 사이를 보간. `--weight_schedule`로 opt-in(기본 OFF), `--adaptive_reward`와
> 조합 시 스케줄이 balancer의 base weight를 갱신해 penalty-gain이 그 위에 합성됨(bi-level).
> 타겟 문법: `"jump_lin_vel_z"`(weight) / `"stair_climb/growth"`(params). 기본 스케줄은
> level 0 = 챔피언(timing #12) 값, 상위 knot은 스윕에서 "위로 열 신호"가 나온
> jump 임펄스·growth·착지 보상을 증가. Outer-loop 탐색은
> [stair_jump_stage_sched.yaml](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/scripts/co_rl/sweeps/stair_jump_stage_sched.yaml)
> (`env.curriculum.weight_schedule.params.stages.<i>.set.<target>` 오버라이드 — train.py의
> param_overrides가 리스트 인덱스 지원하도록 확장됨).
>
> ```bash
> # 단발 실험 (챔피언 대비 A/B)
> python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo \
>     --algo ppo --headless --num_envs 8192 --adaptive_reward --weight_schedule \
>     --warmstart_ckpt logs/co_rl/Flamingo_Light_Flat_Stand_Drive/ppo/2026-07-02_12-12-49/model_1499.pt
> # outer-loop 탐색 (10 trial TPE)
> nohup python scripts/co_rl/sweep.py --config scripts/co_rl/sweeps/stair_jump_stage_sched.yaml > sweep_sched.out 2>&1 &
> ```

현재 코드에 이미 **terrain curriculum** (step height 점진 증가)이 있습니다. 
여기에 **reward weight도 curriculum과 함께 적응**시키는 프레임워크:

```
Outer: Optuna/PBT로 "초기 weight" + "curriculum stage별 weight schedule" 탐색
Inner: 
  Stage 1 (3cm stair): weight_set_1으로 학습 → 기본 hop 학습
  Stage 2 (8cm stair): weight_set_2로 전환 → 강한 점프 학습  
  Stage 3 (15cm stair): weight_set_3으로 전환 → 최대 높이 도전
  각 stage에서 AdaptiveRewardBalancer가 미세 조정
```

**왜 이게 좋은 contribution인가:**
1. **고정 weight의 한계를 명확히 보여줄 수 있음**: 3cm에 최적인 weight가 15cm에서는 실패하는 것을 
   실험으로 증명 → "reward weight는 task difficulty에 따라 달라져야 한다"
2. **기존 연구와 차별화**: ROGER는 penalty만 조절, 일반 Optuna는 고정 weight 탐색. 
   **Curriculum-coupled reward scheduling**은 둘 다 넘어서는 접근
3. **Sim-to-real narrative 완성**: curriculum stage별로 최적화된 weight가 실제 계단에서도 유효함을 
   보여주면 매우 강한 결과

---

### 📋 접근법 E: Constraint-Based RL (CaT) 활용

README에 이미 **Constraints as Termination (CaT)** 구현이 있습니다. 이걸 활용하면:

```
Reward weight 탐색 대신 constraint로 전환:
  - "base orientation은 ±15° 이내" → constraint (위반 시 terminate)
  - "base height는 0.25m 이상" → constraint
  - "stair_climb" → 유일한 maximize 대상 reward
```

이렇게 하면 탐색 공간이 극적으로 줄어들고, reward hacking 문제도 해결됩니다.

---

## 3. 핵심 참고 논문 & 방법론

### 반드시 읽어야 할 논문들

| 논문 | 핵심 내용 | 관련성 |
|------|----------|--------|
| **ROGER** (2025) | Online reward gain adaptation (penalty-gain controller) | 이미 `AdaptiveRewardBalancer`로 구현됨. 비교 baseline |
| **Chamorro et al. (2024)** "RL for Blind Stair Climbing with Wheeled-Legged Robots" | Ascento 로봇 15cm 계단 등반, position-based RL, asymmetric actor-critic | **직접 경쟁 논문** — 반드시 비교 필요 |
| **PBT** (Jaderberg et al., 2017) | Population-Based Training | reward weight를 학습 중 동적 최적화 |
| **MAML** (Finn et al., 2017) | Model-Agnostic Meta-Learning | few-shot task adaptation |
| **Reptile** (Nichol & Schulman, 2018) | 1차 근사 메타러닝 | MAML보다 구현 용이 |
| **CaT** (Constraints as Termination, 2024) | Constraint-based RL for locomotion | 이미 프로젝트에 구현됨 |
| **Statistical Reward Shaping** (2026, MDPI) | 통계 분석으로 reward 상관관계 파악 | sweep 결과 분석 방법론 |
| **TransCurriculum** (2025) | Transformer-based multi-dimensional curriculum | curriculum + reward 공동 최적화 아이디어 |
| **HDPG** (Hybrid Dynamic Policy Gradient) | Multi-head critic for per-component reward weighting | reward component별 가중치 학습 |

### 추가 참고할 방법론

1. **LLM-based Reward Design (RF-Agent, Eureka)**: LLM으로 reward function 코드 자체를 생성/수정. 
   재미있지만 RAL보다는 CoRL/NeurIPS 스타일
2. **Barrier-based Rewards**: Logarithmic barrier로 constraint를 reward에 통합. 
   CaT와 유사하지만 연속적
3. **Proprioceptive Distribution Matching**: Sim-to-real gap 줄이기. 논문의 sim-to-real 
   section에 활용 가능

---

## 4. RAL 논문 가능성 평가

### ✅ RAL에 낼 수 있는 이유

1. **하드웨어 플랫폼의 독창성**: Two-wheeled legged robot(Flamingo)은 quadruped 대비 연구가 적고, 
   점프는 더더욱 희귀함. Ascento(Chamorro et al.)가 거의 유일한 선행 연구
2. **15cm 계단 점프 + sim-to-real**: 이게 성공하면 그 자체가 RAL급 결과. Ascento 논문도 이걸로 
   냈음
3. **RL 방법론 contribution**: 단순히 "로봇이 점프했다"가 아니라, **어떻게 reward를 자동으로 
   찾았는가**를 체계적으로 보여주면 방법론 + 실험 양쪽 contribution

### ⚠️ 주의할 점

1. **Ascento와의 차별화 필수**: Chamorro et al.이 이미 "two-wheeled robot + 15cm stair + RL"을 
   했으므로, **방법론적 차이**를 명확히 해야 함
   - 그들: position-based RL + asymmetric actor-critic + 수동 reward 튜닝
   - 우리: **자동 reward weight optimization** (Optuna/PBT + adaptive balancer + curriculum)
2. **Sim-to-real이 없으면 RAL은 어려움**: simulation-only 결과로는 workshop/IROS extended 
   abstract 수준. Real robot 데모가 있어야 RAL 통과 가능성 높음
3. **Ablation 필요**: Optuna만 vs Adaptive Balancer만 vs 둘 다 vs baseline(수동 튜닝) — 
   각각의 기여를 분리해서 보여야 함

### 📊 RAL 수준의 실험 구성 (예시)

```
Table 1: Reward weight 탐색 방법 비교
───────────────────────────────────────────────────────
Method          | Max Stair | Success | GPU Hours | Reward  
                |  Height   |  Rate   |  to Find  | Robustness
───────────────────────────────────────────────────────
Manual Tuning   |   8 cm    |  45%    |   ~40h    |  Low
Grid Search     |   10 cm   |  55%    |   ~120h   |  Medium
Optuna (TPE)    |   12 cm   |  65%    |   ~60h    |  Medium
PBT             |   15 cm   |  80%    |   ~30h    |  High
Ours (PBT+Adap) |   15 cm   |  90%    |   ~25h    |  High
───────────────────────────────────────────────────────

Table 2: Sim-to-Real Transfer
───────────────────────────────────────────────
         | Sim Success | Real Success | Gap
───────────────────────────────────────────────
Manual   |    45%      |    20%       | 56%
Ours     |    90%      |    75%       | 17%
───────────────────────────────────────────────
(action smoothness 최적화가 sim-to-real gap을 줄임)
```

---

## 5. 추천 로드맵

> 📊 상세 수치·플롯·재현 커맨드는 [autotuning_results.md](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/docs/autotuning_results.md) 참고.
> 지표 = **final_mean**(terrain 곡선 마지막 20% 평균). terrain level ≈ 계단 높이: 5cm + level×1cm.

### Phase 1: Optuna Sweep 완성 ✅ (2026-07-07 ~ 07-11 완료)
- [x] sweep.py + stair_jump_example.yaml 구현 완료
- [x] 20 trial TPE sweep 실행 → 결과 분석
      — broad(isaacsim 사고로 3 trial만 생존) + focused A(12) + focused_auto B(20).
      **동일 설정에서도 12배 분산** 발견(시드/GPU 비결정) → "단일 run 비교 금지" 프로토콜 확립.
- [x] 상관관계 분석 — `sweep_analyze.py`(Spearman + PedAnova 중요도) 구현·적용
- [x] Top-3 중요 파라미터 식별 → focused sweep
      — **growth(2.4~2.6)·jump_lin_vel_z(중요도 0.96)·entropy_coef(0.007~0.009)**만 중요,
      climb_weight·base_height는 무관. focused2(10/10 완주): best 0.99 + LandingStability
      리워드로 bad_orientation 종료 **-41%**, 실패율 25%→0%.

### Phase 2: PBT 구현 ✅ 구현 / 🔄 실험 진행 중
- [x] PBT 모드 추가 (exploit/explore + checkpoint sharing)
      — `scripts/co_rl/pbt.py`(sweep 인프라 재사용, warmstart 체인, pbt_state.json 재개).
      라운드1 gen0: growth가 **상한 3.0에 클립된 채 승리** → 라운드2 경계 3.4로 확장.
- [ ] Optuna vs PBT 비교 실험 — 🔄 **라운드2(5000it×2gen×4멤버) 실행 중** (2026-07-19~,
      `_pbt/pbt_2026-07-19_15-22-09`). 완료 후 챔피언(1.344)과 동일 프로토콜로 비교.
- [x] Curriculum-coupled weight scheduling **구현** (2026-07-19, 접근법 D →
      `CurriculumWeightSchedule` + `--weight_schedule` + `stair_jump_stage_sched.yaml`)
- [ ] Curriculum-coupled weight scheduling **실험** — PBT 라운드2·capability probe 뒤 GPU 확보 시

**중간 결과(2026-07-19 기준):** 역대 챔피언 = **timing sweep #12, final_mean 1.344**
(수동 baseline 0.65 대비 +107%). 방법별 확정 효과: adaptive balancer(밴드 비교 auc t≈3.3 유의,
최악 시드가 non-adaptive 평균 수준으로 상승), 착지 보상(실패율 0), hop 트리거 기하
(step_threshold↑ = 결정적). **한계: top-5 전부 계단 6.0~6.5cm에서 정체** — 목표 15cm까지는
HP 튜닝만으론 불가 판단 → 10~15cm **capability probe**(고정 높이 6단계, 물리 한계 vs 정렬
문제 판별)가 PBT 종료 후 자동 실행 대기 중(`run_pbt_then_probe.sh`). 다음 구조 후보:
yaw-정렬 게이팅 리워드 + 양바퀴 동시 감지(§2 참고), 그리고 본 접근법 D.

### Phase 3: Sim-to-Real (6-10주)
- [ ] Best policy를 실제 Flamingo에 zero-shot transfer
- [ ] 3cm → 8cm → 15cm 실제 계단 실험
- [ ] Failure mode 분석 + fine-tuning

### Phase 4: 논문 작성 (병행)
- [ ] 실험 결과 정리 + ablation study — 🔄 부분 진행: [autotuning_results.md](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/docs/autotuning_results.md)
      (라운드별 표 + 학습곡선/밴드 플롯) + [paper_checklist.md](file:///home/lch/Isaac-RL-Two-wheel-Legged-Bot/docs/paper_checklist.md)
      작성됨. Table 1 재료(Manual/Optuna/Adaptive/Timing) 확보, PBT 열·multi-seed 확정 남음.
- [ ] RAL 포맷 논문 초안

---

## 6. 추천 논문 구조 (RAL)

```
Title: "Automated Reward Weight Discovery for Dynamic Stair Climbing 
        of a Two-Wheeled Balancing Robot via Population-Based Training"

Abstract: ...

I.   Introduction
     - Two-wheeled legged robot의 agile locomotion 과제
     - Reward engineering의 어려움 (manual tuning의 한계)
     - 우리의 contribution: PBT + adaptive balancer + curriculum

II.  Related Work
     - Wheeled-legged locomotion (Ascento, Flamingo)
     - Reward weight optimization (ROGER, Optuna, PBT)
     - Meta-learning for locomotion (MAML, Reptile)

III. Method
     A. System Overview (Flamingo + Isaac Lab)
     B. Reward Function Design (StairClimbProgress, hop, clearance)
     C. Bi-level Reward Optimization Framework
        - Outer: PBT for reward weight evolution
        - Inner: Adaptive penalty-gain controller (ROGER-style)
     D. Curriculum-Coupled Weight Scheduling

IV.  Experiments
     A. Simulation Setup (Isaac Lab, terrain curriculum)
     B. Baseline Comparisons (Manual / Grid / Optuna / PBT)
     C. Ablation Studies
     D. Sim-to-Real Transfer

V.   Results & Discussion
     - 15cm stair climbing 성공률
     - Sim-to-real gap 분석
     - Reward landscape 시각화

VI.  Conclusion
```

---

## 7. 핵심 요약 & 조언

> [!CAUTION]
> **가장 중요한 것**: "15cm 실제 계단 점프 성공"이 RAL의 핵심입니다. 
> RL 방법론은 그것을 **가능하게 만든 기술적 수단**으로 포지셔닝하세요.

### 추천 Contribution 전략 (우선순위 순)

1. **🥇 PBT + Adaptive Balancer 통합** — 구현 난이도 중, 논문 임팩트 높음
   - sweep.py를 확장하여 PBT 모드 추가
   - 학습 중 reward weight를 동적으로 진화시킴
   - Optuna 대비 수렴 속도 & 최종 성능 비교

2. **🥈 Curriculum-Coupled Weight Schedule** — 구현 난이도 중, 독창성 높음
   - Terrain curriculum stage마다 다른 reward weight 사용
   - "낮은 계단에선 탐색 중심, 높은 계단에선 정밀도 중심" 자동 전환

3. **🥉 Reptile Meta-Learning** — 구현 난이도 중-상, 논문 narrative에 유리
   - 다양한 계단 높이에 빠르게 적응하는 정책
   - "일반화" story가 강함

4. **Constraint-Based (CaT) 전환** — 구현 난이도 낮 (이미 있음), 보조 contribution
   - Reward 수를 줄이고 constraint로 전환 → 탐색 공간 축소
   - Ablation에서 "CaT를 쓰면 Optuna가 더 빨리 수렴" 보여주기

> [!NOTE]
> **현실적 조언**: 1번(PBT) + 4번(CaT 보조)을 메인으로 가져가시고, 
> sim-to-real 성공 영상을 확보하는 것이 가장 중요합니다. 
> 메타러닝은 결과가 좋으면 추가 contribution으로 넣으세요.
