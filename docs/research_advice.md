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

### 📋 접근법 A: Optuna + Adaptive Balancer 통합 (현재 진행 중)

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

### 📋 접근법 B: Population-Based Training (PBT)

> [!TIP]
> sweep.py 주석에 이미 "substrate for a future Population-Based-Training loop"이라고 적혀 있습니다.
> PBT는 Optuna의 **상위 호환**이면서 논문 contribution으로 매우 적합합니다.

**PBT vs Optuna의 핵심 차이:**
- Optuna: 각 trial이 **독립적**으로 처음부터 끝까지 학습 (비효율적)
- PBT: 학습 **도중에** 성적이 나쁜 agent의 weight를 좋은 agent에서 복사(exploit) + 
  하이퍼파라미터를 변이(explore). 동일 GPU 시간 대비 훨씬 효율적

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

### Phase 1: Optuna Sweep 완성 (현재 → 2주)
- [x] sweep.py + stair_jump_example.yaml 구현 완료
- [ ] 20 trial TPE sweep 실행 → 결과 분석
- [ ] 상관관계 분석: 어떤 weight가 성능에 가장 영향을 미치는지 파악
- [ ] Top-3 중요 파라미터 식별 → focused sweep

### Phase 2: PBT 구현 (2-4주)
- [ ] sweep.py에 PBT 모드 추가 (exploit/explore + checkpoint sharing)
- [ ] Optuna vs PBT 비교 실험
- [ ] Curriculum-coupled weight scheduling 실험

### Phase 3: Sim-to-Real (6-10주)
- [ ] Best policy를 실제 Flamingo에 zero-shot transfer
- [ ] 3cm → 8cm → 15cm 실제 계단 실험
- [ ] Failure mode 분석 + fine-tuning

### Phase 4: 논문 작성 (병행)
- [ ] 실험 결과 정리 + ablation study
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
