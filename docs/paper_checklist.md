# 📝 RAL 논문 준비 체크리스트 — Flamingo Stair-Jump

> 목표 논문(가제): *"Automated Reward Weight Discovery for Dynamic Stair Climbing of a
> Two-Wheeled Balancing Robot"* — [research_advice.md](research_advice.md)의 로드맵 실행 문서.
> 상태 갱신하며 사용. (마지막 갱신: 2026-07-12)

---

## 0. 한 줄 스토리 (항상 이것으로 되돌아올 것)

> **"이륜 밸런싱 로봇이 15cm 계단을 지각-트리거 점프로 오른다. 그걸 가능하게 한 reward는
> 사람이 아니라 bi-level 자동 탐색(Outer TPE + Inner online balancer)이 찾았다."**

- 핵심 결과물 = **실기 15cm 등반 영상**. 방법론은 그걸 만든 수단으로 포지셔닝.
- 직접 경쟁: Chamorro et al. 2024 (Ascento, blind 15cm, 수동 reward 튜닝).
  차별점 = ①지각 트리거 discrete hop ②reward 자동 탐색 ③커리큘럼 진행률(auc) meta-objective.

---

## 1. 방법론 체크리스트 (구현 상태)

| 구성요소 | 상태 | 위치 / 비고 |
|---|---|---|
| 2-phase 학습 (stand_drive → stair_jump warm-start) | ✅ | flat warmstart가 안정 확정 (2026-07 실험) |
| StairClimbProgress (지수 등반 보상, non-farmable) | ✅ | stair_rewards.py |
| 지각-트리거 hop (StairDetectEventCommand, 배포가능) | ✅ | stair_event_command.py |
| LandingStability (착지 안정 보상) | ✅ | 2026-07-10 추가. bad_orient 종료 -41% 실증 |
| Terrain-level 커리큘럼 (step height 5→15cm, 10 rows) | ✅ | stair_terrain_cfg.py + stair_terrain_levels_climb |
| Optuna TPE sweep 인프라 (`sweep.py` + `--param_overrides`) | ✅ | 재개/시딩/미러링/ABORT 안전장치 포함 |
| Sweep 분석 도구 (상관/중요도/focused 자동생성) | ✅ | sweep_analyze.py |
| AdaptiveRewardBalancer (ROGER식, 옵션 `--adaptive_reward`) | ✅ | curriculums.py. 단독 사용 시 farming 붕괴 실증(§3) |
| Forward-progress 게이팅 (옵션 `--forward_gate`) | ✅ | jump_rewards.py, 기본 OFF |
| **Bi-level sweep (Outer TPE × Inner balancer)** | 🔲 | `sweeps/stair_jump_adaptive.yaml` 준비됨 — 실행 대기 |
| PBT 모드 (exploit/explore) | 🔲 | Phase 2. sweep.py 위에 checkpoint 공유로 확장 |
| Curriculum-coupled weight schedule | 🔲 | Phase 2 후보 (접근법 D) |

## 2. 실험 체크리스트 (표/그림이 되는 것들)

### 2.1 메인 비교표 (Table 1: 탐색 방법 비교)
같은 예산(GPU-hours) 기준, 지표 = terrain auc + 최종 level + 실패율:

- [x] **Manual baseline** — 07-03_17-37-58 (terrain 0.672@5000it) ✅ 확보
- [x] **Optuna broad → focused** — 3라운드 완료. 최고 sweep003 (auc 2521, terrain max 1.047) ✅
- [ ] **Optuna + Adaptive (bi-level)** — `stair_jump_adaptive.yaml` 8 trials 실행
- [ ] **Adaptive 단독** — 이미 1 run 있음(farming 붕괴, terrain 0.00) — negative result로 표에 포함
- [ ] **PBT** (Phase 2) — 구현 후 동일 예산 비교
- [ ] (선택) Grid/Random 대비 TPE 효율 — 기존 study 데이터로 재구성 가능

### 2.2 통계적 엄밀성 ⚠️ 현재 최대 약점
- [ ] **Multi-seed 검증**: best 설정 × seed {42, 123, 7} × 5000it — **아직 0회**.
      동일-설정 12배 분산(#14 vs #15)을 발견했으므로 리뷰어가 반드시 물음.
      모든 표는 mean±std (n≥3)로 보고할 것.
- [ ] 각 비교 조건도 최소 2-3 seed (예산 부족 시 상위 2개 방법만이라도)

### 2.3 Ablation (Table 2)
- [x] LandingStability 유/무 — 이미 데이터 있음 (07-03 vs sweep003: bad_orient 0.069→0.041) — 단 weight 외 변수도 다르므로 **통제 재실험 1회 필요**
- [ ] LandingStability만 끈 best 설정 1 run (완전 통제 ablation)
- [ ] penalty_budget / g_min 민감도 — adaptive sweep 결과에서 도출
- [ ] promote_steps 3 vs 5 — broad sweep 데이터 일부 재활용 가능
- [ ] (선택) forward_gate 유/무

### 2.4 최종 성능 run
- [ ] **best 설정 8000-10000it 긴 학습** (`best_focused2.json` 준비됨) → 최종 terrain level / 15cm 도달 여부
- [ ] Play 재현 + 영상 캡처 (성공/실패 모드 시각화)

### 2.5 Sim-to-Real (Phase 3 — RAL 통과의 관건)
- [ ] Best policy zero-shot 실기 이식 (policy obs가 실센서만 쓰는지 최종 감사)
- [ ] 3cm → 8cm → 15cm 실제 계단, 각 높이 성공률 (n≥10 시도)
- [ ] Failure mode 기록 (영상 + 분류)
- [ ] Action smoothness / 토크 피크 비교 (sim vs real) — sim2real gap 서사

## 3. 논문에 쓸 "발견"들 (이미 확보된 재료)

1. **Farming 붕괴 (negative result, 중요)**: 페널티를 온라인으로 완화하면(adaptive 단독)
   reward는 98→120으로 오르는데 terrain은 0.67→0.00으로 붕괴. 이 태스크의 자세 페널티는
   안전 제약이 아니라 **제자리-farming을 막는 구조적 발판**임을 실증. → bi-level에서
   g_min 하한을 높게 잡은 근거.
2. **착지 보상의 효과**: 이륙 보상 3개만으로는 착지가 우연에 방치됨. LandingStability 추가로
   bad_orientation 종료 -41%, 실패 trial 25%→0%.
3. **파라미터 중요도**: growth(등반 보상의 지수 성장률)와 hop 임펄스 weight가 지배적,
   등반 보상의 절대 크기는 무관 — "보상의 모양 > 크기".
4. **학습 분산**: 동일 설정 12배 차이 → single-run 비교의 위험성, multi-seed 필수 근거.
5. **auc-of-terrain-level meta-objective**: 커리큘럼 진행 속도 자체를 HPO 목적함수로 쓰면
   farming에 강건한 탐색이 됨 (reward 크기를 목적으로 쓰면 속음).

## 4. 글쓰기 체크리스트

- [ ] 관련연구 정리: Ascento(Chamorro'24) / ROGER / PBT(Jaderberg'17) / ADD / CaT
      — 각각 "우리와 뭐가 다른가" 한 문단씩
- [ ] Method 그림: 시스템 다이어그램 (2-phase + bi-level 루프 + 커리큘럼)
- [ ] Reward 표: 전체 term, weight, farmable 여부, 게이팅
- [ ] 그림: terrain-level 학습곡선 (방법별, seed 밴드), penalty_gain 궤적,
      파라미터 중요도 bar, 실기 스냅샷 시퀀스
- [ ] 재현성: sweep yaml + seed + 코드 공개 준비 (RAL 권장)
- [ ] RAL 포맷 (8p, LaTeX) 초안 골격

## 5. 지금 당장의 순서 (우선순위)

1. `best_focused2.json`으로 **8000it 긴 학습** (GPU ~4.5h) → 최종 성능 확인
2. **adaptive sweep 8 trials** (`stair_jump_adaptive.yaml`, ~23h) → Table 1의 bi-level 행
3. **multi-seed 3 runs** (best 설정) → 모든 주장의 통계적 기반
4. LandingStability 통제 ablation 1 run
5. 결과가 terrain ≥ 5 (10cm+)에 도달하면 → Phase 3 (sim2real) 착수
6. 정체되면 → 지형 계단 수 확대(현재 타일당 3-4개) 또는 PBT(Phase 2)로 전환
