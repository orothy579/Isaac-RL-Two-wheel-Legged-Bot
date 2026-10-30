# H200 이전 및 Flamingo 동작 참조 조사 — 2026-10-29

## 현재 실행과 이전 경로

- 원본: `/home/midbar/ws/Isaac-RL-Two-wheel-Legged-Bot` (aarch64 GB10).
- 대상: `work@proxy2.ainexus.ktcloud.com:10537` (x86_64 H200 1장, 143771 MiB; 2026-10-30 접속 갱신).
- 사용자가 확정한 영구 저장소: **`/home/work/postech`**.
- 독립 작업 폴더: `/home/work/postech/flamingo_light`.
- 프로젝트: `flamingo_light/Isaac-RL-Two-wheel-Legged-Bot`.
- Python·venv·패키지 캐시·결과도 모두 `flamingo_light/` 안에 둔다.
- 기존 `dataset`, `lerobot`, `training`, `outputs` 등은 건드리지 않는다.
- 원본은 보존하고 복사한다. 현재 훈련 중인 파일을 이동하거나 변경하지 않는다.
- 비밀번호는 스크립트에 저장하지 않는다. 인증된 SSH 제어 연결을 재사용한다.

`scripts/co_rl/h200/after_current.py`는 현재 `flamingo-two-stage.service`의
감독 프로세스가 종료되고 다른 co_rl 훈련 프로세스도 사라질 때까지 기다린다.
단순히 1단계 Python 프로세스가 종료되는 것을 전체 완료로 취급하지 않는다.
새 2단계 실행의 `batch_complete.json`, 요청한 5000 iteration, 최종 checkpoint와
감독 로그의 정상 종료를 함께 확인한 다음 복사한다. 실패한 훈련에서 자동으로
다음 작업을 시작하지 않는다. 큐는 CPU 감시와 SSH 연결 유지 이외에는 작업하지 않는다.

```bash
# 이미 실행 중인 큐가 없는 경우에만 실행. PID는 실제 감독 PID로 바꾼다.
systemd-run --user --unit=flamingo-h200-after-current \
  --working-directory=/home/midbar/ws/Isaac-RL-Two-wheel-Legged-Bot \
  /usr/bin/python3 -u scripts/co_rl/h200/after_current.py --wait-pid 2891702

systemctl --user status flamingo-h200-after-current
journalctl --user -u flamingo-h200-after-current -n 30
cat logs/co_rl/_runlogs/h200_queue/status.json
```

큐가 시작하는 원격 작업은: 전체 프로젝트·PDF·USD·로그·체크포인트 복사 →
최종 checkpoint SHA-256 확인 → x86_64 환경 신규 설치 → 실제 태스크 64환경 smoke test →
4096/8192/16384/32768/65536환경에서 각 30 iteration 측정 → 선택한 크기로 5000 iteration
추가 학습이다. 대규모 실행은 현재 2단계 최종 정책을 warm start한다.
`docs/h200_handoff.md` 1번의 blind student PPO 이어 학습도 큐에 포함한다.
설치 후 `dispatch.py`가 GPU 수를 조회하여 1장이면 Gain 다음 student,
2장 이상이면 서로 다른 GPU에서 동시에 실행한다. student는 지정된 DAgger checkpoint,
policy stacks 49, critic stacks 2, std scale 1.0, 3000 iteration으로 실행하며
별도의 smoke test와 환경 수 벤치마크를 거친다. 명령 0 수정은 기존 소스를 그대로 쓴다.
65536은 목표 상한이며 성공한 측정 중 최고 처리량의 90% 이상인 가장 큰 크기를 선택한다.
환경 수에 맞춰 PPO minibatch와 PhysX buffer를 조정하고 메모리 여유를 감시한다.
30 iteration 통과는 장기 안정성 보장이 아니다. 최종 결과도 완료 마커와 유한한 weight로 확인한다.

**중요한 제한:** 이 자동 작업은 기존 PPO의 규모 확장 실험이다. 아래 ACP, 접촉 planner,
reference tracking, 검증된 reset bank가 구현되어 있다고 주장하지 않는다.
애니메이션 GT를 확보했다거나 두 논문을 재현했다고 볼 수 없다.

## H200 호환성

현재 소프트웨어는 Isaac Sim 5.1.0 / Isaac Lab 2.3.2.post1 / Torch 2.7.0 CUDA 12.8이다.
Isaac Sim 5.1 공식 요구사항은 RT Core 없는 GPU를 지원하지 않는다.
H200의 CUDA 메모리가 크다는 사실만으로 Isaac Sim 실행을 보장할 수 없다.
GPU를 쓰는 실제 Flamingo 물리/PPO rollout을 완료해야 다음 단계로 진행한다.
기본 headless가 실패하면 렌더러 비활성화로 한 번 더 검증하며, 둘 다 실패하면
로그와 실패 상태를 남긴다. 호환성 오류를 환경 수 축소로 숨기지 않는다.
영상 렌더링은 지원 GPU가 있는 로컬 장비에서 별도로 수행하는 경로가 적합하다.

원격 상태 파일: `bootstrap_status.json`, `training_status.json`, `bootstrap.log`,
`benchmark_results.json`, `h200_*.log`. SSH 세션이나 서버가 소멸하면 큐/작업이 실패할 수 있다.
두 작업의 배정·결과는 `job_queue_status.json`, student 상태는
`student_training_status.json`, 환경 수 측정은 `student_benchmark_results.json`에 남긴다.

공식 근거: [Isaac Sim 5.1 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
[NVIDIA headless 실행 안내](https://github.com/isaac-sim/IsaacSim/blob/develop/skills/isaac-sim-headless-deployment/SKILL.md).
후자는 최신 소스의 안내이며 현재 5.1에서의 실행 성공 증거는 아니다.

## 두 논문에서 실제로 한 일

### Humanoid Mobility via Attention-based Contacts

지정된 로컬 PDF의 Table S1: 계단·사다리·앉기 동작은 Mixamo에서 얻고 로봇에
리타게팅한 뒤 접촉 라벨을 생성했다. 사람 애니메이션의 관절값을 로봇에 그대로 넣지 않았다.
Methods의 리타게팅은 프레임별 nonlinear IK로 관절 제한, 지면 관통, 장애물 거리와
시간적 연속성을 다룬다. 이는 토크까지 동역학적으로 실행 가능하다는 독립적인 보증은 아니다.
접촉 tracker 사전학습 → attention contact planner 학습 → 참조 외 지형에서 RL fine-tuning이다.
배포 시 reference trajectory는 입력하지 않는다. Table S6의 환경 수는 단계별로
worker당 12000/2400/2100이며, 이 논문의 설정이 65536환경은 아니다.

[논문](https://arxiv.org/abs/2610.31608)

### Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning

OmniReset은 사람 시연 없이 reaching/near-object/grasped/near-goal 등 다양한 초기 상태를
오프라인 생성하고 충돌 검사와 짧은 물리 안정화로 유효성을 검사한다.
검증된 상태를 GPU에 캐시하고 대규모 PPO에 제공한다. Figure 8은 4096/16384/65536 환경을
비교한다. 환경 수 확대와 reset 다양성을 함께 쓰는 것이 핵심이다.
논문의 manipulation reset을 Flamingo 계단에 그대로 적용할 수는 없다.

[프로젝트](https://omnireset.github.io/)

## NVIDIA 보조 도구와 직접 애니메이션 제작

| 도구 | 실제 기능 | Flamingo에 대한 한계 |
| --- | --- | --- |
| Isaac Sim Physics Inspector | joint state / drive target 슬라이더로 관절과 자세 확인 | 자동 계단 보행 합성기가 아님 |
| Gain Tuner | 관절 stiffness/damping 설정과 sinusoidal/step 추종 시험 | 동작 시나리오를 생성하지 않음 |
| Lula / cuMotion / cuRobo | waypoint 보간, IK, 충돌 회피 경로·궤적 생성 | 주로 manipulator용. 부유 base·구름 접촉·균형·닫힌 링크를 포함한 Flamingo 보행 문제를 바로 해결하지 않음 |
| Kimodo | 문장, sparse keyframe, 손발 위치·경로 조건으로 동작 생성 | 공개 모델은 SOMA / G1 / SMPL-X. 임의의 URDF 입력 기능이 아님 |
| MotionBricks | 최근 4자세와 목표 1~4자세 사이 동작과 시간 생성. 출력 30Hz | 공개 모델은 SOMA / G1. Flamingo용 모델이나 자동 변환을 확인하지 못함 |

Isaac Sim의 관절 제어 기능과 AI 애니메이션 생성기를 구분해야 한다.
Blender에서 USD를 불러와 시각적으로 움직이는 것도 가능하지만, skeletal animation은
PhysX articulation의 joint command나 실행 가능한 rollout과 별개이다.
Mixamo 업로드 포맷은 FBX/OBJ/ZIP이며 auto-rigger는 사람형용이다. URDF/USD를 직접
올려 Flamingo의 물리 관절을 보존한 채 자동 애니메이션을 생성하는 경로는 아니다.

근거:

- [Physics Inspector와 슬라이더](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/extensions/ux/source/omni.physx.supportui/docs/dev_guide/authoring_tools.html)
- [Gain Tuner 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html)
- [Trajectory Generation](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/concepts/trajectory_interface.html)
- [cuMotion](https://github.com/nvidia-isaac/cumotion/)
- [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/)
- [Kimodo 지원 골격](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/skeleton.html)
- [MotionBricks](https://github.com/NVlabs/motionbricks)
- [Mixamo 업로드](https://helpx.adobe.com/creative-cloud/help/mixamo-rigging-animation.html)
- [Mixamo 사람형 제한](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)

## Flamingo에서 수작업을 줄이는 제안

매 프레임의 관절값을 사람이 만들 필요는 없다. 현재 로봇은 어깨 2개 + 바퀴 2개를
능동 제어하며, 다리는 4절 링크로 바퀴를 거의 수직 이동시킨다. 따라서 사람의
발을 앞뒤로 벌리는 보행을 복제하기보다 바퀴 접촉 순서와 높이를 설계해야 한다.
현재 계단 태스크의 기하 분석상 두 바퀴가 정면으로 나란히 막히면 한 바퀴를
올리기 어려워 약 17도 이상의 사선 접근이 필요하다 (`docs/stair_step_zmp.md`).

초기 설계안은 한 단에 6~8개의 주요 상태다. 이는 측정된 최적 개수가 아니다:

1. 사선 접근, 양쪽 접촉.
2. 뒤에 남을 바퀴로 지지 하중 이동.
3. 선행 바퀴 상승, 모서리 여유 확보.
4. 선행 바퀴가 위 tread에 접촉.
5. 위 바퀴로 지지 하중 이동.
6. 나머지 바퀴 상승·안착, 양쪽 지지로 회복.

사람은 원하는 접촉과 대략적인 높이·방향을 선택하고, 관절값·시간·바퀴 속도는
보간과 물리 rollout 기반 최적화로 채우는 구성이 적합하다. 키프레임은 이벤트 기준으로
두며 일률적으로 10프레임마다 만들 필요가 없다. 예를 들어 3초 동작을 영상 30fps로
보면 90프레임이지만 편집할 목표는 6~8개일 수 있다. 현재 태스크는 제어 50Hz와
물리 200Hz를 사용하므로 제어 150샘플·물리 600스텝은 자동 생성한다.
단순 보간은 동역학적 타당성을 보장하지 않으므로 토크·관절 제한·접촉·관통을 검증한다.

기존 성공 정책에서 상태·접촉을 기록해 이 주요 상태의 초기 후보를 얻으면
모든 자세를 처음부터 만들지 않아도 된다. 좌우 반전, yaw·riser·진행 속도 변형으로
유효 후보를 늘릴 수 있다. 성공 궤적과 물리적으로 안정한 중간 상태만 검증된
reference/reset bank로 사용하는 다음 실험이 두 논문의 아이디어를 연결한다.
이 파이프라인은 제안이며 현재 큐에 구현된 기능이 아니다.

저장소의 `scripts/co_rl/stair_oracle/waypoint_editor.html`은 기존 오프라인 관절 편집기다.
해당 v1 replay는 과거 v03 자산과 정면 점프용이므로 현재 v01_2_2의 사선 한 바퀴씩
오르기에 바로 사용하면 안 된다. `record_stair_step.py --trace`는 현재 정책의
시계열 분석용이며 완전한 reference/reset 데이터셋 exporter는 아니다.
