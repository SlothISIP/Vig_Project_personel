## Reinforcement Learning-based Production Scheduling

Deep Reinforcement Learning으로 생산 스케줄을 최적화하는 모듈입니다. Ray RLlib과 PPO 알고리즘을 사용하여 실시간 동적 스케줄링을 학습합니다.

## 주요 기능

- **커스텀 Gymnasium 환경**: 생산 스케줄링 문제를 강화학습 환경으로 모델링
- **PPO 에이전트**: Proximal Policy Optimization 알고리즘으로 정책 학습
- **동적 스케줄링**: 실시간으로 변화하는 공장 상태에 대응
- **다목적 보상 함수**: 완료율, 지연, 가동률, 대기시간 등 종합 최적화
- **Fallback 메커니즘**: RL 실패 시 규칙 기반 스케줄러로 대체

## 아키텍처

```
RL Scheduling Module
├── State Representation
│   ├── MachineStateRL: 기계 상태 (가동률, 큐, 건강도)
│   ├── JobStateRL: 작업 상태 (우선순위, 마감일, 대기시간)
│   └── SchedulingStateRL: 전체 공장 상태
│
├── Environment
│   └── ProductionSchedulingEnv: Gymnasium 환경
│       ├── Observation: 기계/작업 특징 벡터
│       ├── Action: (작업, 기계) 쌍 할당
│       └── Reward: 다목적 보상 함수
│
├── Training
│   └── SchedulingRLTrainer: Ray RLlib PPO 학습
│       ├── Config: 하이퍼파라미터 설정
│       ├── Train: 반복 학습
│       └── Evaluate: 성능 평가
│
└── Inference
    ├── RLSchedulingPolicy: 학습된 정책
    └── DynamicScheduler: 실시간 스케줄러
```

## 상태 공간 (Observation Space)

환경 상태는 다음 요소로 구성됩니다:

### 기계 특징 (Machine Features)
각 기계마다 5개 특징:
- `is_available`: 사용 가능 여부 (0/1)
- `utilization`: 현재 가동률 (0-1)
- `queue_length`: 대기 큐 길이 (정규화)
- `failure_probability`: 고장 확률 (0-1)
- `health_score`: 건강 점수 (0-1)

**총 기계 특징**: `num_machines × 5`

### 작업 특징 (Job Features)
상위 N개 대기 작업(기본 10개)마다 4개 특징:
- `priority`: 우선순위 (정규화, 0-1)
- `time_to_deadline`: 마감까지 남은 시간 (정규화)
- `completion_ratio`: 완료 비율 (0-1)
- `waiting_time`: 대기 시간 (정규화)

**총 작업 특징**: `10 × 4 = 40`

### 전역 특징 (Global Features)
- `total_utilization`: 전체 가동률 (0-1)
- `average_queue_length`: 평균 큐 길이 (정규화)
- `num_overdue_jobs`: 지연 작업 수 (정규화)
- `current_time`: 현재 시간 (정규화)

**총 전역 특징**: `4`

### 전체 관측 크기
```
observation_size = (num_machines × 5) + (10 × 4) + 4
                 = 3 × 5 + 40 + 4  (기본 3대 기계)
                 = 59
```

## 행동 공간 (Action Space)

**Discrete**: `(job_idx, machine_idx)` 쌍의 인덱스

```
action_index = job_idx × num_machines + machine_idx
max_actions = max_pending_jobs × num_machines + 1  (no-op 포함)
```

예: 10개 대기 작업, 3대 기계 → `10 × 3 + 1 = 31` 가능 행동

## 보상 함수 (Reward Function)

### 양의 보상 (Positive Rewards)
1. **작업 완료**: +10 per job
2. **조기 완료 보너스**: +5 × (margin / processing_time)
3. **고우선순위 완료**: +15 (우선순위 ≥ 8)
4. **가동률 향상**: +2 × Δutilization
5. **처리량**: +3 × (completed / time_delta)

### 음의 보상 (Penalties)
1. **지연 페널티**: -15 per overdue job
2. **대기시간**: -0.1 × waiting_time
3. **유휴 기계**: -1 × (0.5 - utilization) if < 50%
4. **큐 불균형**: -2 × (queue_std / queue_mean)
5. **고장 위험**: -5 × failure_probability (사용 중 기계)

### 에피소드 보너스
- 완료율 보너스: +20 × completion_rate
- 정시 배송률: +30 × on_time_rate
- 고가동률: +10 × (utilization - 0.7) if > 70%
- 다수 지연: -50 if overdue > 5

## 사용 방법

### 1. 학습 (Training)

```bash
# 기본 학습
python scripts/train_rl_scheduler.py \
    --iterations 100 \
    --num-machines 3 \
    --workers 4

# 고급 설정
python scripts/train_rl_scheduler.py \
    --iterations 200 \
    --num-machines 5 \
    --episode-duration 3000 \
    --lr 0.0003 \
    --batch-size 8000 \
    --workers 8 \
    --gpus 1 \
    --checkpoint-freq 10
```

### 2. 평가 (Evaluation)

```bash
# 학습된 모델 평가
python scripts/train_rl_scheduler.py \
    --evaluate-only \
    --resume ./checkpoints/rl_scheduling/checkpoint_000100 \
    --num-eval-episodes 20
```

### 3. 데모 (Demo)

```bash
# 데모 1: 기본 환경 시뮬레이션
python scripts/demo_rl_scheduler.py --demo 1

# 데모 2: 학습된 정책 실행
python scripts/demo_rl_scheduler.py \
    --demo 2 \
    --checkpoint ./checkpoints/rl_scheduling/checkpoint_000100

# 데모 3: 동적 스케줄러
python scripts/demo_rl_scheduler.py \
    --demo 3 \
    --checkpoint ./checkpoints/rl_scheduling/checkpoint_000100
```

### 4. Python API 사용

#### 학습

```python
from src.rl_scheduling import SchedulingRLTrainer

# Trainer 초기화
trainer = SchedulingRLTrainer(
    checkpoint_dir="./checkpoints/rl_scheduling",
    num_workers=4,
    num_gpus=0
)

# 설정 생성
config = trainer.create_config(
    num_machines=3,
    episode_duration=2000,
    learning_rate=3e-4
)

# 학습
results = trainer.train(
    num_iterations=100,
    checkpoint_freq=10,
    config=config
)

print(f"Final checkpoint: {results['final_checkpoint']}")
```

#### 추론

```python
from src.rl_scheduling import RLSchedulingPolicy, DynamicScheduler

# 정책 로드
policy = RLSchedulingPolicy(
    checkpoint_path="./checkpoints/rl_scheduling/checkpoint_000100",
    use_ray=True
)

# 스케줄러 생성
scheduler = DynamicScheduler(policy, fallback_enabled=True)

# 스케줄링 결정
from src.rl_scheduling import SchedulingStateRL

state = SchedulingStateRL(...)  # 현재 공장 상태
action = scheduler.schedule_next_job(state)

if action:
    print(f"Assign job {action.job_id} to machine {action.machine_id}")

# 통계
stats = scheduler.get_statistics()
print(f"RL decision rate: {stats['rl_decision_rate']:.2%}")
```

## 하이퍼파라미터

### 환경 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_machines` | 3 | 기계 수 |
| `max_jobs_per_episode` | 50 | 에피소드당 최대 작업 수 |
| `episode_duration` | 2000 | 에피소드 길이 (분) |
| `job_arrival_rate` | 0.05 | 작업 도착 확률 (per step) |

### PPO 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `learning_rate` | 3e-4 | 학습률 |
| `gamma` | 0.99 | 할인 계수 |
| `lambda` | 0.95 | GAE λ |
| `train_batch_size` | 4000 | 학습 배치 크기 |
| `sgd_minibatch_size` | 128 | SGD 미니배치 |
| `num_sgd_iter` | 10 | SGD 반복 횟수 |
| `clip_param` | 0.2 | PPO 클리핑 파라미터 |
| `entropy_coeff` | 0.01 | 엔트로피 계수 |

### 신경망 구조

```python
model = {
    "fcnet_hiddens": [256, 256, 128],  # 3개 은닉층
    "fcnet_activation": "relu",         # ReLU 활성화
    "vf_share_layers": False            # Value/Policy 분리
}
```

## 성능 지표

### 학습 중 모니터링

- `episode_reward_mean`: 평균 에피소드 보상
- `episode_len_mean`: 평균 에피소드 길이
- `num_env_steps_trained`: 총 학습 스텝 수
- `custom_metrics`:
  - `utilization_mean`: 평균 가동률
  - `num_completed_mean`: 평균 완료 작업 수
  - `num_overdue_mean`: 평균 지연 작업 수

### 평가 지표

- 평균 보상 (Mean Reward)
- 완료율 (Completion Rate)
- 정시 배송률 (On-time Delivery Rate)
- 평균 가동률 (Average Utilization)
- 평균 대기시간 (Average Waiting Time)
- 지연 작업 비율 (Tardiness Rate)

## TensorBoard 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir ./tensorboard/rl_scheduling --port 6006

# 브라우저에서 열기
# http://localhost:6006
```

모니터링 가능 메트릭:
- 에피소드 보상 곡선
- 정책 손실 (Policy Loss)
- 가치 손실 (Value Loss)
- 엔트로피 (Entropy)
- 가동률, 완료율 등 커스텀 메트릭

## 파일 구조

```
src/rl_scheduling/
├── __init__.py              # 모듈 exports
├── state.py                 # 상태 표현
├── reward.py                # 보상 함수
├── environment.py           # Gymnasium 환경
├── trainer.py               # Ray RLlib 학습
├── inference.py             # 추론 및 배포
└── README.md                # 이 문서

scripts/
├── train_rl_scheduler.py    # 학습 스크립트
└── demo_rl_scheduler.py     # 데모 스크립트

tests/
└── test_rl_scheduling.py    # 단위 테스트

checkpoints/rl_scheduling/   # 체크포인트 저장
tensorboard/rl_scheduling/   # TensorBoard 로그
```

## 의존성

```toml
ray = {extras = ["rllib"], version = "^2.9.0"}
gymnasium = "^0.29.0"
tensorboard = "^2.15.0"
torch = "^2.0.0"
numpy = "^1.24.0"
```

## 테스트

```bash
# 전체 테스트 실행
pytest tests/test_rl_scheduling.py -v

# 특정 테스트만 실행
pytest tests/test_rl_scheduling.py::TestEnvironment::test_env_step -v

# 커버리지 포함
pytest tests/test_rl_scheduling.py --cov=src.rl_scheduling --cov-report=html
```

## 트러블슈팅

### 1. Ray 초기화 오류

```python
# Ray가 이미 초기화된 경우
import ray
ray.shutdown()
ray.init()
```

### 2. GPU 메모리 부족

```bash
# CPU only 모드로 실행
python scripts/train_rl_scheduler.py --gpus 0 --workers 2
```

### 3. 학습이 수렴하지 않음

- Learning rate 조정: `--lr 0.0001`
- Batch size 증가: `--batch-size 8000`
- 더 긴 학습: `--iterations 500`

### 4. 보상이 음수로 수렴

- 보상 함수 가중치 조정 (reward.py의 `RewardConfig`)
- 환경 난이도 조정 (`episode_duration`, `job_arrival_rate`)

## 성능 최적화

### 병렬 학습
```bash
# Worker 수 증가
python scripts/train_rl_scheduler.py --workers 16

# GPU 사용
python scripts/train_rl_scheduler.py --gpus 1 --workers 8
```

### 배치 크기 튜닝
```bash
# 큰 배치로 안정적 학습
python scripts/train_rl_scheduler.py --batch-size 16000 --sgd-minibatch 256
```

## 향후 개선 방향

1. **Multi-agent RL**: 여러 스케줄러가 협력
2. **Curriculum Learning**: 점진적 난이도 증가
3. **Transfer Learning**: 다른 공장 설정에 전이
4. **Hierarchical RL**: 상위/하위 정책 계층 구조
5. **Model-based RL**: 환경 모델 학습 및 계획
6. **Offline RL**: 기존 데이터로 학습

## 참고 자료

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 라이선스

MIT License - 자유롭게 사용하세요!
