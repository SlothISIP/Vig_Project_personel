# 🏭 AI-Driven Digital Twin Factory System

> **Vision Transformer 기반 제조 공정 최적화 시스템**
> Computer Vision + Digital Twin + Production Scheduling

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 프로젝트 개요

실시간 컴퓨터 비전 AI를 활용한 스마트 팩토리 디지털 트윈 시스템입니다.

### 주요 기능

- **🔍 AI 결함 검출**: Vision Transformer 기반 실시간 품질 검사
- **🏭 Digital Twin**: 공장 상태 실시간 시뮬레이션
- **📊 예측 유지보수**: ML 기반 설비 고장 예측
- **📅 스마트 스케줄링**: OR-Tools 기반 생산 계획 최적화
- **📈 실시간 대시보드**: React + Three.js 3D 시각화

---

## 🏗️ 시스템 아키텍처

```
┌─────────────┐
│  Frontend   │  React + Three.js
│  Dashboard  │  Real-time 3D Visualization
└──────┬──────┘
       │ HTTP/WebSocket
┌──────▼──────────────────────────────┐
│       FastAPI Gateway                │
│  ┌────────┬────────────┬──────────┐ │
│  │ Vision │ Digital    │ Schedule │ │
│  │ AI     │ Twin Core  │ Optimizer│ │
│  └────────┴────────────┴──────────┘ │
└──────┬──────────────────────────────┘
       │
┌──────▼────────┬──────────┬──────────┐
│ PostgreSQL    │ Redis    │ RabbitMQ │
│ (TimescaleDB) │ (Cache)  │ (Queue)  │
└───────────────┴──────────┴──────────┘
```

상세 아키텍처는 [ARCHITECTURE.md](ARCHITECTURE.md) 참조

---

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.10+
- CUDA 11.8+ (GPU 사용 시)
- Docker & Docker Compose (선택)
- Poetry (Python 패키지 관리)

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/digital-twin-factory.git
cd digital-twin-factory

# 2. Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 3. 의존성 설치
poetry install

# 4. 환경 활성화
poetry shell

# 5. 환경 변수 설정
cp .env.example .env
# .env 파일 편집하여 설정

# 6. 데이터셋 다운로드
python scripts/download_datasets.py

# 7. 데이터베이스 초기화
make db-init

# 8. 첫 모델 학습
python scripts/train_baseline.py
```

### Docker로 빠르게 시작

```bash
# 전체 스택 실행
docker-compose up -d

# 서비스 접속
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Frontend: http://localhost:3000
# - Grafana: http://localhost:3001
```

---

## 📂 프로젝트 구조

```
digital-twin-factory/
├── src/                    # 소스 코드
│   ├── vision/            # Vision AI 엔진
│   ├── digital_twin/      # Digital Twin 코어
│   ├── scheduling/        # 스케줄링 최적화
│   ├── api/               # FastAPI 서버
│   └── workers/           # Background workers
├── frontend/              # React 대시보드
├── tests/                 # 테스트 코드
├── data/                  # 데이터셋 (gitignored)
├── models/                # 학습된 모델
├── notebooks/             # Jupyter 노트북
├── deploy/                # 배포 설정
└── docs/                  # 문서

상세 구조는 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 참조
```

---

## 🔧 개발 가이드

### 개발 환경 설정

```bash
# 개발 의존성 설치
poetry install --with dev

# Pre-commit hooks 설치
pre-commit install

# 테스트 실행
make test

# 코드 포맷팅
make format

# 타입 체크
make typecheck

# 전체 검증
make check
```

### API 서버 실행

```bash
# 개발 모드 (hot reload)
uvicorn src.api.main:app --reload --port 8000

# 프로덕션 모드
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### MLflow 실험 추적

```bash
# MLflow 서버 시작
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 --port 5000

# 브라우저에서 http://localhost:5000 접속
```

### 프론트엔드 개발

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build
```

---

## 📊 모델 성능

### Baseline (Week 1)

| Model | Accuracy | F1-Score | Latency | Size |
|-------|----------|----------|---------|------|
| Swin-Tiny | 93.5% | 0.91 | 42ms | 28M |
| ViT-Base | 95.2% | 0.94 | 67ms | 86M |
| EfficientViT | 89.1% | 0.87 | 18ms | 12M |

### Optimized (Week 4+)

| Model | Format | Latency | Throughput |
|-------|--------|---------|------------|
| Swin-Tiny | ONNX FP32 | 42ms | 24 FPS |
| Swin-Tiny | ONNX FP16 | 28ms | 36 FPS |
| Swin-Tiny | TensorRT INT8 | 15ms | 67 FPS |

---

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html

# 특정 모듈만 테스트
pytest tests/unit/test_vision/

# 통합 테스트
pytest tests/integration/

# 성능 테스트
pytest tests/performance/

# E2E 테스트
pytest tests/e2e/
```

### 부하 테스트

```bash
# Locust로 API 부하 테스트
locust -f tests/performance/locustfile.py \
       --host http://localhost:8000 \
       --users 100 \
       --spawn-rate 10
```

---

## 📦 배포

### Docker 배포

```bash
# 이미지 빌드
docker-compose build

# 프로덕션 실행
docker-compose -f docker-compose.prod.yml up -d

# 로그 확인
docker-compose logs -f api
```

### Kubernetes 배포

```bash
# Namespace 생성
kubectl apply -f deploy/kubernetes/namespace.yaml

# ConfigMap & Secrets
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/secrets.yaml

# 서비스 배포
kubectl apply -f deploy/kubernetes/

# 상태 확인
kubectl get pods -n digital-twin

# 서비스 접속
kubectl port-forward svc/api 8000:8000 -n digital-twin
```

---

## 📚 API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서 확인:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 주요 엔드포인트

```bash
# 결함 검출
POST /api/v1/vision/detect
Content-Type: multipart/form-data

# Digital Twin 상태 조회
GET /api/v1/digital-twin/state

# 스케줄 최적화
POST /api/v1/scheduling/optimize

# 실시간 WebSocket
WS /api/v1/ws/stream
```

---

## 🎓 논문 & 연구

### 목표 학회 (2026)

- **IEEE CASE** (주 타겟): Automation Science and Engineering
- **ICRA**: Robotics and Automation
- **CVPR Workshop**: Computer Vision Applications

### 논문 주제

> "Vision Transformers for Real-time Manufacturing Defect Detection: A Digital Twin Approach"

### 연구 기여도

- ✅ ViT를 제조업 결함 검출에 체계적으로 적용
- ✅ Attention map 기반 설명 가능한 AI
- ✅ Digital Twin과 통합된 end-to-end 시스템
- ✅ 재현 가능한 오픈소스 벤치마크

---

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

상세 가이드: [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일 참조.

---

## 📞 문의

- **작성자**: Your Name
- **이메일**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **협업 교수**: 이덕우 교수님 (계명대학교)

---

## 🙏 감사의 글

이 프로젝트는 다음 연구와 도구를 기반으로 합니다:

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) - Google Research
- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Microsoft Research
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)

---

## 📈 로드맵

### Phase 1: Vision AI (Week 1-4) ✅ In Progress
- [x] Baseline model training
- [x] ONNX optimization
- [x] API deployment
- [ ] Attention map extraction
- [ ] Model ensemble

### Phase 2: Digital Twin (Week 5-10)
- [ ] Factory state machine
- [ ] Discrete-event simulator
- [ ] Predictive maintenance
- [ ] Real-time visualization

### Phase 3: Scheduling (Week 11-12)
- [ ] OR-Tools integration
- [ ] Multi-objective optimization
- [ ] Dynamic rescheduling

### Phase 4: Integration (Week 13-14)
- [ ] Full-stack integration
- [ ] Performance optimization
- [ ] Documentation
- [ ] Paper writing

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
