# 🚀 Implementation Guide: Getting Started

> **"ultrathink" 모드로 설계된 프로젝트를 실제로 구현하기 위한 단계별 가이드**

이 문서는 설계 문서들을 바탕으로 실제 프로젝트를 시작하는 방법을 안내합니다.

---

## 📋 준비 사항 체크리스트

시작하기 전에 다음을 확인하세요:

### 하드웨어 요구사항
- [ ] GPU: NVIDIA GPU with 6GB+ VRAM (권장: RTX 3060 이상)
  - CPU only로도 가능하나 학습 속도가 10-20배 느림
- [ ] RAM: 16GB+ (권장: 32GB)
- [ ] 디스크: 100GB+ 여유 공간 (데이터셋 + 모델)

### 소프트웨어 요구사항
- [ ] OS: Ubuntu 20.04+ / macOS 12+ / Windows 10+ (WSL2)
- [ ] Python: 3.10 or 3.11
- [ ] CUDA: 11.8+ (GPU 사용 시)
- [ ] Git: 최신 버전
- [ ] Docker & Docker Compose (선택사항)

### 계정 준비
- [ ] GitHub 계정 (코드 버전 관리)
- [ ] Weights & Biases 계정 (선택 - 실험 추적)
- [ ] AWS/GCP 계정 (선택 - 클라우드 배포)

---

## 🎬 Step 1: 프로젝트 생성 및 초기화

### 1.1 프로젝트 디렉토리 생성

```bash
# 새 디렉토리 생성
mkdir -p ~/projects/digital-twin-factory
cd ~/projects/digital-twin-factory

# Git 초기화
git init
git branch -M main

# GitHub에 새 repository 생성 후 연결
git remote add origin https://github.com/YOUR_USERNAME/digital-twin-factory.git
```

### 1.2 설계 문서 복사

현재 `Vig_Project_personel` 폴더의 다음 문서들을 프로젝트로 복사:

```bash
# 설계 문서 복사 (이 문서들이 있는 위치에서 실행)
cp ARCHITECTURE.md ~/projects/digital-twin-factory/docs/
cp PROJECT_STRUCTURE.md ~/projects/digital-twin-factory/docs/
cp WEEK1_PLAN.md ~/projects/digital-twin-factory/docs/
cp README.md ~/projects/digital-twin-factory/
cp Makefile ~/projects/digital-twin-factory/
cp docker-compose.yml ~/projects/digital-twin-factory/
cp .env.example ~/projects/digital-twin-factory/
cp .gitignore ~/projects/digital-twin-factory/
```

### 1.3 디렉토리 구조 생성

```bash
cd ~/projects/digital-twin-factory

# PROJECT_STRUCTURE.md의 구조를 기반으로 디렉토리 생성
mkdir -p {src,tests,data,models,notebooks,config,scripts,docs,deploy,frontend}
mkdir -p src/{core,vision,digital_twin,scheduling,data,api,workers,iot,utils}
mkdir -p src/vision/{models,preprocessing,postprocessing,inference,training,utils}
mkdir -p src/digital_twin/{state,simulation,predictive,events}
mkdir -p src/scheduling/{algorithms,constraints,objectives}
mkdir -p src/data/{database,repositories,cache,storage}
mkdir -p src/api/{routes,schemas,services,middleware}
mkdir -p src/workers/{tasks,schedulers}
mkdir -p src/iot/{mqtt,amqp,simulators}
mkdir -p tests/{unit,integration,e2e,performance}
mkdir -p data/{raw,processed,annotations}
mkdir -p models/{checkpoints,onnx,tensorrt,mlflow}
mkdir -p deploy/{docker,kubernetes,terraform,ansible,monitoring}
mkdir -p config/models

# __init__.py 파일 생성
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

echo "✅ 디렉토리 구조 생성 완료"
```

### 1.4 .gitignore 생성

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
ENV/
env/
.venv

# Poetry
poetry.lock

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# PyTorch
*.pth
*.pt
*.ckpt

# Data
data/raw/
data/processed/
!data/.gitkeep

# Models
models/checkpoints/
models/onnx/
models/tensorrt/
!models/.gitkeep

# MLflow
mlruns/
mlflow.db

# DVC
.dvc/cache

# Logs
*.log
logs/

# Environment variables
.env
.env.local

# Coverage
.coverage
htmlcov/
.pytest_cache/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# ruff
.ruff_cache/

# Docker
.dockerignore

# Temporary files
*.tmp
*.bak
.cache/
EOF
```

---

## 🐍 Step 2: Python 환경 설정

### 2.1 Poetry 설치

```bash
# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# PATH 추가 (zsh 사용 시)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# bash 사용 시
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
poetry --version
```

### 2.2 pyproject.toml 생성

```bash
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "digital-twin-factory"
version = "0.1.0"
description = "AI-Driven Digital Twin Factory System with Vision Transformer"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
# Core
fastapi = "^0.104.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.4.0"
pydantic-settings = "^2.0.0"

# Vision AI
torch = "^2.1.0"
torchvision = "^0.16.0"
timm = "^0.9.0"
onnx = "^1.15.0"
onnxruntime = "^1.16.0"
opencv-python = "^4.8.0"
albumentations = "^1.3.0"
Pillow = "^10.1.0"

# Database
sqlalchemy = "^2.0.0"
asyncpg = "^0.29.0"
psycopg2-binary = "^2.9.0"
redis = "^5.0.0"
alembic = "^1.12.0"

# ML Ops
mlflow = "^2.8.0"
dvc = {extras = ["s3"], version = "^3.30.0"}
optuna = "^3.4.0"

# Scheduling
ortools = "^9.8.0"
scipy = "^1.11.0"
numpy = "^1.26.0"

# Workers
celery = {extras = ["redis"], version = "^5.3.0"}
pika = "^1.3.0"

# IoT
paho-mqtt = "^1.6.0"

# Monitoring
prometheus-client = "^0.18.0"

# Utils
python-dotenv = "^1.0.0"
pyyaml = "^6.0.0"
click = "^8.1.0"
python-multipart = "^0.0.6"
passlib = {extras = ["bcrypt"], version = "^1.7.0"}
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
pandas = "^2.1.0"
matplotlib = "^3.8.0"
seaborn = "^0.13.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
black = "^23.10.0"
ruff = "^0.1.0"
mypy = "^1.6.0"
ipython = "^8.16.0"
jupyter = "^1.0.0"
pre-commit = "^3.5.0"

[tool.poetry.group.test.dependencies]
httpx = "^0.25.0"
locust = "^2.17.0"
faker = "^19.13.0"

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "--cov=src --cov-report=html --cov-report=term -v"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF
```

### 2.3 의존성 설치

```bash
# 모든 의존성 설치 (개발 도구 포함)
poetry install

# 가상환경 활성화
poetry shell

# 설치 확인
python --version
pip list | grep torch
```

---

## 📥 Step 3: 데이터셋 준비

### 3.1 데이터 다운로드 스크립트

```bash
cat > scripts/download_datasets.py << 'EOF'
#!/usr/bin/env python3
"""Download MVTec AD and DAGM datasets"""

import urllib.request
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_mvtec_ad():
    """Download MVTec Anomaly Detection dataset"""
    base_url = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    data_dir = Path('data/raw/mvtec_ad')
    data_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading MVTec AD dataset...")
    print("⚠️  NOTE: MVTec AD requires manual download from official website")
    print(f"Please download from: {base_url}")
    print(f"Extract all categories to: {data_dir}")
    print("\nAfter download, your structure should be:")
    print("  data/raw/mvtec_ad/")
    print("    ├── bottle/")
    print("    ├── cable/")
    print("    └── ...")

if __name__ == "__main__":
    download_mvtec_ad()
EOF

chmod +x scripts/download_datasets.py
python scripts/download_datasets.py
```

**Manual Download Steps:**
1. MVTec AD 웹사이트 방문: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. 모든 카테고리 다운로드 (또는 필요한 것만)
3. `data/raw/mvtec_ad/` 에 압축 해제

### 3.2 데이터 검증

```bash
cat > scripts/verify_data.py << 'EOF'
#!/usr/bin/env python3
"""Verify dataset structure and integrity"""

from pathlib import Path

def verify_mvtec():
    data_dir = Path('data/raw/mvtec_ad')

    if not data_dir.exists():
        print("❌ MVTec AD dataset not found!")
        return False

    categories = list(data_dir.iterdir())
    print(f"✅ Found {len(categories)} categories")

    for cat in categories[:3]:  # Check first 3
        train_good = cat / 'train' / 'good'
        test_dir = cat / 'test'

        if train_good.exists() and test_dir.exists():
            n_train = len(list(train_good.glob('*.png')))
            print(f"  {cat.name}: {n_train} training images")
        else:
            print(f"  ❌ {cat.name}: Invalid structure")
            return False

    return True

if __name__ == "__main__":
    if verify_mvtec():
        print("\n✅ Dataset verification passed!")
    else:
        print("\n❌ Dataset verification failed!")
EOF

python scripts/verify_data.py
```

---

## 🏋️ Step 4: 첫 모델 학습 (Day 1-2 작업)

### 4.1 기본 모델 코드 생성

```bash
# Vision 모델 구현
cat > src/vision/models/swin_transformer.py << 'EOF'
import torch
import torch.nn as nn
import timm

class SwinDefectDetector(nn.Module):
    """Swin Transformer for defect detection"""

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()

        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove head
        )

        # Custom classification head
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
EOF

# 테스트
python -c "
from src.vision.models.swin_transformer import SwinDefectDetector
model = SwinDefectDetector()
print(f'✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

### 4.2 학습 스크립트 (간단 버전)

```bash
cat > scripts/train_simple.py << 'EOF'
#!/usr/bin/env python3
"""Simple training script for testing setup"""

import torch
from src.vision.models.swin_transformer import SwinDefectDetector

def main():
    print("🚀 Starting simple training test...")

    # Create model
    model = SwinDefectDetector(num_classes=2, pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"✅ Model on device: {device}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✅ Forward pass successful: {output.shape}")
    print("\n✅ Setup verification complete!")
    print("\nNext steps:")
    print("1. Download dataset (see WEEK1_PLAN.md)")
    print("2. Run full training: python scripts/train_baseline.py")

if __name__ == "__main__":
    main()
EOF

python scripts/train_simple.py
```

---

## 🐳 Step 5: Docker 환경 설정 (선택사항)

### 5.1 Dockerfile 생성

```bash
mkdir -p deploy/docker

cat > deploy/docker/Dockerfile.api << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

### 5.2 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 비밀번호 변경 (중요!)
sed -i '' 's/your_secure_password_here/MySecurePass123!/g' .env
sed -i '' 's/your_rabbitmq_password_here/RabbitMQPass123!/g' .env
sed -i '' 's/your_minio_password_here/MinIOPass123456!/g' .env
```

### 5.3 Docker Compose로 전체 스택 실행

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f api

# 서비스 상태 확인
docker-compose ps

# 접속 테스트
curl http://localhost:8000/health
```

---

## ✅ Step 6: 설정 완료 검증

모든 설정이 완료되었는지 확인:

```bash
cat > scripts/verify_setup.sh << 'EOF'
#!/bin/bash

echo "🔍 Verifying setup..."

# Check Python
if python --version | grep -q "3.10\|3.11"; then
    echo "✅ Python version OK"
else
    echo "❌ Python version incorrect"
fi

# Check Poetry
if poetry --version &> /dev/null; then
    echo "✅ Poetry installed"
else
    echo "❌ Poetry not found"
fi

# Check CUDA (if available)
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA available"
else
    echo "⚠️  CUDA not available (CPU only)"
fi

# Check directory structure
if [ -d "src" ] && [ -d "tests" ] && [ -d "data" ]; then
    echo "✅ Directory structure OK"
else
    echo "❌ Directory structure incomplete"
fi

# Check dependencies
if poetry run python -c "import torch, timm, fastapi" &> /dev/null; then
    echo "✅ Key dependencies installed"
else
    echo "❌ Dependencies missing"
fi

echo ""
echo "🎉 Setup verification complete!"
EOF

chmod +x scripts/verify_setup.sh
./scripts/verify_setup.sh
```

---

## 🚀 다음 단계

설정이 완료되었다면 `WEEK1_PLAN.md`를 따라 개발을 시작하세요:

### Day 1 작업:
1. ✅ 환경 설정 (완료!)
2. 📥 데이터셋 다운로드
3. 📊 데이터 탐색 (Jupyter Notebook)
4. 🔧 전처리 파이프라인 구현

### Day 2 작업:
1. 🏋️ Baseline 모델 학습
2. 📈 MLflow 실험 추적
3. 📊 모델 평가

### Day 3 작업:
1. ⚡ ONNX 최적화
2. 🌐 FastAPI 엔드포인트
3. 🧪 성능 벤치마킹

---

## 📚 추가 리소스

- **Architecture**: `docs/ARCHITECTURE.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Week 1 Plan**: `docs/WEEK1_PLAN.md`
- **API Documentation**: http://localhost:8000/docs (after starting server)
- **MLflow UI**: http://localhost:5000 (after starting MLflow)

---

## 🆘 문제 해결

### CUDA out of memory
```bash
# batch size 줄이기
# scripts/train_baseline.py에서:
batch_size = 16  # 32에서 16으로
```

### Poetry install 실패
```bash
# Cache 삭제 후 재시도
poetry cache clear pypi --all
poetry install
```

### Docker 빌드 실패
```bash
# 캐시 없이 재빌드
docker-compose build --no-cache
```

### 데이터셋 다운로드 실패
```bash
# 수동으로 다운로드 후 압축 해제
# https://www.mvtec.com/company/research/datasets/mvtec-ad
```

---

**🎉 축하합니다! 모든 설정이 완료되었습니다.**

이제 `WEEK1_PLAN.md`를 따라 프로젝트 개발을 시작하세요!
