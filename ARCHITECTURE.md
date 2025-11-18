# AI-Driven Digital Twin Factory System - System Architecture

## 📐 Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production Line (Physical)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Camera 1 │  │ Camera 2 │  │ IoT      │  │ PLC      │        │
│  │ (Vision) │  │ (3D)     │  │ Sensors  │  │ Control  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                          │
                    [MQTT/AMQP]
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼───────┐                  ┌────────▼────────┐
│  Edge Device  │                  │  Message Queue  │
│  (Optional)   │                  │  (RabbitMQ)     │
│               │                  │                 │
│ - TensorRT    │                  │ - Data Buffer   │
│ - ONNX        │                  │ - Load Balance  │
│ - Local Infer │                  └────────┬────────┘
└───────────────┘                           │
                                            │
┌───────────────────────────────────────────┴──────────────────────┐
│                       BACKEND SERVICES                            │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────┐  ┌────────────────────┐                 │
│  │  Vision AI Engine  │  │  Digital Twin Core │                 │
│  ├────────────────────┤  ├────────────────────┤                 │
│  │ • ViT Defect Det.  │  │ • Factory Sim      │                 │
│  │ • Swin Transformer │  │ • State Management │                 │
│  │ • 3D Reconstruction│  │ • Physics Engine   │                 │
│  │ • Real-time Infer  │  │ • Event Handler    │                 │
│  └─────────┬──────────┘  └──────────┬─────────┘                 │
│            │                        │                            │
│  ┌─────────▼────────────────────────▼─────────┐                 │
│  │         FastAPI Gateway                    │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │                 │
│  │  │ Auth     │  │ WebSocket│  │ REST API │ │                 │
│  │  │ Middleware│  │ Handler  │  │ Endpoints│ │                 │
│  │  └──────────┘  └──────────┘  └──────────┘ │                 │
│  └───────────────────┬────────────────────────┘                 │
│                      │                                           │
│  ┌───────────────────▼────────────────────────┐                 │
│  │         AI/ML Pipeline                     │                 │
│  ├────────────────────────────────────────────┤                 │
│  │ • Training Pipeline (MLflow)               │                 │
│  │ • Model Registry                           │                 │
│  │ • A/B Testing                              │                 │
│  │ • Data Versioning (DVC)                    │                 │
│  └────────────────────────────────────────────┘                 │
│                                                                   │
│  ┌───────────────────────────────────────────┐                  │
│  │  Scheduling & Optimization Engine         │                  │
│  ├───────────────────────────────────────────┤                  │
│  │ • OR-Tools Scheduler                      │                  │
│  │ • Genetic Algorithm                       │                  │
│  │ • Constraint Solver                       │                  │
│  │ • What-if Simulator                       │                  │
│  └───────────────────────────────────────────┘                  │
│                                                                   │
└───────────────────────────┬───────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐           ┌─────────▼──────────┐
    │  Data Layer    │           │  Cache Layer       │
    ├────────────────┤           ├────────────────────┤
    │ PostgreSQL     │           │ Redis              │
    │ • Time-series  │           │ • Session          │
    │ • Metadata     │           │ • Real-time Data   │
    │ • User/Auth    │           │ • Pub/Sub          │
    └────────────────┘           └────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐           ┌─────────▼──────────┐
    │  Object Store  │           │  Monitoring        │
    │  (MinIO/S3)    │           │  (Prometheus)      │
    ├────────────────┤           ├────────────────────┤
    │ • Model Blobs  │           │ • Metrics          │
    │ • Images       │           │ • Alerts           │
    │ • Videos       │           │ • Logs (Loki)      │
    └────────────────┘           └────────────────────┘
                            │
┌───────────────────────────┴───────────────────────────────────────┐
│                       FRONTEND LAYER                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                React Dashboard (SPA)                       │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ Real-time   │  │ 3D Digital  │  │ Analytics       │   │  │
│  │  │ Monitoring  │  │ Twin View   │  │ Dashboard       │   │  │
│  │  │             │  │             │  │                 │   │  │
│  │  │ • Live Feed │  │ • Three.js  │  │ • Charts (D3)   │   │  │
│  │  │ • Alerts    │  │ • Factory   │  │ • KPI Metrics   │   │  │
│  │  │ • Status    │  │   Layout    │  │ • Reports       │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ Defect      │  │ Scheduling  │  │ Admin Panel     │   │  │
│  │  │ Analysis    │  │ Optimizer   │  │                 │   │  │
│  │  │             │  │             │  │                 │   │  │
│  │  │ • Attention │  │ • Gantt     │  │ • User Mgmt     │   │  │
│  │  │   Maps      │  │ • What-if   │  │ • Model Config  │   │  │
│  │  │ • History   │  │ • Simulator │  │ • System Logs   │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components Deep Dive

### 1. Vision AI Engine Architecture

```
Vision AI Pipeline:
┌─────────────┐
│ Raw Image   │ (1920x1080 RGB)
└──────┬──────┘
       │
┌──────▼──────────┐
│ Preprocessor    │
├─────────────────┤
│ • Resize        │
│ • Normalize     │
│ • Augmentation  │
└──────┬──────────┘
       │
┌──────▼──────────────────┐
│ Model Router            │
├─────────────────────────┤
│ if edge_device:         │
│   → ONNX/TensorRT       │
│ else:                   │
│   → PyTorch (GPU)       │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│ Backbone Selection      │
├─────────────────────────┤
│ • Swin-T (Fast)         │ ← Default
│ • ViT-B (Accuracy)      │
│ • EfficientViT (Edge)   │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│ Detection Head          │
├─────────────────────────┤
│ • Classification        │
│ • Segmentation          │
│ • Bounding Box          │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│ Post-processing         │
├─────────────────────────┤
│ • NMS                   │
│ • Confidence Filter     │
│ • Attention Extraction  │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│ Output                  │
├─────────────────────────┤
│ {                       │
│   defect_type: str,     │
│   confidence: float,    │
│   bbox: [x,y,w,h],      │
│   attention_map: array, │
│   inference_time: ms    │
│ }                       │
└─────────────────────────┘
```

**Key Technical Decisions:**

1. **Swin Transformer as Default**
   - Hierarchical structure → better for multi-scale defects
   - Window attention → 4x faster than vanilla ViT
   - Proven on ImageNet (87.3% top-1)

2. **ONNX Runtime Optimization**
   ```python
   # Target metrics:
   - FP32: ~50ms per image (acceptable)
   - FP16: ~25ms per image (optimal)
   - INT8: ~15ms per image (edge target)
   ```

3. **Attention Map Extraction**
   - Last layer attention → defect localization
   - GradCAM backup for CNN baselines
   - Real-time visualization for operators

---

### 2. Digital Twin Core Architecture

```
State Management:
┌────────────────────────────────────────┐
│        Factory State Machine           │
├────────────────────────────────────────┤
│                                        │
│  Current State = {                     │
│    machines: {                         │
│      machine_id: {                     │
│        status: RUNNING|IDLE|ERROR,     │
│        current_job: job_id,            │
│        health_score: float,            │
│        last_maintenance: timestamp     │
│      }                                 │
│    },                                  │
│    materials: {                        │
│      material_id: {                    │
│        quantity: int,                  │
│        location: str,                  │
│        quality_grade: A|B|C            │
│      }                                 │
│    },                                  │
│    jobs: [job queue],                  │
│    kpis: {                             │
│      oee: float,                       │
│      defect_rate: float,               │
│      throughput: units/hour            │
│    }                                   │
│  }                                     │
│                                        │
└────────────────────────────────────────┘
         │
         │ State Updates (Event-driven)
         │
┌────────▼────────────────────────────────┐
│      Event Processor                    │
├─────────────────────────────────────────┤
│                                         │
│ handle_defect_detected(event):          │
│   • Update machine health               │
│   • Trigger alert                       │
│   • Log to time-series DB               │
│                                         │
│ handle_job_completed(event):            │
│   • Update KPIs                         │
│   • Schedule next job                   │
│   • Predict maintenance window          │
│                                         │
│ handle_material_consumed(event):        │
│   • Update inventory                    │
│   • Check reorder threshold             │
│   • Optimize batch sizing               │
│                                         │
└─────────────────────────────────────────┘
         │
         │ Simulation Step
         │
┌────────▼────────────────────────────────┐
│   Physics/Logic Simulator               │
├─────────────────────────────────────────┤
│                                         │
│ class FactorySimulator:                 │
│   def step(self, dt: float):            │
│     # Discrete-event simulation         │
│     for machine in self.machines:       │
│       machine.process(dt)               │
│                                         │
│     # Continuous processes              │
│     self.update_environmental()         │
│     self.update_material_flow()         │
│                                         │
│     # Predictive models                 │
│     self.forecast_next_failure()        │
│     self.optimize_schedule()            │
│                                         │
└─────────────────────────────────────────┘
```

**Digital Twin Maturity Level: 2.5**
- ✅ Level 1: Descriptive (현재 상태 모니터링)
- ✅ Level 2: Diagnostic (문제 원인 파악)
- 🔄 Level 3: Predictive (미래 예측 - 부분적)
- ❌ Level 4: Prescriptive (최적 행동 제안 - 향후 구현)

---

### 3. Data Flow Architecture

```
Real-time Data Pipeline:

Camera Feed (30 FPS)
    │
    ├─→ Frame Buffer (Redis Queue)
    │       │
    │       ├─→ [Worker 1] Vision AI
    │       ├─→ [Worker 2] Vision AI
    │       └─→ [Worker 3] Vision AI
    │               │
    │               ├─→ Defect? YES → Alert + Log
    │               └─→ Defect? NO  → Metrics only
    │
IoT Sensors (1 Hz - 100 Hz)
    │
    ├─→ MQTT Broker
    │       │
    │       └─→ TimescaleDB (PostgreSQL extension)
    │               │
    │               └─→ Downsampling (1s → 1min → 1hour)
    │
PLC Data (Event-based)
    │
    └─→ RabbitMQ
            │
            └─→ Digital Twin State Update
                    │
                    └─→ WebSocket Broadcast to Clients
```

**Data Retention Policy:**
```
Raw Images:      7 days (then archive to S3 Glacier)
Defect Images:   1 year (S3 Standard)
Sensor Data:
  - 1s interval: 30 days
  - 1m interval: 1 year
  - 1h interval: Forever (aggregated)
Logs:            90 days (Loki)
Model Versions:  Forever (MLflow)
```

---

## 🗂️ Database Schema Design

### PostgreSQL Tables

```sql
-- Users & Authentication
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('admin', 'engineer', 'operator', 'viewer')),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Factory Machines
CREATE TABLE machines (
    machine_id VARCHAR(50) PRIMARY KEY,
    machine_type VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'IDLE',
    health_score FLOAT DEFAULT 1.0,
    last_maintenance TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Defect Detection Results
CREATE TABLE defects (
    defect_id BIGSERIAL PRIMARY KEY,
    machine_id VARCHAR(50) REFERENCES machines(machine_id),
    image_path VARCHAR(255),
    defect_type VARCHAR(50),
    confidence FLOAT,
    bbox JSONB,  -- {x, y, width, height}
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    resolved BOOLEAN DEFAULT FALSE,
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

-- Production Jobs
CREATE TABLE jobs (
    job_id BIGSERIAL PRIMARY KEY,
    product_type VARCHAR(100),
    quantity INT,
    status VARCHAR(20) CHECK (status IN ('queued', 'in_progress', 'completed', 'failed')),
    assigned_machine_id VARCHAR(50) REFERENCES machines(machine_id),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    metadata JSONB
);

-- IoT Sensor Data (TimescaleDB Hypertable)
CREATE TABLE sensor_data (
    time TIMESTAMPTZ NOT NULL,
    machine_id VARCHAR(50) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    value FLOAT,
    unit VARCHAR(20),
    PRIMARY KEY (time, machine_id, sensor_type)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sensor_data', 'time');

-- Continuous aggregate for downsampling
CREATE MATERIALIZED VIEW sensor_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    machine_id,
    sensor_type,
    AVG(value) AS avg_value,
    MAX(value) AS max_value,
    MIN(value) AS min_value
FROM sensor_data
GROUP BY bucket, machine_id, sensor_type;

-- AI Model Registry
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    framework VARCHAR(20),  -- pytorch, onnx, tensorrt
    storage_path VARCHAR(255),
    metrics JSONB,  -- {accuracy, f1_score, inference_time, etc}
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_name, version)
);
```

### Redis Data Structures

```python
# Real-time frame buffer (List)
LPUSH frame_queue:camera_1 <binary_image_data>

# Machine status cache (Hash)
HSET machine:M001 status RUNNING
HSET machine:M001 health_score 0.95
HSET machine:M001 last_update 1638360000

# WebSocket connections (Set)
SADD websocket:connections <connection_id>

# Rate limiting (String with TTL)
INCR rate_limit:api:<user_id>
EXPIRE rate_limit:api:<user_id> 60

# Pub/Sub for real-time alerts
PUBLISH alerts:defects '{"machine": "M001", "severity": "high"}'
```

---

## 🔐 Security Architecture

```
Security Layers:

1. Network Layer
   ├─→ HTTPS/TLS 1.3 (mandatory)
   ├─→ API Gateway with rate limiting
   └─→ VPC/Private subnets for internal services

2. Authentication
   ├─→ JWT tokens (15min access, 7day refresh)
   ├─→ OAuth2 with PKCE
   └─→ MFA for admin accounts

3. Authorization
   ├─→ Role-based Access Control (RBAC)
   │   • admin: full access
   │   • engineer: read + limited write
   │   • operator: read + alerts
   │   • viewer: read only
   └─→ Resource-level permissions

4. Data Protection
   ├─→ At-rest: AES-256 encryption (PostgreSQL, S3)
   ├─→ In-transit: TLS 1.3
   └─→ PII anonymization in logs

5. Infrastructure
   ├─→ Docker image scanning (Trivy)
   ├─→ Secrets management (Vault or AWS Secrets Manager)
   └─→ Audit logging (all API calls)
```

---

## 📊 Performance Requirements

| Component | Metric | Target | Stretch Goal |
|-----------|--------|--------|--------------|
| **Vision AI** | Inference latency | <100ms | <50ms |
| | Throughput | 30 FPS | 60 FPS |
| | Accuracy (F1) | >0.90 | >0.95 |
| **API** | Response time (p95) | <200ms | <100ms |
| | Throughput | 1000 req/s | 5000 req/s |
| **Database** | Query latency | <50ms | <20ms |
| **Digital Twin** | State update freq | 1 Hz | 10 Hz |
| **Dashboard** | Page load | <2s | <1s |
| | Real-time lag | <500ms | <200ms |

---

## 🚀 Deployment Architecture

```
Production Deployment (Docker Compose → Kubernetes):

┌────────────────────────────────────────┐
│         Load Balancer (Nginx)          │
│         - SSL Termination              │
│         - Rate Limiting                │
└───────────┬────────────────────────────┘
            │
    ┌───────┴───────┐
    │               │
┌───▼────┐    ┌─────▼─────┐
│ API 1  │    │  API 2    │  (Auto-scaling: 2-10 pods)
└───┬────┘    └─────┬─────┘
    │               │
    └───────┬───────┘
            │
    ┌───────▼────────┐
    │ Message Queue  │  (RabbitMQ cluster)
    └───────┬────────┘
            │
    ┌───────┴────────┬──────────────┬──────────────┐
    │                │              │              │
┌───▼─────┐  ┌───────▼───┐  ┌──────▼────┐  ┌──────▼────┐
│Vision   │  │Digital    │  │Scheduler  │  │Monitoring │
│Worker 1 │  │Twin       │  │Engine     │  │(Grafana)  │
└─────────┘  └───────────┘  └───────────┘  └───────────┘

            ┌──────────────────────────┐
            │    Persistent Storage     │
            ├──────────────────────────┤
            │ PostgreSQL (Primary)     │
            │ PostgreSQL (Replica)     │
            │ Redis (Cluster)          │
            │ MinIO (S3-compatible)    │
            └──────────────────────────┘
```

**Infrastructure as Code:**
- Docker Compose (Development)
- Kubernetes manifests (Production)
- Terraform (Cloud infrastructure)
- Ansible (Configuration management)

---

## 🧪 Testing Strategy

```
Testing Pyramid:

         ┌─────────────┐
         │   E2E (5%)  │  ← Selenium/Playwright
         │             │    Full user journeys
         └─────────────┘
       ┌───────────────────┐
       │ Integration (15%) │  ← FastAPI TestClient
       │                   │    API endpoint tests
       └───────────────────┘
   ┌───────────────────────────┐
   │    Unit Tests (80%)       │  ← pytest
   │                           │    Component logic
   └───────────────────────────┘

Special Tests:
• Vision Model: Test-time augmentation validation
• Load Testing: Locust (1000 concurrent users)
• Chaos Engineering: Randomly kill containers
• Security: OWASP ZAP automated scans
```

---

## 📈 Monitoring & Observability

```
Three Pillars of Observability:

1. Metrics (Prometheus)
   • System: CPU, Memory, Disk, Network
   • Application: Request rate, latency, errors
   • Business: Defect rate, OEE, throughput
   • ML: Model drift, inference time, confidence distribution

2. Logs (Loki + Promtail)
   • Structured JSON logs
   • Correlation IDs for request tracing
   • Log levels: DEBUG, INFO, WARN, ERROR, CRITICAL

3. Traces (Jaeger or Tempo)
   • OpenTelemetry instrumentation
   • Distributed tracing across services
   • Bottleneck identification

Visualization:
  └─→ Grafana Dashboards
       ├─→ Operations Dashboard (real-time)
       ├─→ ML Performance Dashboard
       ├─→ Business KPI Dashboard
       └─→ Alert Management
```

---

## 🔄 CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml

on: [push, pull_request]

jobs:
  test:
    - Run unit tests (pytest)
    - Run integration tests
    - Code coverage >80%

  lint:
    - black (code formatting)
    - ruff (linting)
    - mypy (type checking)

  security:
    - bandit (Python security)
    - safety (dependency scan)
    - trivy (container scan)

  build:
    - Build Docker images
    - Tag with git SHA
    - Push to registry

  deploy-dev:
    if: branch == 'develop'
    - Deploy to dev environment
    - Run smoke tests

  deploy-prod:
    if: branch == 'main' AND tag
    - Deploy to production (blue-green)
    - Health checks
    - Rollback on failure
```

---

## 📚 Technology Stack Summary

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Vision AI** | Swin Transformer | Best speed/accuracy trade-off |
| | ONNX Runtime | Cross-platform optimization |
| | OpenCV | Image preprocessing |
| **Backend** | FastAPI | Async, type hints, auto-docs |
| | Pydantic | Data validation |
| | Celery | Async task queue |
| **Database** | PostgreSQL | Reliability + TimescaleDB |
| | Redis | Caching + pub/sub |
| | MinIO | S3-compatible object storage |
| **Frontend** | React | Component-based UI |
| | Three.js | 3D visualization |
| | D3.js | Data visualization |
| | TailwindCSS | Rapid UI development |
| **ML Ops** | MLflow | Experiment tracking |
| | DVC | Data versioning |
| | Weights & Biases | Alternative to MLflow |
| **DevOps** | Docker | Containerization |
| | Kubernetes | Orchestration |
| | Prometheus | Metrics |
| | Grafana | Visualization |
| **Testing** | pytest | Unit/integration |
| | Locust | Load testing |
| | Playwright | E2E testing |

---

## 🎯 Success Criteria

**Technical Milestones:**
- [ ] Vision model achieves >90% F1 score on test set
- [ ] API handles 1000 req/s with p95 latency <200ms
- [ ] System uptime >99.5%
- [ ] Docker deployment under 5 minutes

**Business Milestones:**
- [ ] Detects defects 50% faster than manual inspection
- [ ] Reduces false positives by 70%
- [ ] Provides 24-hour maintenance prediction window

**Research Milestones:**
- [ ] Novel attention-based explainability method
- [ ] IEEE CASE 2026 paper submission ready
- [ ] Open-source dataset contribution

---

*This architecture is designed to scale from MVP (single machine) to production (multi-factory deployment) with minimal refactoring.*
