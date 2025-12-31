# Project Directory Structure & Module Design

## 📁 Complete Directory Tree

```
digital-twin-factory/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker-compose.yml
├── docker-compose.prod.yml
├── Makefile
├── pyproject.toml
├── poetry.lock
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── CONTRIBUTING.md
│   └── research/
│       ├── paper_draft.md
│       └── experiment_logs/
│
├── scripts/
│   ├── setup_dev.sh
│   ├── run_tests.sh
│   ├── benchmark.py
│   ├── data_migration.py
│   └── model_export.py
│
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   ├── production.yaml
│   ├── logging.yaml
│   └── models/
│       ├── swin_tiny.yaml
│       └── vit_base.yaml
│
├── data/                           # Git-ignored, managed by DVC
│   ├── raw/
│   │   ├── mvtec_ad/
│   │   ├── dagm/
│   │   └── custom/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── annotations/
│   └── dvc/
│       └── .dvc                    # DVC metadata
│
├── models/                         # Trained model artifacts
│   ├── checkpoints/
│   │   ├── swin_defect_v1.0.pth
│   │   └── vit_defect_best.pth
│   ├── onnx/
│   │   ├── swin_fp32.onnx
│   │   └── swin_fp16.onnx
│   ├── tensorrt/
│   │   └── swin_int8.trt
│   └── mlflow/                     # MLflow tracking
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   ├── 04_attention_visualization.ipynb
│   └── 05_deployment_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                       # Core business logic
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   ├── logging.py              # Logging setup
│   │   ├── exceptions.py           # Custom exceptions
│   │   └── constants.py            # System constants
│   │
│   ├── vision/                     # Vision AI Engine
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract base model
│   │   │   ├── swin_transformer.py
│   │   │   ├── vit.py
│   │   │   ├── efficientvit.py     # Edge-optimized
│   │   │   └── ensemble.py         # Model ensemble
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── transforms.py       # Augmentations
│   │   │   ├── normalization.py
│   │   │   └── tiling.py           # Large image handling
│   │   ├── postprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── nms.py              # Non-max suppression
│   │   │   ├── attention_extractor.py
│   │   │   └── explainability.py   # GradCAM, etc.
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── pytorch_infer.py
│   │   │   ├── onnx_infer.py
│   │   │   ├── tensorrt_infer.py
│   │   │   └── batch_processor.py  # Batch inference
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py          # Training loop
│   │   │   ├── losses.py           # Custom loss functions
│   │   │   ├── metrics.py          # Evaluation metrics
│   │   │   ├── callbacks.py        # Early stopping, etc.
│   │   │   └── augmentation_policies.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── visualization.py    # Attention map viz
│   │       └── model_converter.py  # PyTorch → ONNX → TRT
│   │
│   ├── digital_twin/               # Digital Twin Core
│   │   ├── __init__.py
│   │   ├── state/
│   │   │   ├── __init__.py
│   │   │   ├── factory_state.py    # State machine
│   │   │   ├── machine.py          # Machine entity
│   │   │   ├── material.py         # Material entity
│   │   │   └── job.py              # Job entity
│   │   ├── simulation/
│   │   │   ├── __init__.py
│   │   │   ├── simulator.py        # Main simulator
│   │   │   ├── discrete_event.py   # Event-driven sim
│   │   │   ├── physics.py          # Physics engine (basic)
│   │   │   └── scenarios.py        # What-if scenarios
│   │   ├── predictive/
│   │   │   ├── __init__.py
│   │   │   ├── maintenance_predictor.py  # ML-based
│   │   │   ├── failure_detector.py
│   │   │   └── time_series_forecaster.py
│   │   └── events/
│   │       ├── __init__.py
│   │       ├── event_bus.py        # Event dispatcher
│   │       ├── handlers.py         # Event handlers
│   │       └── event_types.py      # Event definitions
│   │
│   ├── scheduling/                 # Optimization Engine
│   │   ├── __init__.py
│   │   ├── scheduler.py            # Main scheduler
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── ortools_solver.py   # OR-Tools based
│   │   │   ├── genetic_algorithm.py
│   │   │   ├── simulated_annealing.py
│   │   │   └── heuristics.py       # Fast heuristics
│   │   ├── constraints/
│   │   │   ├── __init__.py
│   │   │   ├── machine_constraints.py
│   │   │   ├── material_constraints.py
│   │   │   └── time_constraints.py
│   │   └── objectives/
│   │       ├── __init__.py
│   │       ├── makespan.py         # Minimize total time
│   │       ├── tardiness.py        # Minimize delays
│   │       └── multi_objective.py  # Pareto optimization
│   │
│   ├── data/                       # Data Access Layer
│   │   ├── __init__.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py         # PostgreSQL connection
│   │   │   ├── timescale.py        # TimescaleDB specific
│   │   │   ├── redis_client.py
│   │   │   └── models.py           # SQLAlchemy models
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── user_repository.py
│   │   │   ├── machine_repository.py
│   │   │   ├── defect_repository.py
│   │   │   ├── job_repository.py
│   │   │   └── sensor_repository.py
│   │   ├── cache/
│   │   │   ├── __init__.py
│   │   │   ├── cache_manager.py
│   │   │   └── strategies.py       # LRU, LFU, etc.
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── s3_client.py        # MinIO/S3
│   │       └── file_manager.py
│   │
│   ├── api/                        # FastAPI Application
│   │   ├── __init__.py
│   │   ├── main.py                 # App entry point
│   │   ├── dependencies.py         # Dependency injection
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py             # JWT validation
│   │   │   ├── cors.py
│   │   │   ├── rate_limit.py
│   │   │   └── logging_middleware.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py             # Login, logout
│   │   │   ├── vision.py           # Defect detection API
│   │   │   ├── digital_twin.py     # Twin state API
│   │   │   ├── scheduling.py       # Scheduling API
│   │   │   ├── analytics.py        # KPI metrics
│   │   │   ├── admin.py            # Admin panel
│   │   │   └── websocket.py        # Real-time updates
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py             # Pydantic models
│   │   │   ├── vision.py
│   │   │   ├── machine.py
│   │   │   ├── job.py
│   │   │   └── responses.py        # Standard responses
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── auth_service.py
│   │       ├── vision_service.py
│   │       ├── twin_service.py
│   │       └── notification_service.py
│   │
│   ├── workers/                    # Background Workers
│   │   ├── __init__.py
│   │   ├── celery_app.py           # Celery configuration
│   │   ├── tasks/
│   │   │   ├── __init__.py
│   │   │   ├── vision_tasks.py     # Async inference
│   │   │   ├── simulation_tasks.py
│   │   │   ├── training_tasks.py   # Model retraining
│   │   │   └── maintenance_tasks.py
│   │   └── schedulers/
│   │       ├── __init__.py
│   │       └── periodic_tasks.py   # Cron-like tasks
│   │
│   ├── iot/                        # IoT Integration
│   │   ├── __init__.py
│   │   ├── mqtt/
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   ├── topics.py
│   │   │   └── handlers.py
│   │   ├── amqp/
│   │   │   ├── __init__.py
│   │   │   ├── producer.py
│   │   │   └── consumer.py
│   │   └── simulators/
│   │       ├── __init__.py
│   │       ├── camera_simulator.py
│   │       ├── sensor_simulator.py
│   │       └── plc_simulator.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── decorators.py           # Retry, cache, etc.
│       ├── helpers.py
│       ├── validators.py
│       └── profiler.py             # Performance profiling
│
├── frontend/                       # React Dashboard
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── .env.example
│   │
│   ├── public/
│   │   ├── favicon.ico
│   │   └── assets/
│   │
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── vite-env.d.ts
│   │   │
│   │   ├── api/                    # API client
│   │   │   ├── client.ts           # Axios setup
│   │   │   ├── auth.ts
│   │   │   ├── vision.ts
│   │   │   ├── digital-twin.ts
│   │   │   └── websocket.ts        # WebSocket client
│   │   │
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Card.tsx
│   │   │   │   ├── Modal.tsx
│   │   │   │   └── Spinner.tsx
│   │   │   ├── layout/
│   │   │   │   ├── Navbar.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── Footer.tsx
│   │   │   ├── vision/
│   │   │   │   ├── DefectViewer.tsx
│   │   │   │   ├── AttentionMap.tsx
│   │   │   │   └── ImageUpload.tsx
│   │   │   ├── digital-twin/
│   │   │   │   ├── FactoryView3D.tsx    # Three.js
│   │   │   │   ├── MachineStatus.tsx
│   │   │   │   └── FlowDiagram.tsx
│   │   │   ├── analytics/
│   │   │   │   ├── KPIDashboard.tsx
│   │   │   │   ├── LineChart.tsx        # D3.js
│   │   │   │   ├── HeatMap.tsx
│   │   │   │   └── GanttChart.tsx
│   │   │   └── scheduling/
│   │   │       ├── ScheduleView.tsx
│   │   │       └── WhatIfSimulator.tsx
│   │   │
│   │   ├── pages/
│   │   │   ├── Login.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── VisionMonitoring.tsx
│   │   │   ├── DigitalTwin.tsx
│   │   │   ├── Analytics.tsx
│   │   │   ├── Scheduling.tsx
│   │   │   └── Admin.tsx
│   │   │
│   │   ├── hooks/
│   │   │   ├── useAuth.ts
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useRealTimeData.ts
│   │   │   └── useThreeJS.ts
│   │   │
│   │   ├── store/                  # State management
│   │   │   ├── index.ts
│   │   │   ├── authSlice.ts        # Redux Toolkit
│   │   │   ├── visionSlice.ts
│   │   │   └── twinSlice.ts
│   │   │
│   │   ├── types/
│   │   │   ├── api.ts
│   │   │   ├── models.ts
│   │   │   └── index.ts
│   │   │
│   │   └── styles/
│   │       ├── index.css
│   │       └── tailwind.css
│   │
│   └── tests/
│       ├── unit/
│       └── e2e/
│
├── tests/                          # Backend Tests
│   ├── __init__.py
│   ├── conftest.py                 # pytest fixtures
│   │
│   ├── unit/
│   │   ├── test_vision/
│   │   │   ├── test_models.py
│   │   │   ├── test_inference.py
│   │   │   └── test_preprocessing.py
│   │   ├── test_digital_twin/
│   │   │   ├── test_state.py
│   │   │   ├── test_simulation.py
│   │   │   └── test_events.py
│   │   ├── test_scheduling/
│   │   │   └── test_algorithms.py
│   │   └── test_api/
│   │       └── test_schemas.py
│   │
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_database.py
│   │   ├── test_redis.py
│   │   └── test_vision_pipeline.py
│   │
│   ├── e2e/
│   │   ├── test_user_journey.py
│   │   └── test_real_time_flow.py
│   │
│   └── performance/
│       ├── test_vision_latency.py
│       ├── test_api_throughput.py
│       └── locustfile.py           # Load testing
│
├── deploy/                         # Deployment Configs
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.worker
│   │   ├── Dockerfile.frontend
│   │   └── Dockerfile.nginx
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── api-deployment.yaml
│   │   ├── worker-deployment.yaml
│   │   ├── postgres-statefulset.yaml
│   │   ├── redis-deployment.yaml
│   │   ├── rabbitmq-deployment.yaml
│   │   ├── ingress.yaml
│   │   └── hpa.yaml                # Horizontal Pod Autoscaler
│   │
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── modules/
│   │       ├── vpc/
│   │       ├── eks/
│   │       └── rds/
│   │
│   ├── ansible/
│   │   ├── inventory.ini
│   │   ├── playbook.yml
│   │   └── roles/
│   │
│   └── monitoring/
│       ├── prometheus/
│       │   └── prometheus.yml
│       ├── grafana/
│       │   └── dashboards/
│       │       ├── operations.json
│       │       ├── ml_performance.json
│       │       └── business_kpi.json
│       └── alertmanager/
│           └── config.yml
```

---

## 🔧 Core Module Specifications

### 1. Vision Module (`src/vision/`)

**Responsibilities:**
- Load and preprocess images
- Run inference with multiple backends (PyTorch, ONNX, TensorRT)
- Extract attention maps for explainability
- Provide training utilities

**Key Interfaces:**

```python
# src/vision/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseVisionModel(ABC):
    """Abstract base class for all vision models"""

    @abstractmethod
    def load_model(self, checkpoint_path: str) -> None:
        """Load model weights"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocess input image"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Returns:
            {
                'defect_type': str,
                'confidence': float,
                'bbox': [x, y, w, h],
                'attention_map': np.ndarray,
                'inference_time_ms': float
            }
        """
        pass

    @abstractmethod
    def export_onnx(self, output_path: str) -> None:
        """Export model to ONNX format"""
        pass
```

```python
# src/vision/inference/pytorch_infer.py
from .base import BaseVisionModel
import torch
import timm

class SwinTransformerInference(BaseVisionModel):
    def __init__(self, config: Dict):
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=config['num_classes']
        )
        self.device = torch.device(config.get('device', 'cuda'))
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            start_time = time.time()

            # Preprocess
            tensor = self.preprocess(image)

            # Forward pass
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)

            # Get prediction
            confidence, pred_class = torch.max(probs, dim=1)

            # Extract attention
            attention_map = self._extract_attention()

            inference_time = (time.time() - start_time) * 1000

            return {
                'defect_type': self.class_names[pred_class.item()],
                'confidence': confidence.item(),
                'bbox': self._compute_bbox(attention_map),
                'attention_map': attention_map,
                'inference_time_ms': inference_time
            }

    def _extract_attention(self) -> np.ndarray:
        """Extract attention from last Swin block"""
        # Implementation depends on model architecture
        pass
```

---

### 2. Digital Twin Module (`src/digital_twin/`)

**Responsibilities:**
- Maintain real-time factory state
- Simulate production processes
- Predict equipment failures
- Handle state change events

**Key Interfaces:**

```python
# src/digital_twin/state/factory_state.py
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

@dataclass
class MachineState:
    machine_id: str
    status: str  # RUNNING, IDLE, ERROR, MAINTENANCE
    current_job_id: Optional[str] = None
    health_score: float = 1.0
    last_maintenance: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class FactoryState:
    machines: Dict[str, MachineState] = field(default_factory=dict)
    materials: Dict[str, MaterialState] = field(default_factory=dict)
    jobs: List[JobState] = field(default_factory=list)
    kpis: KPIMetrics = field(default_factory=KPIMetrics)

    def update_machine_status(self, machine_id: str, new_status: str):
        """Update machine status and trigger events"""
        old_status = self.machines[machine_id].status
        self.machines[machine_id].status = new_status

        # Emit event
        self._emit_event(MachineStatusChangedEvent(
            machine_id=machine_id,
            old_status=old_status,
            new_status=new_status,
            timestamp=datetime.now()
        ))
```

```python
# src/digital_twin/simulation/simulator.py
class FactorySimulator:
    """Discrete-event simulator for factory operations"""

    def __init__(self, initial_state: FactoryState):
        self.state = initial_state
        self.event_queue = PriorityQueue()
        self.current_time = 0.0

    def step(self, dt: float = 1.0) -> FactoryState:
        """
        Advance simulation by dt seconds

        Args:
            dt: Time step in seconds

        Returns:
            Updated factory state
        """
        self.current_time += dt

        # Process all events scheduled before current_time
        while not self.event_queue.empty():
            event_time, event = self.event_queue.get()
            if event_time > self.current_time:
                # Put back and break
                self.event_queue.put((event_time, event))
                break

            # Handle event
            self._handle_event(event)

        # Update continuous processes
        self._update_material_flow(dt)
        self._update_machine_degradation(dt)
        self._update_kpis()

        return self.state

    def run_until(self, end_time: float) -> FactoryState:
        """Run simulation until end_time"""
        while self.current_time < end_time:
            self.step()
        return self.state

    def predict_next_failure(self) -> Dict[str, Any]:
        """Use ML model to predict next equipment failure"""
        # Call predictive maintenance model
        pass
```

---

### 3. Scheduling Module (`src/scheduling/`)

**Responsibilities:**
- Solve job shop scheduling problems
- Handle dynamic rescheduling
- Optimize multiple objectives

**Key Interfaces:**

```python
# src/scheduling/scheduler.py
from ortools.sat.python import cp_model

class JobShopScheduler:
    """OR-Tools based job shop scheduler"""

    def __init__(self, machines: List[Machine], jobs: List[Job]):
        self.machines = machines
        self.jobs = jobs
        self.model = cp_model.CpModel()

    def solve(
        self,
        objective: str = 'makespan',
        time_limit: int = 60
    ) -> ScheduleSolution:
        """
        Solve scheduling problem

        Args:
            objective: 'makespan', 'tardiness', or 'multi'
            time_limit: Solver time limit in seconds

        Returns:
            ScheduleSolution with task assignments and timings
        """
        # Define variables
        all_tasks = {}
        for job in self.jobs:
            for task_id, task in enumerate(job.tasks):
                start_var = self.model.NewIntVar(
                    0, self._horizon(),
                    f'start_{job.id}_{task_id}'
                )
                end_var = self.model.NewIntVar(
                    0, self._horizon(),
                    f'end_{job.id}_{task_id}'
                )
                interval_var = self.model.NewIntervalVar(
                    start_var, task.duration, end_var,
                    f'interval_{job.id}_{task_id}'
                )
                all_tasks[(job.id, task_id)] = {
                    'start': start_var,
                    'end': end_var,
                    'interval': interval_var,
                    'machine': task.machine_id
                }

        # Add constraints
        self._add_precedence_constraints(all_tasks)
        self._add_machine_capacity_constraints(all_tasks)
        self._add_material_constraints(all_tasks)

        # Define objective
        if objective == 'makespan':
            makespan = self.model.NewIntVar(0, self._horizon(), 'makespan')
            self.model.AddMaxEquality(
                makespan,
                [all_tasks[(job.id, len(job.tasks)-1)]['end']
                 for job in self.jobs]
            )
            self.model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(self.model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver, all_tasks)
        else:
            raise SchedulingError(f"Solver failed with status {status}")
```

---

### 4. API Module (`src/api/`)

**Responsibilities:**
- Expose REST and WebSocket endpoints
- Handle authentication and authorization
- Validate requests and responses

**Key Interfaces:**

```python
# src/api/routes/vision.py
from fastapi import APIRouter, UploadFile, Depends, HTTPException
from ..schemas.vision import DefectDetectionResponse
from ..services.vision_service import VisionService
from ..dependencies import get_current_user, get_vision_service

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])

@router.post("/detect", response_model=DefectDetectionResponse)
async def detect_defects(
    file: UploadFile,
    model_name: str = "swin_tiny",
    user: User = Depends(get_current_user),
    vision_service: VisionService = Depends(get_vision_service)
):
    """
    Detect defects in uploaded image

    - **file**: Image file (JPEG, PNG)
    - **model_name**: Model to use (swin_tiny, vit_base, etc.)

    Returns defect detection results with attention map
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Invalid file type")

    # Read image
    image_bytes = await file.read()
    image = decode_image(image_bytes)

    # Run detection
    result = await vision_service.detect_defects(
        image=image,
        model_name=model_name,
        user_id=user.id
    )

    return DefectDetectionResponse(**result)

@router.websocket("/stream")
async def vision_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time video stream processing"""
    await websocket.accept()

    try:
        while True:
            # Receive frame
            data = await websocket.receive_bytes()
            image = decode_image(data)

            # Process
            result = await vision_service.detect_defects_async(image)

            # Send result
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
```

---

## 🔗 Inter-Module Communication

### Event-Driven Architecture

```python
# src/digital_twin/events/event_bus.py
from typing import Callable, List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    event_type: str
    data: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"

class EventBus:
    """Central event bus for inter-module communication"""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._async_handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to synchronous events"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_async(self, event_type: str, handler: Callable):
        """Subscribe to asynchronous events (will be queued)"""
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)

    def publish(self, event: Event):
        """Publish event to all subscribers"""
        # Sync handlers (blocking)
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                handler(event)

        # Async handlers (queue to Celery/RabbitMQ)
        if event.event_type in self._async_handlers:
            for handler in self._async_handlers[event.event_type]:
                # Queue to background worker
                celery_app.send_task(
                    'workers.tasks.event_handler',
                    args=[handler.__name__, event.dict()]
                )

# Usage example
event_bus = EventBus()

# Vision module publishes defect detection
def on_defect_detected(event: Event):
    defect_data = event.data
    # Update digital twin state
    factory_state.update_machine_status(
        defect_data['machine_id'],
        'WARNING'
    )
    # Trigger alert
    notification_service.send_alert(defect_data)

event_bus.subscribe('defect_detected', on_defect_detected)
```

---

## 📦 Dependency Management

### pyproject.toml

```toml
[tool.poetry]
name = "digital-twin-factory"
version = "0.1.0"
description = "AI-Driven Digital Twin Factory System"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"

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

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
black = "^23.10.0"
ruff = "^0.1.0"
mypy = "^1.6.0"
ipython = "^8.16.0"
jupyter = "^1.0.0"

[tool.poetry.group.test.dependencies]
httpx = "^0.25.0"
locust = "^2.17.0"
faker = "^19.13.0"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "--cov=src --cov-report=html --cov-report=term"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

---

## 🎯 Module Dependency Graph

```
┌─────────────┐
│   Frontend  │
│   (React)   │
└──────┬──────┘
       │ HTTP/WebSocket
       │
┌──────▼──────┐      ┌──────────────┐
│  API Layer  │◄─────┤ Auth Service │
│  (FastAPI)  │      └──────────────┘
└──────┬──────┘
       │
       ├──────────────────────┬─────────────────────┐
       │                      │                     │
┌──────▼─────┐      ┌─────────▼────────┐  ┌────────▼────────┐
│   Vision   │      │  Digital Twin    │  │   Scheduling    │
│   Module   │      │     Module       │  │     Module      │
└──────┬─────┘      └─────────┬────────┘  └────────┬────────┘
       │                      │                     │
       │           ┌──────────▼──────────┐          │
       └──────────►│    Event Bus        │◄─────────┘
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Background        │
                   │   Workers (Celery)  │
                   └──────────┬──────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼──────┐
│   PostgreSQL    │  │     Redis       │  │   RabbitMQ   │
└─────────────────┘  └─────────────────┘  └──────────────┘
```

---

*This modular architecture ensures separation of concerns, testability, and scalability from MVP to production.*
