---
name: integration-specialist
description: Expert knowledge for system integration between Vision AI, Digital Twin, RL Scheduling, and Predictive Maintenance. Use when working with feedback loops, unified pipelines, API integration, or cross-component data flow.
---

# Integration Specialist Skill

## Overview
This skill provides deep expertise in integrating the Vision AI, Digital Twin, RL Scheduling, and Predictive Maintenance components into a unified closed-loop system.

## Domain Knowledge

### Architecture Overview

```
Vision AI (Defect Detection)
         |
    Explainability (Grad-CAM)
         |
    DefectFeedback
         |
IntegratedFeedbackLoop
         |
    StationHealthModel Update
         |
    Digital Twin (Health Score)
         |
RL Scheduling (DigitalTwinRLEnv)
         |
    Optimal Schedule
         |
    WebSocket Broadcast
```

### Core Components

#### UnifiedPipeline (`/src/integration/unified_pipeline.py`)
- End-to-end orchestration (723 lines)
- Manages component lifecycle
- Coordinates data flow

```python
class PipelineMode(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    SIMULATION = "simulation"
    ANALYSIS = "analysis"

pipeline = UnifiedPipeline(config)
pipeline.run(mode=PipelineMode.INFERENCE)
```

#### IntegratedFeedbackLoop (`/src/integration/feedback_loop.py`)
- Closed-loop defect-to-health-to-scheduling (699 lines)
- Vision to Digital Twin bridge
- Health model updates

```python
feedback_loop = IntegratedFeedbackLoop(
    production_lines=simulator.production_lines,
    defect_explainer=explainer
)

# Process defect feedback
feedback_loop.process_defect(
    station_id="station_001",
    defect_type=DefectType.SCRATCH,
    severity=DefectSeverity.MODERATE,
    confidence=0.85
)
```

#### VisionDigitalTwinBridge
- Connects defect detection to DT state
- Updates machine health based on defects
- Severity-weighted degradation

#### StationHealthModel
- Per-station health tracking
- Degradation calculation
- Recovery after maintenance

```python
# Health update formula
health_loss = defect_impact_factor * severity_multiplier * confidence
new_health = max(0.1, current_health - health_loss)
```

### API Gateway Integration (`/src/api/main_integrated.py`)

#### Global Services (Line 169-246)
```python
# Service initialization order (CRITICAL)
inference_engine = ONNXInferenceEngine()
machine_state_manager = MachineStateManager()
factory_simulator = create_factory_simulator()
predictive_system = PredictiveMaintenanceSystem()
production_scheduler = ProductionScheduler()
```

#### Key Endpoints
- `POST /api/v1/integration/process-defect` - End-to-end defect processing
- `WebSocket /ws` - Real-time updates
- `GET /api/v1/dashboard/stats` - Aggregated stats (5s cache)

### Known Integration Issues

#### CRITICAL: State Synchronization
**Location**: feedback_loop.py + digital_twin_env.py
**Problem**: No sync mechanism between Feedback Loop and RL Environment

```python
# Feedback Loop updates:
station.machine_state.health_score -= degradation  # Line 143

# RL Environment reads:
station.machine_state.health_score  # Line 514

# RL also updates:
station.processing_time_mean *= speed_factor  # Line 470

# Result: Updates can be lost or inconsistent
```

**Solution**: Add state synchronization callback
```python
def sync_callback(station_id, health_score):
    rl_env.update_station_health(station_id, health_score)

feedback_loop.register_sync_callback(sync_callback)
```

#### HIGH: External Lines Validation
**Location**: digital_twin_env.py:94-146
**Problem**: No validation of external_production_lines

**Solution**:
```python
def validate_production_lines(lines):
    for line in lines:
        assert isinstance(line, ProductionLine)
        assert hasattr(line, 'stations')
        for station in line.stations.values():
            assert hasattr(station, 'machine_state')
```

#### HIGH: Service Initialization
**Location**: main_integrated.py:169-246
**Problem**: No dependency graph

**Solution**: Use dependency injection
```python
@dataclass
class ServiceContainer:
    inference_engine: ONNXInferenceEngine
    state_manager: MachineStateManager
    simulator: FactorySimulator
    # ... with proper initialization order
```

### Best Practices

1. **Shared State Management**
   - Use external_production_lines for shared objects
   - Never replace shared references on reset
   - Implement sync callbacks for updates

2. **Error Handling**
   - Catch and log integration errors
   - Use fallback behavior when components fail
   - Maintain partial functionality

3. **Performance**
   - Batch updates when possible
   - Use async for I/O operations
   - Cache expensive computations (5s TTL)

4. **Testing**
   - Test each integration point independently
   - Use mock components for unit tests
   - Run full E2E tests for regression

## Common Patterns

### Full Integration Flow
```python
# 1. Initialize shared components
simulator = FactorySimulator(config)
production_lines = simulator.production_lines

# 2. Create feedback loop with shared lines
feedback_loop = IntegratedFeedbackLoop(
    production_lines=production_lines
)

# 3. Create RL environment with same lines
rl_env = DigitalTwinRLEnv(
    external_production_lines=production_lines
)

# 4. Register sync callback
def on_health_update(station_id, health):
    # Notify RL environment
    rl_env.update_external_health(station_id, health)

feedback_loop.register_callback(on_health_update)

# 5. Process defects
defect = vision_engine.detect(image)
feedback_loop.process_defect(defect)

# 6. RL takes action
action = rl_policy.predict(rl_env.get_observation())
rl_env.step(action)
```

### WebSocket Broadcasting
```python
async def broadcast_update(message: dict):
    # Thread-safe client management
    async with clients_lock:
        active_clients = list(websocket_clients)

    tasks = [safe_send(ws, message) for ws in active_clients]
    await asyncio.gather(*tasks, return_exceptions=True)
```

## Troubleshooting

### Common Issues
1. **Health not updating**: Check feedback loop callback registration
2. **RL sees stale state**: Verify external lines are shared, not copied
3. **WebSocket disconnects**: Check client list management
4. **API timeouts**: Review service initialization order

### Integration Test Commands
```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Run E2E tests
python -m pytest tests/test_integration_e2e.py -v

# Test feedback loop
python -c "from src.integration.feedback_loop import IntegratedFeedbackLoop; print('OK')"
```

### Debug Logging
```python
import logging
logging.getLogger("integration").setLevel(logging.DEBUG)
```

## Integration Points Summary

| Source | Target | Mechanism | Data |
|--------|--------|-----------|------|
| Vision AI | Feedback Loop | Function call | DefectFeedback |
| Feedback Loop | Digital Twin | Shared reference | Health score |
| Digital Twin | RL Env | external_lines | Production state |
| RL Env | Scheduler | Action output | Job assignments |
| Scheduler | API | REST endpoint | Schedule data |
| API | Frontend | WebSocket | Real-time updates |
