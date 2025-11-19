# 🧪 Testing Guide

> **Comprehensive testing documentation for quality assurance**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Testing Philosophy](#testing-philosophy)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [E2E Testing](#e2e-testing)
- [Performance Testing](#performance-testing)
- [Test Coverage](#test-coverage)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The AI-Driven Digital Twin Factory System employs a comprehensive testing strategy to ensure reliability, performance, and correctness across all components.

**Testing Stack**:
- **Framework**: pytest (Python), Jest (JavaScript)
- **Mocking**: unittest.mock, pytest-mock
- **Async Testing**: pytest-asyncio
- **Coverage**: pytest-cov, coverage.py
- **Load Testing**: Locust, k6
- **E2E Testing**: Standalone Python scripts

---

## Testing Philosophy

### Testing Pyramid

```
         ┌─────────────┐
         │   E2E (5%)  │  ← Full system integration
         │             │    (Slow, expensive, high confidence)
         └─────────────┘
       ┌───────────────────┐
       │ Integration (15%) │  ← API endpoints, service integration
       │                   │    (Medium speed, medium confidence)
       └───────────────────┘
   ┌───────────────────────────┐
   │    Unit Tests (80%)       │  ← Component logic, pure functions
   │                           │    (Fast, cheap, moderate confidence)
   └───────────────────────────┘
```

### Testing Principles

1. **Fast Feedback**: Unit tests should run in < 1 second
2. **Isolation**: Tests should not depend on external services
3. **Deterministic**: Same input → same output (no flaky tests)
4. **Readable**: Test name should describe what's being tested
5. **Maintainable**: Tests should be easy to update when code changes

---

## Test Categories

### 1. Unit Tests

Test individual functions and classes in isolation.

**Location**: `tests/unit/`
**Coverage**: 80% of code
**Run Time**: < 10 seconds

```python
# tests/unit/test_digital_twin/test_machine_state.py
def test_machine_health_score_decreases_on_defect():
    """When a defect is reported, health score should decrease."""
    machine = MachineState("M001", "Assembly_Line")
    machine.cycle_count = 100
    machine.defect_count = 0
    machine.health_score = 1.0

    machine.report_defect()

    assert machine.defect_count == 1
    assert machine.health_score < 1.0
    assert machine.health_score == 0.99  # 1/100 = 0.01 defect rate
```

### 2. Integration Tests

Test multiple components working together (with mocks for external deps).

**Location**: `tests/integration/`
**Coverage**: All critical paths
**Run Time**: < 60 seconds

```python
# tests/integration/test_api/test_vision_api.py
async def test_defect_detection_endpoint():
    """Test vision API detects defects and updates digital twin."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Upload defect image
        files = {"file": ("test.jpg", fake_image_bytes, "image/jpeg")}
        response = await client.post("/api/v1/vision/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["defect_detected"] is True
        assert data["confidence"] > 0.5
```

### 3. E2E Tests

Test complete user workflows end-to-end.

**Location**: `tests/e2e/`, `tests/test_e2e_*.py`
**Coverage**: 4 core scenarios
**Run Time**: < 120 seconds

```python
# tests/test_e2e_standalone.py
async def test_complete_defect_flow():
    """
    E2E: Image upload → Defect detected → State updated →
         Prediction triggered → Schedule adjusted → WebSocket broadcast
    """
    # 1. Upload defect image (mocked)
    # 2. Verify machine state updated
    # 3. Check predictive model triggered
    # 4. Verify schedule adjusted
    # 5. Confirm WebSocket message sent
    assert all_steps_completed
```

### 4. Performance Tests

Test system performance under load.

**Location**: `tests/performance/`
**Coverage**: API endpoints, critical paths
**Run Time**: 5-10 minutes

```python
# tests/performance/test_load.py
from locust import HttpUser, task, between

class FactoryUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_factory_state(self):
        self.client.get("/api/v1/digital-twin/state")

    @task(3)  # 3x weight
    def get_dashboard_stats(self):
        self.client.get("/api/v1/dashboard/stats")
```

---

## Running Tests

### All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v
python tests/test_e2e_standalone.py
python tests/test_e2e_critical_scenario.py

# Performance tests
pytest tests/performance/ -v
```

### Filtered Tests

```bash
# Run tests matching pattern
pytest -k "test_machine" -v

# Run specific test file
pytest tests/unit/test_digital_twin/test_machine_state.py -v

# Run specific test function
pytest tests/unit/test_digital_twin/test_machine_state.py::test_machine_health_score -v

# Run tests with markers
pytest -m "slow" -v  # Run only slow tests
pytest -m "not slow" -v  # Skip slow tests
```

### Parallel Testing

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/ -n 4
```

### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw tests/ -- --cov=src
```

---

## Unit Testing

### Example: Testing Machine State

```python
# tests/unit/test_digital_twin/test_machine_state.py
import pytest
from datetime import datetime
from src.digital_twin.state.machine_state import MachineState, MachineStatus

class TestMachineState:
    """Test suite for MachineState class."""

    def test_initial_state(self):
        """New machine should have default values."""
        machine = MachineState("M001", "Assembly_Line")

        assert machine.machine_id == "M001"
        assert machine.status == MachineStatus.IDLE
        assert machine.health_score == 1.0
        assert machine.cycle_count == 0
        assert machine.defect_count == 0

    def test_report_defect_increments_count(self):
        """Reporting defect should increment defect_count."""
        machine = MachineState("M001", "Assembly_Line")
        machine.cycle_count = 100

        machine.report_defect()

        assert machine.defect_count == 1
        assert machine.last_defect_time is not None

    def test_health_score_calculation(self):
        """Health score should be 1.0 - defect_rate."""
        machine = MachineState("M001", "Assembly_Line")
        machine.cycle_count = 100

        # 5 defects out of 100 cycles = 5% defect rate
        for _ in range(5):
            machine.report_defect()

        assert machine.defect_count == 5
        assert machine.health_score == pytest.approx(0.95)

    def test_status_changes_to_warning_on_low_health(self):
        """Status should change to WARNING when health < 0.7."""
        machine = MachineState("M001", "Assembly_Line")
        machine.cycle_count = 100

        # 35 defects = 35% defect rate = 0.65 health
        for _ in range(35):
            machine.report_defect()

        assert machine.health_score < 0.7
        assert machine.status == MachineStatus.WARNING

    def test_perform_maintenance_resets_state(self):
        """Maintenance should reset health and defect count."""
        machine = MachineState("M001", "Assembly_Line")
        machine.cycle_count = 100
        machine.defect_count = 10
        machine.health_score = 0.9
        machine.status = MachineStatus.WARNING

        machine.perform_maintenance()

        assert machine.health_score == 1.0
        assert machine.defect_count == 0
        assert machine.status == MachineStatus.IDLE
        assert machine.last_maintenance is not None

    def test_to_dict_contains_all_fields(self):
        """to_dict() should include all machine state fields."""
        machine = MachineState("M001", "Assembly_Line")
        machine.cycle_count = 100
        machine.defect_count = 5

        data = machine.to_dict()

        assert "machine_id" in data
        assert "status" in data
        assert "health_score" in data
        assert "defect_rate" in data
        assert data["defect_rate"] == pytest.approx(0.05)
```

### Testing Async Functions

```python
# tests/unit/test_api/test_main_integrated.py
import pytest
from httpx import AsyncClient
from src.api.main_integrated import app

@pytest.mark.asyncio
async def test_health_endpoint():
    """Health endpoint should return 200 with status."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
```

### Mocking External Dependencies

```python
# tests/unit/test_vision/test_onnx_model.py
from unittest.mock import Mock, patch
import numpy as np

def test_onnx_inference_with_mock():
    """Test ONNX inference without loading actual model."""
    # Mock ONNX session
    mock_session = Mock()
    mock_session.run.return_value = [np.array([[0.1, 0.9]])]  # Defect detected

    with patch('onnxruntime.InferenceSession', return_value=mock_session):
        engine = VisionAIEngine()
        result = engine.detect(fake_image)

        assert result.defect_detected is True
        assert result.confidence == pytest.approx(0.9)
```

---

## Integration Testing

### Testing API Endpoints

```python
# tests/integration/test_api/test_digital_twin_api.py
import pytest
from httpx import AsyncClient
from src.api.main_integrated import app

@pytest.mark.asyncio
async def test_get_factory_state():
    """GET /api/v1/digital-twin/state should return factory state."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/digital-twin/state")

        assert response.status_code == 200
        data = response.json()
        assert "factory_id" in data
        assert "machines" in data
        assert "statistics" in data

@pytest.mark.asyncio
async def test_update_machine_state():
    """POST /api/v1/digital-twin/machines/{id}/update should update state."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {
            "status": "maintenance",
            "temperature": 75.0
        }
        response = await client.post(
            "/api/v1/digital-twin/machines/M001/update",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["machine_id"] == "M001"
        assert "updated_fields" in data

@pytest.mark.asyncio
async def test_invalid_machine_id_returns_404():
    """Accessing non-existent machine should return 404."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/digital-twin/machines/INVALID")

        assert response.status_code == 404
```

### Testing WebSocket

```python
# tests/integration/test_api/test_websocket.py
import pytest
from fastapi.testclient import TestClient
from src.api.main_integrated import app

def test_websocket_connection():
    """WebSocket connection should receive factory updates."""
    client = TestClient(app)

    with client.websocket_connect("/api/v1/ws/stream") as websocket:
        # Should receive initial message
        data = websocket.receive_json()

        assert "type" in data
        assert "data" in data
        assert data["type"] == "factory_update"
```

---

## E2E Testing

### Standalone E2E Test

```python
# tests/test_e2e_standalone.py
async def test_integration_flow_simulation():
    """
    Complete integration flow:
    1. Initialize factory state
    2. Simulate defect detection
    3. Trigger predictive maintenance
    4. Optimize schedule
    5. Aggregate dashboard data
    """
    # 1. Initialize state manager
    state_manager = MachineStateManager()
    for i in range(1, 4):
        state_manager.add_machine(f"M{i:03d}", f"Type_{i}", "running")

    # 2. Simulate defect detection
    machine = state_manager.get_machine_state("M001")
    machine.defect_count += 1
    machine.health_score = max(0.5, machine.health_score - 0.1)

    assert machine.defect_count == 1
    print("✅ Defect detection simulated")

    # 3. Predictive maintenance
    predictions = []
    if machine.health_score < 0.7:
        urgency = "high" if machine.health_score < 0.5 else "medium"
        predictions.append({
            "machine_id": "M001",
            "urgency": urgency,
            "failure_probability": 1.0 - machine.health_score
        })

    assert len(predictions) > 0
    print(f"✅ Maintenance prediction: {predictions[0]['urgency']} urgency")

    # 4. Scheduling
    jobs = [
        {"job_id": "J001", "processing_time": 60, "machine_id": "M002"},
        {"job_id": "J002", "processing_time": 90, "machine_id": "M003"}
    ]

    scheduler = ProductionScheduler()
    result = scheduler.schedule_jobs(jobs)

    assert result["success"] is True
    print(f"✅ {len(jobs)} jobs scheduled")

    # 5. Dashboard aggregation
    dashboard_stats = {
        "overall_health": state_manager.factory_state.get_overall_health(),
        "total_machines": len(state_manager.factory_state.machines),
        "maintenance_alerts": len(predictions)
    }

    assert dashboard_stats["overall_health"] > 0
    print(f"✅ Dashboard stats aggregated: {dashboard_stats['overall_health']:.2f} health")

    print("\n🎉 ALL INTEGRATION TESTS PASSED")
    return True

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integration_flow_simulation())
```

### Critical Scenario Test

```bash
# Run critical failure scenario
python tests/test_e2e_critical_scenario.py

# Expected output:
# ✅ M001 critical defects simulated
# ✅ Critical maintenance alert generated
# ✅ 2 jobs redistributed to M002 and M003
# ✅ Dashboard reflects critical state
# 🎉 CRITICAL SCENARIO TEST PASSED
```

---

## Performance Testing

### Locust Load Testing

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class FactoryUser(HttpUser):
    """Simulates factory monitoring user behavior."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    host = "http://localhost:8000"

    @task(5)  # 5x weight (most common request)
    def get_dashboard_stats(self):
        """Fetch dashboard statistics."""
        self.client.get("/api/v1/dashboard/stats")

    @task(3)
    def get_factory_state(self):
        """Fetch full factory state."""
        self.client.get("/api/v1/digital-twin/state")

    @task(1)
    def get_predictions(self):
        """Fetch maintenance predictions."""
        self.client.get("/api/v1/predictive/predictions")

    @task(2)
    def upload_image(self):
        """Upload image for defect detection."""
        files = {"file": ("test.jpg", open("test_images/sample.jpg", "rb"), "image/jpeg")}
        self.client.post("/api/v1/vision/detect", files=files)
```

**Running Load Test**:
```bash
# Install Locust
pip install locust

# Run load test
locust -f tests/performance/locustfile.py \
  --host http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m

# View results at http://localhost:8089
```

**Performance Targets**:
- **Throughput**: > 100 RPS (requests per second)
- **Latency (p95)**: < 200ms
- **Latency (p99)**: < 500ms
- **Error Rate**: < 0.1%

### Benchmarking

```python
# tests/performance/test_benchmarks.py
import pytest
import time

@pytest.mark.benchmark
def test_machine_state_update_benchmark(benchmark):
    """Benchmark machine state update performance."""
    state_manager = MachineStateManager()
    state_manager.add_machine("M001", "Type_1", "running")

    def update():
        state_manager.update_machine_state("M001", temperature=75.0)

    result = benchmark(update)

    # Should complete in < 1ms
    assert result < 0.001
```

**Running Benchmarks**:
```bash
pip install pytest-benchmark
pytest tests/performance/ --benchmark-only
```

---

## Test Coverage

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html

# Generate terminal report
pytest --cov=src --cov-report=term

# Generate XML for CI/CD
pytest --cov=src --cov-report=xml
```

### Coverage Targets

| Component | Current | Target |
|-----------|---------|--------|
| **src/vision/** | 65% | 75% |
| **src/digital_twin/** | 85% | 90% |
| **src/predictive/** | 70% | 80% |
| **src/scheduling/** | 75% | 85% |
| **src/api/** | 80% | 85% |
| **Overall** | 75% | 80% |

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src
omit =
    */tests/*
    */venv/*
    */__pycache__/*
    */site-packages/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
```

---

## CI/CD Integration

### GitHub Actions

Tests automatically run on every push/PR via `.github/workflows/ci-cd.yaml`:

```yaml
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

**Status Badge**:
```markdown
[![Tests](https://github.com/yourusername/Vig_Project_personel/workflows/CI/badge.svg)](https://github.com/yourusername/Vig_Project_personel/actions)
[![Coverage](https://codecov.io/gh/yourusername/Vig_Project_personel/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/Vig_Project_personel)
```

---

## Best Practices

### Test Naming

```python
# ❌ Bad
def test_1():
    ...

# ✅ Good
def test_machine_health_decreases_when_defect_reported():
    ...
```

### Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange: Set up test data
    machine = MachineState("M001", "Type_1")
    machine.cycle_count = 100

    # Act: Perform the action
    machine.report_defect()

    # Assert: Verify the result
    assert machine.defect_count == 1
    assert machine.health_score < 1.0
```

### Fixtures for Reusability

```python
# conftest.py
import pytest

@pytest.fixture
def sample_machine():
    """Create a sample machine for testing."""
    return MachineState("M001", "Assembly_Line")

@pytest.fixture
def factory_with_machines():
    """Create a factory state with 3 machines."""
    factory = FactoryState("Factory_01")
    for i in range(1, 4):
        machine = MachineState(f"M{i:03d}", f"Type_{i}")
        factory.add_machine(machine)
    return factory

# Usage in tests
def test_with_fixture(sample_machine):
    sample_machine.report_defect()
    assert sample_machine.defect_count == 1
```

### Parametrized Tests

```python
@pytest.mark.parametrize("defects,expected_health", [
    (0, 1.0),
    (5, 0.95),
    (10, 0.90),
    (50, 0.50),
])
def test_health_score_calculation(defects, expected_health):
    """Test health score with various defect counts."""
    machine = MachineState("M001", "Type_1")
    machine.cycle_count = 100

    for _ in range(defects):
        machine.report_defect()

    assert machine.health_score == pytest.approx(expected_health)
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'src'

# Fix: Add project root to PYTHONPATH
export PYTHONPATH=/path/to/Vig_Project_personel:$PYTHONPATH
pytest tests/
```

#### 2. Async Test Failures

```bash
# Error: RuntimeWarning: coroutine was never awaited

# Fix: Mark test as async and use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

#### 3. Flaky Tests

```python
# ❌ Bad: Time-dependent test
def test_timestamp():
    result = get_current_timestamp()
    assert result == "2025-11-19T12:00:00Z"  # Fails at different times!

# ✅ Good: Mock time
from unittest.mock import patch

@patch('datetime.datetime')
def test_timestamp(mock_datetime):
    mock_datetime.now.return_value = datetime(2025, 11, 19, 12, 0, 0)
    result = get_current_timestamp()
    assert result == "2025-11-19T12:00:00Z"
```

#### 4. Slow Tests

```bash
# Identify slow tests
pytest --durations=10

# Mark slow tests
@pytest.mark.slow
def test_expensive_operation():
    ...

# Skip slow tests in development
pytest -m "not slow"
```

---

## Test Maintenance

### Regular Tasks

```bash
# Weekly
- Review coverage reports
- Fix failing tests immediately
- Remove obsolete tests

# Monthly
- Refactor duplicate test code
- Update test dependencies
- Review performance benchmarks
```

### Debugging Tests

```bash
# Run with verbose output
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Run specific test with print statements
pytest tests/unit/test_example.py::test_function -s
```

---

## Summary

**Testing Checklist**:
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] E2E scenarios verified
- [ ] Coverage > 80%
- [ ] Performance benchmarks met
- [ ] No flaky tests
- [ ] CI/CD pipeline green

**Quick Commands**:
```bash
# Full test suite
pytest tests/ -v --cov=src --cov-report=html

# Fast feedback (unit tests only)
pytest tests/unit/ -v

# Pre-commit checks
pytest tests/unit/ tests/integration/ -v && ruff check src/ && black --check src/
```

---

**Last Updated**: 2025-11-19
