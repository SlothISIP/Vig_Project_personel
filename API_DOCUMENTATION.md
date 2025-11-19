# 📡 API Documentation

> **Complete API Reference for the AI-Driven Digital Twin Factory System**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Base URL & Authentication](#base-url--authentication)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Health & Status](#health--status)
- [Vision AI Endpoints](#vision-ai-endpoints)
- [Digital Twin Endpoints](#digital-twin-endpoints)
- [Predictive Maintenance Endpoints](#predictive-maintenance-endpoints)
- [Scheduling Endpoints](#scheduling-endpoints)
- [Dashboard Endpoints](#dashboard-endpoints)
- [WebSocket API](#websocket-api)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

---

## Overview

The API follows REST principles with JSON request/response format. All endpoints support:
- **HTTP/2**: For improved performance
- **CORS**: Configurable allowed origins
- **Compression**: GZIP for responses > 1KB
- **Caching**: ETag support for GET requests

**API Version**: v1
**Protocol**: HTTP/HTTPS
**Content-Type**: application/json (unless multipart/form-data for file uploads)

---

## Base URL & Authentication

### Base URL

```
Development:  http://localhost:8000
Production:   https://api.yourdomain.com
```

### Authentication

**Current**: No authentication (development mode)

**Production (Recommended)**:

```http
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <your-api-key>
```

**Obtaining a token** (future):
```bash
POST /auth/token
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "your_password"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 900
}
```

---

## Response Format

### Success Response

```json
{
  "data": {
    ...  // Response payload
  },
  "meta": {
    "timestamp": "2025-11-19T12:00:00Z",
    "processing_time_ms": 15
  }
}
```

### Paginated Response

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 150,
    "total_pages": 8
  },
  "links": {
    "self": "/api/v1/defects?page=1",
    "next": "/api/v1/defects?page=2",
    "prev": null,
    "first": "/api/v1/defects?page=1",
    "last": "/api/v1/defects?page=8"
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Validation Error",
  "detail": "Invalid file format. Expected JPEG or PNG.",
  "type": "validation_error",
  "timestamp": "2025-11-19T12:00:00Z",
  "path": "/api/v1/vision/detect"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| **200** | OK | Request succeeded |
| **201** | Created | Resource created successfully |
| **204** | No Content | Deletion succeeded |
| **400** | Bad Request | Invalid request payload or parameters |
| **401** | Unauthorized | Missing or invalid authentication |
| **403** | Forbidden | Insufficient permissions |
| **404** | Not Found | Resource doesn't exist |
| **409** | Conflict | State conflict (e.g., machine already in maintenance) |
| **422** | Unprocessable Entity | Valid syntax but semantic errors (e.g., scheduling conflict) |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Server-side error |
| **503** | Service Unavailable | Service temporarily unavailable (e.g., model loading) |

### Error Types

```typescript
type ErrorType =
  | "validation_error"      // 400 - Invalid input
  | "authentication_error"  // 401 - Auth failed
  | "authorization_error"   // 403 - Permission denied
  | "not_found_error"       // 404 - Resource missing
  | "state_error"           // 409 - Invalid state
  | "scheduling_error"      // 422 - Scheduling conflict
  | "model_error"           // 503 - Model inference failed
  | "internal_error";       // 500 - Unexpected error
```

---

## Health & Status

### GET /health

Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400
}
```

**Status Codes**:
- `200`: Service is healthy
- `503`: Service degraded or unhealthy

---

### GET /health/ready

Readiness probe for Kubernetes.

**Response**:
```json
{
  "ready": true,
  "checks": {
    "database": "ok",
    "redis": "ok",
    "model_loaded": "ok"
  }
}
```

**Status Codes**:
- `200`: Ready to serve traffic
- `503`: Not ready (e.g., models still loading)

---

### GET /health/live

Liveness probe for Kubernetes.

**Response**:
```json
{
  "alive": true
}
```

**Status Codes**:
- `200`: Process is alive
- `503`: Process should be restarted

---

## Vision AI Endpoints

### POST /api/v1/vision/detect

Detect defects in an uploaded image using Vision Transformer models.

**Request**:
```http
POST /api/v1/vision/detect
Content-Type: multipart/form-data

file: <image_file> (JPEG/PNG, max 10MB)
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api/v1/vision/detect \
  -F "file=@sample_defect.jpg"
```

**Response** (200 OK):
```json
{
  "defect_detected": true,
  "confidence": 0.95,
  "defect_type": "scratch",
  "bbox": [120, 80, 200, 150],
  "attention_map": null,
  "processing_time_ms": 15,
  "model_version": "swin-tiny-v1.0",
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**Response Fields**:
- `defect_detected` (boolean): Whether a defect was found
- `confidence` (float): Model confidence score (0.0-1.0)
- `defect_type` (string): Type of defect (e.g., "scratch", "crack", "dent")
- `bbox` (array): Bounding box `[x, y, width, height]` in pixels
- `attention_map` (array|null): Attention heatmap (if requested)
- `processing_time_ms` (int): Inference time in milliseconds

**Error Responses**:
- `400`: Invalid file format or size
- `503`: Model not loaded or inference failed

---

### POST /api/v1/vision/batch-detect

Detect defects in multiple images (batch processing).

**Request**:
```http
POST /api/v1/vision/batch-detect
Content-Type: multipart/form-data

files: [<image1>, <image2>, ...]
```

**Response** (200 OK):
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "defect_detected": true,
      "confidence": 0.95,
      "defect_type": "scratch"
    },
    {
      "filename": "image2.jpg",
      "defect_detected": false,
      "confidence": 0.12
    }
  ],
  "total_images": 2,
  "total_defects": 1,
  "processing_time_ms": 45
}
```

---

### GET /api/v1/vision/models

List available Vision AI models.

**Response** (200 OK):
```json
{
  "models": [
    {
      "model_id": "swin-tiny-fp16",
      "name": "Swin Transformer Tiny FP16",
      "version": "1.0",
      "accuracy": 0.935,
      "latency_ms": 28,
      "is_active": true
    },
    {
      "model_id": "swin-tiny-int8",
      "name": "Swin Transformer Tiny INT8",
      "version": "1.0",
      "accuracy": 0.932,
      "latency_ms": 15,
      "is_active": false
    }
  ]
}
```

---

## Digital Twin Endpoints

### GET /api/v1/digital-twin/state

Get the current state of the entire factory.

**Response** (200 OK):
```json
{
  "factory_id": "Factory_01",
  "machines": {
    "M001": {
      "machine_id": "M001",
      "machine_type": "Assembly_Line_1",
      "status": "running",
      "health_score": 0.92,
      "cycle_count": 1500,
      "defect_count": 12,
      "defect_rate": 0.008,
      "temperature": 72.5,
      "vibration": 2.1,
      "current_job_id": "J123",
      "last_maintenance": "2025-11-15T10:00:00Z",
      "updated_at": "2025-11-19T12:00:00Z"
    },
    "M002": {
      ...
    }
  },
  "statistics": {
    "total_machines": 3,
    "overall_health": 0.89,
    "status_breakdown": {
      "running": 2,
      "warning": 1,
      "idle": 0
    },
    "total_cycles": 4500,
    "total_defects": 36,
    "overall_defect_rate": 0.008
  },
  "updated_at": "2025-11-19T12:00:00Z"
}
```

---

### GET /api/v1/digital-twin/machines/{machine_id}

Get the state of a specific machine.

**Path Parameters**:
- `machine_id` (string): Machine identifier (e.g., "M001")

**Response** (200 OK):
```json
{
  "machine_id": "M001",
  "machine_type": "Assembly_Line_1",
  "status": "running",
  "health_score": 0.92,
  "cycle_count": 1500,
  "defect_count": 12,
  "defect_rate": 0.008,
  "temperature": 72.5,
  "vibration": 2.1,
  "pressure": 95.0,
  "speed": 1200.0,
  "current_job_id": "J123",
  "last_maintenance": "2025-11-15T10:00:00Z",
  "last_defect_time": "2025-11-19T11:30:00Z",
  "created_at": "2025-11-01T00:00:00Z",
  "updated_at": "2025-11-19T12:00:00Z"
}
```

**Error Responses**:
- `404`: Machine not found

---

### POST /api/v1/digital-twin/machines/{machine_id}/update

Update machine state properties.

**Path Parameters**:
- `machine_id` (string): Machine identifier

**Request**:
```json
{
  "status": "maintenance",
  "temperature": 75.0,
  "vibration": 3.5
}
```

**Response** (200 OK):
```json
{
  "message": "Machine state updated successfully",
  "machine_id": "M001",
  "updated_fields": ["status", "temperature", "vibration"],
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**Allowed Fields**:
- `status`: "idle" | "running" | "warning" | "error" | "maintenance" | "offline"
- `temperature`: float
- `vibration`: float
- `pressure`: float
- `speed`: float

**Error Responses**:
- `400`: Invalid field values
- `404`: Machine not found
- `409`: Invalid state transition

---

### GET /api/v1/digital-twin/statistics

Get factory-wide statistics.

**Query Parameters**:
- `time_range` (optional): "1h" | "24h" | "7d" | "30d" (default: "24h")

**Response** (200 OK):
```json
{
  "factory_id": "Factory_01",
  "time_range": "24h",
  "overall_health": 0.89,
  "total_machines": 3,
  "status_breakdown": {
    "running": 2,
    "warning": 1,
    "idle": 0,
    "error": 0,
    "maintenance": 0,
    "offline": 0
  },
  "total_cycles": 4500,
  "total_defects": 36,
  "overall_defect_rate": 0.008,
  "oee": 0.85,
  "availability": 0.95,
  "performance": 0.92,
  "quality": 0.99,
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**OEE Calculation**:
```
OEE = Availability × Performance × Quality
```

---

## Predictive Maintenance Endpoints

### GET /api/v1/predictive/predictions

Get maintenance predictions for all machines.

**Query Parameters**:
- `urgency` (optional): Filter by urgency ("critical" | "high" | "medium" | "low")
- `machine_id` (optional): Filter by specific machine

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "machine_id": "M001",
      "failure_probability": 0.75,
      "time_to_failure_hours": 48,
      "urgency": "high",
      "recommended_action": "Schedule maintenance within 2 days",
      "features": {
        "temperature": 75.0,
        "vibration": 3.5,
        "defect_rate": 0.01,
        "health_score": 0.75
      },
      "model_confidence": 0.89,
      "predicted_at": "2025-11-19T12:00:00Z"
    },
    {
      "machine_id": "M002",
      ...
    }
  ],
  "summary": {
    "total_machines": 3,
    "predictions_count": 3,
    "urgency_breakdown": {
      "critical": 0,
      "high": 1,
      "medium": 2,
      "low": 0
    }
  },
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**Urgency Classification**:
- **Critical**: < 24 hours to failure
- **High**: 24-72 hours to failure
- **Medium**: 3-7 days to failure
- **Low**: > 7 days to failure

---

### GET /api/v1/predictive/predictions/{machine_id}

Get maintenance prediction for a specific machine.

**Path Parameters**:
- `machine_id` (string): Machine identifier

**Response** (200 OK):
```json
{
  "machine_id": "M001",
  "failure_probability": 0.75,
  "time_to_failure_hours": 48,
  "urgency": "high",
  "recommended_action": "Schedule maintenance within 2 days",
  "features": {
    "temperature": 75.0,
    "vibration": 3.5,
    "pressure": 92.0,
    "speed": 1150.0,
    "defect_rate": 0.01,
    "health_score": 0.75,
    "cycles_since_maintenance": 1500
  },
  "model_details": {
    "xgboost_prediction": 0.78,
    "lstm_prediction": 0.72,
    "ensemble_prediction": 0.75,
    "confidence": 0.89
  },
  "historical_data": {
    "last_maintenance": "2025-11-15T10:00:00Z",
    "maintenance_count": 5,
    "failure_count": 0
  },
  "predicted_at": "2025-11-19T12:00:00Z"
}
```

**Error Responses**:
- `404`: Machine not found

---

### POST /api/v1/predictive/train

Trigger model retraining (admin only).

**Request**:
```json
{
  "model_type": "xgboost",
  "training_data_days": 30,
  "hyperparameters": {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100
  }
}
```

**Response** (202 Accepted):
```json
{
  "message": "Training job started",
  "job_id": "train-20251119-120000",
  "status_url": "/api/v1/predictive/train/status/train-20251119-120000",
  "estimated_duration_minutes": 30
}
```

---

## Scheduling Endpoints

### POST /api/v1/scheduling/optimize

Optimize job-to-machine assignments using CP-SAT solver.

**Request**:
```json
{
  "jobs": [
    {
      "job_id": "J001",
      "processing_time": 60,
      "priority": 1,
      "dependencies": []
    },
    {
      "job_id": "J002",
      "processing_time": 90,
      "priority": 2,
      "dependencies": ["J001"]
    }
  ],
  "machines": ["M001", "M002", "M003"],
  "constraints": {
    "max_makespan": 300,
    "balance_load": true
  }
}
```

**Response** (200 OK):
```json
{
  "schedule": [
    {
      "job_id": "J001",
      "machine_id": "M001",
      "start_time": 0,
      "end_time": 60,
      "priority": 1
    },
    {
      "job_id": "J002",
      "machine_id": "M002",
      "start_time": 60,
      "end_time": 150,
      "priority": 2
    }
  ],
  "makespan": 150,
  "solver_time_ms": 250,
  "optimal": true,
  "objective_value": 150,
  "load_balance": {
    "M001": 60,
    "M002": 90,
    "M003": 0
  },
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**Constraint Types**:
- `max_makespan` (int): Maximum allowed total time
- `balance_load` (bool): Distribute jobs evenly across machines
- `respect_dependencies` (bool): Honor job dependencies (default: true)

**Error Responses**:
- `400`: Invalid job or machine configuration
- `422`: No feasible solution found

---

### POST /api/v1/scheduling/rl/predict

Get RL-based schedule recommendations (experimental).

**Request**:
```json
{
  "factory_state": {
    "machines": [...],
    "job_queue": [...]
  }
}
```

**Response** (200 OK):
```json
{
  "recommended_schedule": [
    {
      "job_id": "J001",
      "machine_id": "M001",
      "confidence": 0.89
    }
  ],
  "expected_reward": 125.5,
  "policy_version": "ppo-v1.0",
  "timestamp": "2025-11-19T12:00:00Z"
}
```

---

### GET /api/v1/scheduling/jobs

List all production jobs.

**Query Parameters**:
- `status` (optional): Filter by status ("queued" | "in_progress" | "completed" | "failed")
- `machine_id` (optional): Filter by assigned machine
- `page` (int, default: 1): Page number
- `page_size` (int, default: 20): Items per page

**Response** (200 OK):
```json
{
  "jobs": [
    {
      "job_id": "J001",
      "product_type": "Widget_A",
      "quantity": 100,
      "status": "in_progress",
      "assigned_machine_id": "M001",
      "started_at": "2025-11-19T10:00:00Z",
      "expected_completion_at": "2025-11-19T12:00:00Z",
      "progress": 0.75
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 50,
    "total_pages": 3
  }
}
```

---

## Dashboard Endpoints

### GET /api/v1/dashboard/stats

Get comprehensive dashboard statistics (cached for 5 seconds).

**Response** (200 OK):
```json
{
  "overall_health": 0.89,
  "total_machines": 3,
  "total_cycles": 4500,
  "total_defects": 36,
  "defect_rate": 0.008,
  "status_breakdown": {
    "running": 2,
    "warning": 1,
    "idle": 0,
    "error": 0,
    "maintenance": 0,
    "offline": 0
  },
  "maintenance_alerts": {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 0
  },
  "oee": 0.85,
  "throughput_last_hour": 120,
  "upcoming_jobs": 15,
  "timestamp": "2025-11-19T12:00:00Z",
  "cache_status": "hit"
}
```

**Cache Behavior**:
- TTL: 5 seconds
- Time-bucket based caching
- Cache hit: ~1ms response time
- Cache miss: ~150ms response time

---

### GET /api/v1/dashboard/charts/defect-trend

Get defect trend data for charts.

**Query Parameters**:
- `interval` (string): "1h" | "24h" | "7d" | "30d" (default: "24h")
- `granularity` (string): "minute" | "hour" | "day" (default: "hour")

**Response** (200 OK):
```json
{
  "interval": "24h",
  "granularity": "hour",
  "data": [
    {
      "timestamp": "2025-11-19T00:00:00Z",
      "defect_count": 3,
      "total_cycles": 120,
      "defect_rate": 0.025
    },
    {
      "timestamp": "2025-11-19T01:00:00Z",
      "defect_count": 2,
      "total_cycles": 115,
      "defect_rate": 0.017
    },
    ...
  ]
}
```

---

## WebSocket API

### WS /api/v1/ws/stream

Real-time factory state updates via WebSocket.

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/stream');

ws.onopen = () => {
  console.log('Connected to factory stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Factory update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from factory stream');
};
```

**Message Format**:
```json
{
  "type": "factory_update",
  "data": {
    "factory_state": {
      "factory_id": "Factory_01",
      "machines": {...},
      "statistics": {...}
    },
    "predictions": [
      {
        "machine_id": "M001",
        "failure_probability": 0.75,
        "urgency": "high"
      }
    ],
    "dashboard_stats": {...}
  },
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**Message Types**:
- `factory_update`: Periodic state updates (every 2 seconds)
- `defect_detected`: Real-time defect alerts
- `maintenance_alert`: Urgent maintenance notifications
- `schedule_updated`: Job schedule changes

**Error Messages**:
```json
{
  "type": "error",
  "error": "Connection limit reached",
  "code": "MAX_CONNECTIONS_EXCEEDED"
}
```

---

## Rate Limiting

### Current Limits

| Endpoint | Rate Limit |
|----------|------------|
| **/api/v1/vision/detect** | 100 requests/minute |
| **/api/v1/vision/batch-detect** | 20 requests/minute |
| **All other endpoints** | 1000 requests/minute |
| **WebSocket connections** | 100 concurrent connections per IP |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638360060
```

### Rate Limit Exceeded Response (429)

```json
{
  "error": "Rate Limit Exceeded",
  "detail": "You have exceeded the rate limit of 100 requests per minute",
  "type": "rate_limit_error",
  "retry_after_seconds": 45,
  "limit": 100,
  "window_seconds": 60
}
```

---

## Examples

### Example 1: Complete Defect Detection Flow

```bash
# 1. Upload image for defect detection
curl -X POST http://localhost:8000/api/v1/vision/detect \
  -F "file=@defect_sample.jpg"

# Response:
{
  "defect_detected": true,
  "confidence": 0.95,
  "defect_type": "scratch",
  "machine_id": "M001"  # Auto-detected from image metadata
}

# 2. Check updated machine state
curl http://localhost:8000/api/v1/digital-twin/machines/M001

# Response:
{
  "machine_id": "M001",
  "health_score": 0.75,  # Decreased after defect
  "defect_count": 13,    # Incremented
  "status": "warning"    # Changed from "running"
}

# 3. Get maintenance prediction
curl http://localhost:8000/api/v1/predictive/predictions/M001

# Response:
{
  "machine_id": "M001",
  "urgency": "high",
  "time_to_failure_hours": 48,
  "recommended_action": "Schedule maintenance within 2 days"
}
```

### Example 2: Scheduling Optimization

```bash
# 1. Submit jobs for optimization
curl -X POST http://localhost:8000/api/v1/scheduling/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "jobs": [
      {"job_id": "J001", "processing_time": 60, "priority": 1},
      {"job_id": "J002", "processing_time": 90, "priority": 2},
      {"job_id": "J003", "processing_time": 45, "priority": 1}
    ],
    "machines": ["M001", "M002", "M003"]
  }'

# Response:
{
  "schedule": [
    {"job_id": "J001", "machine_id": "M001", "start_time": 0, "end_time": 60},
    {"job_id": "J003", "machine_id": "M002", "start_time": 0, "end_time": 45},
    {"job_id": "J002", "machine_id": "M003", "start_time": 0, "end_time": 90}
  ],
  "makespan": 90,
  "optimal": true
}

# 2. Get job status
curl http://localhost:8000/api/v1/scheduling/jobs?status=in_progress

# Response:
{
  "jobs": [
    {
      "job_id": "J001",
      "status": "in_progress",
      "assigned_machine_id": "M001",
      "progress": 0.5
    }
  ]
}
```

### Example 3: Real-Time Dashboard

```javascript
// Connect to WebSocket for live updates
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/stream');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);

  if (update.type === 'factory_update') {
    // Update dashboard UI
    updateDashboard(update.data.dashboard_stats);
    update3DView(update.data.factory_state.machines);
  }

  if (update.type === 'maintenance_alert') {
    // Show alert notification
    showAlert(update.data.urgency, update.data.message);
  }
};

// Fetch initial dashboard data
fetch('http://localhost:8000/api/v1/dashboard/stats')
  .then(res => res.json())
  .then(data => {
    renderDashboard(data);
  });
```

### Example 4: Batch Processing

```bash
# Upload multiple images
curl -X POST http://localhost:8000/api/v1/vision/batch-detect \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"

# Response:
{
  "results": [
    {"filename": "image1.jpg", "defect_detected": true, "confidence": 0.95},
    {"filename": "image2.jpg", "defect_detected": false, "confidence": 0.12},
    {"filename": "image3.jpg", "defect_detected": true, "confidence": 0.88}
  ],
  "total_images": 3,
  "total_defects": 2,
  "processing_time_ms": 45
}
```

---

## SDK & Client Libraries

### Python Client

```python
from digital_twin_client import DigitalTwinAPI

api = DigitalTwinAPI(base_url="http://localhost:8000")

# Detect defects
result = api.vision.detect(image_path="sample.jpg")
print(f"Defect detected: {result.defect_detected}")

# Get factory state
state = api.digital_twin.get_state()
print(f"Overall health: {state.statistics.overall_health}")

# Optimize schedule
schedule = api.scheduling.optimize(jobs=[...])
print(f"Makespan: {schedule.makespan}")
```

### JavaScript/TypeScript Client

```typescript
import { DigitalTwinClient } from 'digital-twin-client';

const client = new DigitalTwinClient({
  baseUrl: 'http://localhost:8000'
});

// Detect defects
const result = await client.vision.detect(imageFile);

// WebSocket streaming
client.stream.onFactoryUpdate((data) => {
  console.log('Factory update:', data);
});
```

---

## Postman Collection

Import the Postman collection for easy testing:

```bash
# Download collection
curl -O https://api.yourdomain.com/postman-collection.json

# Import in Postman: File > Import > Upload Files
```

---

## API Versioning

**Current Version**: v1

**Version Strategy**: URL path versioning (`/api/v1/...`)

**Deprecation Policy**:
- Versions supported for minimum 12 months after deprecation notice
- Breaking changes trigger new version (v2, v3, etc.)
- Non-breaking changes added to current version

---

## Support

**Documentation**: https://docs.yourdomain.com
**GitHub Issues**: https://github.com/yourusername/Vig_Project_personel/issues
**Email**: support@yourdomain.com

---

**Last Updated**: 2025-11-19
**API Version**: 1.0.0
