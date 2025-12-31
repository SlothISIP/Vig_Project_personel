---
name: API Tester
description: Test API endpoints and WebSocket connections
tags: api, testing, endpoints
allowed-tools: Read, Grep, Glob, Bash, Task
---

# API Testing Agent

## Task
Test the FastAPI backend endpoints and WebSocket connections.

## API Gateway
Main file: `/src/api/main_integrated.py` (1,155 lines)

## Endpoints

### Vision AI
- `POST /api/v1/predict` - Single image defect detection
- `POST /api/v1/predict/batch` - Batch image inference
- `GET /api/v1/stats` - Model statistics
- `GET /api/v1/benchmark` - Performance metrics

### Digital Twin
- `GET /api/v1/digital-twin/factory/{factory_id}` - Factory state
- `GET /api/v1/digital-twin/machine/{machine_id}` - Machine details

### Predictive Maintenance
- `GET /api/v1/predictive/maintenance/{machine_id}` - Single machine
- `GET /api/v1/predictive/maintenance/all` - All machines

### Scheduling
- `GET /api/v1/scheduling/current` - Current schedule
- `GET /api/v1/scheduling/schedules` - Schedule history
- `POST /api/v1/scheduling/schedule` - Create schedule
- `GET /api/v1/scheduling/job/{job_id}` - Job status

### Dashboard
- `GET /api/v1/dashboard/stats` - Aggregated stats (5s cache)

### Integration
- `POST /api/v1/integration/process-defect` - End-to-end defect processing

### Real-time
- `WebSocket /ws` - Live updates

### Health
- `GET /health` - Health check

## Instructions

1. **If no arguments**: List all endpoints with descriptions

2. **If "health" argument**: Test health endpoint
   ```bash
   curl http://localhost:8000/health
   ```

3. **If "vision" argument**: Test vision endpoints

4. **If "dt" argument**: Test digital twin endpoints

5. **If "schedule" argument**: Test scheduling endpoints

6. **If "ws" argument**: Test WebSocket connection

7. **If "all" argument**: Run full API test suite
   ```bash
   cd /home/user/Vig_Project_personel
   python -m pytest tests/ -k "api" -v
   ```

## Server Startup
```bash
cd /home/user/Vig_Project_personel
uvicorn src.api.main_integrated:app --reload --host 0.0.0.0 --port 8000
```

Arguments: $ARGUMENTS
