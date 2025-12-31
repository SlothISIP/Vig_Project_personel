---
name: Performance Analyzer
description: Analyze and benchmark system performance
tags: performance, benchmark, optimization
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Performance Analysis Agent

## Task
Analyze performance characteristics and identify optimization opportunities.

## Key Performance Areas

### 1. API Response Times
- `/api/v1/predict` - Single image inference
- `/api/v1/predict/batch` - Batch inference
- `/api/v1/dashboard/stats` - Cached stats (5s TTL)

### 2. Simulation Performance
- SimPy step execution time
- Event processing throughput
- Memory usage during long simulations

### 3. RL Training Performance
- Training steps per second
- GPU utilization (if available)
- Memory footprint

### 4. Frontend Rendering
- Three.js 3D scene FPS
- React component re-renders
- WebSocket message processing

## Instructions

1. **If no arguments**: Full performance audit
   - Analyze API endpoints
   - Review simulation efficiency
   - Check RL training metrics
   - Evaluate frontend performance

2. **If "api" argument**: API performance focus
   ```bash
   cd /home/user/Vig_Project_personel
   python -m pytest tests/performance/ -v --benchmark-only
   ```

3. **If "profile" argument**: Run Python profiler
   ```bash
   cd /home/user/Vig_Project_personel
   python -m cProfile -s cumtime scripts/run_digital_twin_simulation.py
   ```

4. **If "memory" argument**: Memory analysis
   - Check for memory leaks
   - Analyze object allocations
   - Review cache sizes

5. **If "report" argument**: Generate performance report
   - Read PERFORMANCE_ANALYSIS.md
   - Compare with current metrics
   - Identify regressions

## Performance Targets
- API response: < 100ms (single image)
- Batch inference: < 1s (32 images)
- Simulation step: < 10ms
- Dashboard refresh: < 500ms

Arguments: $ARGUMENTS
