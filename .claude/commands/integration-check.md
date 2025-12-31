---
name: Integration Checker
description: Check and test system integration between components
tags: integration, testing, feedback-loop
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Integration Testing Agent

## Task
Verify integration between Vision AI, Digital Twin, RL Scheduling, and Predictive Maintenance.

## Context
Integration components:
- `/src/integration/unified_pipeline.py` (723 lines) - End-to-end orchestration
- `/src/integration/feedback_loop.py` (699 lines) - Closed-loop feedback
- `/src/api/main_integrated.py` (1,155 lines) - API Gateway

## Critical Integration Points

### 1. Vision to Digital Twin Bridge
- Defect detection triggers health updates
- Machine state degradation based on defect severity
- File: feedback_loop.py

### 2. Digital Twin to RL Integration
- DigitalTwinRLEnv uses external_production_lines
- Shared state between feedback loop and RL
- File: digital_twin_env.py

### 3. RL to Scheduling
- Policy optimization for job-machine assignments
- Health predictions from maintenance module
- Quality metrics from feedback loop

## Known Issues
1. **State synchronization missing** between Feedback Loop and RL Environment
2. **External lines validation missing** in DigitalTwinRLEnv
3. **Service initialization order** without dependency graph
4. **WebSocket race condition** in client management

## Instructions

1. **If no arguments**: Full integration analysis
   - Check all integration points
   - Verify data flow between components
   - Identify synchronization issues

2. **If "test" argument**: Run integration tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/integration/ -v
   ```

3. **If "e2e" argument**: Run end-to-end tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/test_integration_e2e.py -v
   ```

4. **If "feedback" argument**: Focus on feedback loop
   - Check VisionDigitalTwinBridge
   - Verify DigitalTwinFeedbackController
   - Analyze StationHealthModel

5. **If "api" argument**: Test integrated API
   - POST /api/v1/integration/process-defect
   - WebSocket /ws connections
   - Dashboard stats endpoint

Arguments: $ARGUMENTS
