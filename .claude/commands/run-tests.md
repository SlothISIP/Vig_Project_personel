---
name: Test Runner
description: Run project test suites with various options
tags: testing, pytest, unit, integration
allowed-tools: Bash, Read, Glob
---

# Test Runner Agent

## Task
Run project test suites.

## Test Structure
```
tests/
├── unit/
│   ├── test_vision/
│   ├── test_digital_twin/
│   ├── test_scheduling/
│   └── test_predictive/
├── integration/
├── performance/
├── test_e2e_simulation.py
├── test_e2e_standalone.py
├── test_e2e_critical_scenario.py
├── test_integration_e2e.py
├── test_rl_scheduling.py
└── conftest.py
```

## Instructions

1. **If no arguments**: Run all tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/ -v
   ```

2. **If "unit" argument**: Run unit tests only
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/unit/ -v
   ```

3. **If "integration" argument**: Run integration tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/integration/ -v
   ```

4. **If "e2e" argument**: Run end-to-end tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/test_e2e*.py tests/test_integration_e2e.py -v
   ```

5. **If "rl" argument**: Run RL scheduling tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/test_rl_scheduling.py -v
   ```

6. **If "cov" argument**: Run with coverage report
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/ --cov=src --cov-report=html -v
   ```

7. **If "fast" argument**: Run quick smoke tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/unit/ -v -x --tb=short
   ```

8. **If specific test name**: Run that test
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/ -k "$ARGUMENTS" -v
   ```

## Common Options
- `-v`: Verbose output
- `-x`: Stop on first failure
- `--tb=short`: Short traceback
- `-k "pattern"`: Run tests matching pattern
- `--cov`: Generate coverage report

Arguments: $ARGUMENTS
