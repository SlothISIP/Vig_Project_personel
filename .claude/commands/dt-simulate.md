---
name: Digital Twin Simulation
description: Run and analyze Digital Twin factory simulation
tags: digital-twin, simulation, analysis
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Digital Twin Simulation Agent

## Task
Analyze and run Digital Twin factory simulation for this smart manufacturing project.

## Context
This project uses SimPy-based discrete event simulation located at:
- `/src/digital_twin/simulation/simulator.py` - Main FactorySimulator
- `/src/digital_twin/simulation/production_line.py` - ProductionLine, WorkStation
- `/src/digital_twin/simulation/sensor.py` - SensorNetwork
- `/src/digital_twin/state/machine_state.py` - MachineState, MachineStateManager

## Instructions

1. **If no arguments provided**: Analyze the current Digital Twin architecture
   - Review simulator configuration
   - Check production line setup
   - Verify sensor network integration
   - Report machine state management

2. **If "run" argument**: Execute simulation demo
   ```bash
   cd /home/user/Vig_Project_personel && python scripts/run_digital_twin_simulation.py
   ```

3. **If "test" argument**: Run Digital Twin unit tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/unit/test_digital_twin/ -v
   ```

4. **If "status" argument**: Check current simulation state
   - Review event bus configuration
   - Check machine health scores
   - Analyze throughput metrics

## Output Format
Provide structured analysis with:
- Current state summary
- Identified issues (if any)
- Recommendations for improvement
- Code locations with line numbers

Arguments: $ARGUMENTS
