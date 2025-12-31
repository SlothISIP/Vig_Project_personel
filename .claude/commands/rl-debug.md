---
name: RL Scheduling Debugger
description: Debug and analyze RL scheduling components
tags: rl, scheduling, debug, reinforcement-learning
allowed-tools: Read, Grep, Glob, Bash, Task
---

# RL Scheduling Debug Agent

## Task
Debug and analyze the Reinforcement Learning scheduling system.

## Context
Key RL scheduling files:
- `/src/rl_scheduling/environment.py` - ProductionSchedulingEnv (Gymnasium)
- `/src/rl_scheduling/digital_twin_env.py` - DigitalTwinRLEnv (861 lines, core innovation)
- `/src/rl_scheduling/trainer.py` - SchedulingRLTrainer (Ray RLlib PPO)
- `/src/rl_scheduling/reward.py` - RewardCalculator (5 components)
- `/src/rl_scheduling/state.py` - SchedulingStateRL, MachineStateRL, JobStateRL
- `/src/rl_scheduling/inference.py` - RLSchedulingPolicy

## Known Issues to Check
1. **External Lines Lifecycle** (digital_twin_env.py:140-145, 318-323)
   - External production lines reset behavior
   - State inconsistencies with feedback loop

2. **State Synchronization**
   - RL environment and Feedback Loop health score sync
   - Products cleared during reset may break feedback references

## Instructions

1. **If no arguments**: Full RL system analysis
   - Check environment configuration
   - Verify reward function weights
   - Analyze observation/action spaces
   - Review trainer settings

2. **If "env" argument**: Focus on environment issues
   - Check ProductionSchedulingEnv setup
   - Verify DigitalTwinRLEnv integration
   - Test observation conversion

3. **If "reward" argument**: Analyze reward function
   - Review multi-objective weights
   - Check reward calculation logic
   - Verify normalization

4. **If "train" argument**: Run training test
   ```bash
   cd /home/user/Vig_Project_personel && python scripts/train_rl_scheduler.py --test
   ```

5. **If "demo" argument**: Run RL scheduler demo
   ```bash
   cd /home/user/Vig_Project_personel && python scripts/demo_rl_scheduler.py
   ```

## Critical Code Locations
- Line 140-145: external_production_lines initialization
- Line 318-323: Reset behavior for external lines
- Line 261-264: Products/defects clearing

Arguments: $ARGUMENTS
