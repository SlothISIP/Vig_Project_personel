---
name: rl-scheduling-expert
description: Expert knowledge for Reinforcement Learning scheduling with Ray RLlib and Gymnasium. Use when working with RL environments, training, rewards, policies, or job scheduling optimization.
---

# RL Scheduling Expert Skill

## Overview
This skill provides deep expertise in the Reinforcement Learning scheduling system using Ray RLlib and Gymnasium for production optimization.

## Domain Knowledge

### Core Components

#### ProductionSchedulingEnv (`/src/rl_scheduling/environment.py`)
- Gymnasium-compatible RL environment
- Abstract scheduling simulation
- Fast training iterations

```python
# Environment interface
env = ProductionSchedulingEnv(config)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

#### DigitalTwinRLEnv (`/src/rl_scheduling/digital_twin_env.py`)
**Core Innovation**: Uses actual factory simulator as RL environment

```python
# Key configuration
config = DigitalTwinRLConfig(
    num_production_lines=3,
    num_stations_per_line=5,
    sim_step_duration=1.0,
    domain_randomization=True,
    curriculum_learning=True
)

# With external lines (for integration)
env = DigitalTwinRLEnv(
    config=config,
    external_production_lines=simulator.production_lines
)
```

**Critical Features:**
- Domain randomization: +/-20% processing time, +/-30% defect rate
- Curriculum learning: Difficulty increases over 10-episode milestones
- External lines support: Shared state with feedback loop
- Multi-fidelity: Combines fast abstract env with high-fidelity DT

#### SchedulingStateRL (`/src/rl_scheduling/state.py`)
- State representation for RL
- Converts simulator state to observation vector

```python
# State components
state.machines: List[MachineStateRL]  # Machine states
state.jobs: Dict[str, List[JobStateRL]]  # pending, active, completed
state.metrics: SchedulingMetrics  # utilization, queue_length, etc.
```

#### RewardCalculator (`/src/rl_scheduling/reward.py`)
Multi-objective reward function with 5 components:

| Component | Weight | Description |
|-----------|--------|-------------|
| throughput | 1.0 | Production output |
| quality | 2.0 | Defect minimization (critical) |
| efficiency | 0.5 | Resource utilization |
| stability | 0.3 | Schedule consistency |
| maintenance | 0.2 | Proactive maintenance |

```python
reward = calculator.calculate(
    throughput_delta=10,
    quality_score=0.95,
    efficiency=0.8,
    stability=0.9,
    maintenance_score=0.7
)
```

#### SchedulingRLTrainer (`/src/rl_scheduling/trainer.py`)
- Ray RLlib PPO training
- Parallel workers
- Checkpoint management
- TensorBoard logging

```python
trainer = SchedulingRLTrainer(config)
trainer.train(num_iterations=100)
trainer.save_checkpoint("checkpoints/")
```

#### RLSchedulingPolicy (`/src/rl_scheduling/inference.py`)
- Load trained policy from checkpoint
- Action prediction interface
- Exploration strategies

```python
policy = RLSchedulingPolicy.load("checkpoints/latest")
action = policy.predict_action(observation)
```

### Action Space

**Discrete Action Space:**
- Size: `num_machines * max_jobs + 1` (includes no-op)
- Hierarchical: [station_idx, action_type, parameter]

**Action Types:**
1. Assign job to machine
2. Adjust processing speed
3. Trigger maintenance
4. Pause/resume production

### Observation Space

**Box Space (continuous):**
- Machine states (health, utilization, queue)
- Job states (priority, deadline, processing time)
- Global metrics (throughput, quality, efficiency)
- Size: 120+ dimensions

### Best Practices

1. **Training Stability**
   - Use curriculum learning for complex scenarios
   - Normalize observations to [0, 1] range
   - Clip rewards to prevent explosion

2. **Integration with Digital Twin**
   - Pass external_production_lines for shared state
   - Use _reset_production_lines_state() on reset
   - Sync health scores with feedback loop

3. **Hyperparameter Tuning**
   - Start with default PPO settings
   - Adjust learning rate: 1e-4 to 3e-4
   - Increase batch size for stability

4. **Domain Randomization**
   - Enable for robust policies
   - Adjust noise levels based on real-world variance
   - Use curriculum to gradually increase difficulty

## Common Patterns

### Training Loop
```python
trainer = SchedulingRLTrainer(config)
for i in range(100):
    result = trainer.train()
    if result["episode_reward_mean"] > threshold:
        trainer.save_checkpoint(f"checkpoint_{i}")
```

### Integrated Environment
```python
# Share production lines between DT and RL
simulator = FactorySimulator(config)
rl_env = DigitalTwinRLEnv(
    external_production_lines=simulator.production_lines
)
feedback_loop = IntegratedFeedbackLoop(
    production_lines=simulator.production_lines
)
```

## Troubleshooting

### Common Issues
1. **Low rewards**: Check reward weights, verify state representation
2. **Training instability**: Reduce learning rate, increase batch size
3. **State mismatch**: Verify observation space matches actual observations
4. **External lines issues**: Check lifecycle and reset behavior

### Critical Code Locations
- Line 140-145: external_production_lines initialization
- Line 318-323: Reset behavior for external lines
- Line 261-264: Products/defects clearing (potential issue)

### Debug Commands
```bash
# Test environment
python -c "from src.rl_scheduling.digital_twin_env import DigitalTwinRLEnv; env = DigitalTwinRLEnv(); print(env.observation_space)"

# Run training demo
python scripts/demo_rl_scheduler.py
```

## Integration Points

- **Digital Twin**: Uses simulator as environment
- **Feedback Loop**: Shares production line references
- **OR-Tools**: Fallback for deterministic scheduling
- **API Gateway**: Exposes scheduling endpoints
