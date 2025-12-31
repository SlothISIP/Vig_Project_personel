---
name: digital-twin-expert
description: Expert knowledge for Digital Twin factory simulation using SimPy. Use when working with simulation, production lines, machine states, sensors, or event-driven factory modeling.
---

# Digital Twin Expert Skill

## Overview
This skill provides deep expertise in the Digital Twin factory simulation system built with SimPy discrete event simulation.

## Domain Knowledge

### Core Components

#### FactorySimulator (`/src/digital_twin/simulation/simulator.py`)
- SimPy-based discrete event simulation engine
- Manages multiple production lines
- Coordinates event bus for inter-component communication
- Collects factory-wide statistics

```python
# Key methods
simulator.step(duration)      # Advance simulation time
simulator.get_statistics()    # Get factory metrics
simulator.run()               # Execute full simulation
```

#### ProductionLine (`/src/digital_twin/simulation/production_line.py`)
- Contains ordered WorkStation dictionary
- Routes products through stations
- Tracks completed and defective products

#### WorkStation
- Individual processing unit with machine state
- Has processing time, defect rate, buffer queue
- Integrates with SensorNetwork for IoT data

```python
# Key attributes
station.machine_state         # MachineState object
station.processing_time_mean  # Average processing duration
station.defect_rate          # Probability of defect
station.buffer               # Input queue (SimPy Store)
station.perform_maintenance() # Maintenance action
```

#### MachineState (`/src/digital_twin/state/machine_state.py`)
- Status: IDLE, RUNNING, WARNING, ERROR, MAINTENANCE
- Health score: 0.0 to 1.0
- Sensor readings: temperature, vibration, pressure
- Degradation modeling

```python
# Status transitions
machine_state.update_status(new_status)
machine_state.degrade(amount)
machine_state.health_score  # Current health (0-1)
```

#### EventBus (`/src/digital_twin/events/event_bus.py`)
- Publish-subscribe pattern for events
- Event types defined in event_types.py
- Enables loose coupling between components

### Best Practices

1. **Simulation Time Management**
   - Use `env.timeout()` for delays
   - Never use `time.sleep()` in simulation code
   - Track simulation time vs wall clock time

2. **State Consistency**
   - Update machine states atomically when possible
   - Use event bus for cross-component notifications
   - Maintain health scores between 0.1 and 1.0

3. **Product Tracking**
   - Each product has unique ID and type
   - Track visited_stations for routing history
   - Mark defective products appropriately

4. **Performance**
   - Batch statistics collection
   - Use generators for product arrivals
   - Minimize event bus overhead

## Common Patterns

### Creating a New Station
```python
station = WorkStation(
    station_id="station_001",
    processing_time_mean=5.0,
    defect_rate=0.02,
    env=simpy_env
)
```

### Handling Machine Failure
```python
if machine_state.health_score < 0.3:
    machine_state.update_status(MachineStatus.WARNING)
    event_bus.publish(MaintenanceRequiredEvent(station_id))
```

### Integrating with RL
```python
# External lines for RL environment
external_lines = simulator.production_lines
rl_env = DigitalTwinRLEnv(external_production_lines=external_lines)
```

## Troubleshooting

### Common Issues
1. **Simulation hangs**: Check for missing `yield` statements
2. **Products stuck**: Verify buffer capacity and processing logic
3. **Health always 1.0**: Ensure degradation is being applied
4. **Events not firing**: Check event bus subscription

### Debug Commands
```bash
# Run simulation with verbose logging
python scripts/run_digital_twin_simulation.py --verbose

# Test specific component
python -m pytest tests/unit/test_digital_twin/ -v -k "simulator"
```

## Integration Points

- **Vision AI**: Defect detection updates machine health
- **RL Scheduling**: Uses DT as training environment
- **Predictive Maintenance**: Consumes sensor data
- **API Gateway**: Exposes DT state via REST/WebSocket
