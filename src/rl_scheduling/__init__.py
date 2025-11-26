"""
Reinforcement Learning-based Production Scheduling Module

Key Innovation: Digital Twin-in-the-Loop RL
- DigitalTwinRLEnv: Uses actual factory simulator as RL environment
- Enables sim-to-real transfer learning
- Domain randomization for robustness
"""

from .state import (
    SchedulingStateRL,
    MachineStateRL,
    JobStateRL,
    SchedulingAction,
    state_from_dict,
)
from .reward import RewardCalculator, RewardConfig
from .environment import ProductionSchedulingEnv
from .digital_twin_env import (
    DigitalTwinRLEnv,
    SimToRealConfig,
    ActionType,
    create_digital_twin_env,
)
from .trainer import SchedulingRLTrainer
from .inference import RLSchedulingPolicy, DynamicScheduler

__all__ = [
    # State
    "SchedulingStateRL",
    "MachineStateRL",
    "JobStateRL",
    "SchedulingAction",
    "state_from_dict",
    # Reward
    "RewardCalculator",
    "RewardConfig",
    # Environment - Standard
    "ProductionSchedulingEnv",
    # Environment - Digital Twin (INNOVATION)
    "DigitalTwinRLEnv",
    "SimToRealConfig",
    "ActionType",
    "create_digital_twin_env",
    # Training
    "SchedulingRLTrainer",
    # Inference
    "RLSchedulingPolicy",
    "DynamicScheduler",
]
