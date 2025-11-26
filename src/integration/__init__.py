"""
Integration Module - Unified Pipeline for Vision-Digital Twin-RL.

This module provides the core innovation: a closed-loop system that
integrates Vision AI, Digital Twin simulation, and RL-based scheduling.

Key Components:
1. FeedbackLoop: Connects defect detection to Digital Twin state updates
2. UnifiedPipeline: End-to-end processing pipeline
3. RealTimeController: Real-time orchestration of all components

Architecture:
    +----------------+     +------------------+     +----------------+
    |   Vision AI    | --> | Feedback Loop    | --> | Digital Twin   |
    | (Defect+XAI)   |     | (Analysis)       |     | (Simulator)    |
    +----------------+     +------------------+     +----------------+
           ^                       |                       |
           |                       v                       v
    +----------------+     +------------------+     +----------------+
    |   Input        |     | Root Cause       |     |   RL Agent     |
    |   Image        |     | Analysis         |     | (Scheduling)   |
    +----------------+     +------------------+     +----------------+

Innovations:
    - Option B: Digital Twin-in-the-Loop RL for scheduling optimization
    - Option C: Explainable AI with feedback for continuous improvement
"""

from .feedback_loop import (
    IntegratedFeedbackLoop,
    VisionDigitalTwinBridge,
    DigitalTwinFeedbackController,
    DefectFeedback,
    QualityMetrics,
    StationHealthModel,
    FeedbackType,
    create_feedback_loop,
)

from .unified_pipeline import (
    UnifiedPipeline,
    PipelineConfig,
    PipelineState,
    PipelineMode,
    create_demo_pipeline,
    run_integration_demo,
)

__all__ = [
    # Feedback Loop
    "IntegratedFeedbackLoop",
    "VisionDigitalTwinBridge",
    "DigitalTwinFeedbackController",
    "DefectFeedback",
    "QualityMetrics",
    "StationHealthModel",
    "FeedbackType",
    "create_feedback_loop",
    # Unified Pipeline
    "UnifiedPipeline",
    "PipelineConfig",
    "PipelineState",
    "PipelineMode",
    "create_demo_pipeline",
    "run_integration_demo",
]
