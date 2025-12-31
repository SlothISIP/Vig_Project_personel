---
name: Vision AI Tester
description: Test and analyze Vision AI defect detection system
tags: vision, ai, testing, defect-detection
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Vision AI Testing Agent

## Task
Test and analyze the Vision AI defect detection system.

## Context
Vision AI components:
- `/src/vision/models/swin_transformer.py` - Swin Vision Transformer
- `/src/vision/inference/onnx_infer.py` - ONNXInferenceEngine
- `/src/vision/explainability/grad_cam.py` - SwinGradCAM
- `/src/vision/explainability/defect_explainer.py` - DefectExplainer, RootCauseHypothesis
- `/src/vision/preprocessing/transforms.py` - Data augmentation
- `/src/vision/training/trainer.py` - Model training
- `/src/vision/training/metrics.py` - Accuracy, precision, F1

## Defect Types
- SCRATCH, STAIN, CRACK, DENT, HOLE, CONTAMINATION

## Severity Levels
- MINOR, MODERATE, MAJOR, CRITICAL

## Instructions

1. **If no arguments**: Full Vision AI analysis
   - Check model configuration
   - Verify preprocessing pipeline
   - Review explainability setup
   - Analyze inference engine

2. **If "test" argument**: Run vision tests
   ```bash
   cd /home/user/Vig_Project_personel && python -m pytest tests/unit/test_vision/ -v
   ```

3. **If "export" argument**: Export model to ONNX
   ```bash
   cd /home/user/Vig_Project_personel && python scripts/export_onnx.py
   ```

4. **If "explain" argument**: Focus on XAI components
   - Review Grad-CAM implementation
   - Check defect explainer logic
   - Verify root cause analysis

5. **If "api" argument**: Test vision API endpoints
   - POST /api/v1/predict
   - POST /api/v1/predict/batch
   - Check response formats

## Model Configuration
- Default: swin_tiny
- Image size: 224x224
- Confidence threshold: 0.7

Arguments: $ARGUMENTS
