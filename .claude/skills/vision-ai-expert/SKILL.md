---
name: vision-ai-expert
description: Expert knowledge for Vision AI defect detection using Swin Transformer, ONNX, and Grad-CAM explainability. Use when working with image processing, model inference, XAI, or defect classification.
---

# Vision AI Expert Skill

## Overview
This skill provides deep expertise in the Vision AI defect detection system using Swin Transformer with ONNX inference and Grad-CAM explainability.

## Domain Knowledge

### Core Components

#### SwinTransformer (`/src/vision/models/swin_transformer.py`)
- Vision Transformer backbone for defect detection
- Pretrained on ImageNet, fine-tuned for defects
- Supports multiple sizes: tiny, small, base

```python
# Model configuration
model = SwinTransformer(
    model_name="swin_tiny",
    num_classes=6,  # Defect types
    pretrained=True
)
```

#### ONNXInferenceEngine (`/src/vision/inference/onnx_infer.py`)
- ONNX Runtime for fast inference
- CPU and GPU support
- Batch processing capability

```python
engine = ONNXInferenceEngine(
    model_path="models/onnx/swin_defect_optimized.onnx",
    device="cuda"  # or "cpu"
)

# Single inference
result = engine.predict(image)

# Batch inference
results = engine.predict_batch(images)
```

#### DefectExplainer (`/src/vision/explainability/defect_explainer.py`)
- Root cause analysis for detected defects
- Pattern classification and hypothesis generation
- Severity assessment

```python
explainer = DefectExplainer()
explanation = explainer.explain(
    image=image,
    prediction=prediction,
    grad_cam_output=attention_map
)
# Returns: DefectRegion, RootCauseHypothesis, severity
```

#### SwinGradCAM (`/src/vision/explainability/grad_cam.py`)
- Gradient-weighted Class Activation Mapping
- Visual explanation of model decisions
- Highlights defect regions

```python
grad_cam = SwinGradCAM(model)
attention_map = grad_cam.generate(image, class_idx)
# Returns: heatmap overlay showing attention
```

### Defect Types
```python
class DefectType(Enum):
    SCRATCH = "scratch"
    STAIN = "stain"
    CRACK = "crack"
    DENT = "dent"
    HOLE = "hole"
    CONTAMINATION = "contamination"
```

### Severity Levels
```python
class DefectSeverity(Enum):
    MINOR = 1      # Cosmetic, no functional impact
    MODERATE = 2   # Some functional concern
    MAJOR = 3      # Significant quality issue
    CRITICAL = 4   # Product unusable
```

### Data Pipeline

#### Transforms (`/src/vision/preprocessing/transforms.py`)
- Albumentations-based augmentation
- Training vs inference transforms
- Normalization for Swin Transformer

```python
# Training transforms
train_transform = get_training_transforms(image_size=224)

# Inference transforms
infer_transform = get_inference_transforms(image_size=224)
```

#### Dataset (`/src/vision/preprocessing/dataset.py`)
- MVTec AD compatible loader
- Defect/normal classification
- Multi-class defect types

### Training

#### Trainer (`/src/vision/training/trainer.py`)
- PyTorch training loop
- Early stopping support
- Checkpoint management

```python
trainer = DefectTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)
trainer.train(epochs=100)
```

#### Metrics (`/src/vision/training/metrics.py`)
- Accuracy, Precision, Recall, F1
- Per-class metrics
- Confusion matrix

### Best Practices

1. **Model Selection**
   - swin_tiny: Fast inference, good accuracy
   - swin_small: Better accuracy, slower
   - swin_base: Best accuracy, production use

2. **Inference Optimization**
   - Use ONNX for production
   - Enable GPU for batch processing
   - Set appropriate batch sizes (32 recommended)

3. **XAI Integration**
   - Always generate Grad-CAM for defects
   - Include root cause hypothesis in reports
   - Store attention maps for quality tracking

4. **Threshold Tuning**
   - Default confidence: 0.7
   - Adjust based on false positive tolerance
   - Use per-class thresholds for imbalanced data

## Common Patterns

### Full Inference Pipeline
```python
# Load image
image = load_image(path)
transform = get_inference_transforms()
input_tensor = transform(image)

# Inference
engine = ONNXInferenceEngine(model_path)
prediction = engine.predict(input_tensor)

# Explainability
if prediction.is_defect:
    grad_cam = SwinGradCAM(model)
    attention = grad_cam.generate(input_tensor, prediction.class_idx)

    explainer = DefectExplainer()
    explanation = explainer.explain(image, prediction, attention)
```

### API Integration
```python
# POST /api/v1/predict
@app.post("/predict")
async def predict(file: UploadFile):
    image = read_image(file)
    result = inference_engine.predict(image)
    if result.is_defect:
        explanation = explainer.explain(image, result)
    return {"prediction": result, "explanation": explanation}
```

## Troubleshooting

### Common Issues
1. **Low accuracy**: Check image preprocessing matches training
2. **Slow inference**: Use ONNX, enable GPU, batch requests
3. **Grad-CAM fails**: Verify model has gradients enabled
4. **Wrong defect type**: Retrain with more balanced data

### Model Configuration
```python
# Default settings
DEFAULT_MODEL_NAME = "swin_tiny"
MODEL_IMAGE_SIZE = 224
VISION_CONFIDENCE_THRESHOLD = 0.7
VISION_MAX_BATCH_SIZE = 64
VISION_ENABLE_ATTENTION_MAPS = True
```

### Debug Commands
```bash
# Test inference
python -c "from src.vision.inference.onnx_infer import ONNXInferenceEngine; print('OK')"

# Export to ONNX
python scripts/export_onnx.py

# Run vision tests
python -m pytest tests/unit/test_vision/ -v
```

## Integration Points

- **Digital Twin**: Updates machine health based on defect rate
- **Feedback Loop**: Provides defect data for closed-loop control
- **API Gateway**: Exposes prediction endpoints
- **Dashboard**: Displays defect statistics and visualizations
