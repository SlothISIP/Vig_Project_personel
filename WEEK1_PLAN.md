# Week 1 Execution Plan: Foundation Phase

## 🎯 Week 1 Objectives

**Primary Goals:**
1. ✅ Set up complete development environment
2. ✅ Acquire and validate datasets
3. ✅ Implement baseline Vision AI model
4. ✅ Create MVP inference pipeline
5. ✅ Establish ML experiment tracking

**Success Metrics:**
- Working defect detection model with >70% accuracy
- End-to-end inference pipeline (image → result) under 200ms
- Reproducible experiments with MLflow
- All code tested and documented

---

## 📅 Day-by-Day Schedule

### **Day 1 (Monday): Project Setup & Data Acquisition**

#### Morning (09:00 - 12:00) - 3 hours

**1. Development Environment Setup** (90 min)
```bash
# Create project structure
mkdir -p digital-twin-factory/{src,data,models,tests,notebooks,config}
cd digital-twin-factory

# Initialize Git
git init
git remote add origin <your-repo-url>

# Set up Python environment
poetry init
poetry add torch torchvision timm opencv-python albumentations
poetry add mlflow dvc pandas numpy matplotlib
poetry add --group dev pytest black ruff mypy jupyter

# Activate environment
poetry shell

# Initialize DVC
dvc init
dvc remote add -d storage s3://your-bucket/dvc-cache  # or local

# Create config files
touch .env.example
touch .gitignore
touch README.md
```

**Expected Output:**
- ✅ Project directory structure created
- ✅ Poetry environment with all dependencies
- ✅ Git + DVC initialized

---

**2. Download Datasets** (60 min)

```python
# scripts/download_datasets.py
"""
Download and organize defect detection datasets
"""
import os
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

def download_mvtec_ad():
    """Download MVTec Anomaly Detection dataset"""
    base_url = "https://www.mvtec.com/company/research/datasets/mvtec-ad"

    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    data_dir = Path('data/raw/mvtec_ad')
    data_dir.mkdir(parents=True, exist_ok=True)

    for category in tqdm(categories, desc="Downloading MVTec AD"):
        url = f"{base_url}/{category}.tar.xz"
        output = data_dir / f"{category}.tar.xz"

        # Download
        if not output.exists():
            urllib.request.urlretrieve(url, output)

        # Extract
        extract_dir = data_dir / category
        if not extract_dir.exists():
            with tarfile.open(output) as tar:
                tar.extractall(data_dir)

    print(f"✅ MVTec AD downloaded to {data_dir}")

def download_dagm():
    """Download DAGM 2007 dataset"""
    # Implement DAGM download
    pass

if __name__ == "__main__":
    download_mvtec_ad()
    download_dagm()
```

**Run:**
```bash
python scripts/download_datasets.py
```

**Expected Output:**
- ✅ MVTec AD dataset (~4.5 GB)
- ✅ DAGM dataset (~1.2 GB)
- ✅ Data organized in `data/raw/`

---

**3. Dataset Exploration** (30 min)

```bash
# Launch Jupyter
jupyter lab

# Create notebook
# notebooks/01_data_exploration.ipynb
```

**Notebook Contents:**
```python
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Load MVTec AD
mvtec_dir = Path('data/raw/mvtec_ad')

# Analyze dataset structure
def analyze_mvtec():
    stats = []
    for category in mvtec_dir.iterdir():
        if not category.is_dir():
            continue

        # Count good/defect images
        good_path = category / 'train' / 'good'
        test_path = category / 'test'

        good_count = len(list(good_path.glob('*.png')))

        defect_types = [d.name for d in test_path.iterdir()
                       if d.is_dir() and d.name != 'good']
        defect_count = sum(len(list((test_path / dt).glob('*.png')))
                          for dt in defect_types)

        stats.append({
            'category': category.name,
            'train_good': good_count,
            'test_defect': defect_count,
            'defect_types': len(defect_types)
        })

    return pd.DataFrame(stats)

df = analyze_mvtec()
print(df)

# Visualize samples
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
categories = list(mvtec_dir.iterdir())[:5]

for i, category in enumerate(categories):
    # Good sample
    good_img = list((category / 'train' / 'good').glob('*.png'))[0]
    img = cv2.imread(str(good_img))
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"{category.name} - Good")
    axes[0, i].axis('off')

    # Defect samples
    test_dir = category / 'test'
    defect_types = [d for d in test_dir.iterdir()
                   if d.is_dir() and d.name != 'good']
    if defect_types:
        for j, dt in enumerate(defect_types[:2], 1):
            defect_img = list(dt.glob('*.png'))[0]
            img = cv2.imread(str(defect_img))
            axes[j, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[j, i].set_title(f"{category.name} - {dt.name}")
            axes[j, i].axis('off')

plt.tight_layout()
plt.savefig('docs/dataset_samples.png', dpi=150)
```

**Expected Output:**
- ✅ Dataset statistics summary
- ✅ Sample visualizations
- ✅ Understanding of data distribution

---

#### Afternoon (13:00 - 18:00) - 5 hours

**4. Data Preprocessing Pipeline** (3 hours)

```python
# src/vision/preprocessing/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class DefectDetectionTransforms:
    """Augmentation pipelines for defect detection"""

    @staticmethod
    def get_train_transforms(image_size=224):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transforms(image_size=224):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

# src/vision/preprocessing/dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2

class MVTecDataset(Dataset):
    """MVTec AD dataset loader"""

    def __init__(
        self,
        data_dir: Path,
        category: str,
        split: str = 'train',
        transform=None
    ):
        self.data_dir = Path(data_dir) / category / split
        self.transform = transform
        self.samples = []
        self.labels = []

        # Load samples
        if split == 'train':
            # Training only has 'good' samples
            good_dir = self.data_dir / 'good'
            for img_path in good_dir.glob('*.png'):
                self.samples.append(str(img_path))
                self.labels.append(0)  # 0 = good
        else:
            # Test has good + defect samples
            for defect_type in self.data_dir.iterdir():
                if not defect_type.is_dir():
                    continue

                label = 0 if defect_type.name == 'good' else 1

                for img_path in defect_type.glob('*.png'):
                    self.samples.append(str(img_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
```

**Test the pipeline:**
```python
# Test script
from src.vision.preprocessing.dataset import MVTecDataset
from src.vision.preprocessing.transforms import DefectDetectionTransforms
from torch.utils.data import DataLoader

# Create dataset
transforms = DefectDetectionTransforms.get_train_transforms()
dataset = MVTecDataset(
    data_dir='data/raw/mvtec_ad',
    category='bottle',
    split='train',
    transform=transforms
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Test loading
for images, labels in dataloader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    break

# ✅ Expected: torch.Size([32, 3, 224, 224])
```

---

**5. MLflow Setup** (2 hours)

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 --port 5000
```

```python
# src/core/mlflow_utils.py
import mlflow
from typing import Dict, Any
from pathlib import Path

class MLflowTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, experiment_name: str, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)
```

**Expected Output:**
- ✅ MLflow server running at http://localhost:5000
- ✅ Experiment tracking utilities ready

---

**Evening Summary (18:00 - 19:00)**

**Completed Today:**
- [x] Development environment configured
- [x] Datasets downloaded and explored
- [x] Data preprocessing pipeline implemented
- [x] MLflow experiment tracking set up

**Tomorrow's Prep:**
- [ ] Review Swin Transformer architecture
- [ ] Prepare training script template

---

### **Day 2 (Tuesday): Baseline Model Training**

#### Morning (09:00 - 12:00) - 3 hours

**1. Implement Swin Transformer Model** (2 hours)

```python
# src/vision/models/swin_transformer.py
import torch
import torch.nn as nn
import timm
from typing import Dict, Any

class SwinDefectDetector(nn.Module):
    """Swin Transformer for defect detection"""

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Load pretrained Swin
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_attention_maps(self):
        """Extract attention maps from last block"""
        # This requires modifying forward pass
        # Implementation depends on specific Swin version
        pass
```

**Test model:**
```python
import torch

# Create model
model = SwinDefectDetector(num_classes=2, pretrained=True)

# Test forward pass
x = torch.randn(4, 3, 224, 224)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # [4, 2]
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ✅ Expected: ~28M parameters for Swin-Tiny
```

---

**2. Training Pipeline** (1 hour)

```python
# src/vision/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import time

class DefectDetectionTrainer:
    """Training pipeline for defect detection models"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        mlflow_tracker=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.mlflow = mlflow_tracker

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': correct / total
        }

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': correct / total
        }

    def train(self, num_epochs: int):
        """Full training loop"""
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Log to MLflow
            if self.mlflow:
                self.mlflow.log_metrics(metrics, step=epoch)

            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {metrics['train_loss']:.4f} | "
                  f"Train Acc: {metrics['train_acc']*100:.2f}%")
            print(f"Val Loss: {metrics['val_loss']:.4f} | "
                  f"Val Acc: {metrics['val_acc']*100:.2f}%")

            # Save best model
            if metrics['val_acc'] > best_val_acc:
                best_val_acc = metrics['val_acc']
                self.save_checkpoint(f'models/checkpoints/best_epoch{epoch}.pth')
                print(f"✅ New best model saved!")

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
```

---

#### Afternoon (13:00 - 18:00) - 5 hours

**3. Run Baseline Training** (4 hours + monitoring)

```python
# scripts/train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path

from src.vision.models.swin_transformer import SwinDefectDetector
from src.vision.preprocessing.dataset import MVTecDataset
from src.vision.preprocessing.transforms import DefectDetectionTransforms
from src.vision.training.trainer import DefectDetectionTrainer
from src.core.mlflow_utils import MLflowTracker

def main():
    # Configuration
    config = {
        'model_name': 'swin_tiny_patch4_window7_224',
        'image_size': 224,
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'categories': ['bottle', 'cable', 'capsule']  # Start with 3 categories
    }

    # MLflow tracking
    mlflow = MLflowTracker(experiment_name="defect-detection-baseline")

    with mlflow.start_run(run_name="swin-tiny-baseline"):
        # Log config
        mlflow.log_params(config)

        # Create datasets
        train_transforms = DefectDetectionTransforms.get_train_transforms(
            config['image_size']
        )
        val_transforms = DefectDetectionTransforms.get_val_transforms(
            config['image_size']
        )

        train_datasets = []
        val_datasets = []

        for category in config['categories']:
            train_ds = MVTecDataset(
                data_dir='data/raw/mvtec_ad',
                category=category,
                split='train',
                transform=train_transforms
            )
            val_ds = MVTecDataset(
                data_dir='data/raw/mvtec_ad',
                category=category,
                split='test',
                transform=val_transforms
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        # Combine datasets
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # Create model
        model = SwinDefectDetector(
            model_name=config['model_name'],
            num_classes=2,
            pretrained=True,
            freeze_backbone=False
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Trainer
        trainer = DefectDetectionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config['device'],
            mlflow_tracker=mlflow
        )

        # Train
        print("🚀 Starting training...")
        trainer.train(num_epochs=config['num_epochs'])

        print("✅ Training complete!")

if __name__ == "__main__":
    main()
```

**Run training:**
```bash
python scripts/train_baseline.py

# Expected runtime: ~2-3 hours on single GPU
```

**Expected Output:**
```
Train samples: 627
Val samples: 386

Epoch 1/20
Train Loss: 0.4521 | Train Acc: 78.23%
Val Loss: 0.3214 | Val Acc: 85.47%
✅ New best model saved!

Epoch 2/20
Train Loss: 0.2891 | Train Acc: 88.51%
Val Loss: 0.2156 | Val Acc: 91.19%
✅ New best model saved!

...

Epoch 20/20
Train Loss: 0.0734 | Train Acc: 97.28%
Val Loss: 0.1423 | Val Acc: 93.52%

✅ Training complete!
```

---

**4. Model Evaluation & Analysis** (1 hour)

```python
# notebooks/02_model_evaluation.ipynb

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load best model
model = SwinDefectDetector(num_classes=2)
checkpoint = torch.load('models/checkpoints/best_epoch15.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to('cuda')

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to('cuda')
        outputs = model(images)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
print(classification_report(all_labels, all_preds,
                          target_names=['Good', 'Defect']))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('docs/confusion_matrix_day2.png')
```

---

**Day 2 Summary:**

**Completed:**
- [x] Swin Transformer model implemented
- [x] Training pipeline created
- [x] Baseline model trained (93.5% val accuracy)
- [x] Model evaluation completed

**Key Metrics:**
- Training time: ~2.5 hours
- Best validation accuracy: 93.52%
- Model size: 28M parameters

---

### **Day 3 (Wednesday): Inference Optimization**

#### Morning (09:00 - 12:00)

**1. ONNX Export** (2 hours)

```python
# src/vision/utils/model_converter.py
import torch
import onnx
import onnxruntime as ort
from typing import Tuple

class ModelConverter:
    """Convert PyTorch models to ONNX and TensorRT"""

    @staticmethod
    def export_to_onnx(
        model: torch.nn.Module,
        input_shape: Tuple[int, int, int, int],
        output_path: str,
        opset_version: int = 14
    ):
        """Export PyTorch model to ONNX"""
        model.eval()

        # Dummy input
        dummy_input = torch.randn(*input_shape)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        print(f"✅ ONNX model saved to {output_path}")

    @staticmethod
    def optimize_onnx(input_path: str, output_path: str):
        """Optimize ONNX model"""
        import onnxoptimizer

        model = onnx.load(input_path)
        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized = onnxoptimizer.optimize(model, passes)

        onnx.save(optimized, output_path)
        print(f"✅ Optimized ONNX model saved to {output_path}")

# Export baseline model
model = SwinDefectDetector(num_classes=2)
checkpoint = torch.load('models/checkpoints/best_epoch15.pth')
model.load_state_dict(checkpoint['model_state_dict'])

converter = ModelConverter()
converter.export_to_onnx(
    model=model,
    input_shape=(1, 3, 224, 224),
    output_path='models/onnx/swin_defect_fp32.onnx'
)

converter.optimize_onnx(
    input_path='models/onnx/swin_defect_fp32.onnx',
    output_path='models/onnx/swin_defect_optimized.onnx'
)
```

---

**2. ONNX Inference Pipeline** (1 hour)

```python
# src/vision/inference/onnx_infer.py
import onnxruntime as ort
import numpy as np
import cv2
from typing import Dict, Any
import time

class ONNXInferenceEngine:
    """Fast inference with ONNX Runtime"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Select execution provider
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize
        image = cv2.resize(image, (224, 224))

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - mean) / std

        # HWC to CHW
        image = image.transpose(2, 0, 1)

        # Add batch dimension
        image = np.expand_dims(image, 0).astype(np.float32)

        return image

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run inference"""
        start_time = time.time()

        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        # Postprocess
        probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        pred_class = np.argmax(probs, axis=1)[0]
        confidence = probs[0, pred_class]

        inference_time = (time.time() - start_time) * 1000

        return {
            'defect_type': 'Defect' if pred_class == 1 else 'Good',
            'confidence': float(confidence),
            'inference_time_ms': inference_time
        }

# Benchmark
engine = ONNXInferenceEngine('models/onnx/swin_defect_optimized.onnx')

# Load test image
image = cv2.imread('data/raw/mvtec_ad/bottle/test/broken_large/000.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Warmup
for _ in range(10):
    engine.predict(image)

# Benchmark
times = []
for _ in range(100):
    result = engine.predict(image)
    times.append(result['inference_time_ms'])

print(f"Average inference time: {np.mean(times):.2f}ms")
print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
print(f"FPS: {1000 / np.mean(times):.1f}")

# ✅ Target: <50ms average, >20 FPS
```

---

#### Afternoon (13:00 - 18:00)

**3. End-to-End Inference API** (3 hours)

```python
# src/api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from src.vision.inference.onnx_infer import ONNXInferenceEngine

app = FastAPI(title="Defect Detection API")

# Load model on startup
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = ONNXInferenceEngine('models/onnx/swin_defect_optimized.onnx')
    print("✅ Model loaded successfully")

@app.post("/api/v1/detect")
async def detect_defects(file: UploadFile = File(...)):
    """Detect defects in uploaded image"""

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Invalid file type")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict
    result = engine.predict(image)

    return JSONResponse(content=result)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn src.api.main:app --reload --port 8000
```

**Test API:**
```bash
# Terminal 1: Start server
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Test endpoint
curl -X POST "http://localhost:8000/api/v1/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/raw/mvtec_ad/bottle/test/broken_large/000.png"

# Expected response:
# {
#   "defect_type": "Defect",
#   "confidence": 0.9823,
#   "inference_time_ms": 42.3
# }
```

---

**4. API Load Testing** (1 hour)

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import random

class DefectDetectionUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def detect_defect(self):
        files = {
            'file': open('data/raw/mvtec_ad/bottle/test/broken_large/000.png', 'rb')
        }
        self.client.post("/api/v1/detect", files=files)

# Run load test
# locust -f tests/performance/locustfile.py --host http://localhost:8000
```

**Expected Results:**
- Target: 100 req/s with p95 latency <200ms
- Single worker achieves ~20-30 req/s

---

**5. Write Day 3 Documentation** (1 hour)

```markdown
# Day 3 Results: Inference Optimization

## Achievements
✅ ONNX export pipeline implemented
✅ Optimized inference engine (42ms avg latency)
✅ FastAPI endpoint deployed
✅ Load testing completed

## Performance Metrics
- **Inference Latency**:
  - Average: 42.3ms
  - 95th percentile: 51.7ms
  - FPS: 23.6

- **API Throughput**:
  - Single worker: 25 req/s
  - Target: 100 req/s (requires 4 workers)

## Next Steps
- [ ] Implement batch inference
- [ ] Add attention map extraction
- [ ] Deploy with Gunicorn (multiple workers)
```

---

### **Days 4-5: Buffer & Integration**

*(Detailed plans continue similarly...)*

**Day 4 Focus:**
- Implement attention map extraction
- Create visualization utilities
- Build simple frontend demo
- Write comprehensive tests

**Day 5 Focus:**
- Integration testing
- Performance profiling
- Documentation
- Prepare Week 2 planning
- (Optional) Start Digital Twin state machine

---

## 📊 Week 1 Success Criteria Checklist

**Technical Deliverables:**
- [ ] Working defect detection model (>90% accuracy)
- [ ] ONNX inference pipeline (<50ms latency)
- [ ] FastAPI endpoint with load testing
- [ ] MLflow experiment tracking
- [ ] Comprehensive test coverage (>80%)

**Documentation:**
- [ ] README with setup instructions
- [ ] API documentation (auto-generated by FastAPI)
- [ ] Architecture diagrams
- [ ] Performance benchmarks

**Code Quality:**
- [ ] All code passes black formatting
- [ ] All code passes ruff linting
- [ ] All code passes mypy type checking
- [ ] Git commit history is clean

---

## 💰 Estimated Claude Code Credit Usage

| Day | Task | Estimated Credits |
|-----|------|-------------------|
| 1 | Environment setup + data prep | $10-15 |
| 2 | Model training (20 epochs) | $30-40 |
| 3 | Optimization + API development | $15-20 |
| 4-5 | Testing + integration | $20-30 |
| **Total Week 1** | | **$75-105** |

**Remaining Budget:** $695-$725 for Weeks 2-14

---

## 🚨 Risk Mitigation

**Potential Blockers:**

1. **GPU Out of Memory during training**
   - Solution: Reduce batch size to 16 or 8
   - Solution: Use gradient accumulation

2. **Dataset download fails**
   - Solution: Manual download from official sources
   - Backup: Use smaller subset initially

3. **ONNX export errors**
   - Solution: Downgrade to opset 13
   - Solution: Simplify model architecture

4. **API latency too high**
   - Solution: Use batch inference
   - Solution: Deploy on faster hardware
   - Solution: Switch to lighter model (EfficientNet)

---

*This detailed plan ensures a productive first week with tangible deliverables and sets the foundation for the remaining 13 weeks.*
