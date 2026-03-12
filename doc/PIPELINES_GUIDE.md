# Image Processing Pipelines Guide

This document describes all available pipelines for deepfake image detection in this project.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Pipeline 1: MobileNetV3 (Client-Side)](#pipeline-1-mobilenetv3-client-side)
3. [Pipeline 2: EfficientNet-B0](#pipeline-2-efficientnet-b0)
4. [Pipeline 3: Classic ML Pipeline](#pipeline-3-classic-ml-pipeline)
5. [Pipeline 4: Face-Specific Detection](#pipeline-4-face-specific-detection)
6. [Pipeline 5: Anomaly Detection](#pipeline-5-anomaly-detection)
7. [Preprocessing Options](#preprocessing-options)
8. [Quick Command Reference](#quick-command-reference)

---

## Pipeline Overview

| Pipeline | Model | Size | Use Case | Accuracy | Speed |
|----------|-------|------|----------|----------|-------|
| MobileNetV3 | Deep Learning | ~4 MB | Browser deployment | High | ⚡⚡⚡ Fast |
| EfficientNet-B0 | Deep Learning | ~20 MB | Server deployment | Very High | ⚡⚡ Medium |
| Classic ML | Traditional ML | <1 MB | Quick baseline | Medium-High | ⚡⚡⚡ Fast |
| Face-Specific | Hybrid | Varies | Face manipulation | Very High | ⚡⚡ Medium |
| Anomaly Detection | Unsupervised | <1 MB | Imbalanced data | Medium | ⚡⚡⚡ Fast |

---

## Pipeline 1: MobileNetV3 (Client-Side)

**Best for**: Browser-based deployment, client-side inference

### Architecture
```
Input Image (RGB)
    ↓
[Face Detection] (optional)
    ↓
[6-Channel Preprocessing]
    ├─ Channel 0-2: RGB (normalized 0-1)
    ├─ Channel 3: Saturation (HSV)
    ├─ Channel 4: SRM Noise Filter
    └─ Channel 5: FFT Magnitude
    ↓
[MobileNetV3-Small Backbone]
    ↓
[Classification Head]
    ↓
Binary Output (Real/Fake)
```

### Key Features
- **Ultra-lightweight**: ~4 MB model size
- **6-channel input**: RGB + Saturation + SRM + FFT
- **ONNX export**: Ready for browser deployment (ONNX.js)
- **Pre-trained backbone**: ImageNet weights for RGB channels
- **Fast inference**: Optimized for mobile/browser

### Usage

#### Basic Training
```bash
python scripts/train_mobilenet_client.py \
    --data data/dataset \
    --epochs 20 \
    --batch-size 32 \
    --export-onnx
```

#### Quick Test (200 samples, 3 epochs)
```bash
bash train_mobilenet.sh
```

#### With Face Detection
```bash
python scripts/train_mobilenet_client.py \
    --data data/dataset \
    --epochs 20 \
    --use-face-detection \
    --face-detector mediapipe \
    --skip-failed
```

### Parameters
- `--data`: Path to dataset (ImageFolder format)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.2)
- `--pretrained`: Use ImageNet weights (default: True)
- `--use-face-detection`: Enable face detection preprocessing
- `--face-detector`: Detector type (mediapipe, mtcnn, retinaface)
- `--skip-failed`: Skip images with failed preprocessing
- `--export-onnx`: Export to ONNX format
- `--output-dir`: Output directory (default: client_side_model)

### Output Files
- `best_model.pth`: PyTorch model weights
- `deepfake_mobilenet.onnx`: ONNX model for browser
- `preprocessing_info.json`: 6-channel preprocessing details
- `results.json`: Evaluation metrics

### When to Use
✅ Browser-based applications  
✅ Mobile deployment  
✅ Real-time inference required  
✅ Limited bandwidth/storage  
❌ Maximum accuracy critical  

---

## Pipeline 2: EfficientNet-B0

**Best for**: Server-side deployment, better accuracy

### Architecture
```
Input Image (RGB)
    ↓
[Face Detection] (optional)
    ↓
[6-Channel Preprocessing]
    ├─ Channel 0-2: RGB (normalized 0-1)
    ├─ Channel 3: Saturation (HSV)
    ├─ Channel 4: SRM Noise Filter
    └─ Channel 5: FFT Magnitude
    ↓
[EfficientNet-B0 Backbone]
    ↓
[Classification Head]
    ↓
Binary Output (Real/Fake)
```

### Key Features
- **Better accuracy**: ~20 MB model size
- **6-channel input**: Same preprocessing as MobileNetV3
- **Efficient architecture**: State-of-the-art efficiency
- **ONNX export**: Server deployment ready
- **Pre-trained backbone**: ImageNet weights

### Usage

#### Basic Training
```bash
python scripts/train_efficientnet.py \
    --data data/dataset \
    --epochs 20 \
    --batch-size 32 \
    --export-onnx
```

#### Quick Test
```bash
bash train_efficientnet.sh
```

#### With Face Detection
```bash
python scripts/train_efficientnet.py \
    --data data/dataset \
    --epochs 20 \
    --use-face-detection \
    --face-detector mediapipe \
    --skip-failed
```

### Parameters
Same as MobileNetV3 (see above)

### Output Files
- `best_model.pth`: PyTorch model weights
- `deepfake_efficientnet.onnx`: ONNX model
- `preprocessing_info.json`: Preprocessing details
- `results.json`: Evaluation metrics

### When to Use
✅ Server-side deployment  
✅ Higher accuracy needed  
✅ <100MB size constraint  
✅ GPU available for inference  
❌ Browser deployment  
❌ Ultra-low latency required  

---

## Pipeline 3: Classic ML Pipeline

**Best for**: Quick baselines, interpretable models

### Architecture
```
Input Image
    ↓
[Feature Extraction]
    ├─ Frequency Analysis (FFT)
    ├─ CNN Features (ResNet50)
    ├─ Vision Transformer (ViT)
    ├─ TIMM Models (any backbone)
    └─ Multi-scale Analysis
    ↓
[Classic Classifier]
    ├─ Logistic Regression
    ├─ Linear SVM
    └─ Random Forest
    ↓
Binary Output (Real/Fake)
```

### Available Extractors

#### 1. Frequency Extractor
```python
from models.extractors import FrequencyExtractor

extractor = FrequencyExtractor(
    method='fft',        # 'fft' or 'wavelet'
    output_size=128,     # Feature dimension
    pooling='mean'       # 'mean' or 'max'
)
```

#### 2. ResNet50 Extractor
```python
from models.extractors import ResNet50Extractor

extractor = ResNet50Extractor(
    pretrained=True,
    freeze_backbone=True,
    pooling='avg'
)
```

#### 3. Vision Transformer
```python
from models.extractors import ViTExtractor

extractor = ViTExtractor(
    model_name='vit_base_patch16_224',
    pretrained=True
)
```

#### 4. Multi-scale Extractor
```python
from models.extractors import MultiScaleExtractor

base_extractor = FrequencyExtractor()
extractor = MultiScaleExtractor(
    base_extractor=base_extractor,
    scales=[224, 448, 896],
    fusion='concat'  # 'concat' or 'mean'
)
```

### Available Classifiers

#### Logistic Regression
```python
from models.classifiers import LogisticRegressionClassifier

classifier = LogisticRegressionClassifier(
    C=1.0,
    max_iter=1000
)
```

#### Linear SVM
```python
from models.classifiers import LinearSVMClassifier

classifier = LinearSVMClassifier(
    C=1.0,
    probability=True
)
```

#### Random Forest
```python
from models.classifiers import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=None
)
```

### Usage

#### Command Line
```bash
python scripts/train_and_evaluate.py \
    --data data/dataset \
    --extractor frequency \
    --classifier logreg \
    --test-size 0.2
```

#### Python API
```python
from models.pipeline import ImageDetectionPipeline
from models.extractors import FrequencyExtractor
from models.classifiers import LogisticRegressionClassifier
from utils.data_loader import collect_image_paths_and_labels

# Load data
paths, labels, class_to_idx, _ = collect_image_paths_and_labels('data/dataset')

# Create pipeline
extractor = FrequencyExtractor()
classifier = LogisticRegressionClassifier()
pipeline = ImageDetectionPipeline(extractor, classifier)

# Train
pipeline.fit(train_paths, train_labels)

# Predict
predictions = pipeline.predict(test_paths)
```

### When to Use
✅ Quick baseline experiments  
✅ Interpretable models needed  
✅ Limited computational resources  
✅ Small datasets  
❌ Maximum accuracy required  
❌ Complex visual patterns  

---

## Pipeline 4: Face-Specific Detection

**Best for**: Face manipulation detection (swap, reenactment, deepfakes)

### Architecture
```
Input Image
    ↓
[Face Detection]
    ├─ MediaPipe (fastest)
    ├─ MTCNN (balanced)
    └─ RetinaFace (best quality)
    ↓
[Face Alignment]
    └─ Eye-based similarity transform
    ↓
[Fixed Crop & Resize] (224×224)
    ↓
[Face-Aware Feature Extraction]
    ├─ Face-Frequency: FFT on aligned faces
    ├─ Face-Spatial: CNN on aligned faces
    └─ Face-Multimodal: Fusion of all
    ↓
[Classifier]
    ↓
Binary Output (Real/Fake)
```

### Available Face Detectors

#### MediaPipe (Default)
- **Speed**: ⚡⚡⚡ Fastest
- **Accuracy**: Good
- **Dependencies**: Lightweight
```python
from models.face_preprocessor import create_face_preprocessor

preprocessor = create_face_preprocessor(
    detector_name='mediapipe',
    output_size=(224, 224)
)
```

#### MTCNN
- **Speed**: ⚡⚡ Medium
- **Accuracy**: Very Good
- **Features**: Returns facial landmarks
```python
preprocessor = create_face_preprocessor(
    detector_name='mtcnn',
    output_size=(224, 224)
)
```

#### RetinaFace
- **Speed**: ⚡ Slower
- **Accuracy**: Best
- **Features**: High-quality detection + landmarks
```python
preprocessor = create_face_preprocessor(
    detector_name='retinaface',
    output_size=(224, 224)
)
```

### Face-Aware Extractors

#### Face-Frequency
```python
from models.face_extractors import FaceFrequencyExtractor

extractor = FaceFrequencyExtractor(
    face_detector_name='mediapipe',
    output_size=(224, 224)
)
```

#### Face-Spatial
```python
from models.face_extractors import FaceSpatialExtractor

extractor = FaceSpatialExtractor(
    backbone='resnet50',  # or 'vit'
    face_detector_name='mediapipe'
)
```

#### Face-Multimodal (Best Accuracy)
```python
from models.face_extractors import FaceMultimodalExtractor

extractor = FaceMultimodalExtractor(
    face_detector_name='mediapipe',
    fusion_method='concat'  # or 'mean'
)
```

### Usage

#### Quick Baseline (Frequency)
```bash
python scripts/train_and_evaluate.py \
    --data data/dataset \
    --extractor face_frequency \
    --face-detector mediapipe \
    --classifier logreg
```

#### Best Accuracy (Multimodal)
```bash
python scripts/train_and_evaluate.py \
    --data data/dataset \
    --extractor face_multimodal \
    --face-detector mediapipe \
    --fusion-method concat \
    --classifier rf
```

### When to Use
✅ Face swap detection  
✅ Face reenactment detection  
✅ Deepfake videos (frame-by-frame)  
✅ Portrait/selfie datasets  
❌ Full scene images  
❌ Non-face manipulations  

---

## Pipeline 5: Anomaly Detection

**Best for**: Imbalanced datasets, one-class learning

### Architecture
```
Input Image
    ↓
[Feature Extraction]
    (same extractors as Classic ML)
    ↓
[Anomaly Detection Classifier]
    ├─ One-Class SVM
    ├─ Elliptic Envelope
    └─ Isolation Forest
    ↓
Binary Output (Normal/Anomaly)
```

### Available Anomaly Detectors

#### One-Class SVM
```python
from models.classifiers import OneClassSVMClassifier

classifier = OneClassSVMClassifier(
    nu=0.1,          # Expected anomaly fraction
    kernel='rbf',    # 'linear', 'rbf', 'poly'
    gamma='scale'
)
```

#### Elliptic Envelope
```python
from models.classifiers import EllipticEnvelopeClassifier

classifier = EllipticEnvelopeClassifier(
    contamination=0.1,  # Expected anomaly fraction
    support_fraction=None
)
```

#### Isolation Forest
```python
from models.classifiers import IsolationForestClassifier

classifier = IsolationForestClassifier(
    contamination=0.1,
    n_estimators=100
)
```

### Training Strategy
Train on **normal class only** (e.g., real faces), then detect anomalies (e.g., fake faces).

### Usage

```bash
python scripts/train_and_evaluate.py \
    --data data/dataset \
    --extractor frequency \
    --classifier one_class_svm \
    --anomaly-detection
```

### When to Use
✅ Highly imbalanced datasets  
✅ Only "real" data available for training  
✅ Novel attack detection  
✅ Zero-shot deepfake detection  
❌ Balanced datasets available  
❌ Maximum accuracy required  

---

## Preprocessing Options

### 6-Channel Preprocessing (MobileNetV3/EfficientNet)

Transforms RGB images into 6 channels:

```python
from models.extractors import SixChannelPreprocessor

processor = SixChannelPreprocessor()
six_channel = processor.process_face_to_6_channels(rgb_image)
```

**Channel Description**:
- **Channel 0-2**: RGB (normalized 0-1)
- **Channel 3**: Saturation from HSV color space
- **Channel 4**: SRM Noise Filter (5×5 high-pass kernel)
- **Channel 5**: FFT Magnitude (log scale, centered, normalized)

### Face Preprocessing

```python
from models.face_preprocessor import create_face_preprocessor

preprocessor = create_face_preprocessor(
    detector_name='mediapipe',
    output_size=(224, 224),
    normalize_imagenet=True,
    margin=0.2  # Expand bbox by 20%
)

# Process image
face_crop, landmarks, metadata = preprocessor.preprocess_image(image)
```

---

## Quick Command Reference

### MobileNetV3 Training
```bash
# Basic
python scripts/train_mobilenet_client.py --data data/dataset --epochs 20 --export-onnx

# With face detection
python scripts/train_mobilenet_client.py --data data/dataset --use-face-detection --skip-failed

# Quick test (200 samples)
bash train_mobilenet.sh
```

### EfficientNet Training
```bash
# Basic
python scripts/train_efficientnet.py --data data/dataset --epochs 20 --export-onnx

# With face detection
python scripts/train_efficientnet.py --data data/dataset --use-face-detection --skip-failed

# Quick test
bash train_efficientnet.sh
```

### Classic ML Pipeline
```bash
# Frequency + Logistic Regression
python scripts/train_and_evaluate.py --data data/dataset --extractor frequency --classifier logreg

# ResNet50 + SVM
python scripts/train_and_evaluate.py --data data/dataset --extractor resnet50 --classifier linear_svm

# ViT + Random Forest
python scripts/train_and_evaluate.py --data data/dataset --extractor vit --classifier rf
```

### Face-Specific Pipeline
```bash
# Face-Frequency (fastest)
python scripts/train_and_evaluate.py --data data/dataset --extractor face_frequency --face-detector mediapipe

# Face-Spatial (balanced)
python scripts/train_and_evaluate.py --data data/dataset --extractor face_spatial --face-detector mtcnn

# Face-Multimodal (best accuracy)
python scripts/train_and_evaluate.py --data data/dataset --extractor face_multimodal --fusion-method concat
```

### Anomaly Detection
```bash
# One-Class SVM
python scripts/train_and_evaluate.py --data data/dataset --extractor frequency --classifier one_class_svm --anomaly-detection

# Isolation Forest
python scripts/train_and_evaluate.py --data data/dataset --extractor resnet50 --classifier isolation_forest
```

---

## Dataset Format

All pipelines expect ImageFolder format:

```
data/dataset/
├── fake/
│   ├── fake_001.jpg
│   ├── fake_002.jpg
│   └── ...
└── real/
    ├── real_001.jpg
    ├── real_002.jpg
    └── ...
```

---

## Performance Comparison

Based on typical face deepfake datasets:

| Pipeline | Accuracy | Inference Time | Model Size | Training Time |
|----------|----------|----------------|------------|---------------|
| MobileNetV3 | 85-92% | ~5ms | 4 MB | 30 min |
| EfficientNet-B0 | 90-95% | ~15ms | 20 MB | 1 hour |
| Face-Frequency + LogReg | 75-85% | ~50ms | <1 MB | 10 min |
| Face-Spatial + SVM | 85-90% | ~100ms | 2 MB | 20 min |
| Face-Multimodal + RF | 90-95% | ~150ms | 5 MB | 30 min |
| Anomaly (One-Class SVM) | 70-80% | ~50ms | <1 MB | 10 min |

*Performance varies by dataset and hardware*

---

## Next Steps

1. **Start with a baseline**: Try MobileNetV3 or Face-Frequency
2. **Iterate for accuracy**: Move to EfficientNet or Face-Multimodal
3. **Optimize for deployment**: Export to ONNX, quantize if needed
4. **Tune hyperparameters**: Learning rate, dropout, batch size
5. **Ensemble models**: Combine multiple pipelines for best results

For more details, see:
- [README.md](README.md) - Project overview
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [FACE_DETECTION_PIPELINE.md](FACE_DETECTION_PIPELINE.md) - Face pipeline details
- [ANOMALY_DETECTION.md](ANOMALY_DETECTION.md) - Anomaly detection guide
