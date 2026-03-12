# Pipeline B: Face-Specific Deepfake Detection

## Overview

A complete face-aware preprocessing and detection pipeline that optimizes for detecting manipulated or AI-generated faces (face swap, reenactment, diffusion-generated portraits).

## Architecture

```
Input Image
    ↓
[Face Detection] ← MTCNN / RetinaFace / MediaPipe
    ↓
[Face Alignment] ← Eye-based similarity transform
    ↓
[Fixed Crop & Resize] ← 224×224 output
    ↓
[Parallel Feature Extraction]
    ├→ [Spatial Model: ResNet50/ViT] → Texture & lighting artifacts
    ├→ [Frequency Model: FFT/Wavelet] → Periodic GAN patterns
    └→ [Multi-scale Analysis] → Low & high-level cues
    ↓
[Multi-modal Fusion] ← concat or mean pooling
    ↓
[Classifier: Linear SVM / RF / LogReg]
    ↓
Output: Fake/Real Probability
```

## Components

### 1. Face Preprocessor (`face_preprocessor.py`)

**Purpose**: Detect, align, and normalize face regions.

**Key Classes**:
- `FaceDetector`: Base interface
  - `MediapipeFaceDetector`: Lightweight, no heavy deps
  - `MTCNNFaceDetector`: High accuracy with landmarks
  - `RetinaFaceDetector`: State-of-the-art detection
- `FacePreprocessor`: Main preprocessing pipeline
  - Detects faces
  - Aligns using eye landmarks (similarity transform)
  - Crops with margin and resizes to fixed size
  - Applies ImageNet normalization

**Key Functions**:
- `create_face_preprocessor()`: Factory to create preprocessor
- `align_and_crop_face()`: Crop and align a single face
- `face_alignment_transform()`: Compute similarity transform
- `create_landmark_mask()`: Create masks for focus zones (eyes, lips, jaw)

**Example**:
```python
from deepfake_image_detector.models.face_preprocessor import create_face_preprocessor

preprocessor = create_face_preprocessor(
    detector_name="mediapipe",
    output_size=(224, 224),
    normalize_imagenet=True,
)

image = Image.open("face.jpg")
face_crop, landmarks, metadata = preprocessor.preprocess_image(image)
# face_crop: (224, 224, 3) normalized image
# landmarks: (5, 2) array of [left_eye, right_eye, nose, mouth_l, mouth_r]
# metadata: dict with detection info and alignment status
```

### 2. Face-Aware Extractors (`face_extractors.py`)

**Purpose**: Extract features optimized for face deepfake detection.

**Key Classes**:

#### FaceAwareFrequencyExtractor
- Detects periodic patterns from GANs (StyleGAN, ProGAN, etc.)
- Uses FFT on aligned face region
- Adds detection confidence as metadata
- **Speed**: Fast (no torch required)
- **Use when**: Quick baseline, limited compute

#### FaceAwareSpatialExtractor
- Uses ResNet50 or ViT on aligned face region
- Learns texture and lighting artifacts
- Adds alignment status as metadata
- **Speed**: Medium (requires torch)
- **Use when**: Good balance needed

#### FaceAwareMultiModalExtractor
- **Combines both spatial and frequency features**
- Fusion: concatenation or mean pooling
- Includes alignment metadata
- **Speed**: Slower but best accuracy
- **Use when**: Highest accuracy needed

**Example**:
```python
from deepfake_image_detector.models.face_extractors import build_face_aware_extractor

# Frequency only
extractor = build_face_aware_extractor("face_frequency")

# Spatial (ResNet50)
extractor = build_face_aware_extractor(
    "face_spatial",
    model_name="resnet50",
    detector_name="mediapipe",
)

# Multi-modal fusion
extractor = build_face_aware_extractor(
    "face_multimodal",
    model_name="resnet50",
    detector_name="mediapipe",
    fusion_method="concat",  # or "mean"
)

features = extractor.extract(image)  # Returns numpy array
```

### 3. Enhanced Pipeline (`pipeline.py`)

Updated to support:
- Face preprocessing with failure handling
- Metadata tracking per image
- Configurable skip-on-failure for robust training

**Example**:
```python
from deepfake_image_detector.models import ImageDetectionPipeline

pipeline = ImageDetectionPipeline(
    extractor=extractor,
    classifier=classifier,
    skip_failed=True,  # Skip images that fail preprocessing
)

pipeline.fit(train_paths, train_labels)
predictions = pipeline.predict(test_paths)
```

### 4. Training Script (`scripts/train_and_evaluate.py`)

**New command-line arguments**:
- `--extractor {face_frequency, face_spatial, face_multimodal}`
- `--face-detector {mediapipe, mtcnn, retinaface}`
- `--fusion-method {concat, mean}`
- `--skip-failed`: Skip images that fail preprocessing

## Preprocessing Steps

| Step | Purpose | Implementation | Notes |
|------|---------|-----------------|-------|
| **1. Face Detection** | Extract ROI | MTCNN / RetinaFace / MediaPipe | Returns bounding box(es) |
| **2. Face Alignment** | Normalize orientation | Similarity transform using eye landmarks | Eyes made horizontal |
| **3. Crop & Resize** | Fixed input size | OpenCV resize to 224×224 | Margin applied (~20%) |
| **4. Normalization** | ImageNet standardization | (x - mean) / std | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

## Feature Extractors

### Frequency Analysis (Face-Frequency)
- **Input**: Aligned face region (224×224)
- **Method**: 2D FFT → radial power spectrum
- **Output**: 64 radial bins + 3 stats + 1 confidence = ~68 features
- **Detects**: Periodic patterns from GANs, compression artifacts
- **Speed**: ⚡ Very fast (no torch)
- **Dependencies**: numpy, PIL, OpenCV

### Spatial Analysis (Face-Spatial)
- **Input**: Aligned face region (224×224)
- **Method**: ResNet50 (ImageNet pretrained) → global pooling
- **Output**: 2048 features + 1 alignment status
- **Detects**: Texture inconsistencies, lighting artifacts, blending seams
- **Speed**: ⚡⚡ Medium (requires GPU for speed)
- **Dependencies**: torch, torchvision, timm

### Multi-Modal Fusion (Face-Multimodal)
- **Combines**: Spatial + Frequency
- **Fusion**: Concatenation (recommended) or mean pooling
- **Output (concat)**: ~2048 + ~68 + 2 metadata = ~2118 features
- **Output (mean)**: ~2050 features (more compact)
- **Speed**: ⚡⚡⚡ Slower but best accuracy
- **Use case**: When highest accuracy is needed

## Face Detectors

| Detector | Speed | Accuracy | Landmarks | Dependencies | Notes |
|----------|-------|----------|-----------|--------------|-------|
| **MediaPipe** | ⚡⚡ Fast | Good | 5 points | mediapipe | Default, lightweight, recommended |
| **MTCNN** | ⚡ Medium | Good | 5 points | facenet-pytorch | Classic, reliable |
| **RetinaFace** | ⚡⚡ Medium | ⭐ Excellent | 5 points | retinaface | SOTA, best for difficult angles |

### When to use which detector:
- **MediaPipe**: Default choice, no heavy dependencies
- **MTCNN**: When you want reliable results with minimal setup
- **RetinaFace**: When you need best accuracy (e.g., profile faces, extreme angles)

## Usage Examples

### Command Line

**Fast baseline (Frequency only)**:
```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_frequency \
  --face-detector mediapipe \
  --classifier logreg
```

**Good balance (Spatial + ResNet50)**:
```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --face-detector mediapipe \
  --classifier linear_svm
```

**Best accuracy (Multi-modal fusion)**:
```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --face-detector mediapipe \
  --fusion-method concat \
  --classifier rf
```

**With different detector**:
```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --face-detector retinaface \
  --classifier logreg
```

### Python API

```python
from PIL import Image
from deepfake_image_detector.models.face_extractors import build_face_aware_extractor
from deepfake_image_detector.models import build_classifier, ImageDetectionPipeline

# Create components
extractor = build_face_aware_extractor(
    "face_multimodal",
    img_size=224,
    model_name="resnet50",
    detector_name="mediapipe",
    fusion_method="concat",
)
classifier = build_classifier("rf")

# Create pipeline
pipeline = ImageDetectionPipeline(
    extractor=extractor,
    classifier=classifier,
    skip_failed=True,  # Skip failed images
)

# Train
pipeline.fit(train_image_paths, train_labels)

# Predict
predictions = pipeline.predict(test_image_paths)
proba = pipeline.predict_proba(test_image_paths)
accuracy = pipeline.score(test_image_paths, test_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Predictions: {predictions}")
print(f"Probabilities:\n{proba}")
```

### Face Preprocessing (Standalone)

```python
from deepfake_image_detector.models.face_preprocessor import create_face_preprocessor
from PIL import Image

preprocessor = create_face_preprocessor(
    detector_name="mediapipe",
    output_size=(224, 224),
    normalize_imagenet=True,
)

image = Image.open("face.jpg")
face_crop, landmarks, metadata = preprocessor.preprocess_image(image)

if face_crop is not None:
    print(f"Face shape: {face_crop.shape}")  # (224, 224, 3)
    print(f"Aligned: {metadata['aligned']}")
    print(f"Number of faces: {metadata['n_faces']}")
else:
    print("No face detected")
```

## Key Features

### ✅ Robustness
- **Multiple face detectors** for different scenarios
- **Face alignment** for consistent inputs
- **Skip-failed mode** for handling problematic images
- **Metadata tracking** for debugging

### ✅ Flexibility
- **Three extractor options** (frequency, spatial, multi-modal)
- **Multiple classifiers** (LogReg, LinearSVM, RandomForest)
- **Configurable fusion** (concat or mean)
- **Modular design** for easy extension

### ✅ Performance
- **CPU-friendly options** (frequency extractor)
- **GPU acceleration** for deep models
- **Scalable** from prototype to production

### ✅ Interpretability
- **Metadata** on alignment, detection, confidence
- **Frequency analysis** detects specific GAN artifacts
- **Spatial features** reveal texture inconsistencies

## Architecture Decision: Why This Design?

1. **Two-stage pipeline**: Detection → Feature extraction
   - Separates concerns
   - Makes it easy to swap detectors/extractors
   - Handles "no face found" gracefully

2. **Three extractor options**:
   - **Frequency**: Fast baseline, detects periodic patterns
   - **Spatial**: Deep learning, detects texture artifacts
   - **Multi-modal**: Best of both, fusion for robustness

3. **Face alignment critical**:
   - Ensures consistent eye position across images
   - Helps spatial models learn discriminative features
   - Improves frequency analysis stability

4. **Metadata tracking**:
   - Understand preprocessing failures
   - Debug model behavior
   - Monitor data quality

## Extension Points

### Add a new face detector:
```python
from deepfake_image_detector.models.face_preprocessor import FaceDetector

class CustomDetector(FaceDetector):
    def detect(self, image: np.ndarray):
        # Your detection code
        return [(x1, y1, x2, y2), ...]
    
    def detect_with_landmarks(self, image):
        # Optional: return landmarks
        return bboxes, landmarks
```

### Add a new extractor:
```python
from deepfake_image_detector.models.extractors import BaseImageFeatureExtractor

class CustomExtractor(BaseImageFeatureExtractor):
    def extract(self, image: Image.Image) -> np.ndarray:
        # Your extraction code
        return features
```

### Custom pipeline:
```python
from deepfake_image_detector.models.face_preprocessor import create_face_preprocessor
from deepfake_image_detector.models.extractors import BaseImageFeatureExtractor

class CustomFaceExtractor(BaseImageFeatureExtractor):
    def __init__(self):
        self.preprocessor = create_face_preprocessor()
    
    def extract(self, image):
        face, lms, meta = self.preprocessor.preprocess_image(image)
        # Your custom feature extraction on face_crop
        return features
```

## Dependencies

### Core (always required):
- `numpy>=1.22`
- `Pillow>=9.0`
- `scikit-learn>=1.1`
- `opencv-python>=4.5.0`

### Face detection (choose at least one):
- `mediapipe>=0.10.0` (lightweight, default)
- `facenet-pytorch>=2.5.0` (MTCNN, optional)
- `retinaface>=0.0.13` (SOTA, optional)

### Deep extractors (optional, for torch-based models):
- `torch>=2.0`
- `torchvision>=0.15`
- `timm>=0.9.2`

### To install all:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### No face detected in images
- Use a different detector: `--face-detector retinaface`
- Check image quality and resolution
- Enable `--skip-failed` to continue training

### Out of memory with face_multimodal
- Use `face_spatial` or `face_frequency` instead
- Reduce batch size in your training loop
- Use smaller classifier (LogReg instead of RF)

### Poor accuracy
- Check if faces are properly aligned (see metadata)
- Try different detector (retinaface > mtcnn > mediapipe)
- Increase training data
- Try different classifiers (RF > LinearSVM > LogReg)

### Very slow preprocessing
- Use `mediapipe` detector (fastest)
- Use `face_frequency` extractor (no torch)
- Process images in parallel if possible

## Performance Benchmarks (Indicative)

On a typical laptop (4-core CPU, no GPU):

| Extractor | Per-image time | VRAM | Notes |
|-----------|----------------|------|-------|
| face_frequency | ~50ms | <100MB | Fastest |
| face_spatial | ~500ms | ~2GB | Requires GPU for speed |
| face_multimodal | ~550ms | ~2GB | Slowest but best |

With GPU (e.g., RTX 3080):
- face_spatial: ~10ms
- face_multimodal: ~15ms

## Citation & References

- **Face alignment**: Similarity transform based on eye landmarks (standard CV technique)
- **Frequency analysis**: FFT-based detection of periodic GAN artifacts
- **ResNet50**: He et al., 2016 - "Deep Residual Learning for Image Recognition"
- **MediaPipe**: Google's lightweight ML framework
- **MTCNN**: Zhang et al., 2016 - "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- **RetinaFace**: Deng et al., 2020 - "RetinaFace: Single-stage Dense Face Localisation in the Wild"

## Next Steps

1. Prepare dataset in ImageFolder format
2. Install dependencies: `pip install -r requirements.txt`
3. Choose pipeline based on your needs:
   - Fast: `face_frequency`
   - Balanced: `face_spatial`
   - Best: `face_multimodal`
4. Train and evaluate
5. Monitor metadata for preprocessing quality
6. Iterate: adjust detector, extractor, or classifier

For questions or issues, refer to `examples_face_detection.py` for detailed usage examples.
