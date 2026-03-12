# Deepfake Image Detector (modular)

A lightweight, modular image deepfake detection toolkit mirroring the text pipeline's flexibility. It lets you:

## Choose among multiple feature extractors:

- **Standard extractors:**
  - Frequency Analysis (FFT radial power spectrum)
  - CNN-based (ResNet50)
  - Vision Transformer (ViT)
  - SOTA models via timm (any backbone supported by timm)
  - Optional Multi-scale Analysis wrapper (mean or concat pooling)

- **Face-aware extractors** (NEW - optimized for face deepfake detection):
  - **Face-Frequency Extractor**: Detects periodic patterns from GANs on aligned faces
  - **Face-Spatial Extractor**: Learns texture/lighting artifacts from CNN/ViT on aligned faces
  - **Face-Multimodal Extractor**: Combines spatial + frequency + landmarks for robust detection

- Plug in classic classifiers (scikit-learn): Logistic Regression, Linear SVM, Random Forest
- Run quick train/eval from a CLI against a folder dataset

This is a starter you can grow into full experiments and API later.

## Folder layout

- `deepfake_image_detector/`
  - `models/`
    - `extractors.py` — Standard feature extractors (frequency, ResNet50, ViT, timm/SOTA)
    - `face_preprocessor.py` — Face detection, alignment, and preprocessing (NEW)
    - `face_extractors.py` — Face-aware feature extractors (NEW)
    - `classifiers.py` — Thin wrappers over sklearn classifiers
    - `pipeline.py` — Pipeline to extract features + train/test a classifier
  - `utils/`
    - `data_loader.py` — Minimal dataset loader from an image folder (class subfolders)
  - `scripts/`
    - `train_and_evaluate.py` — CLI to train/evaluate on a folder dataset
- `tests/` — Minimal tests (safe to skip heavy deps if missing)

## Install

Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Notes:

- `torch`, `torchvision`, and `timm` are only needed for deep model extractors. Frequency extractor works with just `numpy` and `Pillow`.
- For face-aware extractors, install **at least one** face detector:
  - **MediaPipe** (recommended, lightweight, no heavy deps): Already in requirements
  - **MTCNN** (via facenet-pytorch): For higher accuracy
  - **RetinaFace** (state-of-the-art): For best detection quality
- If CUDA is required, install the matching torch build from pytorch.org.

## Dataset format

Assumes an ImageFolder-style directory:

```
/path/to/dataset/
  real/
    img1.jpg
    img2.jpg
  fake/
    img3.jpg
    img4.jpg
```

Class names map to integer labels in alphabetical order.

## Pipeline B: Face-Specific Deepfake Detection

### Goal
Detect manipulated or AI-generated faces (swap, reenactment, diffusion-generated portraits).

### Use cases
- Face swap or morphing detection
- ID verification integrity
- Portrait authenticity scoring

### ⚙️ Preprocessing Steps

| Step | Purpose | Implementation |
|------|---------|-----------------|
| 1. Face detection | Extract region of interest | MTCNN, RetinaFace, or Mediapipe |
| 2. Face alignment | Normalize orientation (eyes horizontal) | Similarity transform (cv2) |
| 3. Crop & resize | Fixed input (e.g., 224×224) | OpenCV resize |
| 4. Optional landmarks | Focus on high-impact zones (eyes, lips, jawline) | Mediapipe landmarks |
| 5. Normalize | ImageNet standardization | mean/std normalization |

### 🧠 Architecture

| Component | Description |
|-----------|-------------|
| **Spatial model** (ResNet or ViT) | Learns texture & lighting artifacts |
| **Frequency model** (FFT) | Detects periodic GAN patterns |
| **Multi-scale fusion** | Combine low- and high-level cues |
| **Classifier** | Binary head → fake/real probability |

## Quick start

### Standard extractors (existing behavior):

Train with Frequency features + Logistic Regression (fast and CPU-only):

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor frequency \
  --classifier logreg
```

Use ResNet50 features (ImageNet pretrained) + Linear SVM:

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor resnet50 \
  --classifier linear_svm
```

### Face-aware extractors (NEW):

#### Face-Frequency (fast, detect GAN patterns):

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_frequency \
  --face-detector mediapipe \
  --classifier logreg
```

#### Face-Spatial (ResNet50 on aligned faces):

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --face-detector mediapipe \
  --classifier linear_svm
```

#### Face-Multimodal (spatial + frequency fusion):

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --face-detector mediapipe \
  --fusion-method concat \
  --classifier rf
```

#### With different face detectors:

```bash
# MTCNN (high accuracy with landmarks)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --face-detector mtcnn \
  --classifier logreg

# RetinaFace (state-of-the-art)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --face-detector retinaface \
  --classifier linear_svm
```

### Multi-scale + face extraction:

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --face-detector mediapipe \
  --fusion-method concat \
  --classifier logreg
```

## API Usage (Python)

```python
from PIL import Image
from deepfake_image_detector.models.face_extractors import build_face_aware_extractor
from deepfake_image_detector.models import build_classifier, ImageDetectionPipeline

# Create face-aware extractor
extractor = build_face_aware_extractor(
    "face_multimodal",
    img_size=224,
    model_name="resnet50",
    detector_name="mediapipe",
    fusion_method="concat",
)

# Create pipeline
classifier = build_classifier("logreg")
pipeline = ImageDetectionPipeline(extractor=extractor, classifier=classifier)

# Train on images
pipeline.fit(train_image_paths, train_labels)

# Predict
predictions = pipeline.predict(test_image_paths)
proba = pipeline.predict_proba(test_image_paths)
```

### Face preprocessing (standalone):

```python
from deepfake_image_detector.models.face_preprocessor import create_face_preprocessor

# Create preprocessor
preprocessor = create_face_preprocessor(
    detector_name="mediapipe",
    output_size=(224, 224),
    normalize_imagenet=True,
)

# Process image
image = Image.open("face.jpg")
face_crop, landmarks, metadata = preprocessor.preprocess_image(image)

print(f"Face detected: {metadata['n_faces']}")
print(f"Aligned: {metadata['aligned']}")
if face_crop is not None:
    print(f"Face shape: {face_crop.shape}")
```

## Extending

- Add a new standard extractor: implement `BaseImageFeatureExtractor` and register it in `build_extractor`.
- Add a new face-aware extractor: extend `FaceAwareFrequencyExtractor` or `FaceAwareSpatialExtractor` and register in `build_face_aware_extractor`.
- Add a new face detector: implement `FaceDetector` interface with `.detect()` and optionally `.detect_with_landmarks()`.
- Add a new classifier: wrap a sklearn estimator in `BaseClassifier` and register in `build_classifier`.
- For end-to-end fine-tuning, replace the sklearn step with a small torch head and trainer.

## License

This repo contains only original, generic scaffolding code. You're responsible for dataset licensing and model usage.
