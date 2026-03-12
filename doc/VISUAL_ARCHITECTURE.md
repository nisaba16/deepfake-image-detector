# Visual Architecture Guide: Face-Specific Deepfake Detection

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT: Raw Image File                              │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FACE PREPROCESSING                                │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 1. FACE DETECTION (Choose one)                                        │   │
│ │    ├─ 🚀 MediaPipe (Fast, lightweight) ← DEFAULT                     │   │
│ │    ├─ 🎯 MTCNN (Accurate, landmarks)                                 │   │
│ │    └─ ⭐ RetinaFace (SOTA quality)                                   │   │
│ └────────────────────────┬─────────────────────────────────────────────┘   │
│                          ↓                                                  │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 2. FACE ALIGNMENT                                                      │   │
│ │    ├─ Detect eye landmarks (left_eye, right_eye)                      │   │
│ │    ├─ Compute similarity transform (rotation + scale + translation)    │   │
│ │    └─ Warp image → eyes horizontal                                    │   │
│ └────────────────────────┬─────────────────────────────────────────────┘   │
│                          ↓                                                  │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 3. CROP & RESIZE                                                       │   │
│ │    ├─ Apply margin (~20% expansion)                                    │   │
│ │    ├─ Crop to face region                                              │   │
│ │    └─ Resize to 224×224 (fixed size)                                  │   │
│ └────────────────────────┬─────────────────────────────────────────────┘   │
│                          ↓                                                  │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 4. NORMALIZATION                                                       │   │
│ │    └─ ImageNet standardization: (x - mean) / std                      │   │
│ │       mean = [0.485, 0.456, 0.406]                                     │   │
│ │       std  = [0.229, 0.224, 0.225]                                     │   │
│ └────────────────────────┬─────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                  ✓ Preprocessed Face: (224, 224, 3)                        │
│                    + Metadata (alignment, confidence, bbox)                │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION (3 Options)                         │
│                                                                              │
│  ╔════════════════════╗   ╔════════════════════╗   ╔═════════════════════╗│
│  ║ OPTION 1: FREQ     ║   ║ OPTION 2: SPATIAL  ║   ║ OPTION 3: MULTI     ║│
│  ║ (Fast, CPU)        ║   ║ (Medium, GPU✓)     ║   ║ (Best, GPU✓)        ║│
│  ╠════════════════════╣   ╠════════════════════╣   ╠═════════════════════╣│
│  ║ Input: Aligned    ║   ║ Input: Aligned     ║   ║ Input: Aligned      ║│
│  ║ Face 224×224      ║   ║ Face 224×224       ║   ║ Face 224×224        ║│
│  ║         ↓         ║   ║         ↓          ║   ║         ↓↓          ║│
│  ║ 2D FFT            ║   ║ ResNet50           ║   ║ FFT + ResNet50      ║│
│  ║ (frequency shift) ║   ║ (ImageNet pre)     ║   ║ (dual stream)       ║│
│  ║         ↓         ║   ║         ↓          ║   ║         ↓↓          ║│
│  ║ Radial Power      ║   ║ Global Avg Pool    ║   ║ Freq: ~68 feats     ║│
│  ║ Spectrum (64 bins)║   ║ → 2048 features    ║   ║ Spatial: 2048 feats ║│
│  ║         ↓         ║   ║         ↓          ║   ║         ↓           ║│
│  ║ Global Stats:     ║   ║ + Alignment Status ║   ║ FUSION: concat/mean ║│
│  ║ - mean, std       ║   ║ (metadata)         ║   ║         ↓           ║│
│  ║ - high-freq ratio ║   ║         ↓          ║   ║ 2118 or 2050        ║│
│  ║         ↓         ║   ║ Output: (2049,)    ║   ║ features            ║│
│  ║ + Confidence      ║   ║                    ║   ║         ↓           ║│
│  ║ (metadata)        ║   ║ Learns:            ║   ║ Output: (2118,)     ║│
│  ║         ↓         ║   ║ - Texture artifacts║   ║                     ║│
│  ║ Output: (68,)     ║   ║ - Lighting oddities║   ║ Learns:             ║│
│  ║                   ║   ║ - Blending seams   ║   ║ - Both patterns +    ║│
│  ║ Detects:          ║   ║                    ║   ║   textures          ║│
│  ║ - GAN patterns    ║   ║                    ║   ║                     ║│
│  ║ - Periodic    ║   ║                    ║   ║ Best overall        ║│
│  ║   artifacts   ║   ║                    ║   ║ detection            ║│
│  ║ - Compression ║   ║                    ║   ║                     ║│
│  ║   artifacts   ║   ║                    ║   ║                     ║│
│  ╚════════════════════╝   ╚════════════════════╝   ╚═════════════════════╝│
│           │                       │                       │                │
│           └───────────────────────┼───────────────────────┘                │
│                                   ↓                                        │
│                          Feature Vector (n_features,)                     │
└────────────────────────────────────────┬───────────────────────────────────┘
                                         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLASSIFICATION                                     │
│  ┌──────────────────────┬──────────────────────┬──────────────────────┐    │
│  │ Logistic Regression  │ Linear SVM           │ Random Forest        │    │
│  │ (Fast)               │ (Balanced)           │ (Accurate)           │    │
│  │                      │                      │                      │    │
│  │ Best for:            │ Best for:            │ Best for:            │    │
│  │ - Quick baseline     │ - Good balance       │ - Maximum accuracy   │    │
│  │ - Limited data       │ - Production         │ - Sufficient data    │    │
│  │ - Real-time inference│ - Balanced speed     │ - Off-line analysis  │    │
│  └──────────────┬───────┴──────────────┬───────┴──────────────┬──────┘    │
│                 └──────────────────────┼──────────────────────┘            │
│                                        ↓                                    │
│                         Binary Classification Head                         │
│                              ↓         ↓                                    │
│                         PREDICTED CLASS + PROBABILITY                      │
│                         (Real: 0) or (Fake: 1)                           │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         ↓
                          OUTPUT: {pred, confidence}
```

## Data Flow: Training

```
TRAINING DATA
   │
   ├─→ [Dataset Root]
   │       real/
   │       └── img_*.jpg
   │       fake/
   │       └── img_*.jpg
   │
   ↓
[stratified_split(test_size=0.2)]
   │
   ├─→ X_train (80%)  →──┐
   │                     │
   └─→ X_test (20%)   →──┤
                        ↓
                   [Image Pipeline]
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    [Preprocess]  [Preprocess]   [Preprocess]
        │               │               │
        ├─→ Face 1 Feats  ├─→ Face 2 Feats  ├─→ Face 3 Feats
        │               │               │
        └───────────────┴───────────────┘
                        ↓
                   Stack Features
                   Align to (n, d)
                        │
        ┌───────────────┴───────────────┐
        ↓                               ↓
   X_TRAIN FEATURES              X_TEST FEATURES
   (n_train, d)                  (n_test, d)
        │                               │
        ↓                               ↓
   [classifier.fit()]            [classifier.predict()]
        │                               │
        ↓                               ↓
   TRAINED MODEL                PREDICTIONS
                                        ↓
                               [Evaluate: Accuracy, F1, etc.]
```

## Preprocessing Detail: Face Alignment

```
BEFORE ALIGNMENT           AFTER ALIGNMENT
─────────────────          ─────────────────

  Raw Face Image         Normalized Face
  (may be rotated)       (eyes horizontal)

         🔄
      / | \                  __|__
     /  |  \                  /O O\
        |                        |
        |                       \_/
       / \

KEY POINTS USED:
- Left Eye:     ~100, ~100
- Right Eye:    ~300, ~100
- Transform:    Computed via cv2.estimateAffinePartial2D()
- Warp:         cv2.warpAffine() with similarity matrix

TARGET EYE POSITIONS (in 224×224 output):
- Left Eye:  (67, 89)     ← 0.3*224, 0.4*224
- Right Eye: (157, 89)    ← 0.7*224, 0.4*224
```

## Feature Comparison Matrix

```
┌─────────────────┬──────────────┬───────────┬──────────┬─────────┐
│ Aspect          │ Frequency    │ Spatial   │ Multimod │ Notes   │
├─────────────────┼──────────────┼───────────┼──────────┼─────────┤
│ Speed (CPU)     │ 50ms ⚡⚡    │ 500ms ⚡  │ 550ms ⚡ │ Per img │
│ Speed (GPU)     │ 50ms ⚡⚡    │ 10ms ⚡⚡⚡│ 15ms ⚡⚡⚡│         │
│ Accuracy        │ 70-80% 📊    │ 85-92% 📊 │ 90-95% 📊│ Typical │
│ Dependencies    │ 2 (minimal)  │ 5 (torch) │ 6 (both) │         │
│ VRAM            │ <100MB       │ ~2GB      │ ~2GB     │ w/ model│
│ Use Case        │ Baseline     │ Production│ Best     │         │
│ Detects         │ GAN patterns │ Texture   │ Both     │         │
├─────────────────┼──────────────┼───────────┼──────────┼─────────┤
│ Recommended     │ For quick    │ For good  │ For max  │ Choose  │
│ When            │ testing      │ accuracy  │ accuracy │ based   │
│                 │ Limited CPU  │ Limited   │ No       │ on      │
│                 │              │ constraints│         │ goal    │
└─────────────────┴──────────────┴───────────┴──────────┴─────────┘
```

## Face Detector Selection Guide

```
┌──────────────┬────────────┬───────────────┬──────────────────────┐
│ MediaPipe ✨ │ MTCNN      │ RetinaFace 🏆 │                      │
├──────────────┼────────────┼───────────────┼──────────────────────┤
│ Speed        │ FAST ⚡⚡  │ MEDIUM ⚡     │ MEDIUM ⚡            │
│ Quality      │ GOOD       │ GOOD          │ EXCELLENT ⭐⭐⭐    │
│ Landmarks    │ 5 pts      │ 5 pts         │ 5 pts                │
│              │ (basic)    │ (reliable)    │ (refined)            │
│ Dependencies │ mediapipe  │ facenet-torch │ retinaface           │
│ (pip)        │ (1 package)│ (1 package)   │ (1 package)          │
├──────────────┼────────────┼───────────────┼──────────────────────┤
│ DEFAULT ✓    │            │               │                      │
│ NO INSTALL   │            │               │                      │
│ NEEDED       │            │               │                      │
├──────────────┼────────────┼───────────────┼──────────────────────┤
│ USE WHEN     │ You want   │ You want      │ You want best        │
│              │ speed and  │ reliability   │ accuracy (e.g.,      │
│              │ simplicity │ and good      │ extreme angles,      │
│              │ (default)  │ accuracy      │ difficult scenarios) │
│              │            │               │                      │
│ BEST FOR     │ Most cases │ Production    │ Critical apps       │
│              │ Prototypes │ systems       │ (ID verification)    │
│              │ Quick PoC  │ Balanced      │ High-stakes uses    │
└──────────────┴────────────┴───────────────┴──────────────────────┘

FACE DETECTION EXAMPLES:

        MEDIAPIPE        MTCNN         RETINAFACE
        ─────────        ─────         ──────────
        
  :)      :)              :)             :)
 Face    Face           Face           Face
 Good    Good           Good           Perfect
 
        Profile       Profile        Profile
        ─────────     ─────          ──────
        
  /(    /(            /(             /(
 Face  ~Face        Face            Face
 OK    WEAK         Good            Good

Extreme Angle
──────────────

 \(     \(          \(              \(
 Face  FAIL        WEAK            Good
 FAIL             Maybe           OK
```

## Workflow Decision Tree

```
START: Want to detect deepfake faces?
│
├─ LIMITED TIME / QUICK TEST?
│  └─ YES → face_frequency + mediapipe + logreg ⚡
│           (70-80% accuracy, 50ms/image)
│
├─ WANT GOOD ACCURACY?
│  └─ YES → face_spatial + mediapipe + linear_svm ⚡⚡
│           (85-92% accuracy, 500ms/image)
│
├─ NEED MAXIMUM ACCURACY?
│  └─ YES → face_multimodal + retinaface + rf ⚡⚡⚡
│           (90-95% accuracy, 550ms/image)
│
├─ HAVE MANY EDGE CASE FACES?
│  └─ YES → Use retinaface detector
│           (handles extreme angles/profiles)
│
├─ NEED REAL-TIME PROCESSING?
│  └─ YES → face_frequency (no GPU needed)
│           OR face_spatial on GPU
│
└─ PRODUCTION DEPLOYMENT?
   └─ YES → face_multimodal with skip_failed=True
            Use retinaface for best quality
            Monitor metadata for debugging
```

## Feature Extraction Comparison

```
                    FREQUENCY               SPATIAL                MULTIMODAL
                    ─────────               ────────               ──────────

INPUT:              Aligned 224×224         Aligned 224×224        Aligned 224×224
                    Face Region             Face Region            Face Region

PROCESSING:
                    FFT → Power             ResNet50 →             FFT → Freq
                    Spectrum →              Global Avg Pool        ResNet50 → Spatial
                    Radial Bins             
                    
                    ↓                       ↓                      ↓↓
                    
FEATURES:           64 radial bins          2048 CNN               2048 (spatial) +
                    +mean, std,             features               68 (frequency)
                    high-freq ratio         +alignment status      +alignment status
                    +confidence
                    
                    ~68 features            ~2049 features         ~2118 features

DETECTS:            • GAN patterns          • Texture              • All of above
                    • Periodicity           • Lighting             • Texture
                    • Compression           • Blending seams       • Patterns
                    • Artifact rings        • Inconsistencies      • Lighting
                    
BEST AT:            PATTERN                 ARTIFACT               OVERALL
                    DETECTION               DETECTION              DETECTION
                    
SPEED:              FASTEST                 SLOWER                 SLOWEST
                    (no torch)              (torch needed)         (both needed)
                    
ACCURACY:           70-80%                  85-92%                 90-95%
                    
MEMORY:             MINIMAL                 ~2GB                   ~2GB
                    
TYPICAL USE:        Quick baseline          Production             High-stakes
                    PoC testing             systems                detection
```

## Installation Paths

```
┌─ MINIMAL (Frequency only)
│  └─ pip install numpy pillow scikit-learn opencv-python mediapipe
│     → Only works with face_frequency + mediapipe
│     → No torch, no MTCNN, no RetinaFace
│     → ~500MB disk, <1GB RAM
│
├─ STANDARD (Recommended)
│  └─ pip install -r requirements.txt
│     → All features enabled
│     → Torch (4GB), all detectors optional
│     → ~6GB disk, <3GB RAM
│
└─ FULL (All options)
   └─ pip install torch torchvision timm mediapipe \
        facenet-pytorch retinaface scikit-learn opencv-python
      → Maximum flexibility
      → All detectors + extractors
      → ~8GB disk, <4GB RAM
```

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║               FACE DEEPFAKE DETECTION QUICK REFERENCE          ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║ PREPROCESSING STEPS:                                          ║
║   1. Detect face (bbox)        → 3 detectors available        ║
║   2. Align face (similarity)   → Eyes normalized horizontally  ║
║   3. Crop & resize             → Fixed 224×224 output        ║
║   4. Normalize                 → ImageNet standardization     ║
║                                                                ║
║ FEATURE EXTRACTORS:                                           ║
║   face_frequency   ⚡  Fast, CPU-only, patterns (70-80%)     ║
║   face_spatial    ⚡⚡ Balanced, GPU, textures (85-92%)      ║
║   face_multimodal ⚡⚡⚡ Best, GPU, combined (90-95%)        ║
║                                                                ║
║ FACE DETECTORS:                                               ║
║   mediapipe        ← Default, lightweight                      ║
║   mtcnn           → Accurate, established                      ║
║   retinaface      → Best quality, handles profiles            ║
║                                                                ║
║ CLASSIFIERS:                                                  ║
║   logreg          → Fast                                       ║
║   linear_svm      → Balanced                                   ║
║   rf              → Most accurate                              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```
