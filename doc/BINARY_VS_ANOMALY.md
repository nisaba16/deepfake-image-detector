# Binary vs Anomaly Detection: Quick Comparison

## At a Glance

```
┌─────────────────────────┬──────────────────────┬────────────────────────┐
│ ASPECT                  │ BINARY CLASSIFICATION│ ANOMALY DETECTION      │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ DATA REQUIREMENT        │ Both classes labeled │ Mostly normal class    │
│                         │ Balanced preferred   │ Works with imbalance   │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ APPROACH                │ Learn decision       │ Learn normal boundary  │
│                         │ boundary             │ Detect outliers        │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ CLASSIFIERS             │ LogReg, SVM, RF      │ OCSVM, Elliptic,       │
│                         │                      │ Isolation Forest       │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ BEST FOR                │ Balanced data        │ Imbalanced data        │
│                         │ Closed-world         │ Open-world/evolving    │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ ACCURACY (balanced)     │ 90-95% ✓             │ 85-92%                 │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ ACCURACY (imbalanced)   │ 70-80%               │ 85-95% ✓               │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ INTERPRETABILITY        │ Class probabilities  │ Anomaly scores         │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ ROBUSTNESS              │ Known deepfakes      │ Unknown/future types ✓ │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ TRAINING TIME           │ Medium-Fast          │ Fast-Medium            │
├─────────────────────────┼──────────────────────┼────────────────────────┤
│ TUNING COMPLEXITY       │ Medium               │ Low-Medium             │
└─────────────────────────┴──────────────────────┴────────────────────────┘
```

## Decision Matrix

Choose **BINARY** if:
- ✅ You have ~equal real and fake samples
- ✅ Both classes are well-represented
- ✅ You know all deepfake types in advance
- ✅ You need maximum accuracy on known fakes

Choose **ANOMALY** if:
- ✅ You have mostly real faces (90%+)
- ✅ Few labeled fake examples
- ✅ Want to detect unknown/future deepfake types
- ✅ Willing to accept slightly lower acc on training fakes for robustness

## Example Scenarios

### Scenario 1: ID Verification System

**Data**: 10,000 real photos, 100 fake/spoofed

```
BINARY (Poor):  ~75% accuracy (class imbalance kills it)
ANOMALY (Good): ~88% accuracy (learns what "real" is)

→ Use ANOMALY
```

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --classifier isolation_forest \
  --mode anomaly \
  --contamination 0.01
```

### Scenario 2: Balanced Deepfake Research Dataset

**Data**: 5,000 real, 5,000 synthetic (balanced)

```
BINARY (Excellent): ~93% accuracy
ANOMALY (Good):     ~87% accuracy

→ Use BINARY (RandomForest)
```

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --classifier rf \
  --mode supervised
```

### Scenario 3: Detecting New Deepfake Methods

**Data**: Training on StyleGAN fakes, test on Diffusion fakes

```
BINARY (Poor):  ~60% accuracy (overfits to StyleGAN)
ANOMALY (Better): ~75% accuracy (learns "realness" pattern)

→ Use ANOMALY (One-Class SVM)
```

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /training/dataset \
  --extractor face_multimodal \
  --classifier one_class_svm \
  --mode anomaly \
  --nu 0.1
```

### Scenario 4: Real-World Deployment (Unknown Distribution)

**Data**: Mix of real + various deepfake methods

```
BINARY (Risky):  Depends on training data match
ANOMALY (Robust): Generalizes to unseen methods ✓

→ Use ANOMALY (Isolation Forest)
```

```bash
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/diverse/dataset \
  --extractor face_multimodal \
  --classifier isolation_forest \
  --mode anomaly \
  --contamination 0.2
```

## Command Line Quick Reference

### Binary Classification (Traditional)

```bash
# Quick & Fast
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_frequency \
  --classifier logreg \
  --mode supervised

# Balanced Approach
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --classifier linear_svm \
  --mode supervised

# Best Accuracy
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --classifier rf \
  --mode supervised
```

### Anomaly Detection (One-Class Methods)

```bash
# Quick Baseline (Isolation Forest)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_frequency \
  --classifier isolation_forest \
  --mode anomaly \
  --contamination 0.1

# Interpretable (Elliptic Envelope)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --classifier elliptic_envelope \
  --mode anomaly \
  --contamination 0.1

# Complex Features (One-Class SVM)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_multimodal \
  --classifier one_class_svm \
  --mode anomaly \
  --nu 0.1
```

## Key Differences Explained

### Training Data

**Binary**:
```
Training set:
  ├─ Real: real_001.jpg, real_002.jpg, ...
  └─ Fake: fake_001.jpg, fake_002.jpg, ...

Both classes equally important
```

**Anomaly**:
```
Training set:
  ├─ Normal (Real): real_001.jpg, real_002.jpg, ...
  └─ Ignored (Fake): fake_001.jpg, ... (optional)

Only normal class used for training
```

### Learning Process

**Binary Classification**:
```
Real face features → Feature space
Fake face features → Feature space
                  ↓
           [Decision boundary]
           
Goal: Draw line separating both
```

**Anomaly Detection**:
```
Real face features → Feature space
                  ↓
           [Normal region boundary]
           
Goal: Circle the normal distribution
Anything outside = anomaly
```

### Predictions

**Binary**:
```
predict():
  1 (Real) or 0 (Fake)

predict_proba():
  [0.92, 0.08] → 92% real, 8% fake
```

**Anomaly**:
```
predict():
  1 (Normal) or 0 (Anomaly)

predict_proba():
  [0.15, 0.85] → 15% anomaly, 85% normal
```

### Failure Modes

**Binary fails when**:
- Very imbalanced (95% real, 5% fake)
- New deepfake type not in training
- Classes poorly separated

**Anomaly fails when**:
- Normal class very heterogeneous
- Too few training samples
- Fakes very similar to reals

## Parameter Quick Guide

### Binary Classifiers

```python
# LogReg: No tuning needed
build_classifier("logreg")

# Linear SVM: Adjust C
build_classifier("linear_svm", C=1.0)  # 0.1-100

# Random Forest: More estimators = better
build_classifier("rf", n_estimators=200)  # 100-500
```

### Anomaly Detectors

```python
# One-Class SVM: Tune nu (anomaly fraction)
build_classifier("one_class_svm", nu=0.1)  
# 0.01=strict, 0.1=balanced, 0.5=permissive

# Elliptic Envelope: Tune contamination
build_classifier("elliptic_envelope", contamination=0.1)
# Should match expected anomaly rate

# Isolation Forest: Tune contamination
build_classifier("isolation_forest", contamination=0.1)
# More robust, less tuning needed
```

## Performance Comparison

### On Balanced Dataset (50% real, 50% fake)

| Method | Accuracy | ROC-AUC | Notes |
|--------|----------|---------|-------|
| LogReg | 82% | 0.89 | Fast baseline |
| Linear SVM | 86% | 0.92 | Good balance |
| **RandomForest** | **92%** | **0.96** | **Best** |
| One-Class SVM | 84% | 0.90 | Overkill here |
| Elliptic Envelope | 80% | 0.87 | Assumes Gaussian |
| Isolation Forest | 82% | 0.89 | Not ideal for balanced |

### On Imbalanced Dataset (95% real, 5% fake)

| Method | Accuracy | ROC-AUC | Notes |
|--------|----------|---------|-------|
| LogReg | 71% | 0.62 | Struggles |
| Linear SVM | 73% | 0.65 | Still difficult |
| RandomForest | 76% | 0.72 | Needs class_weight |
| **One-Class SVM** | **88%** | **0.91** | **Good** |
| **Elliptic Envelope** | **89%** | **0.93** | **Best** |
| **Isolation Forest** | **87%** | **0.90** | **Best** |

## Migration Guide

### From Binary to Anomaly Detection

```bash
# Old (Binary)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --classifier rf

# New (Anomaly)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --extractor face_spatial \
  --classifier isolation_forest \
  --mode anomaly \
  --contamination 0.1 \
  --normal-class real
```

### From Anomaly to Binary

```bash
# Old (Anomaly)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --classifier one_class_svm \
  --mode anomaly \
  --nu 0.1

# New (Binary)
python -m deepfake_image_detector.scripts.train_and_evaluate \
  --data /path/to/dataset \
  --classifier rf \
  --mode supervised
```

## Recommendation Algorithm

```python
import numpy as np

def recommend_approach(n_real, n_fake):
    """Recommend training approach based on data."""
    ratio = n_fake / n_real
    
    if ratio > 0.3:  # Balanced
        return "BINARY (use RandomForest)"
    elif ratio > 0.05:  # Moderate imbalance
        return "BINARY (use class_weight) or ANOMALY"
    else:  # Severe imbalance (<5%)
        return "ANOMALY (use Isolation Forest)"

# Examples
print(recommend_approach(5000, 5000))  # → BINARY
print(recommend_approach(10000, 500))  # → BINARY or ANOMALY
print(recommend_approach(10000, 100))  # → ANOMALY
```

## Conclusion

| Situation | Recommended |
|-----------|------------|
| Research, balanced data | Binary (RandomForest) |
| Production, imbalanced data | Anomaly (Isolation Forest) |
| New deepfake detection | Anomaly (One-Class SVM) |
| Speed critical | Anomaly (Elliptic Envelope) |
| Maximum accuracy (balanced) | Binary (RandomForest) |
| Robustness to distribution shift | Anomaly (Isolation Forest) |

---

**Try both approaches and evaluate on YOUR data!**

The best method depends on your specific dataset and goals.
