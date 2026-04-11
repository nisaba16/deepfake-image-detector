# Deepfake Image Detector

Binary deepfake classification toolkit with a full experiment pipeline:
**FP32 training → QAT fine-tuning → ONNX export → INT8 quantization → accuracy / size / latency comparison**.

Supported models: `resnet50`, `vit_b_16`, `mobilenet_v3_small`, `dinov2_vitb14`, `forensic_mobilenet`

---

## Project Structure

```
deepfake-image-detector/
├── scripts/
│   ├── train_fp32.py                  # Phase 1: full-precision training (standard models)
│   ├── train_qat.py                   # Phase 2: quantization-aware fine-tuning
│   ├── train_forensic_mobilenet.py    # Phase 1: ForensicMobileNetV3 (multi-channel)
│   └── evaluate_all.py               # Cross-dataset evaluation of all checkpoints
├── common/
│   ├── forensic_mobilenet.py          # ForensicMobileNetV3 + ForensicNoiseAugment
│   ├── solution.py                    # Quantization primitives (linear_quantize, STE, …)
│   └── utils.py                       # replace_with_quantized_modules
├── onnx_experiments/
│   ├── export_to_onnx.py              # Step 1 – PyTorch → ONNX (FP32 or QAT)
│   ├── data_reader.py                 # Calibration data reader for ORT static quant
│   ├── quantize_onnx.py               # Step 2 – FP32 ONNX → INT8 ONNX (ORT static)
│   └── run_experiments.py             # Step 3 – accuracy + size + latency table
├── utils/
│   └── data_loader.py                 # collect_image_paths_and_labels, stratified_split
├── checkpoints/                       # Saved PyTorch weights (.pth)
├── data/
│   ├── dataset/                       # Training data  (fake/ real/)
│   └── ddata/test/                    # External evaluation data (fake/ real/)
├── slurm_train_models.sh              # SLURM: FP32, QAT, or forensic_mobilenet training
└── slurm_onnx_experiments.sh          # SLURM: full ONNX experiment pipeline
```

---

## Setup

```bash
conda activate deepfake_env
pip install -r requirements.txt
```

### Dataset Format

```
data/dataset/
├── fake/
│   ├── img_001.jpg
│   └── ...
└── real/
    ├── img_001.jpg
    └── ...
```

Labels are assigned alphabetically by folder name: `fake=0`, `real=1`.

---

## Experiment Runs

Each experiment compares **FP32** (trained and run in full precision) against **QAT INT8**
(fine-tuned with simulated quantization, then converted to real INT8 ops via ORT static
quantization). The pipeline is always the same 4 steps:

```
Train FP32 → QAT fine-tune → Export QAT ONNX → ORT static INT8
```

All commands assume you are at the repo root with the virtualenv active.

---

### DINOv2 ViT-B/14 — FP32 témoin (no INT8)

```bash
# 1 — Train
python scripts/train_fp32.py \
    --model dinov2_vitb14 \
    --data_dir data/dataset \
    --epochs 10 --batch_size 32 \
    --save_dir checkpoints \
    --wandb --wandb_name dinov2_fp32

# 2 — Export to ONNX
python onnx_experiments/export_to_onnx.py \
    --model dinov2_vitb14 \
    --checkpoint checkpoints/best_dinov2_vitb14_fp32.pth \
    --output onnx_experiments/models/dinov2_vitb14_fp32.onnx
```

> DINOv2 is FP32-only. No QAT variant — it serves as the accuracy ceiling.

On SLURM:
```bash
sbatch --export=MODEL=dinov2_vitb14,PHASE=fp32 slurm_train_models.sh
```

---

### ViT-B/16 — FP32 vs QAT INT8

```bash
# 1 — Train FP32
python scripts/train_fp32.py \
    --model vit_b_16 \
    --data_dir data/dataset \
    --epochs 10 --batch_size 32 \
    --save_dir checkpoints \
    --wandb --wandb_name vit_b_16_fp32

# 2 — Export FP32 to ONNX (baseline)
python onnx_experiments/export_to_onnx.py \
    --model vit_b_16 \
    --checkpoint checkpoints/best_vit_b_16_fp32.pth \
    --output onnx_experiments/models/vit_b_16_fp32.onnx

# 3 — QAT fine-tune from FP32 checkpoint
python scripts/train_qat.py \
    --model vit_b_16 \
    --data_dir data/dataset \
    --epochs 3 --batch_size 32 \
    --pretrained checkpoints/best_vit_b_16_fp32.pth \
    --save_dir checkpoints \
    --wandb --wandb_name vit_b_16_qat

# 4 — Export QAT model to ONNX (float graph, simulated quant baked in)
python onnx_experiments/export_to_onnx.py \
    --model vit_b_16 \
    --checkpoint checkpoints/best_vit_b_16_qat.pth \
    --output onnx_experiments/models/vit_b_16_qat.onnx \
    --qat

# 5 — ORT static INT8 quantization of the QAT ONNX → real INT8 ops
python onnx_experiments/quantize_onnx.py \
    --input  onnx_experiments/models/vit_b_16_qat.onnx \
    --output onnx_experiments/models/vit_b_16_int8.onnx \
    --data_dir data/dataset \
    --quant_format QDQ --per_channel
```

On SLURM:
```bash
# FP32 training
sbatch --export=MODEL=vit_b_16,PHASE=fp32 slurm_train_models.sh

# QAT fine-tuning (run after FP32 job completes)
sbatch --export=MODEL=vit_b_16,PHASE=qat,EPOCHS=3 slurm_train_models.sh

# ONNX export + INT8 quantization + comparison table
sbatch --export=MODELS=vit_b_16 slurm_onnx_experiments.sh
```

---

### ResNet-50 — FP32 vs QAT INT8

```bash
# 1 — Train FP32
python scripts/train_fp32.py \
    --model resnet50 \
    --data_dir data/dataset \
    --epochs 10 --batch_size 32 \
    --save_dir checkpoints \
    --wandb --wandb_name resnet50_fp32

# 2 — Export FP32 to ONNX (baseline)
python onnx_experiments/export_to_onnx.py \
    --model resnet50 \
    --checkpoint checkpoints/best_resnet50_fp32.pth \
    --output onnx_experiments/models/resnet50_fp32.onnx

# 3 — QAT fine-tune from FP32 checkpoint
python scripts/train_qat.py \
    --model resnet50 \
    --data_dir data/dataset \
    --epochs 3 --batch_size 32 \
    --pretrained checkpoints/best_resnet50_fp32.pth \
    --save_dir checkpoints \
    --wandb --wandb_name resnet50_qat

# 4 — Export QAT model to ONNX
python onnx_experiments/export_to_onnx.py \
    --model resnet50 \
    --checkpoint checkpoints/best_resnet50_qat.pth \
    --output onnx_experiments/models/resnet50_qat.onnx \
    --qat

# 5 — ORT static INT8 quantization of the QAT ONNX
python onnx_experiments/quantize_onnx.py \
    --input  onnx_experiments/models/resnet50_qat.onnx \
    --output onnx_experiments/models/resnet50_int8.onnx \
    --data_dir data/dataset \
    --quant_format QDQ --per_channel
```

On SLURM:
```bash
sbatch --export=MODEL=resnet50,PHASE=fp32 slurm_train_models.sh
sbatch --export=MODEL=resnet50,PHASE=qat,EPOCHS=3 slurm_train_models.sh
sbatch --export=MODELS=resnet50 slurm_onnx_experiments.sh
```

---

### MobileNet-V3-Small — FP32 vs QAT INT8

```bash
# 1 — Train FP32
python scripts/train_fp32.py \
    --model mobilenet_v3_small \
    --data_dir data/dataset \
    --epochs 10 --batch_size 32 \
    --save_dir checkpoints \
    --wandb --wandb_name mobilenet_fp32

# 2 — Export FP32 to ONNX (baseline)
python onnx_experiments/export_to_onnx.py \
    --model mobilenet_v3_small \
    --checkpoint checkpoints/best_mobilenet_v3_small_fp32.pth \
    --output onnx_experiments/models/mobilenet_v3_small_fp32.onnx

# 3 — QAT fine-tune from FP32 checkpoint
python scripts/train_qat.py \
    --model mobilenet_v3_small \
    --data_dir data/dataset \
    --epochs 3 --batch_size 32 \
    --pretrained checkpoints/best_mobilenet_v3_small_fp32.pth \
    --save_dir checkpoints \
    --wandb --wandb_name mobilenet_qat

# 4 — Export QAT model to ONNX
python onnx_experiments/export_to_onnx.py \
    --model mobilenet_v3_small \
    --checkpoint checkpoints/best_mobilenet_v3_small_qat.pth \
    --output onnx_experiments/models/mobilenet_v3_small_qat.onnx \
    --qat

# 5 — ORT static INT8 quantization of the QAT ONNX
python onnx_experiments/quantize_onnx.py \
    --input  onnx_experiments/models/mobilenet_v3_small_qat.onnx \
    --output onnx_experiments/models/mobilenet_v3_small_int8.onnx \
    --data_dir data/dataset \
    --quant_format QDQ --per_channel
```

On SLURM:
```bash
sbatch --export=MODEL=mobilenet_v3_small,PHASE=fp32,EPOCHS=50 slurm_train_models.sh
sbatch --export=MODEL=mobilenet_v3_small,PHASE=qat,EPOCHS=3 slurm_train_models.sh
sbatch --export=MODELS=mobilenet_v3_small slurm_onnx_experiments.sh
```

---

### ForensicMobileNet (custom) — FP32 vs QAT INT8

```bash
# 1 — Train FP32
python scripts/train_forensic_mobilenet.py \
    --features rgb hsv fft noise srm \
    --data_dir data/dataset \
    --epochs 10 --batch_size 32 \
    --save_dir checkpoints \
    --wandb --wandb_name forensic_mobilenet_fp32

# 2 — Export FP32 to ONNX (baseline)
python onnx_experiments/export_to_onnx.py \
    --model forensic_mobilenet \
    --features rgb hsv fft noise srm \
    --checkpoint checkpoints/best_forensic_mobilenet_rgb-hsv-fft-noise-srm_fp32.pth \
    --output onnx_experiments/models/forensic_mobilenet_fp32.onnx

# 3 — QAT fine-tune from FP32 checkpoint
python scripts/train_qat.py \
    --model forensic_mobilenet \
    --features rgb hsv fft noise srm \
    --data_dir data/dataset \
    --epochs 3 --batch_size 32 \
    --pretrained checkpoints/best_forensic_mobilenet_rgb-hsv-fft-noise-srm_fp32.pth \
    --save_dir checkpoints \
    --wandb --wandb_name forensic_mobilenet_qat

# 4 — Export QAT model to ONNX
python onnx_experiments/export_to_onnx.py \
    --model forensic_mobilenet \
    --features rgb hsv fft noise srm \
    --checkpoint checkpoints/best_forensic_mobilenet_rgb-hsv-fft-noise-srm_qat.pth \
    --output onnx_experiments/models/forensic_mobilenet_qat.onnx \
    --qat

# 5 — ORT static INT8 quantization of the QAT ONNX
python onnx_experiments/quantize_onnx.py \
    --input  onnx_experiments/models/forensic_mobilenet_qat.onnx \
    --output onnx_experiments/models/forensic_mobilenet_int8.onnx \
    --data_dir data/dataset \
    --quant_format QDQ --per_channel
```

On SLURM:
```bash
sbatch --export=MODEL=forensic_mobilenet,PHASE=fp32,EPOCHS=50,"FEATURES=rgb hsv fft noise srm" slurm_train_models.sh
sbatch --export=MODEL=forensic_mobilenet,PHASE=qat,EPOCHS=3,"FEATURES=rgb hsv fft noise srm" slurm_train_models.sh
sbatch --export="MODELS=forensic_mobilenet","FEATURES=rgb hsv fft noise srm" slurm_onnx_experiments.sh
```

---

### Compare all experiments

Once all ONNX models are in `onnx_experiments/models/`:

```bash
python onnx_experiments/run_experiments.py \
    --models_dir onnx_experiments/models \
    --data_dir   data/dataset \
    --latency_runs 100 \
    --warmup 20 \
    --output onnx_experiments/results.json
```

The output table groups models by family and prints accuracy drop and speedup of `_int8`
relative to `_fp32` for each.

On SLURM (all models at once):
```bash
sbatch --export="MODELS=resnet50 vit_b_16 mobilenet_v3_small forensic_mobilenet dinov2_vitb14" \
    slurm_onnx_experiments.sh
```

---

## Phase 1a — FP32 Training (standard models)

```bash
python scripts/train_fp32.py \
    --model resnet50 \
    --data_dir data/dataset \
    --epochs 10 \
    --batch_size 32 \
    --disable_cudnn
```

**Outputs:** `checkpoints/best_{model}_fp32.pth`

On SLURM:
```bash
sbatch slurm_train_models.sh                                    # default: vit_b_16, fp32
sbatch --export=MODEL=resnet50,PHASE=fp32 slurm_train_models.sh
```

---

## Phase 1b — ForensicMobileNetV3 (multi-channel input)

A modified MobileNetV3-Small that internally computes forensic feature channels from the
RGB input before the backbone. Useful for cross-dataset generalization experiments.

### Feature modalities

| Flag | Channels | What it captures |
|---|---|---|
| `rgb` | 3 | Standard ImageNet-normalized RGB |
| `hsv` | 3 | Hue/Saturation/Value — color manipulation cues |
| `fft` | 3 | Log-magnitude 2-D FFT — GAN checkerboard artifacts |
| `noise` | 1 | High-freq noise residual — camera PRNU fingerprint |
| `srm` | 3 | SRM high-pass filters — classic forgery-detection kernels |

### Training-time noise augmentation

To improve cross-dataset generalization, `ForensicNoiseAugment` is applied in the pixel
domain during `model.train()` only — completely off at inference:

| Augmentation | Default prob | Purpose |
|---|---|---|
| Gaussian noise | 0.5 | Simulates unseen camera sensors |
| JPEG approximation | 0.5 | Robustness to compression artifacts |
| Gaussian blur | 0.3 | Covers upsampling differences between generators |
| Random erase | 0.3 | Forces spatial robustness |

```bash
# All 5 feature channels (recommended)
python scripts/train_forensic_mobilenet.py \
    --features rgb hsv fft noise srm \
    --data_dir data/dataset \
    --epochs 10 \
    --batch_size 32

# Without noise augmentation (ablation)
python scripts/train_forensic_mobilenet.py --features rgb hsv fft noise srm --no_noise_augment

# RGB only — equivalent to standard MobileNetV3 (ablation baseline)
python scripts/train_forensic_mobilenet.py --features rgb
```

**Outputs:** `checkpoints/best_forensic_mobilenet_{features-tag}_fp32.pth`
e.g. `best_forensic_mobilenet_rgb-hsv-fft-noise-srm_fp32.pth`

On SLURM:
```bash
# All features (default)
sbatch --export=MODEL=forensic_mobilenet slurm_train_models.sh

# Ablation: run all combinations in parallel
sbatch --export=MODEL=forensic_mobilenet,FEATURES="rgb"                   slurm_train_models.sh
sbatch --export=MODEL=forensic_mobilenet,FEATURES="rgb hsv"               slurm_train_models.sh
sbatch --export=MODEL=forensic_mobilenet,FEATURES="rgb hsv fft"           slurm_train_models.sh
sbatch --export=MODEL=forensic_mobilenet,FEATURES="rgb noise srm"         slurm_train_models.sh
sbatch --export=MODEL=forensic_mobilenet,FEATURES="rgb hsv fft noise srm" slurm_train_models.sh
```

---

## Phase 2 — Quantization-Aware Training (QAT)

Fine-tunes a pretrained FP32 model with simulated INT8 quantization using custom
`Quantized_Linear` / `Quantized_Conv2d` layers (symmetric 8-bit, STE gradients).

```bash
python scripts/train_qat.py \
    --model resnet50 \
    --data_dir data/dataset \
    --epochs 3 \
    --batch_size 32 \
    --pretrained checkpoints/best_resnet50_fp32.pth \
    --disable_cudnn
```

**Outputs:**
- `checkpoints/best_{model}_qat.pth` — QAT weights (Quantized_* layers, method=sym)
- `checkpoints/best_{model}_quantized.pth` — identical copy for inference

On SLURM:
```bash
sbatch --export=MODEL=resnet50,PHASE=qat slurm_train_models.sh
sbatch --export=MODEL=vit_b_16,PHASE=qat slurm_train_models.sh
```

> The FP32 pretrained checkpoint is loaded automatically if it exists in `checkpoints/`.

---

## Phase 3 — ONNX Experiments (accuracy / size / latency)

Three ONNX variants are produced per model and compared:

| Variant | Source | ONNX dtype | Real INT8 ops? |
|---|---|---|---|
| `{model}_fp32.onnx` | `best_{model}_fp32.pth` | float32 | no |
| `{model}_int8.onnx` | ORT static quant of FP32 ONNX | int8 | **yes** |
| `{model}_qat.onnx` | `best_{model}_qat.pth` | float32 (simulated) | no |

INT8 latency gains require hardware with INT8 SIMD support (x86 AVX512-VNNI, ARM NEON).

### Run the full pipeline on SLURM

```bash
# Both resnet50 and vit_b_16 (default)
sbatch slurm_onnx_experiments.sh

# Single model
sbatch --export=MODELS=resnet50 slurm_onnx_experiments.sh

# Choose quantization format (QDQ recommended for GPU/accelerators)
sbatch --export=MODELS="resnet50 vit_b_16",QUANT_FORMAT=QOperator slurm_onnx_experiments.sh
```

**Results:** `onnx_experiments/results.json` + printed table with speedup summary.

### Run steps manually

**Step 1 — Export to ONNX**
```bash
# FP32
python onnx_experiments/export_to_onnx.py \
    --model resnet50 \
    --checkpoint checkpoints/best_resnet50_fp32.pth

# QAT (float graph with simulated quantization baked in)
python onnx_experiments/export_to_onnx.py \
    --model resnet50 \
    --checkpoint checkpoints/best_resnet50_qat.pth \
    --qat
```

**Step 2 — ORT static INT8 quantization** (on the FP32 ONNX)
```bash
python onnx_experiments/quantize_onnx.py \
    --input  onnx_experiments/models/resnet50_fp32.onnx \
    --output onnx_experiments/models/resnet50_int8.onnx \
    --data_dir data/dataset \
    --quant_format QDQ
```

**Step 3 — Compare all models (accuracy + size + latency on 100 images)**
```bash
python onnx_experiments/run_experiments.py \
    --models_dir   onnx_experiments/models \
    --data_dir     data/dataset \
    --max_val_samples 100 \
    --latency_runs 100 \
    --warmup 20 \
    --output onnx_experiments/results.json
```

Add `--gpu` to measure latency with `CUDAExecutionProvider`.

### Output table (example)

```
============================================================
EXPERIMENT RESULTS
============================================================
Model                                          Size (MB)  Accuracy (%)   Median (ms)  Mean (ms)   P95 (ms)
----------------------------------------------  ---------  -------------  -----------  ----------  --------
resnet50_fp32                                      97.70        94.20          18.40       18.90      21.10
resnet50_int8                                      24.80        93.50           9.20        9.50      11.30
resnet50_qat                                       97.70        93.80          19.10       19.40      21.80
vit_b_16_fp32                                     329.60        95.10          82.30       83.10      86.70
vit_b_16_int8                                      84.10        94.40          41.50       42.20      44.90
vit_b_16_qat                                      329.60        94.80          84.00       84.50      87.30

Speedup over FP32 baseline (latency):
  resnet50_int8     speedup=2.00x  size_reduction=74.6%  acc_drop=0.70pp
  vit_b_16_int8     speedup=1.98x  size_reduction=74.5%  acc_drop=0.70pp
```

---

## Experiment Parameters Reference

### `run_experiments.py`

| Argument | Default | Description |
|---|---|---|
| `--models_dir` | `onnx_experiments/models` | Folder with `.onnx` files |
| `--data_dir` | `data/dataset` | Dataset root |
| `--max_val_samples` | all | Cap number of validation images |
| `--latency_runs` | 100 | Inference iterations per model |
| `--warmup` | 20 | Warm-up iterations (discarded) |
| `--gpu` | off | Use `CUDAExecutionProvider` |
| `--output` | `onnx_experiments/results.json` | JSON results path |

### `quantize_onnx.py`

| Argument | Default | Description |
|---|---|---|
| `--quant_format` | `QDQ` | `QDQ` (hardware-friendly) or `QOperator` (fused) |
| `--per_channel` | off | Per-channel weights (more accurate, slower calibration) |
| `--calibration_method` | `MinMax` | `MinMax`, `Entropy`, or `Percentile` |
| `--num_calibration_samples` | 256 | Images used to calibrate activations |

---

## Cross-Dataset Evaluation

Evaluate all checkpoints in `checkpoints/` on an external dataset and compare them
in a single ranked table. Model type is auto-detected from the filename — including
QAT checkpoints (which use `Quantized_*` layers) and `forensic_mobilenet` variants.

```bash
# Evaluate on 1000 images from ddata/test (500 fake, 500 real)
python scripts/evaluate_all.py \
    --data_dir   data/ddata/test \
    --n_samples  1000 \
    --batch_size 64

# Save results to JSON
python scripts/evaluate_all.py --data_dir data/ddata/test --n_samples 1000 \
    --output results/eval_ddata.json
```

On SLURM:
```bash
sbatch --wrap="/home/infres/billy-22/miniconda3/envs/deepfake_env/bin/python \
  scripts/evaluate_all.py --data_dir data/ddata/test --n_samples 1000 \
  --output results/eval_ddata.json" \
  --job-name=eval --partition=P100 --gres=gpu:1 --mem=16G
```

**Output:**
```
===========================================================================
Checkpoint                                            Acc   Fake   Real
---------------------------------------------------------------------------
best_forensic_mobilenet_rgb-hsv-fft-noise-srm_fp32  74.20% 68.40% 80.00%
best_vit_b_16_fp32                                  53.00%  6.20% 99.80%
best_mobilenet_v3_small_fp32                        50.40%  0.80% 100.00%
...
===========================================================================
```

> **Note on cross-dataset accuracy:** models trained on one deepfake generator often
> predict "real" for everything on unseen generators (~50% accuracy, 0% fake recall).
> This is a known domain-generalization problem. The forensic features (FFT, SRM, noise)
> and `ForensicNoiseAugment` are designed to improve this.

---

## Weights & Biases Tracking

```bash
python scripts/train_fp32.py              --model resnet50 ... --wandb --wandb_project deepfake-detection
python scripts/train_qat.py               --model resnet50 ... --wandb --wandb_project deepfake-detection
python scripts/train_forensic_mobilenet.py --features rgb hsv fft noise srm ... --wandb --wandb_project deepfake-detection
```

Setup: `wandb login` or `export WANDB_API_KEY=...` or copy `.env.example` → `.env`.

---

## SLURM Quick Reference

```bash
# Check job status
squeue -u $USER

# Live training log
tail -f logs/deepfake_train_<job_id>.out

# Cancel all jobs
scancel -u $USER
```

Override any variable at submission time:
```bash
sbatch --export=MODEL=vit_b_16,EPOCHS=5,BATCH_SIZE=64 slurm_train_models.sh
sbatch --export=MODELS="resnet50 vit_b_16",NUM_CAL_SAMPLES=512 slurm_onnx_experiments.sh
```
