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
│   ├── export.py                      # Step 1 – fp32.pth + qat.pth → fp32/ptq_int8/qat_int8 ONNX
│   ├── bench_latency.py               # Step 2 – latency table (dummy inputs, TRT INT8 for quant models)
│   ├── bench_accuracy.py              # Step 3 – accuracy table on validation split
│   ├── export_to_onnx.py              # (internal) PyTorch → ONNX export helpers
│   ├── quantize_onnx.py               # (internal) ORT static quantization with NaN-safe calibration
│   └── data_reader.py                 # (internal) calibration data reader
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
sbatch --export=MODEL=dinov2_vitb14,PHASE=fp32,,EPOCHS=50 slurm_train_models.sh
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
sbatch --export=MODEL=vit_b_16,PHASE=fp32,EPOCHS=30 slurm_train_models.sh

# QAT fine-tuning (run after FP32 job completes)
sbatch --export=MODEL=vit_b_16,PHASE=qat,EPOCHS=3 slurm_train_models.sh

# ONNX export + INT8 quantization + comparison table (GPU partition)
sbatch --export=ALL,MODELS=vit_b_16 slurm_onnx_experiments.sh

# On CPU partition (EPYC 9274F, VNNI support for INT8)
sbatch --partition=cpu-high --nodelist=nodecpu01 --export=ALL,MODELS=vit_b_16,USE_GPU=0 slurm_onnx_experiments.sh
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
sbatch --export=ALL,MODEL=resnet50,PHASE=fp32,EPOCHS=30 slurm_train_models.sh
sbatch --export=ALL,MODEL=resnet50,PHASE=qat,EPOCHS=3 slurm_train_models.sh
sbatch --export=ALL,MODELS=resnet50 slurm_onnx_experiments.sh
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
sbatch --export=ALL,MODEL=mobilenet_v3_small,PHASE=fp32,EPOCHS=30 slurm_train_models.sh
sbatch --export=ALL,MODEL=mobilenet_v3_small,PHASE=qat,EPOCHS=3 slurm_train_models.sh
sbatch --export=ALL,MODELS=mobilenet_v3_small slurm_onnx_experiments.sh
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
sbatch --export=ALL,MODEL=forensic_mobilenet,PHASE=fp32,EPOCHS=50,"FEATURES=rgb hsv fft noise srm" slurm_train_models.sh
sbatch --export=ALL,MODEL=forensic_mobilenet,PHASE=qat,EPOCHS=3,"FEATURES=rgb hsv fft noise srm" slurm_train_models.sh
sbatch --export=ALL,MODELS=forensic_mobilenet,"FEATURES=rgb hsv fft noise srm" slurm_onnx_experiments.sh
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
sbatch --export=ALL,"MODELS=resnet50 vit_b_16 mobilenet_v3_small forensic_mobilenet dinov2_vitb14" \
    slurm_onnx_experiments.sh
```

---

## Phase 1a — FP32 Training (standard models)

```bash
python scripts/train_fp32.py \
    --model resnet50 \
    --data_dir data/dataset \
    --epochs 10 \
    --batch_size 32
```

**Outputs:** `checkpoints/best_{model}_fp32.pth`

> If you hit cuDNN architecture mismatch errors (e.g. on older drivers), add `--disable_cudnn` as a workaround. Avoid it otherwise — it disables cuDNN optimizations and slows training.

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
sbatch --export=ALL,MODEL=forensic_mobilenet slurm_train_models.sh

# Ablation: run all combinations in parallel
sbatch --export=ALL,MODEL=forensic_mobilenet,FEATURES="rgb"                   slurm_train_models.sh
sbatch --export=ALL,MODEL=forensic_mobilenet,FEATURES="rgb hsv"               slurm_train_models.sh
sbatch --export=ALL,MODEL=forensic_mobilenet,FEATURES="rgb hsv fft"           slurm_train_models.sh
sbatch --export=ALL,MODEL=forensic_mobilenet,FEATURES="rgb noise srm"         slurm_train_models.sh
sbatch --export=ALL,MODEL=forensic_mobilenet,FEATURES="rgb hsv fft noise srm" slurm_train_models.sh
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
    --pretrained checkpoints/best_resnet50_fp32.pth
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

Three ONNX variants are produced per model:

| File | Source | Quantization |
|---|---|---|
| `{model}_fp32.onnx` | FP32 checkpoint | none — float32 baseline |
| `{model}_ptq_int8.onnx` | FP32 ONNX + ORT static calibration | post-training quantization |
| `{model}_qat_int8.onnx` | QAT checkpoint + ORT static calibration | quantization-aware training |

Quantization scheme: **S8S8 QDQ** — signed INT8 weights and activations, symmetric
(zero\_point = 0), QuantizeLinear/DequantizeLinear nodes.
TensorRT INT8 engine is used automatically for `*_int8` models when `--trt` is set.

### Three focused scripts

**`export.py`** — one command exports all three variants for a model:
```bash
python onnx_experiments/export.py \
    --model      resnet50 \
    --fp32_ckpt  checkpoints/best_resnet50_fp32.pth \
    --qat_ckpt   checkpoints/best_resnet50_qat.pth \
    --data_dir   data/dataset \
    --output_dir onnx_experiments/models
# → resnet50_fp32.onnx  resnet50_ptq_int8.onnx  resnet50_qat_int8.onnx
```

**`bench_latency.py`** — dummy inputs, no dataset needed:
```bash
python onnx_experiments/bench_latency.py \
    --models_dir onnx_experiments/models \
    --models     resnet50 mobilenet_v3_small \
    --gpu --trt \
    --runs 100 --warmup 20
# FP32 models → CUDA EP (float)
# *_int8 models → TensorRT INT8 engine
```

**`bench_accuracy.py`** — accuracy on the validation split:
```bash
python onnx_experiments/bench_accuracy.py \
    --models_dir onnx_experiments/models \
    --data_dir   data/dataset \
    --models     resnet50 \
    --max_samples 1000
```

### Run on SLURM

```bash
# First time: export all models (needs dataset for calibration ~5 min/model)
sbatch --export=ALL,MODELS="resnet50",RUN_EXPORT=1,RUN_ACCURACY=1 slurm_onnx_experiments.sh

# Re-run accuracy on already-exported models (fast, no calibration)
sbatch --export=ALL,"MODELS=resnet50 mobilenet_v3_small",RUN_ACCURACY=1 slurm_onnx_experiments.sh

# GPU latency with TensorRT INT8
sbatch --export=ALL,MODELS="resnet50",RUN_LATENCY=1,USE_GPU=1,USE_TRT=1 slurm_onnx_experiments.sh

# Full pipeline (export + latency + accuracy)
sbatch --export=ALL,"MODELS=resnet50 mobilenet_v3_small",RUN_EXPORT=1,RUN_LATENCY=1,RUN_ACCURACY=1,USE_GPU=1,USE_TRT=1 slurm_onnx_experiments.sh
```

---

## Experiment Parameters Reference

### `export.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Model architecture |
| `--fp32_ckpt` | required | FP32 checkpoint path |
| `--qat_ckpt` | None | QAT checkpoint (skipped if not provided) |
| `--data_dir` | `data/dataset` | Dataset root for PTQ/QAT calibration |
| `--output_dir` | `onnx_experiments/models` | Where to write `.onnx` files |
| `--num_cal_samples` | 256 | Images used for ORT static calibration |

### `bench_latency.py`

| Argument | Default | Description |
|---|---|---|
| `--models_dir` | `onnx_experiments/models` | Folder with `.onnx` files |
| `--models` | all | Filter by model family prefix |
| `--gpu` | off | Use `CUDAExecutionProvider` |
| `--trt` | off | Use TensorRT INT8 for `*_int8` models (requires `--gpu`) |
| `--runs` | 100 | Measurement iterations |
| `--warmup` | 20 | Warm-up iterations (discarded) |

### `bench_accuracy.py`

| Argument | Default | Description |
|---|---|---|
| `--models_dir` | `onnx_experiments/models` | Folder with `.onnx` files |
| `--data_dir` | required | Dataset root |
| `--models` | all | Filter by model family prefix |
| `--max_samples` | all | Cap number of validation images |
| `--gpu` | off | Use GPU for faster inference throughput |

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
sbatch --export=ALL,MODEL=vit_b_16,EPOCHS=5,BATCH_SIZE=64 slurm_train_models.sh
sbatch --export=ALL,"MODELS=resnet50 vit_b_16",NUM_CAL_SAMPLES=512 slurm_onnx_experiments.sh
```
