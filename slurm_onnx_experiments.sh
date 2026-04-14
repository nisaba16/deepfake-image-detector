#!/bin/bash
#SBATCH --job-name=onnx_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

################################################################################
# #SBATCH --nodelist=nodecpu01
# #SBATCH --gres=gpu:1
# CONFIGURATION - override with --export at submission time
# To run on GPU: sbatch --partition=P100 --gres=gpu:1 --export=ALL,USE_GPU=1 slurm_onnx_experiments.sh
# To also use TensorRT: add USE_TRT=1 to the --export list above
# MODELS        : space-separated list of models to process
# FEATURES      : forensic_mobilenet feature tag (space-separated, same as training)
# QUANT_FORMAT  : QDQ (default, recommended) | QOperator
# NUM_CAL_SAMPLES: images used for ORT static calibration
#
# Pipeline per model (except dinov2 which is FP32-only):
#   1. Export FP32 checkpoint  → {model}_fp32.onnx
#   2. Export QAT checkpoint   → {model}_qat.onnx
#   3. ORT static INT8 quant of QAT ONNX → {model}_int8.onnx
#   4. run_experiments.py: accuracy + size + latency table for all .onnx files
################################################################################

MODELS="${MODELS:-resnet50 vit_b_16 mobilenet_v3_small forensic_mobilenet}"
FEATURES="${FEATURES:-rgb hsv fft noise srm}"
QUANT_FORMAT="${QUANT_FORMAT:-QDQ}"
NUM_CAL_SAMPLES="${NUM_CAL_SAMPLES:-256}"
LATENCY_RUNS="${LATENCY_RUNS:-100}"
WARMUP="${WARMUP:-20}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
USE_GPU="${USE_GPU:-0}"      # Set to 1 to use CUDAExecutionProvider
USE_TRT="${USE_TRT:-0}"      # Set to 1 to use TensorrtExecutionProvider (requires USE_GPU=1)
RUN_COMPARE="${RUN_COMPARE:-0}"  # Set to 1 to run compare_quant.py (PTQ vs QAT accuracy table)
CAL_METHOD="${CAL_METHOD:-MinMax}"  # MinMax | Entropy | Percentile

DATA_DIR="data/dataset"
CKPT_DIR="checkpoints_exp"
ONNX_DIR="onnx_experiments/models"
RESULTS="onnx_experiments/results.json"

################################################################################
# SETUP
################################################################################

module purge
mkdir -p logs "$ONNX_DIR"
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Make the conda env's CUDA runtime visible to ORT after module purge strips PATH.
# libcudart.so lives in the conda env lib dir — ORT's CUDAExecutionProvider needs it.
CONDA_ENV_LIB="/home/infres/billy-22/miniconda3/envs/deepfake_env/lib"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH}"

# Suppress ORT pthread_setaffinity_np errors in SLURM cgroup environments.
# Without this ORT tries to pin threads to specific cores, which the cgroup
# denies with EINVAL, producing noisy (but harmless) error messages.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export ORT_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Force Python to flush stdout immediately so log files update in real time.
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/infres/billy-22/miniconda3/envs/deepfake_env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN"
    exit 1
fi

FEATURES_TAG="${FEATURES// /-}"   # "rgb hsv fft noise srm" → "rgb-hsv-fft-noise-srm"

GPU_FLAGS=""
[[ "$USE_GPU" == "1" ]] && GPU_FLAGS="--gpu"
[[ "$USE_GPU" == "1" && "$USE_TRT" == "1" ]] && GPU_FLAGS="--gpu --trt"

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start     : $(date)"
echo "Models    : $MODELS"
echo "Quant fmt : $QUANT_FORMAT"
echo "Cal samples: $NUM_CAL_SAMPLES"
echo "GPU flags : ${GPU_FLAGS:-(none, CPU only)}"
echo "=========================================="

################################################################################
# Helper: resolve checkpoint and ONNX names for a given model
#
# Sets: FP32_CKPT, QAT_CKPT, FP32_ONNX, QAT_ONNX, INT8_ONNX
#       EXPORT_EXTRA (extra args for export_to_onnx.py, e.g. --features)
#       IS_DINOV2 (1 if dinov2, no QAT/INT8)
################################################################################
resolve_model_paths() {
    local MODEL="$1"
    IS_DINOV2=0
    EXPORT_EXTRA=""

    if [[ "$MODEL" == "forensic_mobilenet" ]]; then
        FP32_CKPT="$CKPT_DIR/best_forensic_mobilenet_${FEATURES_TAG}_fp32.pth"
        QAT_CKPT="$CKPT_DIR/best_forensic_mobilenet_${FEATURES_TAG}_qat.pth"
        FP32_ONNX="$ONNX_DIR/forensic_mobilenet_fp32.onnx"
        QAT_ONNX="$ONNX_DIR/forensic_mobilenet_qat.onnx"
        INT8_ONNX="$ONNX_DIR/forensic_mobilenet_int8.onnx"
        EXPORT_EXTRA="--features $FEATURES"
    elif [[ "$MODEL" == "dinov2_vitb14" ]]; then
        FP32_CKPT="$CKPT_DIR/best_dinov2_vitb14_fp32.pth"
        QAT_CKPT=""
        FP32_ONNX="$ONNX_DIR/dinov2_vitb14_fp32.onnx"
        QAT_ONNX=""
        INT8_ONNX=""
        IS_DINOV2=1
    else
        FP32_CKPT="$CKPT_DIR/best_${MODEL}_fp32.pth"
        QAT_CKPT="$CKPT_DIR/best_${MODEL}_qat.pth"
        FP32_ONNX="$ONNX_DIR/${MODEL}_fp32.onnx"
        QAT_ONNX="$ONNX_DIR/${MODEL}_qat.onnx"
        INT8_ONNX="$ONNX_DIR/${MODEL}_int8.onnx"
    fi
}

################################################################################
# STEP 1 — Export checkpoints to ONNX
################################################################################
echo ""
echo ">>> STEP 1: Export PyTorch → ONNX"

for MODEL in $MODELS; do
    resolve_model_paths "$MODEL"

    # FP32 export
    if [[ -f "$FP32_CKPT" ]]; then
        echo "  [$MODEL] Exporting FP32 ..."
        "$PYTHON_BIN" onnx_experiments/export_to_onnx.py \
            --model      "$MODEL" \
            --checkpoint "$FP32_CKPT" \
            --output     "$FP32_ONNX" \
            $EXPORT_EXTRA
    else
        echo "  [$MODEL] [SKIP FP32] checkpoint not found: $FP32_CKPT"
    fi

    # QAT export (skip for dinov2)
    if [[ $IS_DINOV2 -eq 0 ]]; then
        if [[ -f "$QAT_CKPT" ]]; then
            echo "  [$MODEL] Exporting QAT ..."
            "$PYTHON_BIN" onnx_experiments/export_to_onnx.py \
                --model      "$MODEL" \
                --checkpoint "$QAT_CKPT" \
                --output     "$QAT_ONNX" \
                --qat \
                $EXPORT_EXTRA
        else
            echo "  [$MODEL] [SKIP QAT] checkpoint not found: $QAT_CKPT"
        fi
    fi
done

################################################################################
# STEP 2 — ORT static INT8 quantization (applied to QAT ONNX)
################################################################################
echo ""
echo ">>> STEP 2: ORT static INT8 quantization (QAT ONNX → INT8 ONNX)"

for MODEL in $MODELS; do
    resolve_model_paths "$MODEL"

    # Skip dinov2 — FP32 only
    if [[ $IS_DINOV2 -eq 1 ]]; then
        echo "  [$MODEL] [SKIP INT8] dinov2 is FP32-only"
        continue
    fi

    if [[ -f "$QAT_ONNX" ]]; then
        # ViT uses dynamic quantization (recommended for transformers: per-batch activation
        # scales, no calibration data, better accuracy). CNN models use static S8S8 QDQ.
        if [[ "$MODEL" == "vit_b_16" || "$MODEL" == "dinov2_vitb14" ]]; then
            echo "  [$MODEL] Dynamic quantization (transformer model) ..."
            "$PYTHON_BIN" onnx_experiments/quantize_onnx.py \
                --input  "$QAT_ONNX" \
                --output "$INT8_ONNX" \
                --dynamic
        else
            echo "  [$MODEL] Static quantization QAT ONNX → INT8 ($QUANT_FORMAT, per_channel) ..."
            "$PYTHON_BIN" onnx_experiments/quantize_onnx.py \
                --input  "$QAT_ONNX" \
                --output "$INT8_ONNX" \
                --data_dir "$DATA_DIR" \
                --num_calibration_samples "$NUM_CAL_SAMPLES" \
                --quant_format "$QUANT_FORMAT" \
                --per_channel
        fi
        # qat.onnx is a float32 simulation graph — not a deployment artifact.
        # Remove it so only fp32 and int8 remain for comparison.
        rm -f "$QAT_ONNX"
        echo "  [$MODEL] Removed intermediate QAT ONNX ($QAT_ONNX)"
    else
        echo "  [$MODEL] [SKIP INT8] QAT ONNX not found: $QAT_ONNX — run QAT training first"
    fi
done

################################################################################
# STEP 3 — Compare all models: accuracy + size + latency
################################################################################
echo ""
echo ">>> STEP 3: Accuracy / Size / Latency experiments"

"$PYTHON_BIN" onnx_experiments/run_experiments.py \
    --models_dir      "$ONNX_DIR" \
    --data_dir        "$DATA_DIR" \
    --warmup          "$WARMUP" \
    --latency_runs    "$LATENCY_RUNS" \
    --max_val_samples "$MAX_VAL_SAMPLES" \
    --models          $MODELS \
    --output          "$RESULTS" \
    $GPU_FLAGS

################################################################################
# STEP 4 (optional) — Accuracy comparison: FP32 vs PTQ INT8 vs QAT→INT8
################################################################################
if [[ "$RUN_COMPARE" == "1" ]]; then
    echo ""
    echo ">>> STEP 4: Accuracy comparison (FP32 / PTQ INT8 / QAT→INT8)"
    "$PYTHON_BIN" onnx_experiments/compare_quant.py \
        --models          $MODELS \
        --models_dir      "$ONNX_DIR" \
        --data_dir        "$DATA_DIR" \
        --num_cal_samples "$NUM_CAL_SAMPLES" \
        --max_val_samples "$MAX_VAL_SAMPLES" \
        --calibration_method "$CAL_METHOD" \
        --output          "onnx_experiments/compare_quant_results.json"
fi

echo ""
echo "=========================================="
echo "Done. Results → $RESULTS"
echo "End time: $(date)"
echo "=========================================="
exit 0
