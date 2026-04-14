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
# ONNX experiment pipeline — three focused scripts:
#
#   export.py        fp32.pth + qat.pth → fp32.onnx, ptq_int8.onnx, qat_int8.onnx
#   bench_latency.py dummy inputs → latency table  (TRT INT8 for quantized models)
#   bench_accuracy.py val split   → accuracy table
#
# Control which steps run via env vars:
#   RUN_EXPORT=1    re-export and re-quantize all models (needs data_dir for calibration)
#   RUN_LATENCY=1   benchmark latency with dummy inputs
#   RUN_ACCURACY=1  evaluate accuracy on the validation split
#
# Typical usage:
#   # First time — export everything (slow, needs dataset for calibration)
#   sbatch --export=ALL,MODELS="resnet50",RUN_EXPORT=1,RUN_ACCURACY=1 slurm_onnx_experiments.sh
#
#   # Re-run accuracy/latency on already-exported models (fast, no calibration)
#   sbatch --export=ALL,MODELS="resnet50 mobilenet_v3_small",RUN_LATENCY=1,RUN_ACCURACY=1 slurm_onnx_experiments.sh
#
#   # GPU latency with TensorRT INT8 for quantized models
#   sbatch --export=ALL,MODELS="resnet50",RUN_LATENCY=1,USE_GPU=1,USE_TRT=1 slurm_onnx_experiments.sh
################################################################################

MODELS="${MODELS:-resnet50 vit_b_16 mobilenet_v3_small forensic_mobilenet}"
FEATURES="${FEATURES:-rgb hsv fft noise srm}"
NUM_CAL_SAMPLES="${NUM_CAL_SAMPLES:-256}"
LATENCY_RUNS="${LATENCY_RUNS:-100}"
WARMUP="${WARMUP:-20}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
USE_GPU="${USE_GPU:-0}"
USE_TRT="${USE_TRT:-0}"
RUN_EXPORT="${RUN_EXPORT:-0}"
RUN_LATENCY="${RUN_LATENCY:-0}"
RUN_ACCURACY="${RUN_ACCURACY:-1}"

DATA_DIR="data/dataset"
CKPT_DIR="checkpoints_exp"
ONNX_DIR="onnx_experiments/models"

################################################################################
# SETUP
################################################################################

module purge
mkdir -p logs "$ONNX_DIR"
export PYTHONPATH="${PYTHONPATH}:${PWD}"

CONDA_ENV_LIB="/home/infres/billy-22/miniconda3/envs/deepfake_env/lib"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export ORT_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/infres/billy-22/miniconda3/envs/deepfake_env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN"; exit 1
fi

FEATURES_TAG="${FEATURES// /-}"

GPU_FLAGS=""
[[ "$USE_GPU" == "1" ]] && GPU_FLAGS="--gpu"
[[ "$USE_GPU" == "1" && "$USE_TRT" == "1" ]] && GPU_FLAGS="--gpu --trt"

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start      : $(date)"
echo "Models     : $MODELS"
echo "GPU flags  : ${GPU_FLAGS:-(none, CPU only)}"
echo "Steps      : export=$RUN_EXPORT  latency=$RUN_LATENCY  accuracy=$RUN_ACCURACY"
echo "=========================================="

################################################################################
# STEP 1 — Export: fp32 + ptq_int8 + qat_int8 per model
################################################################################
if [[ "$RUN_EXPORT" == "1" ]]; then
    echo ""
    echo ">>> STEP 1: Export (FP32 + PTQ INT8 + QAT INT8)"

    for MODEL in $MODELS; do
        echo ""
        echo "  [$MODEL]"

        if [[ "$MODEL" == "forensic_mobilenet" ]]; then
            FP32_CKPT="$CKPT_DIR/best_forensic_mobilenet_${FEATURES_TAG}_fp32.pth"
            QAT_CKPT="$CKPT_DIR/best_forensic_mobilenet_${FEATURES_TAG}_qat.pth"
            EXTRA="--features $FEATURES"
        elif [[ "$MODEL" == "dinov2_vitb14" ]]; then
            FP32_CKPT="$CKPT_DIR/best_dinov2_vitb14_fp32.pth"
            QAT_CKPT=""
            EXTRA=""
        else
            FP32_CKPT="$CKPT_DIR/best_${MODEL}_fp32.pth"
            QAT_CKPT="$CKPT_DIR/best_${MODEL}_qat.pth"
            EXTRA=""
        fi

        if [[ ! -f "$FP32_CKPT" ]]; then
            echo "  [SKIP] FP32 checkpoint not found: $FP32_CKPT"
            continue
        fi

        QAT_ARG=""
        [[ -n "$QAT_CKPT" && -f "$QAT_CKPT" ]] && QAT_ARG="--qat_ckpt $QAT_CKPT"

        "$PYTHON_BIN" onnx_experiments/export.py \
            --model          "$MODEL" \
            --fp32_ckpt      "$FP32_CKPT" \
            $QAT_ARG \
            --data_dir       "$DATA_DIR" \
            --output_dir     "$ONNX_DIR" \
            --num_cal_samples "$NUM_CAL_SAMPLES" \
            $EXTRA
    done
fi

################################################################################
# STEP 2 — Latency benchmark (dummy inputs, no dataset)
################################################################################
if [[ "$RUN_LATENCY" == "1" ]]; then
    echo ""
    echo ">>> STEP 2: Latency benchmark (dummy inputs)"

    "$PYTHON_BIN" onnx_experiments/bench_latency.py \
        --models_dir  "$ONNX_DIR" \
        --models      $MODELS \
        --runs        "$LATENCY_RUNS" \
        --warmup      "$WARMUP" \
        $GPU_FLAGS
fi

################################################################################
# STEP 3 — Accuracy benchmark (validation split)
################################################################################
if [[ "$RUN_ACCURACY" == "1" ]]; then
    echo ""
    echo ">>> STEP 3: Accuracy benchmark (validation split)"

    "$PYTHON_BIN" onnx_experiments/bench_accuracy.py \
        --models_dir  "$ONNX_DIR" \
        --data_dir    "$DATA_DIR" \
        --models      $MODELS \
        --max_samples "$MAX_VAL_SAMPLES"
fi

echo ""
echo "=========================================="
echo "Done. End time: $(date)"
echo "=========================================="
exit 0
