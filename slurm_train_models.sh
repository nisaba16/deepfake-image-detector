#!/bin/bash
#SBATCH --job-name=deepfake_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

################################################################################
# CONFIGURATION - override with --export at submission time
#
# MODEL  : resnet50 | vit_b_16 | mobilenet_v3_small | dinov2_vitb14
#          | forensic_mobilenet
# PHASE  : fp32 | qat
#          (dinov2_vitb14 is always fp32, qat is not supported)
# FEATURES: space-separated forensic_mobilenet modalities
#           (ignored for all other models)
################################################################################

MODEL="${MODEL:-vit_b_16}"
PHASE="${PHASE:-fp32}"
FEATURES="${FEATURES:-rgb hsv fft noise srm}"

WANDB="${WANDB:-true}"
DATA_DIR="data/dataset"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SUBSAMPLE="${SUBSAMPLE:-0}"

################################################################################
# SETUP
################################################################################

module purge
mkdir -p logs checkpoints
export PYTHONPATH="${PYTHONPATH}:${PWD}"

PYTHON_BIN="${PYTHON_BIN:-/home/infres/billy-22/miniconda3/envs/deepfake_env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN"
    exit 1
fi

FEATURES_TAG="${FEATURES// /-}"   # "rgb hsv fft noise srm" → "rgb-hsv-fft-noise-srm"

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start     : $(date)"
echo "Model     : $MODEL"
echo "Phase     : $PHASE"
echo "Dataset   : $DATA_DIR"
echo "Epochs    : $EPOCHS"
echo "Batch size: $BATCH_SIZE"
[[ "$MODEL" == "forensic_mobilenet" ]] && echo "Features  : $FEATURES"
echo "=========================================="

################################################################################
# SELECT SCRIPT AND BUILD PARAMS
################################################################################

# dinov2 has no QAT support — silently downgrade to fp32
if [[ "$MODEL" == "dinov2_vitb14" && "$PHASE" == "qat" ]]; then
    echo "[WARN] dinov2_vitb14 does not support QAT. Running fp32 instead."
    PHASE="fp32"
fi

if [[ "$MODEL" == "forensic_mobilenet" ]]; then

    if [[ "$PHASE" == "fp32" ]]; then
        SCRIPT="scripts/train_forensic_mobilenet.py"
        PARAMS=(
            --features $FEATURES
            --data_dir "$DATA_DIR"
            --epochs   "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --subsample  "$SUBSAMPLE"
            --disable_cudnn
        )
        if [[ "$WANDB" == "true" ]]; then
            PARAMS+=(--wandb --wandb_project "deepfake-detection"
                     --wandb_name "forensic_mobilenet_${FEATURES_TAG}_fp32_${SLURM_JOB_ID}")
        fi
    else
        # QAT for forensic_mobilenet
        SCRIPT="scripts/train_qat.py"
        PRETRAINED="checkpoints/best_forensic_mobilenet_${FEATURES_TAG}_fp32.pth"
        PARAMS=(
            --model forensic_mobilenet
            --features $FEATURES
            --data_dir "$DATA_DIR"
            --epochs   "${EPOCHS:-3}"
            --batch_size "$BATCH_SIZE"
            --subsample  "$SUBSAMPLE"
            --disable_cudnn
        )
        if [[ -f "$PRETRAINED" ]]; then
            echo "Found FP32 checkpoint: $PRETRAINED"
            PARAMS+=(--pretrained "$PRETRAINED")
        else
            echo "[WARN] FP32 checkpoint not found: $PRETRAINED — starting from scratch"
        fi
        if [[ "$WANDB" == "true" ]]; then
            PARAMS+=(--wandb --wandb_project "deepfake-detection"
                     --wandb_name "forensic_mobilenet_${FEATURES_TAG}_qat_${SLURM_JOB_ID}")
        fi
    fi

else

    if [[ "$PHASE" == "fp32" ]]; then
        SCRIPT="scripts/train_fp32.py"
        PARAMS=(
            --model "$MODEL"
            --data_dir "$DATA_DIR"
            --epochs   "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --subsample  "$SUBSAMPLE"
            --disable_cudnn
        )
        if [[ "$WANDB" == "true" ]]; then
            PARAMS+=(--wandb --wandb_project "deepfake-detection"
                     --wandb_name "${MODEL}_fp32_${SLURM_JOB_ID}")
        fi
    else
        # QAT
        SCRIPT="scripts/train_qat.py"
        PRETRAINED="checkpoints/best_${MODEL}_fp32.pth"
        PARAMS=(
            --model "$MODEL"
            --data_dir "$DATA_DIR"
            --epochs   "${EPOCHS:-3}"
            --batch_size "$BATCH_SIZE"
            --subsample  "$SUBSAMPLE"
            --disable_cudnn
        )
        if [[ -f "$PRETRAINED" ]]; then
            echo "Found FP32 checkpoint: $PRETRAINED"
            PARAMS+=(--pretrained "$PRETRAINED")
        else
            echo "[WARN] FP32 checkpoint not found: $PRETRAINED — starting from scratch"
        fi
        if [[ "$WANDB" == "true" ]]; then
            PARAMS+=(--wandb --wandb_project "deepfake-detection"
                     --wandb_name "${MODEL}_qat_${SLURM_JOB_ID}")
        fi
    fi

fi

################################################################################
# RUN
################################################################################

echo "Executing: $PYTHON_BIN $SCRIPT ${PARAMS[*]}"
"$PYTHON_BIN" "$SCRIPT" "${PARAMS[@]}"

EXIT_CODE=$?
echo "Finished with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
