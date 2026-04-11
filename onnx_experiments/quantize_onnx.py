"""
Apply onnxruntime static INT8 quantization to an FP32 ONNX model.

For QAT models exported from PyTorch with custom Quantized_* layers:
they are already float ONNX graphs that simulate quantization — skip this step,
onnxruntime runs them as-is.

Usage:
    python onnx_experiments/quantize_onnx.py \
        --input  onnx_experiments/models/resnet50_fp32.onnx \
        --output onnx_experiments/models/resnet50_int8.onnx \
        --data_dir data/dataset

    # QDQ format (recommended for GPU / hardware accelerators):
    python onnx_experiments/quantize_onnx.py \
        --input  onnx_experiments/models/vit_b_16_fp32.onnx \
        --output onnx_experiments/models/vit_b_16_int8.onnx \
        --data_dir data/dataset \
        --quant_format QDQ \
        --per_channel
"""

import argparse
import os
import sys

from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_experiments.data_reader import DeepfakeCalibrationReader


def main():
    parser = argparse.ArgumentParser(description="ORT static INT8 quantization")
    parser.add_argument("--input",  required=True, help="Input FP32 ONNX model")
    parser.add_argument("--output", required=True, help="Output INT8 ONNX model")
    parser.add_argument("--data_dir", default="data/dataset",
                        help="Dataset root for calibration")
    parser.add_argument("--num_calibration_samples", type=int, default=256,
                        help="Number of images for calibration")
    parser.add_argument("--quant_format", default="QDQ",
                        choices=["QDQ", "QOperator"],
                        help="QDQ = QuantizeLinear/DequantizeLinear nodes (better HW support); "
                             "QOperator = fused quantized ops")
    parser.add_argument("--per_channel", action="store_true",
                        help="Per-channel weight quantization (more accurate, slower calibration)")
    parser.add_argument("--calibration_method", default="MinMax",
                        choices=["MinMax", "Entropy", "Percentile"])
    args = parser.parse_args()

    fmt = QuantFormat.QDQ if args.quant_format == "QDQ" else QuantFormat.QOperator
    cal_method = {
        "MinMax":     CalibrationMethod.MinMax,
        "Entropy":    CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }[args.calibration_method]

    print(f"Calibrating with {args.num_calibration_samples} samples "
          f"({args.calibration_method}, {args.quant_format}, "
          f"per_channel={args.per_channel}) ...")

    dr = DeepfakeCalibrationReader(
        data_dir=args.data_dir,
        model_path=args.input,
        num_samples=args.num_calibration_samples,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=dr,
        quant_format=fmt,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=cal_method,
        # Keep model optimized for inference
        optimize_model=True,
    )

    in_mb  = os.path.getsize(args.input)  / 1024 ** 2
    out_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"Done.")
    print(f"  FP32: {args.input}  ({in_mb:.2f} MB)")
    print(f"  INT8: {args.output} ({out_mb:.2f} MB)  "
          f"[{(1 - out_mb / in_mb) * 100:.1f}% smaller]")


if __name__ == "__main__":
    main()
