"""
Export PyTorch checkpoints to ONNX format.

Usage:
    # FP32 export
    python onnx_experiments/export_to_onnx.py --model resnet50 --checkpoint checkpoints/best_resnet50_fp32.pth
    python onnx_experiments/export_to_onnx.py --model vit_b_16  --checkpoint checkpoints/best_vit_b_16_fp32.pth

    # QAT export (float graph with simulated quantization baked in)
    python onnx_experiments/export_to_onnx.py --model resnet50 --checkpoint checkpoints/best_resnet50_qat.pth --qat
    python onnx_experiments/export_to_onnx.py --model vit_b_16  --checkpoint checkpoints/best_vit_b_16_qat.pth  --qat
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NUM_CLASSES = 2
INPUT_SHAPE = (1, 3, 224, 224)


def build_model(model_name: str, num_classes: int = NUM_CLASSES,
                features: tuple = ("rgb", "hsv", "fft", "noise", "srm")) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "dinov2_vitb14":
        base = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        class DinoV2Classifier(nn.Module):
            def __init__(self, base_model, n_classes):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Linear(768, n_classes)

            def forward(self, x):
                return self.classifier(self.base_model(x))

        model = DinoV2Classifier(base, num_classes)
    elif model_name == "forensic_mobilenet":
        from common.forensic_mobilenet import ForensicMobileNetV3
        model = ForensicMobileNetV3(
            features=features,
            num_classes=num_classes,
            pretrained_rgb=False,
            noise_augment=False,   # always off at export / inference
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def load_fp32(model_name: str, ckpt_path: str,
              features: tuple = ("rgb", "hsv", "fft", "noise", "srm")) -> nn.Module:
    model = build_model(model_name, features=features)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def load_qat(model_name: str, ckpt_path: str,
             features: tuple = ("rgb", "hsv", "fft", "noise", "srm")) -> nn.Module:
    from common.utils import replace_with_quantized_modules
    from common.solution import Quantized_Conv2d, Quantized_Linear

    model = build_model(model_name, features=features)
    replace_with_quantized_modules(model)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)

    # Activate symmetric 8-bit quantization for all layers (as used during QAT training)
    for name, m in model.named_modules():
        if isinstance(m, (Quantized_Conv2d, Quantized_Linear)):
            if m.weight_N_bits is not None:  # already calibrated
                m.method = "sym"
    return model


def _patch_fft_for_onnx(model: nn.Module):
    """
    aten::fft_rfft2 has no ONNX symbolic — replace the FFT branch with zeros
    of the correct shape so the rest of the graph exports cleanly.
    Only active during export (patch is applied on the live object, not saved).
    """
    try:
        from common.forensic_mobilenet import ForensicFeatureExtractor
    except ImportError:
        return
    for m in model.modules():
        if isinstance(m, ForensicFeatureExtractor) and "fft" in m.features:
            orig_fft = m._fft_features
            m._fft_features = lambda x: torch.zeros(
                x.shape[0], 3, x.shape[2], x.shape[3], device=x.device
            )
            print("  [WARN] FFT branch replaced with zeros for ONNX export "
                  "(aten::fft_rfft2 unsupported). rgb/hsv/noise/srm export correctly.")
            break


def export(model: nn.Module, output_path: str, opset: int = 17):
    model.eval()
    _patch_fft_for_onnx(model)
    dummy = torch.zeros(INPUT_SHAPE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    size_mb = os.path.getsize(output_path) / 1024 ** 2
    print(f"Exported  → {output_path}  ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model", required=True,
                        choices=["resnet50", "vit_b_16", "mobilenet_v3_small",
                                 "dinov2_vitb14", "forensic_mobilenet"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path (default: onnx_experiments/models/<model>_<variant>.onnx)")
    parser.add_argument("--qat", action="store_true",
                        help="Load as QAT model (Quantized_* layers)")
    parser.add_argument("--features", nargs="+",
                        default=["rgb", "hsv", "fft", "noise", "srm"],
                        help="Feature modalities for forensic_mobilenet (ignored for other models)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    variant = "qat" if args.qat else "fp32"
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "models", f"{args.model}_{variant}.onnx"
    )

    features = tuple(args.features)
    print(f"Loading {args.model} ({variant}) from {args.checkpoint} ...")
    if args.qat:
        model = load_qat(args.model, args.checkpoint, features=features)
    else:
        model = load_fp32(args.model, args.checkpoint, features=features)

    export(model, output_path, opset=args.opset)


if __name__ == "__main__":
    main()
