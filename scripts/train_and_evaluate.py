from __future__ import annotations

import argparse
import json
import numpy as np
from typing import Sequence

from ..models import build_extractor, build_classifier, ImageDetectionPipeline
from ..models.face_extractors import build_face_aware_extractor
from ..utils import collect_image_paths_and_labels, stratified_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/Evaluate deepfake image detector (feature + classifier)")
    p.add_argument("--data", required=True, help="Path to dataset root (ImageFolder-style)")

    # Extractor options (standard)
    p.add_argument("--extractor", default="frequency", 
                   choices=["frequency", "resnet50", "vit", "sota", "timm", 
                           "face_frequency", "face_spatial", "face_multimodal"],
                   help="Feature extractor type")
    p.add_argument("--timm-model", default=None, help="timm model name for 'sota' or 'timm' extractor (e.g., convnext_base)")
    p.add_argument("--img-size", type=int, default=224, help="Input image size for extractors")
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights for deep models")

    # Face-specific options
    p.add_argument("--face-detector", default="mediapipe", 
                   choices=["mediapipe", "mtcnn", "retinaface"],
                   help="Face detector for face-aware extractors")
    p.add_argument("--fusion-method", default="concat", 
                   choices=["concat", "mean"],
                   help="Feature fusion method for multimodal extractor")

    # Multiscale
    p.add_argument("--multiscale", action="store_true", help="Enable multi-scale analysis")
    p.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], help="Scale factors for multi-scale")
    p.add_argument("--pool", default="mean", choices=["mean", "concat"], help="Multi-scale pooling")

    # Classifier options
    p.add_argument("--classifier", default="logreg", 
                   choices=["logreg", "linear_svm", "rf", "one_class_svm", "elliptic_envelope", "isolation_forest"],
                   help="Classifier type (binary or anomaly detection)")
    p.add_argument("--mode", default="supervised", 
                   choices=["supervised", "anomaly"],
                   help="Training mode: 'supervised' (binary) or 'anomaly' (one-class detection)")

    # Anomaly detection specific options
    p.add_argument("--normal-class", default=None, 
                   choices=["real", "fake"],
                   help="Which class to treat as 'normal' for anomaly detection mode. "
                        "If not specified, auto-detected as majority class.")
    p.add_argument("--nu", type=float, default=0.1, 
                   help="For One-Class SVM: fraction of anomalies (0.001-0.5)")
    p.add_argument("--contamination", type=float, default=0.1,
                   help="For Elliptic Envelope/Isolation Forest: expected anomaly fraction")

    # Train/eval
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Optional limit on total samples for quick runs")
    p.add_argument("--skip-failed", action="store_true", help="Skip images that fail preprocessing")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths, labels, class_to_idx, idx_to_class = collect_image_paths_and_labels(args.data)

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
        labels = labels[: args.limit]

    # Determine training mode
    is_anomaly_mode = args.mode == "anomaly"
    
    if is_anomaly_mode:
        # ANOMALY DETECTION MODE: detect anomalies (usually "fake") vs normal (usually "real")
        print(f"[INFO] Training mode: ANOMALY DETECTION")
        
        # Auto-detect normal class if not specified
        if args.normal_class is None:
            # Count samples per class
            unique, counts = np.unique(labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            majority_class = idx_to_class[majority_label]
            print(f"[INFO] Auto-detected majority class as 'normal': {majority_class}")
            normal_class = majority_class
        else:
            normal_class = args.normal_class
        
        # Convert labels to binary (1=normal, 0=anomaly)
        normal_label = class_to_idx[normal_class]
        labels_binary = np.array([1 if l == normal_label else 0 for l in labels])
        print(f"[INFO] Normal class: {normal_class} (label={normal_label})")
        print(f"[INFO] Anomaly class: All others")
        print(f"[INFO] Normal samples: {(labels_binary == 1).sum()}, Anomaly samples: {(labels_binary == 0).sum()}")
        
        # Split data
        X_train, X_test, y_train, y_test = stratified_split(
            paths, labels_binary, test_size=args.test_size, seed=args.seed
        )
        original_labels = labels
    else:
        # BINARY CLASSIFICATION MODE: standard supervised learning
        print(f"[INFO] Training mode: BINARY CLASSIFICATION (supervised)")
        X_train, X_test, y_train, y_test = stratified_split(
            paths, labels, test_size=args.test_size, seed=args.seed
        )
        original_labels = labels

    # Build extractor
    if args.extractor.startswith("face_"):
        extractor = build_face_aware_extractor(
            args.extractor,
            img_size=args.img_size,
            pretrained=not args.no_pretrained,
            model_name=args.timm_model or "resnet50",
            detector_name=args.face_detector,
            fusion_method=args.fusion_method,
        )
    else:
        extractor = build_extractor(
            args.extractor,
            img_size=args.img_size,
            pretrained=not args.no_pretrained,
            timm_model=args.timm_model,
            multiscale=args.multiscale,
            scales=args.scales,
            pool=args.pool,
        )

    # Build classifier with mode-specific parameters
    clf_kwargs = {}
    if args.classifier in {"one_class_svm", "ocsvm", "oc_svm"}:
        clf_kwargs["nu"] = args.nu
    elif args.classifier in {"elliptic_envelope", "elliptic", "robust_covariance"}:
        clf_kwargs["contamination"] = args.contamination
    elif args.classifier in {"isolation_forest", "iforest", "iso_forest"}:
        clf_kwargs["contamination"] = args.contamination

    clf = build_classifier(args.classifier, **clf_kwargs)
    pipe = ImageDetectionPipeline(extractor=extractor, classifier=clf, skip_failed=args.skip_failed)

    print("[INFO] Fitting classifier on training features...")
    pipe.fit(X_train, y_train)

    print("[INFO] Evaluating on test set...")
    acc = pipe.score(X_test, y_test)

    # Get predictions
    y_pred = pipe.predict(X_test)
    
    # Optional: detailed report if sklearn is available
    try:
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # type: ignore

        # Create appropriate labels for reporting
        if is_anomaly_mode:
            # For anomaly detection, report as "Normal" and "Anomaly"
            target_names = ["Anomaly", "Normal"]
            try:
                auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
                auc_str = f"\nROC-AUC Score: {auc:.4f}"
            except Exception:
                auc_str = ""
        else:
            # For binary classification, use original class names
            target_names = [idx_to_class[i] for i in sorted(idx_to_class)]
            try:
                auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
                auc_str = f"\nROC-AUC Score: {auc:.4f}"
            except Exception:
                auc_str = ""

        report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
        confusion = confusion_matrix(y_test, y_pred)
    except Exception as e:
        report = None
        auc_str = ""
        confusion = None

    result = {
        "accuracy": acc,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "classes": class_to_idx,
        "extractor": args.extractor,
        "classifier": args.classifier,
        "mode": args.mode,
    }
    
    if is_anomaly_mode:
        result["normal_class"] = normal_class
        result["anomaly_detection_config"] = {
            "normal_label": int(normal_label),
            "normal_samples": int((y_train == 1).sum()),
            "anomaly_samples": int((y_train == 0).sum()),
        }

    print("[RESULT]", json.dumps(result, indent=2))
    if report:
        print("\n[CLASSIFICATION REPORT]\n" + str(report))
    if auc_str:
        print(auc_str)
    if confusion is not None:
        print("\n[CONFUSION MATRIX]")
        print(confusion)


if __name__ == "__main__":
    main()
