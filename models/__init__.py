from .extractors import (
    BaseImageFeatureExtractor,
    FrequencyAnalysisExtractor,
    build_extractor,
)
from .classifiers import (
    BaseClassifier,
    LogisticRegressionClassifier,
    LinearSVMClassifier,
    RandomForestClassifier,
    OneClassSVMClassifier,
    EllipticEnvelopeClassifier,
    IsolationForestClassifier,
    build_classifier,
)
from .pipeline import ImageDetectionPipeline
from .face_preprocessor import (
    FaceDetector,
    MediapipeFaceDetector,
    MTCNNFaceDetector,
    RetinaFaceDetector,
    FacePreprocessor,
    create_face_preprocessor,
    create_landmark_mask,
    align_and_crop_face,
)
# Face extractors deprecated in favor of client-side MobileNetV3 architecture
# from .face_extractors import (
#     FaceAwareFrequencyExtractor,
#     FaceAwareSpatialExtractor,
#     FaceAwareMultiModalExtractor,
#     build_face_aware_extractor,
# )

__all__ = [
    # Standard extractors
    "BaseImageFeatureExtractor",
    "FrequencyAnalysisExtractor",
    "build_extractor",
    # Classifiers - Binary
    "BaseClassifier",
    "LogisticRegressionClassifier",
    "LinearSVMClassifier",
    "RandomForestClassifier",
    # Classifiers - Anomaly Detection
    "OneClassSVMClassifier",
    "EllipticEnvelopeClassifier",
    "IsolationForestClassifier",
    "build_classifier",
    # Pipeline
    "ImageDetectionPipeline",
    # Face preprocessing
    "FaceDetector",
    "MediapipeFaceDetector",
    "MTCNNFaceDetector",
    "RetinaFaceDetector",
    "FacePreprocessor",
    "create_face_preprocessor",
    "create_landmark_mask",
    "align_and_crop_face",
    # Face-aware extractors (deprecated - use MobileNetV3)
    # "FaceAwareFrequencyExtractor",
    # "FaceAwareSpatialExtractor",
    # "FaceAwareMultiModalExtractor",
    # "build_face_aware_extractor",
]
