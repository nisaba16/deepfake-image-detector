from __future__ import annotations

from typing import Optional

import numpy as np


class BaseClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:  # pragma: no cover - interface
        return None


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, n_jobs: int = -1) -> None:
        try:  # local import to keep optional dep
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("scikit-learn is required for LogisticRegressionClassifier") from e
        self.model = LogisticRegression(C=C, max_iter=max_iter, n_jobs=n_jobs, solver="lbfgs")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class LinearSVMClassifier(BaseClassifier):
    def __init__(self, C: float = 1.0, probability: bool = True) -> None:
        try:
            from sklearn.svm import SVC  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("scikit-learn is required for LinearSVMClassifier") from e
        self.model = SVC(kernel="linear", C=C, probability=probability)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class RandomForestClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None, n_jobs: int = -1) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier as SKRF  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("scikit-learn is required for RandomForestClassifier") from e
        self.model = SKRF(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


# ============================================================================
# ANOMALY DETECTION CLASSIFIERS
# ============================================================================
# These treat deepfake detection as an anomaly detection problem:
# - Train on "normal" class (e.g., real faces only)
# - Detect "abnormal" samples (e.g., generated/manipulated faces)
# - No need for balanced labeled data of both real and fake


class OneClassSVMClassifier(BaseClassifier):
    """One-Class SVM for anomaly detection.
    
    Treats the majority class (normal, e.g., real faces) as the target
    and learns its boundary. Samples far from this boundary are considered
    anomalies (fake faces).
    
    Requires:
        y: Binary labels where one class is the "normal" class
        Use positive labels (1) for normal (e.g., real), negative (0) for anomaly (fake)
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
        coef0: float = 0.0,
    ) -> None:
        """
        Args:
            nu: Upper bound on fraction of anomalies in training set (0.001-0.5 typical)
            kernel: 'linear', 'rbf', 'poly', 'sigmoid'
            gamma: Kernel coefficient ('scale', 'auto', or float)
            coef0: Kernel coefficient for poly/sigmoid
        """
        try:
            from sklearn.svm import OneClassSVM  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("scikit-learn is required for OneClassSVMClassifier") from e

        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, coef0=coef0)
        self.nu = nu
        self.kernel = kernel

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit on the majority/normal class.
        
        Args:
            X: Feature matrix
            y: Binary labels (1=normal/real, 0=anomaly/fake)
        
        Note: One-Class SVM learns from the normal class only.
              Unlabeled data is also fine - it will learn the normal distribution.
        """
        # Use only normal samples (y==1) if labeled data is provided
        # Otherwise use all data
        if len(np.unique(y)) > 1:
            normal_mask = y == 1
            if normal_mask.sum() > 0:
                X_normal = X[normal_mask]
            else:
                # If no positive labels, train on all
                X_normal = X
        else:
            # If all same label, train on all
            X_normal = X

        self.model.fit(X_normal)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels: 1=normal, -1=anomaly.
        
        Returns array with:
            1 = normal (real face)
            -1 = anomaly (fake face)
        """
        raw_pred = self.model.predict(X)
        # Convert from {-1, 1} to {1, 0} (real=1, fake=0)
        return ((raw_pred + 1) // 2).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get anomaly scores as pseudo-probabilities.
        
        Returns array with shape (n_samples, 2):
            [:,0] = probability of being anomaly (fake)
            [:,1] = probability of being normal (real)
        """
        # Get decision function (distance from hyperplane)
        # Negative = anomaly, positive = normal
        decision = self.model.decision_function(X)

        # Normalize to [0, 1] range using sigmoid
        from scipy.special import expit  # type: ignore

        # Invert: high decision function -> low anomaly probability
        anomaly_score = 1.0 / (1.0 + np.exp(decision))  # sigmoid(-decision)

        # Return [prob_fake, prob_real]
        proba = np.column_stack([anomaly_score, 1.0 - anomaly_score])
        return proba


class EllipticEnvelopeClassifier(BaseClassifier):
    """Elliptic Envelope (Robust Covariance) for anomaly detection.
    
    Learns a robust covariance estimate of the normal class distribution
    and detects samples with high Mahalanobis distance as anomalies.
    
    More interpretable than One-Class SVM: assumes Gaussian distribution.
    Performs well in low-moderate dimensions.
    
    Requires:
        y: Binary labels where one class is the "normal" class
    """

    def __init__(self, contamination: float = 0.1, support_fraction: Optional[float] = None) -> None:
        """
        Args:
            contamination: Expected fraction of outliers in dataset (0.0-0.5)
            support_fraction: Fraction of samples to include in fit (None=auto)
        """
        try:
            from sklearn.covariance import EllipticEnvelope  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("scikit-learn is required for EllipticEnvelopeClassifier") from e

        self.model = EllipticEnvelope(
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=42,
        )
        self.contamination = contamination

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit robust covariance on normal class.
        
        Args:
            X: Feature matrix
            y: Binary labels (1=normal/real, 0=anomaly/fake)
        
        Note: Learns covariance from the normal class (y==1).
              Best with n_features < n_samples.
        """
        # Use normal samples if labeled
        if len(np.unique(y)) > 1:
            normal_mask = y == 1
            if normal_mask.sum() > 0:
                X_normal = X[normal_mask]
            else:
                X_normal = X
        else:
            X_normal = X

        self.model.fit(X_normal)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels: 1=normal, -1=anomaly.
        
        Returns array with:
            1 = normal (real face)
            -1 = anomaly (fake face)
        """
        raw_pred = self.model.predict(X)
        # Convert from {-1, 1} to {1, 0}
        return ((raw_pred + 1) // 2).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get Mahalanobis distance as pseudo-probabilities.
        
        Returns array with shape (n_samples, 2):
            [:,0] = probability of being anomaly (fake)
            [:,1] = probability of being normal (real)
        """
        # Get Mahalanobis distance
        mahal_dist = self.model.mahalanobis(X)

        # Normalize using sigmoid
        # Higher distance = more likely anomaly
        anomaly_score = 1.0 / (1.0 + np.exp(-mahal_dist))

        # Return [prob_fake, prob_real]
        proba = np.column_stack([anomaly_score, 1.0 - anomaly_score])
        return proba


class IsolationForestClassifier(BaseClassifier):
    """Isolation Forest for anomaly detection.
    
    Uses isolation forests to identify anomalies without computing
    explicit densities. Very efficient for high-dimensional data.
    Works well with limited labeled data.
    
    Requires:
        y: Binary labels (1=normal, 0=anomaly, can be mostly unlabeled)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = "auto",
        n_jobs: int = -1,
    ) -> None:
        """
        Args:
            contamination: Expected fraction of anomalies (0.0-0.5)
            n_estimators: Number of isolation trees
            max_samples: Number of samples per tree
            n_jobs: Parallel jobs
        """
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("scikit-learn is required for IsolationForestClassifier") from e

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=42,
        )
        self.contamination = contamination

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit isolation forest.
        
        Args:
            X: Feature matrix
            y: Can contain labels (ignored), or be all same value (unsupervised)
        
        Note: Isolation Forest learns anomalies from all data.
              Can work with completely unlabeled data.
        """
        # Train on all data (Isolation Forest doesn't require labels)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels: 1=normal, -1=anomaly.
        
        Returns array with:
            1 = normal (real face)
            -1 = anomaly (fake face)
        """
        raw_pred = self.model.predict(X)
        # Convert from {-1, 1} to {1, 0}
        return ((raw_pred + 1) // 2).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get anomaly scores as pseudo-probabilities.
        
        Returns array with shape (n_samples, 2):
            [:,0] = probability of being anomaly (fake)
            [:,1] = probability of being normal (real)
        """
        # Get anomaly scores
        anomaly_score = self.model.score_samples(X)

        # Normalize: more negative = more anomalous
        # Use sigmoid on negative scores
        anomaly_prob = 1.0 / (1.0 + np.exp(anomaly_score))

        # Return [prob_fake, prob_real]
        proba = np.column_stack([anomaly_prob, 1.0 - anomaly_prob])
        return proba


# Factories

def build_classifier(name: str, **kwargs) -> BaseClassifier:
    """Build a classifier by name.
    
    Binary Classification (supervised):
        - logreg, logistic, logistic_regression
        - svm, linear_svm, linsvm
        - rf, random_forest
    
    Anomaly Detection (weakly supervised/unsupervised):
        - one_class_svm, ocsvm
        - elliptic_envelope, elliptic
        - isolation_forest, iforest
    
    Args:
        name: Classifier name (case-insensitive)
        **kwargs: Additional arguments passed to classifier constructor
    
    Returns:
        BaseClassifier instance
    
    Example:
        # Binary classification
        clf = build_classifier("rf")
        
        # Anomaly detection
        clf = build_classifier("one_class_svm", nu=0.05)
        clf = build_classifier("isolation_forest", contamination=0.15)
    """
    name = name.lower()
    
    # Binary classifiers
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegressionClassifier(**kwargs)
    if name in {"svm", "linear_svm", "linsvm"}:
        return LinearSVMClassifier(**kwargs)
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(**kwargs)
    
    # Anomaly detection classifiers
    if name in {"one_class_svm", "ocsvm", "oc_svm"}:
        return OneClassSVMClassifier(**kwargs)
    if name in {"elliptic_envelope", "elliptic", "robust_covariance"}:
        return EllipticEnvelopeClassifier(**kwargs)
    if name in {"isolation_forest", "iforest", "iso_forest"}:
        return IsolationForestClassifier(**kwargs)
    
    raise ValueError(
        f"Unknown classifier: {name}. "
        f"Choose from: logreg, svm, rf (binary) or "
        f"one_class_svm, elliptic_envelope, isolation_forest (anomaly detection)"
    )
