"""Baseline classifiers: Logistic Regression, Random Forest, SVM.

All three wrap scikit-learn estimators behind a consistent interface::

    model = BaselineModel("svm")
    model.fit(X_train, y_train)
    scores = model.predict_scores(X_test)
    preds = model.predict(X_test)

Scores are calibrated to follow the project convention:
higher score → more likely spoof.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ============================================================
# BASELINE MODEL WRAPPER
# ============================================================

class BaselineModel:
    """Unified wrapper around three scikit-learn classifiers.

    Parameters
    ----------
    model_type : str
        ``"logistic_regression"``, ``"random_forest"``, or ``"svm"``.
    """

    _VALID_TYPES = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
    }

    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        if model_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Choose from {list(self._VALID_TYPES)}"
            )

        self.model_type = model_type
        self.scaler = StandardScaler()
        self._fitted = False

        # Configure estimator with sensible defaults
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                **kwargs,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                **kwargs,
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
                **kwargs,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """Train the model (with feature scaling)."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard predictions (0 = bonafide, 1 = spoof)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return spoof-probability scores (higher = more likely spoof)."""
        X_scaled = self.scaler.transform(X)
        # Column 1 = P(class=1) = P(spoof)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]

    def save(self, path: str):
        """Persist the model (including scaler)."""
        joblib.dump(
            {"model": self.model, "scaler": self.scaler,
             "model_type": self.model_type},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        """Load a saved model."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.model_type = data["model_type"]
        instance._fitted = True
        return instance


# ============================================================
# BATCH TRAINING
# ============================================================

def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_types: list[str] | None = None,
) -> dict[str, BaselineModel]:
    """Train multiple baseline models at once.

    Returns
    -------
    dict  mapping model type → trained BaselineModel
    """
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "svm"]

    models = {}
    for mt in model_types:
        model = BaselineModel(mt)
        model.fit(X_train, y_train)
        models[mt] = model
        print(f"  Trained {mt}")

    return models


def evaluate_baselines(
    models: dict[str, BaselineModel],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """Evaluate multiple baseline models.

    Returns
    -------
    dict  mapping model type → result dict from ``evaluate_predictions``
    """
    from ..evaluation.metrics import evaluate_predictions

    results = {}
    for name, model in models.items():
        scores = model.predict_scores(X_test)
        preds = model.predict(X_test)
        res = evaluate_predictions(y_test, scores)
        results[name] = res
        print(f"  {name}: EER={res['eer']:.4f}, "
              f"min-tDCF={res['min_tdcf']:.4f}, "
              f"acc={res['accuracy']:.4f}")

    return results
