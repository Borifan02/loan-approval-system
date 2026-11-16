"""Model and scaler loader with caching, validation, and diagnostic checks.

This module guarantees thread-safe, cached access to the trained Random Forest
model and StandardScaler artifacts without redundant disk I/O.
"""

from __future__ import annotations

import json
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from .config import (
    FALLBACK_MODEL_PATH,
    FALLBACK_SCALER_PATH,
    FINAL_MODEL_FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    SCALED_NUMERICAL_FEATURES,
)
from .utils import logger


class ModelLoadError(FileNotFoundError):
    """Raised when model or scaler files cannot be found on disk."""
    pass


class ArtifactIncompatibleError(ValueError):
    """Raised when loaded artifacts fail consistency or schema checks."""
    pass


@dataclass
class ModelArtifacts:
    """Encapsulates loaded machine learning models and preprocessing transformers."""

    model: Any
    scaler: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_path: Path = field(default_factory=lambda: MODEL_PATH)
    scaler_path: Path = field(default_factory=lambda: SCALER_PATH)

    @property
    def expected_features(self) -> List[str]:
        """Returns the list of features expected by the model."""
        return self.metadata.get("expected_features", FINAL_MODEL_FEATURES)

    @property
    def scaled_features(self) -> List[str]:
        """Returns the list of features transformed by the scaler."""
        return self.metadata.get("scaled_features", SCALED_NUMERICAL_FEATURES)


_lock = threading.Lock()
_cached_artifacts: Optional[ModelArtifacts] = None


def resolve_artifact_path(primary: Path, fallback: Path, description: str) -> Path:
    """Resolves an artifact path, attempting primary then fallback."""
    if primary.is_file():
        return primary
    if fallback.is_file():
        logger.warning(
            "Primary %s path not found at '%s'. Using fallback at '%s'.",
            description,
            primary,
            fallback,
        )
        return fallback
    raise ModelLoadError(
        f"Unable to locate {description}. Checked:\n"
        f"  1. Primary path:  {primary.resolve()}\n"
        f"  2. Fallback path: {fallback.resolve()}\n"
        f"Please verify that the model artifacts exist."
    )


def load_metadata(path: Path) -> Dict[str, Any]:
    """Loads metadata JSON if present; returns default metadata if absent."""
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metadata from %s: %s", path, e)

    # Return default metadata reflecting original training configuration
    return {
        "model_type": "RandomForestClassifier",
        "framework": "scikit-learn",
        "expected_features": FINAL_MODEL_FEATURES,
        "scaled_features": SCALED_NUMERICAL_FEATURES,
        "classes": [0, 1],
        "class_names": {0: "Rejected", 1: "Approved"},
        "description": "Random Forest model retrained with 4 financial history features",
    }


def validate_artifacts(model: Any, scaler: Any) -> None:
    """Verifies that loaded artifacts conform to expected interfaces and dimensions."""
    if not hasattr(model, "predict"):
        raise ArtifactIncompatibleError("Loaded model object does not implement 'predict()'")
    if not hasattr(model, "predict_proba"):
        raise ArtifactIncompatibleError("Loaded model object does not implement 'predict_proba()'")

    # Verify number of input features for the model
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != len(FINAL_MODEL_FEATURES):
        raise ArtifactIncompatibleError(
            f"Model expects {n_features} features, but {len(FINAL_MODEL_FEATURES)} "
            f"were specified: {FINAL_MODEL_FEATURES}"
        )

    # Verify scaler
    if not hasattr(scaler, "transform"):
        raise ArtifactIncompatibleError("Loaded scaler object does not implement 'transform()'")

    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features is not None and scaler_features != len(SCALED_NUMERICAL_FEATURES):
        raise ArtifactIncompatibleError(
            f"Scaler expects {scaler_features} features, but {len(SCALED_NUMERICAL_FEATURES)} "
            f"were specified: {SCALED_NUMERICAL_FEATURES}"
        )


def load_artifacts(force_reload: bool = False) -> ModelArtifacts:
    """Loads and caches the model and scaler artifacts in a thread-safe manner."""
    global _cached_artifacts

    if _cached_artifacts is not None and not force_reload:
        return _cached_artifacts

    with _lock:
        if _cached_artifacts is not None and not force_reload:
            return _cached_artifacts

        model_file = resolve_artifact_path(MODEL_PATH, FALLBACK_MODEL_PATH, "Random Forest model")
        scaler_file = resolve_artifact_path(SCALER_PATH, FALLBACK_SCALER_PATH, "StandardScaler")

        logger.info("Loading model from %s", model_file)
        model = joblib.load(model_file)

        logger.info("Loading scaler from %s", scaler_file)
        scaler = joblib.load(scaler_file)

        validate_artifacts(model, scaler)

        metadata = load_metadata(METADATA_PATH)

        _cached_artifacts = ModelArtifacts(
            model=model,
            scaler=scaler,
            metadata=metadata,
            model_path=model_file,
            scaler_path=scaler_file,
        )
        return _cached_artifacts
