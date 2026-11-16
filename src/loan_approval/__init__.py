"""Loan Approval Prediction & Explainable AI Package."""

from .config import (
    ALL_INPUT_FIELDS,
    CLASS_LABELS,
    FEATURE_DISPLAY_NAMES,
    FINAL_MODEL_FEATURES,
    OPTIONAL_CONTEXTUAL_FIELDS,
    REQUIRED_FINANCIAL_FIELDS,
    SCALED_NUMERICAL_FEATURES,
)
from .explanation import (
    ExplanationResult,
    FeatureContribution,
    explain_prediction,
    render_waterfall_plot,
)
from .model_loader import (
    ArtifactIncompatibleError,
    ModelArtifacts,
    ModelLoadError,
    load_artifacts,
)
from .predictor import (
    CONFIDENCE_DISCLAIMER,
    PredictionResult,
    predict_batch,
    predict_loan_approval,
)
from .preprocessing import (
    PreprocessingOutput,
    compute_engineered_features,
    preprocess_applicant_data,
)
from .validation import (
    ValidationError,
    ValidationResult,
    validate_applicant_data,
)

__version__ = "1.0.0"
__all__ = [
    # Prediction
    "predict_loan_approval",
    "predict_batch",
    "PredictionResult",
    "CONFIDENCE_DISCLAIMER",
    # Explanation
    "explain_prediction",
    "render_waterfall_plot",
    "ExplanationResult",
    "FeatureContribution",
    # Validation
    "validate_applicant_data",
    "ValidationResult",
    "ValidationError",
    # Preprocessing
    "preprocess_applicant_data",
    "compute_engineered_features",
    "PreprocessingOutput",
    # Model Loading
    "load_artifacts",
    "ModelArtifacts",
    "ModelLoadError",
    "ArtifactIncompatibleError",
    # Config & Constants
    "REQUIRED_FINANCIAL_FIELDS",
    "OPTIONAL_CONTEXTUAL_FIELDS",
    "ALL_INPUT_FIELDS",
    "FINAL_MODEL_FEATURES",
    "SCALED_NUMERICAL_FEATURES",
    "FEATURE_DISPLAY_NAMES",
    "CLASS_LABELS",
    "__version__",
]
