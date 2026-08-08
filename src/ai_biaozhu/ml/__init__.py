"""Isolated YOLO worker package.

The GUI process may import this package without importing PyTorch or
Ultralytics.  Heavy dependencies are loaded lazily inside worker adapters.
"""

from .model_registry import MODEL_REGISTRY, ModelBackend, ModelSpec, get_model
from .protocol import (
    PROTOCOL_VERSION,
    JsonlEmitter,
    ProtocolEvent,
    ProtocolSequenceTracker,
    SequenceDecision,
)
from .training_results import (
    TrainingEndReason,
    TrainingEndResult,
    contains_legacy_early_stopping,
    normalize_yolov5_metrics,
    resolve_training_end,
    training_end_from_ultralytics,
)

__all__ = [
    "MODEL_REGISTRY",
    "PROTOCOL_VERSION",
    "JsonlEmitter",
    "ModelBackend",
    "ModelSpec",
    "ProtocolEvent",
    "ProtocolSequenceTracker",
    "SequenceDecision",
    "TrainingEndReason",
    "TrainingEndResult",
    "contains_legacy_early_stopping",
    "get_model",
    "normalize_yolov5_metrics",
    "resolve_training_end",
    "training_end_from_ultralytics",
]
