"""Backend-neutral training metrics and terminal-reason normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

YOLOV5_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "box_loss": ("box_loss", "train/box_loss"),
    "cls_loss": ("cls_loss", "train/cls_loss"),
    "objectness_loss": ("objectness_loss", "train/obj_loss", "obj_loss"),
    "dfl_loss": ("dfl_loss", "train/dfl_loss"),
    "precision": ("precision", "metrics/precision", "metrics/precision(B)"),
    "recall": ("recall", "metrics/recall", "metrics/recall(B)"),
    "map50": (
        "map50",
        "mAP50",
        "metrics/mAP_0.5",
        "metrics/mAP50",
        "metrics/mAP50(B)",
    ),
    "map50_95": (
        "map50_95",
        "map50-95",
        "mAP50-95",
        "metrics/mAP_0.5:0.95",
        "metrics/mAP50-95",
        "metrics/mAP50-95(B)",
    ),
    "learning_rate": ("learning_rate", "x/lr0", "lr/pg0", "lr0"),
}


def normalize_yolov5_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Add canonical metric aliases while retaining every original CSV field.

    YOLOv5 v7 does not calculate distribution-focal loss.  It is deliberately
    represented as ``None``/``unavailable`` rather than mapping objectness loss
    to DFL, which would make cross-model charts technically incorrect.
    """

    result = {str(key).strip(): value for key, value in metrics.items()}
    lookup = {key.casefold(): key for key in result}
    for canonical, aliases in YOLOV5_METRIC_ALIASES.items():
        current = result.get(canonical)
        if current is not None:
            continue
        for alias in aliases:
            source_key = lookup.get(alias.casefold())
            if source_key is not None and result[source_key] is not None:
                result[canonical] = result[source_key]
                break
    if result.get("dfl_loss") is None:
        result["dfl_loss"] = None
        result["dfl_loss_status"] = "unavailable"
    else:
        result["dfl_loss_status"] = "available"
    return result


class TrainingEndReason(StrEnum):
    MAX_EPOCHS = "max_epochs"
    EARLY_STOPPING = "early_stopping"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TrainingEndResult:
    """Serializable terminal outcome shared by modern and legacy adapters."""

    reason: TrainingEndReason | str
    completed_epochs: int
    requested_epochs: int
    patience: int
    monitor: str = "fitness"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", TrainingEndReason(self.reason))
        for name, value in (
            ("completed_epochs", self.completed_epochs),
            ("requested_epochs", self.requested_epochs),
            ("patience", self.patience),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} 必须是整数")
        if self.completed_epochs < 0:
            raise ValueError("completed_epochs 不能小于 0")
        if self.requested_epochs < 1:
            raise ValueError("requested_epochs 必须大于 0")
        if self.patience < 0:
            raise ValueError("patience 不能小于 0")
        monitor = str(self.monitor).strip()
        if not monitor:
            raise ValueError("monitor 不能为空")
        object.__setattr__(self, "monitor", monitor)
        object.__setattr__(
            self,
            "evidence",
            tuple(str(item) for item in self.evidence if str(item).strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "completed_epochs": self.completed_epochs,
            "requested_epochs": self.requested_epochs,
            "patience": self.patience,
            "monitor": self.monitor,
            "evidence": list(self.evidence),
        }


def resolve_training_end(
    *,
    completed_epochs: int,
    requested_epochs: int,
    patience: int,
    early_stopping: bool | None = None,
    cancelled: bool = False,
    failed: bool = False,
    log_lines: str | Iterable[str] | None = None,
    monitor: str = "fitness",
) -> TrainingEndResult:
    """Resolve a terminal reason without guessing from a short run alone.

    A clean run that ends before its requested epoch count is only called early
    stopping when there is explicit backend or log evidence and early stopping
    was enabled.  Otherwise it remains ``unknown``.
    """

    evidence: list[str] = []
    log_signal = contains_legacy_early_stopping(log_lines)
    if early_stopping is True:
        evidence.append("backend_early_stopping_signal")
    if log_signal:
        evidence.append("early_stopping_log")

    if cancelled:
        reason = TrainingEndReason.CANCELLED
        evidence.append("cancelled")
    elif failed:
        reason = TrainingEndReason.FAILED
        evidence.append("failed")
    elif completed_epochs >= requested_epochs:
        reason = TrainingEndReason.MAX_EPOCHS
        evidence.append("completed_requested_epochs")
    elif patience > 0 and (early_stopping is True or log_signal):
        reason = TrainingEndReason.EARLY_STOPPING
    else:
        reason = TrainingEndReason.UNKNOWN

    return TrainingEndResult(
        reason=reason,
        completed_epochs=completed_epochs,
        requested_epochs=requested_epochs,
        patience=patience,
        monitor=monitor,
        evidence=tuple(evidence),
    )


def training_end_from_ultralytics(
    trainer: object | None,
    *,
    requested_epochs: int,
    patience: int,
) -> TrainingEndResult:
    """Read completion evidence from an Ultralytics trainer instance."""

    completed = ultralytics_completed_epochs(trainer)
    early = ultralytics_early_stopping_detected(
        trainer,
        completed_epochs=completed,
        requested_epochs=requested_epochs,
        patience=patience,
    )
    return resolve_training_end(
        completed_epochs=completed,
        requested_epochs=requested_epochs,
        patience=patience,
        early_stopping=early,
    )


def ultralytics_completed_epochs(trainer: object | None) -> int:
    """Return Ultralytics' zero-based current epoch as a completed count."""

    if trainer is None:
        return 0
    raw_epoch = getattr(trainer, "epoch", None)
    if raw_epoch is None:
        return 0
    try:
        return max(0, int(raw_epoch) + 1)
    except (TypeError, ValueError, OverflowError):
        return 0


def ultralytics_early_stopping_detected(
    trainer: object | None,
    *,
    completed_epochs: int,
    requested_epochs: int,
    patience: int,
) -> bool:
    """Require positive stopper evidence before reporting an early stop."""

    if (
        trainer is None
        or patience <= 0
        or completed_epochs <= 0
        or completed_epochs >= requested_epochs
    ):
        return False
    stopper = getattr(trainer, "stopper", None)
    if stopper is None:
        return False
    for name in ("early_stopped", "stopped", "early_stop"):
        if getattr(stopper, name, False) is True:
            return True
    if (
        getattr(trainer, "stop", False) is True
        and getattr(stopper, "possible_stop", False) is True
    ):
        return True
    best_epoch = getattr(stopper, "best_epoch", None)
    current_epoch = getattr(trainer, "epoch", None)
    try:
        return int(current_epoch) - int(best_epoch) >= patience
    except (TypeError, ValueError, OverflowError):
        return False


_EARLY_STOP_PATTERNS = (
    re.compile(r"\bstopping\s+training\s+early\b", re.IGNORECASE),
    re.compile(r"\bearly[ -]?stopping\s+(?:was\s+)?triggered\b", re.IGNORECASE),
    re.compile(r"\btriggered\s+early[ -]?stopping\b", re.IGNORECASE),
)


def contains_legacy_early_stopping(
    log_lines: str | Iterable[str] | None,
) -> bool:
    """Recognize explicit YOLOv5 early-stop terminal messages."""

    if log_lines is None:
        return False
    text = (
        log_lines
        if isinstance(log_lines, str)
        else "\n".join(str(line) for line in log_lines)
    )
    return any(pattern.search(text) is not None for pattern in _EARLY_STOP_PATTERNS)


__all__ = [
    "TrainingEndReason",
    "TrainingEndResult",
    "YOLOV5_METRIC_ALIASES",
    "contains_legacy_early_stopping",
    "normalize_yolov5_metrics",
    "resolve_training_end",
    "training_end_from_ultralytics",
    "ultralytics_completed_epochs",
    "ultralytics_early_stopping_detected",
]
