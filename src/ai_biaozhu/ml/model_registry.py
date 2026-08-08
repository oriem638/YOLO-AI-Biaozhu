"""The eight model choices exposed by the desktop application."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class ModelBackend(StrEnum):
    """Execution backend required by a model family."""

    LEGACY_YOLOV5 = "legacy_yolov5"
    ULTRALYTICS = "ultralytics"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    size: str
    weight: str
    backend: ModelBackend
    deployment_decoder: str

    @property
    def is_legacy(self) -> bool:
        return self.backend is ModelBackend.LEGACY_YOLOV5


_MODELS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec(
        "YOLOv5n",
        "YOLOv5n",
        "yolov5",
        "n",
        "yolov5n.pt",
        ModelBackend.LEGACY_YOLOV5,
        "yolov5",
    ),
    ModelSpec(
        "YOLOv5s",
        "YOLOv5s",
        "yolov5",
        "s",
        "yolov5s.pt",
        ModelBackend.LEGACY_YOLOV5,
        "yolov5",
    ),
    ModelSpec(
        "YOLOv8n",
        "YOLOv8n",
        "yolov8",
        "n",
        "yolov8n.pt",
        ModelBackend.ULTRALYTICS,
        "yolov8",
    ),
    ModelSpec(
        "YOLOv8s",
        "YOLOv8s",
        "yolov8",
        "s",
        "yolov8s.pt",
        ModelBackend.ULTRALYTICS,
        "yolov8",
    ),
    ModelSpec(
        "YOLO11n",
        "YOLO11n",
        "yolo11",
        "n",
        "yolo11n.pt",
        ModelBackend.ULTRALYTICS,
        "yolo11",
    ),
    ModelSpec(
        "YOLO11s",
        "YOLO11s",
        "yolo11",
        "s",
        "yolo11s.pt",
        ModelBackend.ULTRALYTICS,
        "yolo11",
    ),
    ModelSpec(
        "YOLO26n",
        "YOLO26n",
        "yolo26",
        "n",
        "yolo26n.pt",
        ModelBackend.ULTRALYTICS,
        "yolo26",
    ),
    ModelSpec(
        "YOLO26s",
        "YOLO26s",
        "yolo26",
        "s",
        "yolo26s.pt",
        ModelBackend.ULTRALYTICS,
        "yolo26",
    ),
)

MODEL_REGISTRY: Final = MappingProxyType({model.key: model for model in _MODELS})
DEFAULT_MODEL_KEY: Final = "YOLO26n"

_ALIASES: Final = MappingProxyType(
    {
        model.key.casefold(): model.key
        for model in _MODELS
    }
    | {
        model.weight.casefold(): model.key
        for model in _MODELS
    }
)


def iter_models() -> Iterator[ModelSpec]:
    """Yield models in the exact order used by the UI."""

    return iter(_MODELS)


def get_model(key: str) -> ModelSpec:
    """Resolve a UI key or official weight filename."""

    normalized = key.strip().casefold()
    try:
        return MODEL_REGISTRY[_ALIASES[normalized]]
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"未知模型 {key!r}；可选项：{choices}") from exc
