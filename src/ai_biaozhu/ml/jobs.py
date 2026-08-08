"""Validated, JSON-serializable worker job manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AugmentationOptions, TrainingOptions
from .model_registry import get_model


@dataclass(frozen=True, slots=True)
class TrainingJob:
    job_id: str
    model_key: str
    data_yaml: Path
    output_dir: Path
    run_name: str
    options: TrainingOptions
    augmentation: AugmentationOptions
    checkpoint_source: str | Path | None = None
    legacy_yolov5_repo: Path | None = None
    python_executable: Path | None = None
    cancel_file: Path | None = None
    metrics_path: Path | None = None
    weight_cache_dir: Path | None = None
    weight_lock_path: Path | None = None
    offline_weights: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingJob:
        job_id = _required_text(value, "job_id")
        model_key = str(value.get("model_key", "YOLO26n"))
        get_model(model_key)
        output_dir = Path(_required_text(value, "output_dir"))
        options_raw = dict(value.get("config") or {})
        options_raw["model_key"] = model_key
        options = TrainingOptions.from_value(options_raw)
        options.validate()
        augmentation = AugmentationOptions.from_value(value.get("augmentation"))
        augmentation.validate()
        return cls(
            job_id=job_id,
            model_key=model_key,
            data_yaml=Path(_required_text(value, "data_yaml")),
            output_dir=output_dir,
            run_name=str(value.get("run_name") or job_id),
            options=options,
            augmentation=augmentation,
            checkpoint_source=value.get("checkpoint_source"),
            legacy_yolov5_repo=_optional_path(value.get("legacy_yolov5_repo")),
            python_executable=_optional_path(value.get("python_executable")),
            cancel_file=_optional_path(value.get("cancel_file")),
            metrics_path=_optional_path(value.get("metrics_path"))
            or output_dir / "metrics.jsonl",
            weight_cache_dir=_optional_path(value.get("weight_cache_dir")),
            weight_lock_path=_optional_path(value.get("weight_lock_path")),
            offline_weights=bool(value.get("offline_weights", False)),
        )


@dataclass(frozen=True, slots=True)
class PredictionImage:
    image_id: str
    path: Path
    expected_revision: int
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PredictionImage:
        return cls(
            image_id=_required_text(value, "image_id"),
            path=Path(_required_text(value, "path")),
            expected_revision=int(value.get("expected_revision", 0)),
            width=_optional_positive_int(value.get("width")),
            height=_optional_positive_int(value.get("height")),
        )


@dataclass(frozen=True, slots=True)
class PredictionJob:
    job_id: str
    model_key: str
    checkpoint: Path
    images: tuple[PredictionImage, ...]
    output_dir: Path
    class_ids: tuple[str, ...] = ()
    confidence: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: int | str = 0
    legacy_yolov5_repo: Path | None = None
    python_executable: Path | None = None
    cancel_file: Path | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PredictionJob:
        job_id = _required_text(value, "job_id")
        model_key = str(value.get("model_key", "YOLO26n"))
        get_model(model_key)
        raw_images = value.get("images")
        if not isinstance(raw_images, list):
            raise ValueError("images 必须是列表")
        images = tuple(
            PredictionImage.from_mapping(item)
            for item in raw_images
            if isinstance(item, Mapping)
        )
        if len(images) != len(raw_images):
            raise ValueError("images 中的每一项都必须是对象")
        confidence = float(value.get("confidence", 0.25))
        iou = float(value.get("iou", 0.7))
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("confidence 和 iou 必须在 0 到 1 之间")
        imgsz = int(value.get("imgsz", 640))
        if imgsz <= 0 or imgsz % 32:
            raise ValueError("imgsz 必须是 32 的正整数倍")
        return cls(
            job_id=job_id,
            model_key=model_key,
            checkpoint=Path(_required_text(value, "checkpoint")),
            images=images,
            output_dir=Path(_required_text(value, "output_dir")),
            class_ids=tuple(str(item) for item in value.get("class_ids", [])),
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            device=value.get("device", 0),
            legacy_yolov5_repo=_optional_path(value.get("legacy_yolov5_repo")),
            python_executable=_optional_path(value.get("python_executable")),
            cancel_file=_optional_path(value.get("cancel_file")),
        )


def load_manifest(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("worker manifest 顶层必须是 JSON 对象")
    return value


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = str(value.get(key, "")).strip()
    if not result:
        raise ValueError(f"{key} 不能为空")
    return result


def _optional_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(str(value))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("图片宽高必须大于 0")
    return result
