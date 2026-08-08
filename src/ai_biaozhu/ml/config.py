"""Validation and backend mapping for training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_registry import DEFAULT_MODEL_KEY, get_model


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    model_key: str = DEFAULT_MODEL_KEY
    imgsz: int = 640
    epochs: int = 100
    patience: int = 20
    batch: int | float | str = "auto"
    device: int | str = 0
    workers: int = 0
    seed: int = 42
    deterministic: bool = True
    resume: str | Path | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: TrainingOptions | Mapping[str, Any] | Any | None,
    ) -> TrainingOptions:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw = value.to_dict() if hasattr(value, "to_dict") else dict(value)
        known = {
            "model_key",
            "imgsz",
            "epochs",
            "patience",
            "batch",
            "device",
            "workers",
            "seed",
            "deterministic",
            "resume",
        }
        extra = dict(raw.get("extra") or {})
        extra.update({key: item for key, item in raw.items() if key not in known | {"extra"}})
        return cls(
            model_key=str(raw.get("model_key", DEFAULT_MODEL_KEY)),
            imgsz=int(raw.get("imgsz", 640)),
            epochs=int(raw.get("epochs", 100)),
            patience=int(raw.get("patience", 20)),
            batch=raw.get("batch", "auto"),
            device=raw.get("device", 0),
            workers=int(raw.get("workers", 0)),
            seed=int(raw.get("seed", 42)),
            deterministic=bool(raw.get("deterministic", True)),
            resume=raw.get("resume"),
            extra=extra,
        )

    def validate(self) -> None:
        get_model(self.model_key)
        if self.imgsz <= 0 or self.imgsz % 32:
            raise ValueError("imgsz 必须是 32 的正整数倍")
        if self.epochs <= 0:
            raise ValueError("epochs 必须大于 0")
        if self.patience < 0:
            raise ValueError("patience 不能小于 0")
        normalize_batch(self.batch)
        if self.workers < 0:
            raise ValueError("workers 不能小于 0")
        if not str(self.device).strip():
            raise ValueError("device 不能为空")


@dataclass(frozen=True, slots=True)
class AugmentationOptions:
    enabled: bool = True
    rotation_degrees: float = 0.0
    rotation_probability: float = 0.0
    blur_kernel: int = 3
    blur_probability: float = 0.0
    fliplr: float = 0.0
    flipud: float = 0.0

    @classmethod
    def from_value(
        cls,
        value: AugmentationOptions | Mapping[str, Any] | Any | None,
    ) -> AugmentationOptions:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw = value.to_dict() if hasattr(value, "to_dict") else dict(value)
        return cls(
            enabled=bool(raw.get("enabled", True)),
            rotation_degrees=float(raw.get("rotation_degrees", raw.get("degrees", 0.0))),
            rotation_probability=float(raw.get("rotation_probability", 0.0)),
            blur_kernel=int(raw.get("blur_kernel", 3)),
            blur_probability=float(raw.get("blur_probability", 0.0)),
            fliplr=float(raw.get("fliplr", 0.0)),
            flipud=float(raw.get("flipud", 0.0)),
        )

    def validate(self) -> None:
        if not 0 <= self.rotation_degrees <= 30:
            raise ValueError("旋转角度必须在 0 到 30 度之间")
        for name, value in (
            ("rotation_probability", self.rotation_probability),
            ("blur_probability", self.blur_probability),
            ("fliplr", self.fliplr),
            ("flipud", self.flipud),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel 必须是正奇数")


def normalize_batch(value: int | float | str) -> int | float:
    """Map the product-level ``auto`` value to Ultralytics' ``-1``."""

    if isinstance(value, str):
        if value.strip().casefold() == "auto":
            return -1
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError("batch 必须为 auto、正整数或 0 到 1 的显存比例") from exc
    if isinstance(value, bool):
        raise ValueError("batch 不能是布尔值")
    if isinstance(value, int):
        if value == -1 or value > 0:
            return value
    elif isinstance(value, float) and 0 < value < 1:
        return value
    raise ValueError("batch 必须为 auto、-1、正整数或 0 到 1 的显存比例")


def build_albumentations(
    options: AugmentationOptions,
    *,
    albumentations_module: Any | None = None,
) -> list[Any]:
    """Build optional custom transforms without importing them in the GUI."""

    options.validate()
    if not options.enabled:
        return []
    if options.rotation_probability <= 0 and options.blur_probability <= 0:
        return []
    if albumentations_module is None:
        try:
            import albumentations as albumentations_module
        except ImportError as exc:
            raise RuntimeError(
                "启用了旋转/模糊增强，但 ML 环境中未安装 albumentations"
            ) from exc
    transforms: list[Any] = []
    if options.rotation_probability > 0 and options.rotation_degrees > 0:
        transforms.append(
            albumentations_module.Rotate(
                limit=options.rotation_degrees,
                p=options.rotation_probability,
            )
        )
    if options.blur_probability > 0:
        transforms.append(
            albumentations_module.Blur(
                blur_limit=options.blur_kernel,
                p=options.blur_probability,
            )
        )
    return transforms


def build_ultralytics_train_kwargs(
    options: TrainingOptions | Mapping[str, Any] | Any | None,
    *,
    data_yaml: str | Path,
    project_dir: str | Path,
    run_name: str,
    augmentation: AugmentationOptions | Mapping[str, Any] | Any | None = None,
    albumentations_module: Any | None = None,
) -> dict[str, Any]:
    """Convert application settings to explicit Ultralytics keyword arguments."""

    cfg = TrainingOptions.from_value(options)
    aug = AugmentationOptions.from_value(augmentation)
    cfg.validate()
    aug.validate()
    # Ultralytics treats the exact name "model" as a sentinel and replaces it
    # with the (possibly absolute) checkpoint path.  Prefixing the same
    # directory name with "./" preserves our intended <project>/model output.
    ultralytics_run_name = "./model" if run_name == "model" else run_name
    kwargs: dict[str, Any] = {
        "data": str(Path(data_yaml)),
        "project": str(Path(project_dir)),
        "name": ultralytics_run_name,
        "exist_ok": False,
        "imgsz": cfg.imgsz,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "batch": normalize_batch(cfg.batch),
        "device": cfg.device,
        "workers": cfg.workers,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "fliplr": aug.fliplr if aug.enabled else 0.0,
        "flipud": aug.flipud if aug.enabled else 0.0,
    }
    custom = build_albumentations(aug, albumentations_module=albumentations_module)
    if custom:
        kwargs["augmentations"] = custom
    if cfg.resume is not None:
        kwargs["resume"] = str(Path(cfg.resume))
    kwargs.update(dict(cfg.extra))
    return kwargs


def reduced_oom_batch(batch: int | float | str) -> int:
    """Return the single conservative retry value used after CUDA OOM."""

    normalized = normalize_batch(batch)
    if isinstance(normalized, float) or normalized == -1:
        return 1
    return max(1, normalized // 2)
