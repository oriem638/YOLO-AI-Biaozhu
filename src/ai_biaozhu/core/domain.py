"""Validated, Qt-free value objects for annotation and training data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

PROJECT_FORMAT = "ai-biaozhu-project"
PROJECT_SCHEMA_VERSION = 5


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    DRAFT = "draft"
    VERIFIED = "verified"


class AnnotationOrigin(StrEnum):
    NONE = "none"
    MANUAL = "manual"
    AI = "ai"
    MIXED = "mixed"


class AIStatus(StrEnum):
    NONE = "none"
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


class RunKind(StrEnum):
    TRAIN = "train"
    PREDICT = "predict"
    DEPLOY = "deploy"


class RunStatus(StrEnum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    SNAPSHOTTING = "snapshotting"
    TRAINING = "training"
    EVALUATING = "evaluating"
    INFERENCING = "inferencing"
    IMPORTING = "importing"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelKey(StrEnum):
    YOLOV5N = "YOLOv5n"
    YOLOV5S = "YOLOv5s"
    YOLOV8N = "YOLOv8n"
    YOLOV8S = "YOLOv8s"
    YOLO11N = "YOLO11n"
    YOLO11S = "YOLO11s"
    YOLO26N = "YOLO26n"
    YOLO26S = "YOLO26s"


MODEL_WEIGHTS: Mapping[ModelKey, str] = MappingProxyType(
    {
        ModelKey.YOLOV5N: "yolov5n.pt",
        ModelKey.YOLOV5S: "yolov5s.pt",
        ModelKey.YOLOV8N: "yolov8n.pt",
        ModelKey.YOLOV8S: "yolov8s.pt",
        ModelKey.YOLO11N: "yolo11n.pt",
        ModelKey.YOLO11S: "yolo11s.pt",
        ModelKey.YOLO26N: "yolo26n.pt",
        ModelKey.YOLO26S: "yolo26s.pt",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    project_id: str
    name: str
    created_at: str
    updated_at: str
    format: str = PROJECT_FORMAT
    schema_version: int = PROJECT_SCHEMA_VERSION
    database: str = "annotations.db"
    images_dir: str = "images"
    runs_dir: str = "runs"
    exports_dir: str = "exports"
    deployments_dir: str = "deployments"
    thumbnails_dir: str = "thumbnails"

    def __post_init__(self) -> None:
        if self.format != PROJECT_FORMAT:
            raise ValueError(f"不支持的项目格式：{self.format}")
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"不支持的项目版本：{self.schema_version}")
        if not self.project_id.strip():
            raise ValueError("project_id 不能为空")
        if not self.name.strip():
            raise ValueError("项目名称不能为空")
        for attr in (
            "database",
            "images_dir",
            "runs_dir",
            "exports_dir",
            "deployments_dir",
            "thumbnails_dir",
        ):
            _validate_project_relative_path(getattr(self, attr), attr)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProjectConfig:
        required = {"project_id", "name", "created_at", "updated_at"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"project.json 缺少字段：{', '.join(missing)}")
        stored_schema_version = int(value.get("schema_version", PROJECT_SCHEMA_VERSION))
        if not 1 <= stored_schema_version <= PROJECT_SCHEMA_VERSION:
            raise ValueError(f"不支持的项目版本：{stored_schema_version}")
        return cls(
            project_id=str(value["project_id"]),
            name=str(value["name"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            format=str(value.get("format", PROJECT_FORMAT)),
            # Older supported descriptors are normalized in memory; project.open
            # rewrites project.json after the SQLite migration succeeds.
            schema_version=PROJECT_SCHEMA_VERSION,
            database=str(value.get("database", "annotations.db")),
            images_dir=str(value.get("images_dir", "images")),
            runs_dir=str(value.get("runs_dir", "runs")),
            exports_dir=str(value.get("exports_dir", "exports")),
            deployments_dir=str(value.get("deployments_dir", "deployments")),
            thumbnails_dir=str(value.get("thumbnails_dir", "thumbnails")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Category:
    id: str
    name: str
    color: str
    position: int
    enabled: bool
    created_at: str
    updated_at: str
    display_name: str | None = None

    @property
    def effective_display_name(self) -> str:
        """Return the UI-only label without changing the dataset class."""

        return self.display_name or self.name


ClassRecord = Category


@dataclass(frozen=True, slots=True)
class ImageRecord:
    id: str
    relative_path: str
    original_name: str
    source_path: str | None
    sha256: str
    width: int
    height: int
    review_status: ReviewStatus
    origin: AnnotationOrigin
    ai_status: AIStatus
    revision: int
    imported_at: str
    updated_at: str
    training_selected: bool = True


@dataclass(frozen=True, slots=True, init=False)
class BoxInput:
    class_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    id: str | None
    origin: AnnotationOrigin
    confidence: float | None
    model_run_id: str | None
    prediction_id: str | None

    def __init__(
        self,
        class_id: str,
        x1: float | None = None,
        y1: float | None = None,
        x2: float | None = None,
        y2: float | None = None,
        *,
        id: str | None = None,
        origin: AnnotationOrigin | str = AnnotationOrigin.MANUAL,
        confidence: float | None = None,
        model_run_id: str | None = None,
        prediction_id: str | None = None,
        xmin: float | None = None,
        ymin: float | None = None,
        xmax: float | None = None,
        ymax: float | None = None,
    ) -> None:
        coords = (
            x1 if x1 is not None else xmin,
            y1 if y1 is not None else ymin,
            x2 if x2 is not None else xmax,
            y2 if y2 is not None else ymax,
        )
        if any(value is None for value in coords):
            raise ValueError("标注框必须提供 x1/y1/x2/y2")
        values = tuple(float(value) for value in coords if value is not None)
        _validate_finite_box(*values)
        if confidence is not None:
            confidence = float(confidence)
            if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence 必须位于 0 到 1")
        object.__setattr__(self, "class_id", str(class_id))
        object.__setattr__(self, "x1", values[0])
        object.__setattr__(self, "y1", values[1])
        object.__setattr__(self, "x2", values[2])
        object.__setattr__(self, "y2", values[3])
        object.__setattr__(self, "id", None if id is None else str(id))
        object.__setattr__(self, "origin", AnnotationOrigin(origin))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self, "model_run_id", None if model_run_id is None else str(model_run_id)
        )
        object.__setattr__(
            self, "prediction_id", None if prediction_id is None else str(prediction_id)
        )

    @property
    def xmin(self) -> float:
        return self.x1

    @property
    def ymin(self) -> float:
        return self.y1

    @property
    def xmax(self) -> float:
        return self.x2

    @property
    def ymax(self) -> float:
        return self.y2

    @classmethod
    def from_value(cls, value: BoxInput | BoundingBox | Mapping[str, Any]) -> BoxInput:
        if isinstance(value, cls):
            return value
        if isinstance(value, BoundingBox):
            return cls(
                id=value.id,
                class_id=value.class_id,
                x1=value.x1,
                y1=value.y1,
                x2=value.x2,
                y2=value.y2,
                origin=value.origin,
                confidence=value.confidence,
                model_run_id=value.model_run_id,
                prediction_id=value.prediction_id,
            )
        return cls(
            id=_optional_string(value.get("id")),
            class_id=str(value["class_id"]),
            x1=_coordinate(value, "x1", "xmin"),
            y1=_coordinate(value, "y1", "ymin"),
            x2=_coordinate(value, "x2", "xmax"),
            y2=_coordinate(value, "y2", "ymax"),
            origin=AnnotationOrigin(value.get("origin", AnnotationOrigin.MANUAL)),
            confidence=_optional_float(value.get("confidence")),
            model_run_id=_optional_string(value.get("model_run_id")),
            prediction_id=_optional_string(value.get("prediction_id")),
        )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    id: str
    image_id: str
    class_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    origin: AnnotationOrigin
    confidence: float | None
    model_run_id: str | None
    prediction_id: str | None
    created_at: str
    updated_at: str

    @property
    def xmin(self) -> float:
        return self.x1

    @property
    def ymin(self) -> float:
        return self.y1

    @property
    def xmax(self) -> float:
        return self.x2

    @property
    def ymax(self) -> float:
        return self.y2


@dataclass(frozen=True, slots=True)
class AIPrediction:
    image_id: str
    class_id: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    prediction_id: str | None = None
    class_index: int | None = None
    expected_revision: int | None = None

    @property
    def xmin(self) -> float:
        return self.x1

    @property
    def ymin(self) -> float:
        return self.y1

    @property
    def xmax(self) -> float:
        return self.x2

    @property
    def ymax(self) -> float:
        return self.y2

    @classmethod
    def from_value(cls, value: AIPrediction | Mapping[str, Any]) -> AIPrediction:
        if isinstance(value, cls):
            return value
        return cls(
            image_id=str(value["image_id"]),
            class_id=_optional_string(value.get("class_id")),
            class_index=(
                int(value["class_index"])
                if value.get("class_index") is not None
                else None
            ),
            x1=_coordinate(value, "x1", "xmin"),
            y1=_coordinate(value, "y1", "ymin"),
            x2=_coordinate(value, "x2", "xmax"),
            y2=_coordinate(value, "y2", "ymax"),
            confidence=_optional_float(value.get("confidence")),
            prediction_id=_optional_string(value.get("prediction_id")),
            expected_revision=(
                int(value["expected_revision"])
                if value.get("expected_revision") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_key: ModelKey | str = ModelKey.YOLO26N
    imgsz: int = 640
    epochs: int = 100
    patience: int = 20
    batch: int | float | str = "auto"
    device: int | str = 0
    workers: int = 0
    seed: int = 42
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", ModelKey(self.model_key))
        integer_fields = {
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "patience": self.patience,
            "workers": self.workers,
            "seed": self.seed,
        }
        invalid_integer = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_integer:
            raise ValueError(f"以下参数必须为整数：{', '.join(invalid_integer)}")
        if not 160 <= self.imgsz <= 2048 or self.imgsz % 32:
            raise ValueError("imgsz 必须是 160 到 2048 之间的 32 倍数")
        if not 1 <= self.epochs <= 5000:
            raise ValueError("epochs 必须位于 1 到 5000")
        if not 0 <= self.patience <= self.epochs:
            raise ValueError("patience 必须位于 0 到 epochs")
        if isinstance(self.batch, str):
            if self.batch != "auto":
                raise ValueError("batch 字符串值只能为 auto")
        elif isinstance(self.batch, bool) or self.batch <= 0:
            raise ValueError("batch 必须为 auto 或正数")
        if isinstance(self.device, bool) or not isinstance(self.device, int | str):
            raise ValueError("device 必须是设备编号或字符串")
        if isinstance(self.device, str) and not self.device.strip():
            raise ValueError("device 不能为空")
        if not 0 <= self.workers <= 32:
            raise ValueError("workers 必须位于 0 到 32")
        known = {
            "model_key",
            "imgsz",
            "epochs",
            "patience",
            "batch",
            "device",
            "workers",
            "seed",
            "extra",
        }
        overlap = known.intersection(self.extra)
        if overlap:
            raise ValueError(f"extra 不能覆盖标准参数：{', '.join(sorted(overlap))}")
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    @property
    def weight_name(self) -> str:
        return MODEL_WEIGHTS[self.model_key]

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model_key": self.model_key.value,
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch": self.batch,
            "device": self.device,
            "workers": self.workers,
            "seed": self.seed,
        }
        result.update(self.extra)
        return result


@dataclass(frozen=True, slots=True, init=False)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def __init__(
        self,
        seed: int = 42,
        val_ratio: float = 0.2,
        train_ratio: float | None = None,
        test_ratio: float = 0.0,
    ) -> None:
        """Create a split.

        ``val_ratio`` intentionally remains the second argument for compatibility
        with the original ``SplitConfig(seed, val_ratio)`` API. If train_ratio is
        omitted it is inferred from the validation and test ratios.
        """

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed 必须为整数")
        val_ratio = float(val_ratio)
        test_ratio = float(test_ratio)
        if train_ratio is None:
            train_ratio = 1.0 - val_ratio - test_ratio
        train_ratio = float(train_ratio)
        if train_ratio <= 0 or val_ratio <= 0 or test_ratio < 0:
            raise ValueError("train/val 必须大于 0，test 可以为 0")
        if not isfinite(train_ratio + val_ratio + test_ratio):
            raise ValueError("数据划分比例必须为有限数值")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError("train/val/test 比例之和必须为 1")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "train_ratio", train_ratio)
        object.__setattr__(self, "val_ratio", val_ratio)
        object.__setattr__(self, "test_ratio", test_ratio)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    enabled: bool = True
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, **dict(self.settings)}


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    kind: RunKind
    model_key: ModelKey
    status: RunStatus
    parameters: Mapping[str, Any]
    snapshot_path: str | None
    metrics_jsonl_path: str | None
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    checkpoint_path: str | None
    progress: float
    error: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class DeploymentPackage:
    id: str
    run_id: str
    target: str
    checkpoint_role: str
    npu_mode: str
    status: str
    model_package_path: str | None
    app_package_path: str | None
    report_path: str | None
    zip_bytes: int | None
    payload_bytes: int | None
    warnings: tuple[str, ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    minimum: int
    verified_count: int
    positive_image_count: int
    negative_image_count: int
    instance_count: int
    class_instance_counts: Mapping[str, int]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ImportFailure:
    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class ImportReport:
    requested: int
    imported: tuple[ImageRecord, ...]
    duplicate_paths: tuple[Path, ...]
    failures: tuple[ImportFailure, ...]
    report_path: Path | None = None

    @property
    def imported_count(self) -> int:
        return len(self.imported)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_paths)

    @property
    def failed_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True, slots=True)
class AIPredictionImportResult:
    run_id: str
    image_id: str
    imported_count: int
    skipped_verified: bool
    revision_conflict: bool


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    root: Path
    data_yaml: Path
    manifest_path: Path
    train_count: int
    val_count: int
    test_count: int
    class_count: int
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class ExportResult:
    root: Path
    data_yaml: Path
    manifest_path: Path
    image_count: int
    box_count: int
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_index: int
    center_x: float
    center_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.class_index < 0:
            raise ValueError("YOLO 类别索引不能小于 0")
        values = (self.center_x, self.center_y, self.width, self.height)
        if not all(isfinite(value) for value in values):
            raise ValueError("YOLO 坐标必须为有限数值")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("YOLO 坐标必须位于 0 到 1")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("YOLO 标注框宽高必须大于 0")
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        tolerance = 1e-7
        if (
            self.center_x - half_w < -tolerance
            or self.center_y - half_h < -tolerance
            or self.center_x + half_w > 1.0 + tolerance
            or self.center_y + half_h > 1.0 + tolerance
        ):
            raise ValueError("YOLO 标注框超出图像边界")

    def to_pixels(self, image_width: int, image_height: int, class_id: str) -> BoxInput:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("图像尺寸必须大于 0")
        half_w = self.width * image_width / 2.0
        half_h = self.height * image_height / 2.0
        center_x = self.center_x * image_width
        center_y = self.center_y * image_height
        return BoxInput(
            class_id=class_id,
            x1=max(0.0, center_x - half_w),
            y1=max(0.0, center_y - half_h),
            x2=min(float(image_width), center_x + half_w),
            y2=min(float(image_height), center_y + half_h),
        )


def jsonable_config(
    value: TrainingConfig | SplitConfig | AugmentationConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, TrainingConfig | SplitConfig | AugmentationConfig):
        return value.to_dict()
    return dict(value)


def validate_box_bounds(box: BoxInput, image_width: int, image_height: int) -> None:
    """Validate a pixel-coordinate box against its original image."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸必须大于 0")
    if box.x1 < 0 or box.y1 < 0:
        raise ValueError("标注框坐标不能小于 0")
    if box.x2 > image_width or box.y2 > image_height:
        raise ValueError("标注框不能超出原图边界")


def _validate_finite_box(x1: float, y1: float, x2: float, y2: float) -> None:
    if not all(isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("标注框坐标必须为有限数值")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("标注框必须满足 x1 < x2 且 y1 < y2")


def _validate_project_relative_path(value: str, field_name: str) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) == "."
        or ":" in path.parts[0]
    ):
        raise ValueError(f"{field_name} 必须是安全的项目相对路径")


def _coordinate(value: Mapping[str, Any], preferred: str, legacy: str) -> float:
    if preferred in value:
        return float(value[preferred])
    if legacy in value:
        return float(value[legacy])
    raise KeyError(preferred)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
