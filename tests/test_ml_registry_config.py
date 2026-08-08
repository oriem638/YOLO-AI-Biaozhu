from __future__ import annotations

from ai_biaozhu.ml.config import (
    AugmentationOptions,
    TrainingOptions,
    build_albumentations,
    build_ultralytics_train_kwargs,
    normalize_batch,
    reduced_oom_batch,
)
from ai_biaozhu.ml.model_registry import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    ModelBackend,
    get_model,
    iter_models,
)


def test_registry_has_exact_eight_choices_and_traditional_v5() -> None:
    assert list(MODEL_REGISTRY) == [
        "YOLOv5n",
        "YOLOv5s",
        "YOLOv8n",
        "YOLOv8s",
        "YOLO11n",
        "YOLO11s",
        "YOLO26n",
        "YOLO26s",
    ]
    assert len(tuple(iter_models())) == 8
    assert DEFAULT_MODEL_KEY == "YOLO26n"
    assert get_model("yolov5n.pt").backend is ModelBackend.LEGACY_YOLOV5
    assert get_model("YOLOv5s").weight == "yolov5s.pt"
    assert get_model("yolo26n.pt").backend is ModelBackend.ULTRALYTICS


def test_training_mapping_is_explicit_and_deterministic(tmp_path) -> None:
    options = TrainingOptions()
    kwargs = build_ultralytics_train_kwargs(
        options,
        data_yaml=tmp_path / "data.yaml",
        project_dir=tmp_path / "runs",
        run_name="run-1",
        augmentation=AugmentationOptions(enabled=True, fliplr=0.25),
    )
    assert kwargs["imgsz"] == 640
    assert kwargs["epochs"] == 100
    assert kwargs["patience"] == 20
    assert kwargs["batch"] == -1
    assert kwargs["device"] == 0
    assert kwargs["workers"] == 0
    assert kwargs["seed"] == 42
    assert kwargs["deterministic"] is True
    assert kwargs["fliplr"] == 0.25

    reserved_name = build_ultralytics_train_kwargs(
        options,
        data_yaml=tmp_path / "data.yaml",
        project_dir=tmp_path / "runs",
        run_name="model",
    )
    assert reserved_name["name"] == "./model"


def test_batch_validation_and_oom_reduction() -> None:
    assert normalize_batch("auto") == -1
    assert normalize_batch(16) == 16
    assert normalize_batch(0.6) == 0.6
    assert reduced_oom_batch("auto") == 1
    assert reduced_oom_batch(16) == 8


def test_custom_augmentations_use_injected_module() -> None:
    class FakeAlbumentations:
        @staticmethod
        def Rotate(**kwargs):
            return ("rotate", kwargs)

        @staticmethod
        def Blur(**kwargs):
            return ("blur", kwargs)

    transforms = build_albumentations(
        AugmentationOptions(
            rotation_degrees=15,
            rotation_probability=0.4,
            blur_kernel=5,
            blur_probability=0.2,
        ),
        albumentations_module=FakeAlbumentations,
    )
    assert transforms == [
        ("rotate", {"limit": 15, "p": 0.4}),
        ("blur", {"blur_limit": 5, "p": 0.2}),
    ]
