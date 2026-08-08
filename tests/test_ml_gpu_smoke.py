"""Opt-in eight-model CUDA smoke test.

Run one model:
    AI_BIAOZHU_GPU_SMOKE=YOLO26n pytest -m gpu tests/test_ml_gpu_smoke.py

Run all eight:
    AI_BIAOZHU_GPU_SMOKE=all pytest -m gpu tests/test_ml_gpu_smoke.py

Traditional YOLOv5 also requires ``AI_BIAOZHU_YOLOV5_REPO`` pointing to an
official v7.0 checkout with the tag lock accepted by the application.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ai_biaozhu.ml.adapters import resolve_adapter
from ai_biaozhu.ml.jobs import PredictionJob, TrainingJob
from ai_biaozhu.ml.model_registry import MODEL_REGISTRY
from ai_biaozhu.ml.protocol import JsonlEmitter, read_jsonl_events


def _selected_models() -> list[str]:
    value = os.environ.get("AI_BIAOZHU_GPU_SMOKE", "").strip()
    if not value:
        return list(MODEL_REGISTRY)
    if value.casefold() == "all":
        return list(MODEL_REGISTRY)
    return [item.strip() for item in value.split(",") if item.strip()]


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", _selected_models())
def test_model_load_one_epoch_predict_and_artifact(tmp_path, model_key) -> None:
    selection = os.environ.get("AI_BIAOZHU_GPU_SMOKE", "").strip()
    if not selection:
        pytest.skip("set AI_BIAOZHU_GPU_SMOKE to a model key or 'all'")
    if model_key not in MODEL_REGISTRY:
        pytest.fail(f"unknown AI_BIAOZHU_GPU_SMOKE model: {model_key}")
    repository = os.environ.get("AI_BIAOZHU_YOLOV5_REPO")
    if MODEL_REGISTRY[model_key].is_legacy and not repository:
        pytest.skip("traditional YOLOv5 needs AI_BIAOZHU_YOLOV5_REPO")
    data_yaml, sample = _tiny_dataset(tmp_path)
    common = {
        "job_id": f"smoke-{model_key}",
        "model_key": model_key,
        "data_yaml": str(data_yaml),
        "output_dir": str(tmp_path / "runs"),
        # Production uses this directory name; Ultralytics itself reserves the
        # exact string "model", so the adapter must neutralize that sentinel.
        "run_name": "model",
        "legacy_yolov5_repo": repository,
        "config": {
            "epochs": 1,
            "imgsz": 64,
            "batch": 1,
            "device": 0,
            "workers": 0,
            "patience": 0,
        },
        "augmentation": {"enabled": False},
    }
    adapter = resolve_adapter(model_key)
    train_job = TrainingJob.from_mapping(common)
    train_stream = io.StringIO()
    try:
        result = adapter.train(
            train_job,
            JsonlEmitter(train_job.job_id, train_stream),
        )
    finally:
        print("worker_train_events_begin")
        print(train_stream.getvalue())
        print("worker_train_events_end")
    checkpoint = result["artifacts"].get("best") or result["artifacts"].get("last")
    assert checkpoint and Path(checkpoint).is_file()
    prediction_job = PredictionJob.from_mapping(
        {
            "job_id": f"predict-{model_key}",
            "model_key": model_key,
            "checkpoint": checkpoint,
            "output_dir": str(tmp_path / "predictions"),
            "legacy_yolov5_repo": repository,
            "class_ids": ["object"],
            "images": [
                {
                    "image_id": "sample",
                    "path": str(sample),
                    "expected_revision": 0,
                    "width": 64,
                    "height": 64,
                }
            ],
            "imgsz": 64,
            "device": 0,
        }
    )
    prediction_stream = io.StringIO()
    try:
        prediction_result = adapter.predict(
            prediction_job,
            JsonlEmitter(prediction_job.job_id, prediction_stream),
        )
    finally:
        print("worker_prediction_events_begin")
        print(prediction_stream.getvalue())
        print("worker_prediction_events_end")
    assert prediction_result["completed_images"] == 1
    assert any(
        event.type == "prediction"
        for event in read_jsonl_events(prediction_stream.getvalue().splitlines())
    )


@pytest.mark.gpu
def test_modern_rotation_and_blur_are_applied_and_persisted(tmp_path) -> None:
    model_key = os.environ.get("AI_BIAOZHU_MODERN_AUG_SMOKE", "").strip()
    if not model_key:
        pytest.skip("set AI_BIAOZHU_MODERN_AUG_SMOKE to a modern model key")
    if model_key not in MODEL_REGISTRY:
        pytest.fail(f"unknown AI_BIAOZHU_MODERN_AUG_SMOKE model: {model_key}")
    if MODEL_REGISTRY[model_key].is_legacy:
        pytest.fail("AI_BIAOZHU_MODERN_AUG_SMOKE must select a modern model")

    data_yaml, _ = _tiny_dataset(tmp_path)
    job = TrainingJob.from_mapping(
        {
            "job_id": f"augmentation-{model_key}",
            "model_key": model_key,
            "data_yaml": str(data_yaml),
            "output_dir": str(tmp_path / "runs"),
            "config": {
                "epochs": 1,
                "imgsz": 64,
                "batch": 1,
                "device": 0,
                "workers": 0,
                "patience": 0,
            },
            "augmentation": {
                "enabled": True,
                "rotation_degrees": 17,
                "rotation_probability": 0.73,
                "blur_kernel": 5,
                "blur_probability": 0.41,
                "fliplr": 0.37,
                "flipud": 0.19,
            },
        }
    )
    stream = io.StringIO()
    result = resolve_adapter(model_key).train(
        job,
        JsonlEmitter(job.job_id, stream),
    )
    checkpoint = result["artifacts"].get("best") or result["artifacts"].get("last")
    assert checkpoint and Path(checkpoint).is_file()

    args_path = Path(result["save_dir"]) / "args.yaml"
    args_text = args_path.read_text(encoding="utf-8")
    assert "augmentations:" in args_text
    assert "Rotate" in args_text
    assert "Blur" in args_text
    assert "fliplr: 0.37" in args_text
    assert "flipud: 0.19" in args_text
    events = read_jsonl_events(stream.getvalue().splitlines())
    assert any(event.type == "metrics" for event in events)


def _tiny_dataset(root: Path) -> tuple[Path, Path]:
    dataset = root / "tiny"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        for index in range(2):
            image_path = dataset / "images" / split / f"{index}.jpg"
            image = Image.new("RGB", (64, 64), "black")
            ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="white")
            image.save(image_path)
            (dataset / "labels" / split / f"{index}.txt").write_text(
                "0 0.5 0.5 0.5 0.5\n",
                encoding="utf-8",
            )
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [object]\n",
        encoding="utf-8",
    )
    return data_yaml, dataset / "images" / "val" / "0.jpg"
