from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ai_biaozhu.ml.adapters import (
    AdapterError,
    UltralyticsAdapter,
    _resolve_runtime_device,
    _ultralytics_amp_check_directory,
    _ultralytics_augmentation_config,
    _uses_cuda_amp,
)
from ai_biaozhu.ml.jobs import PredictionJob, TrainingJob
from ai_biaozhu.ml.protocol import JsonlEmitter, read_jsonl_events
from ai_biaozhu.ml.weights import WeightUnavailableError


class TensorList:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class MultiValueTensor(TensorList):
    def item(self):
        raise RuntimeError("a Tensor with 3 elements cannot be converted to Scalar")


def test_ultralytics_prediction_streams_empty_and_nonempty_results(tmp_path) -> None:
    class FakeModel:
        def predict(self, **kwargs):
            if Path(kwargs["source"]).name == "empty.jpg":
                return [SimpleNamespace(boxes=None)]
            boxes = SimpleNamespace(
                xyxy=TensorList([[1, 2, 11, 22]]),
                cls=TensorList([0]),
                conf=TensorList([0.88]),
            )
            return [SimpleNamespace(boxes=boxes)]

    job = PredictionJob.from_mapping(
        {
            "job_id": "predict-1",
            "model_key": "YOLO11n",
            "checkpoint": str(tmp_path / "best.pt"),
            "output_dir": str(tmp_path / "predictions"),
            "class_ids": ["class-a"],
            "images": [
                {
                    "image_id": "a",
                    "path": str(tmp_path / "a.jpg"),
                    "expected_revision": 1,
                },
                {
                    "image_id": "b",
                    "path": str(tmp_path / "empty.jpg"),
                    "expected_revision": 2,
                },
            ],
        }
    )
    stream = io.StringIO()
    result = UltralyticsAdapter(lambda _: FakeModel()).predict(
        job, JsonlEmitter(job.job_id, stream)
    )
    events = read_jsonl_events(stream.getvalue().splitlines())
    predictions = [event for event in events if event.type == "prediction"]
    running = [
        event
        for event in events
        if event.type == "status" and event.payload.get("stage") == "image_running"
    ]
    assert result == {"completed_images": 2, "failed_images": 0}
    assert [event.payload["image_id"] for event in running] == ["a", "b"]
    assert [event.payload["expected_revision"] for event in running] == [1, 2]
    assert predictions[0].payload["predictions"][0]["class_id"] == "class-a"
    assert predictions[1].payload["predictions"] == []


def test_ultralytics_training_retries_cuda_oom_once(tmp_path) -> None:
    calls = []

    class Result:
        def __init__(self, save_dir):
            self.save_dir = save_dir
            self.results_dict = {"metrics/mAP50(B)": 0.75}

    class FakeModel:
        def __init__(self, fail):
            self.fail = fail
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self, **kwargs):
            calls.append(kwargs)
            if self.fail:
                raise RuntimeError("CUDA out of memory")
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"best")
            (save_dir / "results.png").write_bytes(b"plot")
            (save_dir / "confusion_matrix_normalized.png").write_bytes(b"matrix")
            return Result(save_dir)

    models = iter((FakeModel(True), FakeModel(False)))
    job = TrainingJob.from_mapping(
        {
            "job_id": "train-oom",
            "model_key": "YOLOv8n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "config": {"epochs": 1, "batch": 16},
            "checkpoint_source": str(tmp_path / "pretrained.pt"),
        }
    )
    stream = io.StringIO()
    result = UltralyticsAdapter(lambda _: next(models)).train(
        job, JsonlEmitter(job.job_id, stream)
    )
    assert len(calls) == 2
    assert calls[1]["batch"] == 8
    assert calls[1]["name"].endswith("_oom_retry")
    assert result["artifacts"]["best"].endswith("best.pt")
    events = read_jsonl_events(stream.getvalue().splitlines())
    assert any(
        event.type == "warning"
        and event.payload["code"] == "cuda_oom_retry"
        for event in events
    )
    assert {
        event.payload["name"]
        for event in events
        if event.type == "artifact"
        and event.payload["kind"] == "training_visual"
    } == {
        "results.png",
        "confusion_matrix_normalized.png",
    }


def test_ultralytics_second_cuda_oom_has_actionable_guidance(tmp_path) -> None:
    class FakeModel:
        def add_callback(self, name, callback):
            del name, callback

        def train(self, **kwargs):
            del kwargs
            raise RuntimeError("CUDA out of memory")

    job = TrainingJob.from_mapping(
        {
            "job_id": "train-oom-twice",
            "model_key": "YOLOv8s",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "config": {"epochs": 1, "batch": 4, "imgsz": 640},
            "checkpoint_source": str(tmp_path / "pretrained.pt"),
        }
    )
    with pytest.raises(
        AdapterError,
        match="n 型模型.*降低 imgsz.*更小的 batch",
    ):
        UltralyticsAdapter(lambda _: FakeModel()).train(
            job,
            JsonlEmitter(job.job_id, io.StringIO()),
        )


def test_runtime_auto_device_uses_cuda_when_available_and_cpu_otherwise(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_runtime_device("auto") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_runtime_device("auto") == 0
    assert _resolve_runtime_device("cpu") == "cpu"
    assert _resolve_runtime_device(1) == 1


def test_ultralytics_resume_loads_last_checkpoint_and_passes_resume_true(
    tmp_path,
) -> None:
    loaded = []
    loaded_args = []
    calls = []
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {"model": SimpleNamespace(args={"project": "old", "name": "old"})},
        checkpoint,
    )
    save_dir = tmp_path / "runs" / "resumed"

    class Result:
        results_dict = {}

        def __init__(self):
            self.save_dir = save_dir

    class FakeModel:
        def add_callback(self, name, callback):
            pass

        def train(self, **kwargs):
            calls.append(kwargs)
            (save_dir / "weights").mkdir(parents=True)
            (save_dir / "weights" / "last.pt").write_bytes(b"last")
            return Result()

    def factory(value):
        loaded.append(value)
        copied = torch.load(value, map_location="cpu", weights_only=False)
        loaded_args.append(dict(copied["model"].args))
        return FakeModel()

    job = TrainingJob.from_mapping(
        {
            "job_id": "resume-modern",
            "model_key": "YOLO11n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "config": {"epochs": 1, "resume": str(checkpoint)},
        }
    )
    UltralyticsAdapter(factory).train(
        job,
        JsonlEmitter(job.job_id, io.StringIO()),
    )
    isolated = tmp_path / "runs" / ".resume-modern-resume" / "last.pt"
    assert loaded == [str(isolated)]
    assert calls[0]["resume"] == str(isolated)
    assert loaded_args[0]["project"] == str(tmp_path / "runs")
    assert loaded_args[0]["name"] == "resume-modern"
    assert not isolated.exists()


def test_ultralytics_batch_progress_is_throttled() -> None:
    class FakeModel:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

    model = FakeModel()
    stream = io.StringIO()
    adapter = UltralyticsAdapter()
    adapter._attach_training_callbacks(
        model,
        JsonlEmitter("batch-job", stream),
        SimpleNamespace(raise_if_cancelled=lambda: None),
    )
    trainer = SimpleNamespace(
        train_loader=list(range(100)),
        batch_i=0,
        epoch=0,
        epochs=2,
        tloss=MultiValueTensor([1.0, 2.0, 3.0]),
    )
    for batch in range(20):
        trainer.batch_i = batch
        model.callbacks["on_train_batch_end"](trainer)
    progress = [
        event
        for event in read_jsonl_events(stream.getvalue().splitlines())
        if event.type == "progress"
    ]
    assert [event.payload["current"] for event in progress] == [1, 11]
    assert all("eta_seconds" in event.payload for event in progress)
    assert all("gpu_utilization" in event.payload for event in progress)
    assert all("gpu_memory_gb" in event.payload for event in progress)
    assert progress[0].payload["loss"] == [1.0, 2.0, 3.0]


def test_ultralytics_training_without_checkpoint_artifact_fails(tmp_path) -> None:
    class Result:
        save_dir = tmp_path / "runs" / "missing"
        results_dict = {}

    class Model:
        def add_callback(self, name, callback):
            pass

        def train(self, **kwargs):
            Result.save_dir.mkdir(parents=True)
            return Result()

    job = TrainingJob.from_mapping(
        {
            "job_id": "missing-artifact",
            "model_key": "YOLOv8n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "checkpoint_source": str(tmp_path / "source.pt"),
            "config": {"epochs": 1},
        }
    )
    with pytest.raises(AdapterError, match="best.pt/last.pt"):
        UltralyticsAdapter(lambda _: Model()).train(
            job,
            JsonlEmitter(job.job_id, io.StringIO()),
        )


def test_ultralytics_augmentation_config_is_temporary() -> None:
    import ultralytics.cfg as ultralytics_cfg

    assert "augmentations" not in ultralytics_cfg.DEFAULT_CFG_DICT
    assert not hasattr(ultralytics_cfg.DEFAULT_CFG, "augmentations")
    with _ultralytics_augmentation_config([object()]):
        assert ultralytics_cfg.DEFAULT_CFG_DICT["augmentations"] is None
        assert ultralytics_cfg.DEFAULT_CFG.augmentations is None
    assert "augmentations" not in ultralytics_cfg.DEFAULT_CFG_DICT
    assert not hasattr(ultralytics_cfg.DEFAULT_CFG, "augmentations")


@pytest.mark.parametrize(
    ("device", "amp", "expected"),
    [
        (0, True, True),
        ("cuda:0", True, True),
        ("cpu", True, False),
        ("mps", True, False),
        (0, False, False),
        (0, "false", False),
    ],
)
def test_cuda_amp_reference_requirement(device, amp, expected) -> None:
    assert _uses_cuda_amp({"device": device, "amp": amp}) is expected


def test_amp_check_changes_cwd_only_inside_probe(tmp_path, monkeypatch) -> None:
    import ultralytics.engine.trainer as trainer_module

    original_cwd = Path.cwd()
    observed = []

    def check_amp(model):
        observed.append((model, Path.cwd()))
        return True

    monkeypatch.setattr(trainer_module, "check_amp", check_amp)
    marker = object()
    with _ultralytics_amp_check_directory(tmp_path):
        assert Path.cwd() == original_cwd
        assert trainer_module.check_amp(marker)
        assert Path.cwd() == original_cwd
    assert trainer_module.check_amp is check_amp
    assert observed == [(marker, tmp_path)]


def test_missing_offline_amp_reference_falls_back_to_fp32(tmp_path) -> None:
    class MissingManager:
        def ensure(self, model_key, **kwargs):
            assert model_key == "YOLO26n"
            assert kwargs["offline"] is True
            raise WeightUnavailableError("not cached")

    job = TrainingJob.from_mapping(
        {
            "job_id": "offline-amp",
            "model_key": "YOLOv8n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "offline_weights": True,
            "config": {"device": 0},
        }
    )
    kwargs = {"device": 0}
    stream = io.StringIO()
    directory = UltralyticsAdapter(weight_manager=MissingManager())._prepare_amp_reference_dir(
        job,
        kwargs,
        JsonlEmitter(job.job_id, stream),
    )
    assert directory is None
    assert kwargs["amp"] is False
    events = read_jsonl_events(stream.getvalue().splitlines())
    assert any(
        event.type == "warning"
        and event.payload["code"] == "amp_reference_unavailable"
        for event in events
    )


def test_auto_amp_reference_follows_cuda_availability(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not _uses_cuda_amp({"device": "auto"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _uses_cuda_amp({"device": "auto"})
