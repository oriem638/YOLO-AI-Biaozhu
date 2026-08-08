from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_biaozhu.workers.main as worker_main
from ai_biaozhu.deploy.maix import Cam2NpuMode, MaixTarget
from ai_biaozhu.ml.importer import AIResultImporter
from ai_biaozhu.ml.protocol import (
    JsonlEmitter,
    ProtocolEvent,
    ProtocolSequenceTracker,
    read_jsonl_events,
)
from ai_biaozhu.workers.commands import build_worker_command
from ai_biaozhu.workers.main import (
    _copy_verified_calibration_image,
    _deployment_class_names,
    _deployment_target_and_cam2_mode,
    _ignore_polars_in_pyside_feature_probe,
    _prepare_calibration_directory,
    _run_legacy_script,
    run_job,
)


def test_worker_marks_polars_as_non_pyside_before_training(monkeypatch) -> None:
    class FakeFeatureModule:
        pyside_feature_dict: dict[str, int] = {}

    feature_module = FakeFeatureModule()
    monkeypatch.setitem(sys.modules, "shibokensupport.feature", feature_module)
    _ignore_polars_in_pyside_feature_probe()
    assert feature_module.pyside_feature_dict["polars"] == -1


def test_deployment_target_ignores_legacy_cv181x_cam2_mode_for_maixcam_pro() -> None:
    target, mode = _deployment_target_and_cam2_mode(
        {"target": "maixcam_pro", "cam2_npu_mode": "cv181x"}
    )
    assert target is MaixTarget.MAIXCAM_PRO
    assert mode is Cam2NpuMode.BOTH


def test_deployment_target_validates_cam2_npu_mode() -> None:
    target, mode = _deployment_target_and_cam2_mode(
        {"target": "maixcam2", "cam2_npu_mode": "vnpu"}
    )
    assert target is MaixTarget.MAIXCAM2
    assert mode is Cam2NpuMode.VNPU


def test_deployment_class_aliases_keep_checkpoint_names_separate() -> None:
    checkpoint, deployed = _deployment_class_names(
        {
            "checkpoint_class_names": ["小钢球"],
            "deployment_class_names": ["BALL"],
        }
    )
    assert checkpoint == ("小钢球",)
    assert deployed == ("BALL",)
    assert _deployment_class_names({"class_names": ["legacy"]}) == (
        ("legacy",),
        ("legacy",),
    )


def test_run_deploy_reaches_pro_conversion_with_legacy_cv181x_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    onnx_path = tmp_path / "source.onnx"
    onnx_path.write_bytes(b"onnx")
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "0001.jpg").write_bytes(b"image")

    class NumericReport:
        def require_ok(self):
            return self

        def to_dict(self) -> dict[str, bool]:
            return {"ok": True}

    captured: dict[str, object] = {}

    def capture_request(request):
        captured["request"] = request
        raise RuntimeError("captured conversion request")

    monkeypatch.setattr(
        worker_main,
        "_export_deployment_onnx",
        lambda *_args, **_kwargs: (onnx_path, None),
    )
    monkeypatch.setattr(
        worker_main,
        "inspect_deployment_dependencies",
        lambda: SimpleNamespace(
            require_ready=lambda: SimpleNamespace(to_dict=lambda: {"ready": True})
        ),
    )
    monkeypatch.setattr(
        worker_main,
        "_prepare_calibration_directory",
        lambda *_args, **_kwargs: calibration_dir,
    )
    monkeypatch.setattr(worker_main, "load_rgb_nchw", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        worker_main,
        "inspect_onnx_numerics",
        lambda *_args, **_kwargs: NumericReport(),
    )
    monkeypatch.setattr(worker_main, "build_conversion_plan", capture_request)

    stream = io.StringIO()
    code = worker_main.run_deploy(
        {
            "job_id": "pro-deploy",
            "target": "maixcam_pro",
            "cam2_npu_mode": "cv181x",
            "model_key": "YOLO26n",
            "class_names": ["ball"],
            "input_width": 320,
            "input_height": 224,
            "calibration_count": 20,
            "output_dir": str(tmp_path / "conversion"),
            "audit_dir": str(tmp_path / "audit"),
            "execute": False,
        },
        stream=stream,
    )

    assert code == 1
    request = captured["request"]
    assert request.target is MaixTarget.MAIXCAM_PRO
    assert request.cam2_npu_mode is Cam2NpuMode.BOTH
    assert "captured conversion request" in stream.getvalue()


def test_jsonl_round_trip_and_metrics_persistence(tmp_path) -> None:
    stream = io.StringIO()
    metrics = tmp_path / "metrics.jsonl"
    emitter = JsonlEmitter("job-1", stream, metrics_path=metrics)
    first = emitter.emit("status", {"stage": "start"})
    second = emitter.emit("metrics", {"mAP50": 0.5})
    assert first.seq == 0
    assert second.seq == 1
    events = read_jsonl_events(stream.getvalue().splitlines())
    assert [event.type for event in events] == ["status", "metrics"]
    assert read_jsonl_events(metrics)[0].payload["mAP50"] == 0.5
    assert ProtocolEvent.from_json(first.to_json()) == first


def test_sequence_tracker_ignores_duplicate_with_diagnostic() -> None:
    tracker = ProtocolSequenceTracker()
    event = ProtocolEvent.create(job_id="job-1", seq=4, event_type="status")
    assert tracker.inspect(event).accepted
    duplicate = tracker.inspect(event)
    assert not duplicate.accepted
    assert duplicate.previous_seq == 4
    assert "已忽略重复" in str(duplicate.diagnostic)
    older = tracker.inspect(
        ProtocolEvent.create(job_id="job-1", seq=3, event_type="log")
    )
    assert not older.accepted
    assert "乱序" in str(older.diagnostic)
    assert tracker.inspect(
        ProtocolEvent.create(job_id="job-2", seq=0, event_type="status")
    ).accepted


def test_calibration_snapshot_rechecks_copy_and_uses_fallback(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.jpg"
    good.write_bytes(b"good-image")
    missing = tmp_path / "missing.jpg"
    import hashlib

    stream = io.StringIO()
    emitter = JsonlEmitter("deploy-1", stream)
    directory = _prepare_calibration_directory(
        {
            "calibration_count": 1,
            "calibration_images": [
                {
                    "image_id": "bad",
                    "path": str(missing),
                    "sha256": "0" * 64,
                }
            ],
            "calibration_candidate_images": [
                {
                    "image_id": "good",
                    "path": str(good),
                    "sha256": hashlib.sha256(good.read_bytes()).hexdigest(),
                }
            ],
        },
        output_dir=tmp_path / "conversion",
        emitter=emitter,
    )
    snapshots = list(directory.glob("*.jpg"))
    assert len(snapshots) == 1
    assert snapshots[0].read_bytes() == good.read_bytes()
    snapshot_manifest = json.loads(
        (directory / "calibration-snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot_manifest["used"][0]["image_id"] == "good"
    assert snapshot_manifest["used"][0]["role"] == "fallback"
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert any(
        item["type"] == "warning"
        and item["payload"].get("code") == "calibration_candidate_rejected"
        for item in events
    )
    assert any(
        item["type"] == "artifact"
        and item["payload"].get("kind") == "calibration_snapshot"
        for item in events
    )


def test_calibration_snapshot_rejects_a_corrupted_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import hashlib

    source = tmp_path / "source.jpg"
    source.write_bytes(b"stable")
    destination = tmp_path / "snapshot"
    destination.mkdir()

    def corrupt_copy(_source, target):
        Path(target).write_bytes(b"corrupted")

    monkeypatch.setattr(worker_main.shutil, "copy2", corrupt_copy)
    with pytest.raises(ValueError, match="复制期间发生变化"):
        _copy_verified_calibration_image(
            {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            directory=destination,
            output_index=1,
        )
    assert not list(destination.iterdir())


def test_worker_uses_injected_adapter_without_ml_dependencies(tmp_path) -> None:
    class FakeAdapter:
        def train(self, job, emitter):
            emitter.emit("metrics", {"epoch": 1})
            return {"checkpoint": "best.pt"}

        def predict(self, job, emitter):
            raise AssertionError("not used")

    manifest = {
        "job_id": "train-1",
        "model_key": "YOLO26n",
        "data_yaml": str(tmp_path / "data.yaml"),
        "output_dir": str(tmp_path / "runs"),
        "config": {"epochs": 1},
    }
    stream = io.StringIO()
    code = run_job(
        "train",
        manifest,
        stream=stream,
        adapter_factory=lambda _: FakeAdapter(),
    )
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert code == 0
    assert [item["type"] for item in events] == ["status", "metrics", "completed"]
    assert all(item["protocol_version"] == "1.0" for item in events)


def test_prediction_import_is_process_and_store_idempotent() -> None:
    class Sink:
        def __init__(self) -> None:
            self.calls = []

        def import_ai_predictions(
            self, run_id, image_id, predictions, *, expected_revision
        ):
            self.calls.append((run_id, image_id, predictions, expected_revision))
            return "imported"

    sink = Sink()
    importer = AIResultImporter(sink)
    event = ProtocolEvent.create(
        job_id="run-1",
        seq=4,
        event_type="prediction",
        payload={
            "image_id": "image-1",
            "expected_revision": 3,
            "predictions": [
                {
                    "class_index": 0,
                    "xmin": 1,
                    "ymin": 2,
                    "xmax": 10,
                    "ymax": 20,
                    "confidence": 0.9,
                }
            ],
        },
    )
    assert importer.import_event(event) == "imported"
    assert importer.import_event(event) is None
    assert len(sink.calls) == 1
    assert sink.calls[0][2][0]["image_id"] == "image-1"
    assert sink.calls[0][3] == 3


def test_qprocess_command_uses_unified_entrypoint(tmp_path) -> None:
    command = build_worker_command(
        Path("C:/conda/envs/yolo/python.exe"),
        "train",
        manifest=tmp_path / "job.json",
    )
    assert command.arguments[:3] == ("-m", "ai_biaozhu.workers.main", "train")
    deploy = build_worker_command(
        Path("C:/conda/envs/yolo/python.exe"),
        "deploy",
        manifest=tmp_path / "deploy.json",
    )
    assert deploy.arguments[2] == "deploy"


def test_internal_legacy_script_runner_is_allowlisted_and_restores_process_state(
    tmp_path,
    capsys,
) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    for name in ("train.py", "detect.py", "export.py"):
        (repository / name).write_text(
            "import sys\nprint('|'.join(sys.argv[1:]))\n",
            encoding="utf-8",
        )
    (repository / ".ai-biaozhu-yolov5-tag").write_text(
        "v7.0\n", encoding="utf-8"
    )
    original_argv = list(sys.argv)
    original_cwd = Path.cwd()
    assert (
        _run_legacy_script(
            repository,
            "train",
            ["--", "--resume", "last.pt"],
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "--resume|last.pt"
    assert sys.argv == original_argv
    assert Path.cwd() == original_cwd


def test_frozen_legacy_export_runner_forces_classic_torch_onnx(
    tmp_path,
    monkeypatch,
) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    for name in ("train.py", "detect.py"):
        (repository / name).write_text("# placeholder\n", encoding="utf-8")
    (repository / "export.py").write_text(
        "import torch\ntorch.onnx.export('model', 'model.onnx')\n",
        encoding="utf-8",
    )
    (repository / ".ai-biaozhu-yolov5-tag").write_text(
        "v7.0\n",
        encoding="utf-8",
    )
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))

    torch_module = SimpleNamespace(onnx=SimpleNamespace(export=original))
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert _run_legacy_script(repository, "export", []) == 0
    assert calls == [(("model", "model.onnx"), {"dynamo": False})]
    assert torch_module.onnx.export is original
