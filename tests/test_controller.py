from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController, _device_value
from ai_biaozhu.core import BoxInput, EmptyAnnotationConfirmationRequired
from ai_biaozhu.data.utils import sha256_file
from ai_biaozhu.errors import ValidationError
from ai_biaozhu.ml.environment import EnvironmentCandidate, EnvironmentReport
from ai_biaozhu.ml.protocol import ProtocolEvent
from ai_biaozhu.settings import SettingsStore


def _paths(root: Path) -> AppPaths:
    return AppPaths(
        data=root / "app-data",
        cache=root / "cache",
        logs=root / "logs",
        models=root / "models",
        yolo_config=root / "ultralytics",
    )


def _valid_environment(_value: object) -> EnvironmentReport:
    python = Path(sys.executable).resolve()
    candidate = EnvironmentCandidate(python.parent, python, "test")
    return EnvironmentReport(
        candidate=candidate,
        valid=True,
        python_version="3.11.15",
        torch_version="2.11.0+cu128",
        torchvision_version="0.26.0+cu128",
        ultralytics_version="8.4.82",
        cuda_available=True,
        cuda_version="12.8",
        device_name="test GPU",
        errors=(),
        compatibility_errors=(),
        gpu_ready=True,
        raw={},
    )


def _controller(tmp_path: Path) -> ApplicationController:
    source_root = Path(__file__).resolve().parents[1]
    return ApplicationController(
        _paths(tmp_path),
        settings=SettingsStore(tmp_path / "settings.json"),
        environment_inspector=_valid_environment,
        source_root=source_root,
    )


def test_available_training_devices_lists_cpu_auto_and_detected_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            command,
            0,
            "0, NVIDIA GeForce RTX 5060 Laptop GPU\n1, Test GPU\n",
            "",
        )

    monkeypatch.setattr("ai_biaozhu.controller.subprocess.run", fake_run)
    devices = _controller(tmp_path).available_training_devices()
    assert [device["value"] for device in devices] == ["auto", "cpu", "0", "1"]
    assert "RTX 5060" in devices[2]["label"]
    assert _device_value("auto") == "auto"


def test_training_presets_are_persisted_and_deleted(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.save_training_preset("低显存", {"imgsz": 320, "batch": 2})
    assert controller.training_presets["低显存"] == {
        "imgsz": 320,
        "batch": 2,
    }
    controller.delete_training_preset("低显存")
    assert controller.training_presets == {}


def test_cpu_only_environment_allows_cpu_and_auto_but_rejects_explicit_cuda(
    tmp_path: Path,
) -> None:
    python = Path(sys.executable).resolve()

    def cpu_only_environment(_value: object) -> EnvironmentReport:
        return EnvironmentReport(
            candidate=EnvironmentCandidate(python.parent, python, "test"),
            valid=True,
            python_version="3.11.15",
            torch_version="2.11.0+cu128",
            torchvision_version="0.26.0+cu128",
            ultralytics_version="8.4.82",
            cuda_available=False,
            cuda_version="12.8",
            device_name=None,
            errors=(),
            compatibility_errors=(),
            gpu_ready=False,
            raw={},
        )

    controller = ApplicationController(
        _paths(tmp_path),
        settings=SettingsStore(tmp_path / "settings.json"),
        environment_inspector=cpu_only_environment,
        source_root=Path(__file__).resolve().parents[1],
    )
    assert controller._select_worker_python(python, device="cpu") == python
    assert controller._select_worker_python(python, device="auto") == python
    with pytest.raises(ValidationError, match="需要可用的 CUDA"):
        controller._select_worker_python(python, device=0)


def _seed_verified_project(controller: ApplicationController, root: Path) -> None:
    project = controller.new_project(root, "seed")
    category = project.repository.list_categories()[0]
    template = root / "template.png"
    Image.new("RGB", (32, 32), (40, 80, 120)).save(template)
    for index in range(100):
        relative = f"images/{index:03d}.png"
        destination = root / relative
        shutil.copyfile(template, destination)
        image = project.repository.add_image_record(
            image_id=f"image-{index:03d}",
            relative_path=relative,
            original_name=f"{index:03d}.png",
            source_path=None,
            sha256=f"{index + 1:064x}",
            width=32,
            height=32,
        )
        if index == 0:
            project.repository.save_and_confirm(
                image.id,
                [BoxInput(category.id, 2, 2, 20, 20)],
            )
        else:
            project.repository.confirm_image(image.id, confirm_empty=True)
    template.unlink()


def _deploy_source(
    controller: ApplicationController,
    root: Path,
) -> tuple[Any, Any, list[str]]:
    _seed_verified_project(controller, root)
    project = controller.current_project
    assert project is not None
    source = project.repository.create_run("train", "YOLO26n", run_id="source")
    checkpoint = project.runs_dir / source.id / "training" / "model" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    source = project.repository.update_run(
        source.id,
        status="completed",
        progress=1.0,
        artifacts={"best": str(checkpoint)},
        checkpoint_path=str(checkpoint),
    )
    calibration = [image.id for image in project.list_images()[:20]]
    return project, source, calibration


def _deploy_values(source: Any, calibration: list[str], **updates: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_id": source.id,
        "checkpoint_kind": "best",
        "target": "maixcam2",
        "cam2_npu_mode": "both",
        "input_width": 640,
        "input_height": 480,
        "calibration_image_ids": calibration,
        "ml_environment": sys.executable,
    }
    values.update(updates)
    return values


def test_calibration_recommendation_is_deterministic_and_class_balanced(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "calibration")
    common = project.repository.list_categories()[0]
    rare = project.repository.add_category("rare")
    labels_by_image: dict[str, set[str]] = {}
    for index in range(12):
        image = project.repository.add_image_record(
            image_id=f"image-{index:02d}",
            relative_path=f"images/{index:02d}.png",
            original_name=f"{index:02d}.png",
            source_path=None,
            sha256=f"{index + 1:064x}",
            width=32,
            height=32,
        )
        if index < 8:
            category_id = common.id
            project.repository.save_and_confirm(
                image.id,
                [BoxInput(category_id, 2, 2, 20, 20)],
            )
            labels_by_image[image.id] = {category_id}
        elif index < 11:
            category_id = rare.id
            project.repository.save_and_confirm(
                image.id,
                [BoxInput(category_id, 2, 2, 20, 20)],
            )
            labels_by_image[image.id] = {category_id}
        else:
            project.repository.confirm_image(image.id, confirm_empty=True)
            labels_by_image[image.id] = set()

    first = controller.recommend_calibration_image_ids(3, 123)
    second = controller.recommend_calibration_image_ids(3, 123)

    assert first == second
    assert len(first) == len(set(first)) == 3
    selected_labels = set().union(*(labels_by_image[image_id] for image_id in first))
    assert selected_labels == {common.id, rare.id}
    assert any(not labels_by_image[image_id] for image_id in first)


def test_empty_sample_confirmation_must_be_explicit_at_controller_boundary(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "empty")
    image = project.repository.add_image_record(
        image_id="empty-image",
        relative_path="images/empty.png",
        original_name="empty.png",
        source_path=None,
        sha256="1" * 64,
        width=32,
        height=32,
    )

    with pytest.raises(EmptyAnnotationConfirmationRequired):
        controller.verify_and_next(image.id, [])

    controller.verify_and_next(image.id, [], confirm_empty=True)
    assert project.repository.get_image(image.id).review_status.value == "verified"


def test_maixcam2_converter_requires_archive_import_not_registry_pull(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)

    with pytest.raises(ValidationError, match="tar.*导入镜像"):
        controller.pull_converter_image({"target": "maixcam2", "confirmed": True})

    launch = controller.pull_converter_image(
        {"target": "maixcam_pro", "confirmed": True}
    )
    assert launch["arguments"] == ["pull", "sophgo/tpuc_dev:latest"]


def test_controller_creates_locked_training_manifest(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _seed_verified_project(controller, tmp_path / "project")
    launch = controller.start_training(
        "YOLO26n",
        {
            "model_key": "YOLO26n",
            "imgsz": 640,
            "epochs": 2,
            "patience": 0,
            "batch": "auto",
            "device": "0",
            "workers": 0,
            "seed": 7,
            "split": {
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.1,
                "seed": 7,
            },
            "augmentation": {
                "rotation_degrees": 20,
                "rotation_probability": 0.4,
                "blur_kernel": 5,
                "blur_probability": 0.2,
                "fliplr": 0.5,
                "flipud": 0.1,
            },
            "start_from": "official",
            "ml_environment": sys.executable,
        },
    )
    project = controller.current_project
    assert project is not None
    run = project.repository.get_run(str(launch["job_id"]))
    manifest = json.loads(
        (project.runs_dir / run.id / "job.json").read_text(encoding="utf-8")
    )
    assert manifest["checkpoint_source"] is None
    assert Path(manifest["weight_lock_path"]).name == "weights.lock.json"
    assert manifest["augmentation"]["blur_kernel"] == 5
    assert manifest["config"]["patience"] == 0
    assert run.snapshot_path
    snapshot = json.loads(
        (Path(run.snapshot_path) / "manifest.json").read_text(encoding="utf-8")
    )
    assert snapshot["split"] == {
        "seed": 7,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
    }
    assert launch["arguments"][:3] == ["-m", "ai_biaozhu.workers.main", "train"]


def test_bundled_worker_skips_external_environment_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probes: list[object] = []

    def fail_if_probed(value: object) -> EnvironmentReport:
        probes.append(value)
        raise AssertionError("bundled releases must not probe an external Conda environment")

    controller = ApplicationController(
        _paths(tmp_path),
        settings=SettingsStore(tmp_path / "settings.json"),
        environment_inspector=fail_if_probed,
        source_root=Path(__file__).resolve().parents[1],
    )
    worker = tmp_path / "standalone" / "AI-Biaozhu-Worker.exe"
    worker.parent.mkdir()
    worker.write_bytes(b"MZ")
    monkeypatch.setattr(controller, "_bundled_worker_executable", lambda: worker)

    selected = controller._select_worker_python()
    manifest = tmp_path / "job.json"
    manifest.write_text("{}", encoding="utf-8")
    launch = controller._worker_launch(selected, "train", manifest, "job")

    assert probes == []
    assert Path(launch["program"]) == worker
    assert launch["arguments"] == ["train", "--manifest", str(manifest)]
    assert Path(launch["working_directory"]) == worker.parent
    assert launch["environment"]["AI_BIAOZHU_STANDALONE"] == "1"
    assert launch["environment"]["YOLO_AUTOINSTALL"] == "false"
    assert "PYTHONPATH" not in launch["environment"]
    source_environment = controller._worker_environment(bundled=False)
    assert "AI_BIAOZHU_STANDALONE" not in source_environment
    assert "YOLO_AUTOINSTALL" not in source_environment
    assert "PYTHONPATH" in source_environment


def test_prediction_events_import_ai_draft_once(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "prediction")
    category = project.repository.list_categories()[0]
    image_path = project.images_dir / "sample.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(image_path)
    image = project.repository.add_image_record(
        image_id="image-1",
        relative_path="images/sample.png",
        original_name="sample.png",
        source_path=None,
        sha256="1" * 64,
        width=64,
        height=48,
    )
    source = project.repository.create_run("train", "YOLO26n", run_id="source")
    checkpoint = project.runs_dir / source.id / "training" / "model" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    project.repository.update_run(
        source.id,
        status="completed",
        progress=1.0,
        artifacts={"best": str(checkpoint)},
        checkpoint_path=str(checkpoint),
    )
    launch = controller.start_autolabel(
        {
            "run_id": source.id,
            "checkpoint_kind": "best",
            "checkpoint": str(checkpoint),
            "confidence": 0.3,
            "iou": 0.5,
            "deduplicate": True,
            "dedup_iou": 0.8,
            "ml_environment": sys.executable,
        }
    )
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))
    assert manifest["deduplicate"] is True
    assert manifest["dedup_iou"] == pytest.approx(0.8)
    queued = project.repository.get_image(image.id)
    running = ProtocolEvent.create(
        job_id=str(launch["job_id"]),
        seq=0,
        event_type="status",
        payload={
            "stage": "image_running",
            "image_id": image.id,
            "current": 1,
            "total": 1,
        },
    )
    controller.handle_job_event(running.to_dict())
    in_progress = project.repository.get_image(image.id)
    assert in_progress.ai_status.value == "running"
    assert in_progress.revision == queued.revision

    event = ProtocolEvent.create(
        job_id=str(launch["job_id"]),
        seq=1,
        event_type="prediction",
        payload={
            "image_id": image.id,
            "expected_revision": queued.revision,
            "predictions": [
                {
                    "prediction_id": "prediction-lower-confidence",
                    "class_id": category.id,
                    "xmin": 2,
                    "ymin": 3,
                    "xmax": 20,
                    "ymax": 30,
                    "confidence": 0.4,
                },
                {
                    "prediction_id": "prediction-1",
                    "class_id": category.id,
                    "xmin": 1,
                    "ymin": 2,
                    "xmax": 20,
                    "ymax": 30,
                    "confidence": 0.9,
                }
            ],
        },
    )
    controller.handle_job_event(event.to_dict())
    controller.handle_job_event(event.to_dict())
    updated = project.repository.get_image(image.id)
    assert updated.review_status.value == "draft"
    assert updated.ai_status.value == "ready"
    boxes = project.list_boxes(image.id)
    assert len(boxes) == 1
    assert boxes[0].prediction_id == "prediction-1"

    completed = ProtocolEvent.create(
        job_id=str(launch["job_id"]),
        seq=2,
        event_type="completed",
        payload={"command": "predict", "result": {"completed_images": 1}},
    )
    controller.handle_job_event(completed.to_dict())
    assert project.repository.get_run(str(launch["job_id"])).status.value == "completed"


def test_autolabel_manifest_failure_rolls_back_queue_and_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "prediction rollback")
    Image.new("RGB", (64, 48), (10, 20, 30)).save(
        project.images_dir / "sample.png"
    )
    image = project.repository.add_image_record(
        image_id="image-1",
        relative_path="images/sample.png",
        original_name="sample.png",
        source_path=None,
        sha256="1" * 64,
        width=64,
        height=48,
    )
    source = project.repository.create_run("train", "YOLO26n", run_id="source")
    checkpoint = project.runs_dir / source.id / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    project.repository.update_run(
        source.id,
        status="completed",
        artifacts={"best": str(checkpoint)},
        checkpoint_path=str(checkpoint),
    )

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("manifest disk failure")

    monkeypatch.setattr("ai_biaozhu.controller.write_json", fail_write)
    with pytest.raises(OSError, match="manifest disk failure"):
        controller.start_autolabel(
            {
                "run_id": source.id,
                "checkpoint_kind": "best",
                "checkpoint": str(checkpoint),
                "ml_environment": sys.executable,
            }
        )

    assert project.repository.list_runs(kind="predict") == ()
    assert project.repository.get_image(image.id).ai_status.value == "none"


def test_deployment_events_persist_package_audit_and_device_validation(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    _seed_verified_project(controller, tmp_path / "project")
    project = controller.current_project
    assert project is not None
    source = project.repository.create_run("train", "YOLO26n", run_id="source")
    checkpoint = project.runs_dir / source.id / "training" / "model" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    project.repository.update_run(
        source.id,
        status="completed",
        progress=1.0,
        artifacts={"best": str(checkpoint)},
        checkpoint_path=str(checkpoint),
    )
    calibration = [image.id for image in project.list_images()[:20]]
    launch = controller.start_maix_deploy(
        {
            "run_id": source.id,
            "checkpoint_kind": "best",
            "target": "maixcam2",
            "cam2_npu_mode": "both",
            "input_width": 640,
            "input_height": 480,
            "calibration_image_ids": calibration,
            "ml_environment": sys.executable,
        }
    )
    run_id = str(launch["job_id"])
    package = project.repository.list_deployment_packages(run_id=run_id)[0]
    assert package.status == "queued"

    report_path = project.deployments_dir / run_id / "deployment-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "kind": "full-app",
                        "zip_size": 30_000_001,
                        "unpacked_size": 31_000_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    model_package = report_path.with_name("model.zip")
    app_package = report_path.with_name("app.maixapp")
    events = [
        ProtocolEvent.create(
            job_id=run_id,
            seq=0,
            event_type="status",
            payload={"stage": "exporting_onnx"},
        ),
        ProtocolEvent.create(
            job_id=run_id,
            seq=1,
            event_type="warning",
            payload={
                "code": "package_size_warning",
                "message": "超过 30 MB",
                "packages": [
                    {
                        "zip_size": 30_000_001,
                        "unpacked_size": 31_000_000,
                    }
                ],
            },
        ),
        ProtocolEvent.create(
            job_id=run_id,
            seq=2,
            event_type="artifact",
            payload={"kind": "maix_model_package", "path": str(model_package)},
        ),
        ProtocolEvent.create(
            job_id=run_id,
            seq=3,
            event_type="artifact",
            payload={"kind": "maix_app_package", "path": str(app_package)},
        ),
        ProtocolEvent.create(
            job_id=run_id,
            seq=4,
            event_type="artifact",
            payload={"kind": "deployment_report", "path": str(report_path)},
        ),
        ProtocolEvent.create(
            job_id=run_id,
            seq=5,
            event_type="completed",
            payload={
                "command": "deploy",
                "model_package_path": str(model_package),
                "app_package_path": str(app_package),
            },
        ),
    ]
    for event in events:
        controller.handle_job_event(event.to_dict())

    package = project.repository.list_deployment_packages(run_id=run_id)[0]
    assert package.status == "needs_device_validation"
    assert package.model_package_path == str(model_package)
    assert package.app_package_path == str(app_package)
    assert package.report_path == str(report_path)
    assert package.zip_bytes == 30_000_001
    assert package.payload_bytes == 31_000_000
    assert package.warnings == ("超过 30 MB",)
    assert project.repository.get_run(run_id).status.value == "completed"


def test_deployment_controller_rejects_invalid_values_before_creating_run(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )
    cases = [
        (
            {"calibration_image_ids": [calibration[0]] * 20},
            "重复",
        ),
        (
            {"calibration_count": 21},
            "选择数量",
        ),
        (
            {"confidence": float("nan")},
            "置信度",
        ),
        (
            {"cam2_npu_mode": "NPU3"},
            "NPU2",
        ),
        (
            {"camera_width": 0},
            "相机",
        ),
    ]

    for updates, message in cases:
        with pytest.raises(ValidationError, match=message):
            controller.start_maix_deploy(
                _deploy_values(source, calibration, **updates)
            )

    assert project.repository.list_runs(kind="deploy") == ()


def test_deployment_directory_collision_does_not_create_orphan_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )
    workspace = tmp_path / "conversion"
    (workspace / "fixed-deploy-id").mkdir(parents=True)
    monkeypatch.setattr(
        "ai_biaozhu.controller.uuid4",
        lambda: SimpleNamespace(hex="fixed-deploy-id"),
    )

    with pytest.raises(ValidationError, match="已存在"):
        controller.start_maix_deploy(
            _deploy_values(
                source,
                calibration,
                conversion_workspace=str(workspace),
            )
        )

    assert project.repository.list_runs(kind="deploy") == ()


def test_deployment_manifest_failure_is_audited_and_cancel_uses_deploy_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("ai_biaozhu.controller.write_json", fail_write)
    with pytest.raises(OSError, match="disk full"):
        controller.start_maix_deploy(_deploy_values(source, calibration))
    failed = project.repository.list_runs(kind="deploy")
    assert len(failed) == 1
    assert failed[0].status.value == "failed"
    assert "disk full" in (failed[0].error or "")

    monkeypatch.undo()
    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    deploy_id = str(launch["job_id"])
    assert controller.cancel_job(deploy_id)
    assert (project.deployments_dir / deploy_id / "cancel.requested").is_file()
    assert not (project.runs_dir / deploy_id / "cancel.requested").exists()


def test_deployment_manifest_hashes_normalized_project_files(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )

    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    manifest_path = Path(str(launch["arguments"][-1]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_id = {
        str(item["image_id"]): item
        for item in manifest["calibration_images"]
    }
    assert set(by_id) == set(calibration)
    for image_id in calibration:
        image = project.repository.get_image(image_id)
        project_path = project.image_path(image)
        assert by_id[image_id]["sha256"] == sha256_file(project_path)
        # The fixture deliberately stores source-file hashes that differ from
        # the project copies, matching real EXIF-normalized imports.
        assert by_id[image_id]["sha256"] != image.sha256


def test_new_deployment_uses_current_canonical_name_for_old_checkpoint(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )
    category = project.repository.list_categories()[0]
    snapshot_root = project.runs_dir / source.id / "snapshot"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "manifest.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"index": 0, "id": category.id, "name": category.name}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source = project.repository.update_run(
        source.id,
        snapshot_path=str(snapshot_root),
    )
    rename = controller.rename_category_canonical(category.id, "钢球")
    assert Path(rename["backup"]["path"]).is_file()

    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))

    assert manifest["class_ids"] == [category.id]
    assert manifest["checkpoint_class_names"] == ["目标"]
    assert manifest["deployment_class_names"] == ["钢球"]
    assert manifest["class_names"] == ["目标"]


def test_new_deployment_recovers_legacy_snapshot_name_after_category_rename(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "legacy-project",
    )
    category = project.repository.list_categories()[0]
    snapshot_root = project.runs_dir / source.id / "snapshot"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "manifest.json").write_text(
        json.dumps(
            {"classes": [{"index": 0, "name": category.name}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source = project.repository.update_run(
        source.id,
        snapshot_path=str(snapshot_root),
    )
    controller.rename_category_canonical(category.id, "钢球")

    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))

    assert manifest["class_ids"] == [category.id]
    assert manifest["checkpoint_class_names"] == ["目标"]
    assert manifest["deployment_class_names"] == ["钢球"]


def test_new_deployment_recovers_legacy_data_yaml_after_category_rename(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "legacy-yaml-project",
    )
    category = project.repository.list_categories()[0]
    snapshot_root = project.runs_dir / source.id / "snapshot"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "data.yaml").write_text(
        "names:\n  0: 目标\n",
        encoding="utf-8",
    )
    source = project.repository.update_run(
        source.id,
        snapshot_path=str(snapshot_root),
    )
    controller.rename_category_canonical(category.id, "钢球")

    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))

    assert manifest["checkpoint_class_names"] == ["目标"]
    assert manifest["deployment_class_names"] == ["钢球"]


def test_new_deployment_blocks_renamed_legacy_run_without_class_snapshot(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "unsafe-legacy-project",
    )
    category = project.repository.list_categories()[0]
    source = project.repository.update_run(source.id, snapshot_path=None)
    controller.rename_category_canonical(category.id, "钢球")

    with pytest.raises(Exception, match="缺少可验证的类别快照"):
        controller.start_maix_deploy(_deploy_values(source, calibration))


def test_deployment_freezes_calibration_in_run_local_snapshot(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )

    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    run_id = str(launch["job_id"])
    run_dir = project.deployments_dir / run_id
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))
    frozen = manifest["calibration_images"][0]
    frozen_path = Path(str(frozen["path"]))
    source_path = Path(str(frozen["source_path"]))
    snapshot_manifest = Path(str(manifest["calibration_source_snapshot_manifest"]))

    assert frozen_path.parent == run_dir / "calibration-source-snapshot"
    assert snapshot_manifest == run_dir / "calibration-source-snapshot.json"
    assert frozen_path.is_file()
    assert sha256_file(frozen_path) == frozen["sha256"]

    source_path.write_bytes(b"changed after the deployment job was queued")

    assert sha256_file(source_path) != frozen["sha256"]
    assert sha256_file(frozen_path) == frozen["sha256"]
    frozen_audit = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    assert frozen_audit["selected"][0]["sha256"] == frozen["sha256"]
    assert frozen_audit["selected"][0]["path"] == str(frozen_path)


def test_deployment_replaces_corrupt_selected_calibration_with_healthy_candidate(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )
    corrupt_id = calibration[0]
    corrupt_image = project.repository.get_image(corrupt_id)
    project.image_path(corrupt_image).write_bytes(b"not a decodable image")

    launch = controller.start_maix_deploy(_deploy_values(source, calibration))
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))
    selected = manifest["calibration_images"]
    frozen_audit = json.loads(
        Path(str(manifest["calibration_source_snapshot_manifest"])).read_text(
            encoding="utf-8"
        )
    )

    assert len(selected) == len(calibration)
    assert corrupt_id not in {item["image_id"] for item in selected}
    replacements = [item for item in selected if item["replacement"]]
    assert len(replacements) == 1
    assert replacements[0]["source_role"] == "fallback"
    assert sha256_file(Path(str(replacements[0]["path"]))) == replacements[0]["sha256"]
    assert any(
        item["image_id"] == corrupt_id
        and item["source_role"] == "selected"
        and item["reason"]
        for item in frozen_audit["rejected"]
    )


def test_maixcam_pro_manifest_has_no_cam2_mode_value(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )

    launch = controller.start_maix_deploy(
        _deploy_values(
            source,
            calibration,
            target="maixcam_pro",
            cam2_npu_mode="vnpu",
        )
    )
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))
    package = project.repository.list_deployment_packages(
        run_id=str(launch["job_id"])
    )[0]

    assert manifest["target"] == "maixcam_pro"
    assert manifest.get("cam2_npu_mode") is None
    assert package.npu_mode == "not_applicable"


@pytest.mark.parametrize(
    ("package_outputs", "expects_editable"),
    [
        (["maixapp"], False),
        (["editable_project"], True),
    ],
)
def test_deployment_manifest_honors_independent_package_output_selection(
    tmp_path: Path,
    package_outputs: list[str],
    expects_editable: bool,
) -> None:
    controller = _controller(tmp_path)
    _project, source, calibration = _deploy_source(
        controller,
        tmp_path / "project",
    )

    launch = controller.start_maix_deploy(
        _deploy_values(
            source,
            calibration,
            package_outputs=package_outputs,
        )
    )
    manifest = json.loads(Path(str(launch["arguments"][-1])).read_text(encoding="utf-8"))

    assert manifest["package_outputs"] == package_outputs
    assert bool(manifest["editable_project_path"]) is expects_editable
    assert manifest["package_path"].endswith(".maixapp")


def test_internal_process_logs_are_persisted_separately(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "logs")
    run = project.repository.create_run("train", "YOLO26n", run_id="run-log")
    controller.handle_job_event(
        {
            "_internal": True,
            "job_id": run.id,
            "type": "log",
            "payload": {"message": "native trainer output"},
        }
    )
    assert (project.runs_dir / run.id / "console.log").read_text(
        encoding="utf-8"
    ) == "native trainer output\n"


def test_controller_ignores_duplicate_and_out_of_order_events_with_diagnostic(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "event diagnostics")
    run = project.repository.create_run("train", "YOLO26n", run_id="run-sequence")

    accepted = ProtocolEvent.create(
        job_id=run.id,
        seq=4,
        event_type="status",
        payload={"stage": "training"},
    )
    duplicate_error = ProtocolEvent.create(
        job_id=run.id,
        seq=4,
        event_type="error",
        payload={"message": "must not fail the run"},
    )
    older_error = ProtocolEvent.create(
        job_id=run.id,
        seq=3,
        event_type="error",
        payload={"message": "must not fail the run either"},
    )
    controller.handle_job_event(accepted.to_dict())
    assert controller.handle_job_event(duplicate_error.to_dict()) is None
    assert controller.handle_job_event(older_error.to_dict()) is None

    unchanged = project.repository.get_run(run.id)
    assert unchanged.status.value == "training"
    assert unchanged.error is None
    event_path = project.runs_dir / run.id / "events.jsonl"
    persisted = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(event["seq"], event["type"]) for event in persisted] == [
        (4, "status")
    ]
    diagnostics = (project.runs_dir / run.id / "console.log").read_text(
        encoding="utf-8"
    )
    assert "已忽略重复 worker 事件" in diagnostics
    assert "已忽略乱序 worker 事件" in diagnostics

    next_event = ProtocolEvent.create(
        job_id=run.id,
        seq=5,
        event_type="progress",
        payload={"progress": 0.5},
    )
    controller.handle_job_event(next_event.to_dict())
    updated = project.repository.get_run(run.id)
    assert updated.progress == pytest.approx(0.5)
    persisted = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(event["seq"], event["type"]) for event in persisted] == [
        (4, "status"),
        (5, "progress"),
    ]


def test_resume_training_reuses_only_original_snapshot_and_last_checkpoint(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    _seed_verified_project(controller, tmp_path / "project")
    initial = controller.start_training(
        "YOLO26n",
        {
            "epochs": 3,
            "patience": 0,
            "batch": 1,
            "device": 0,
            "workers": 0,
            "ml_environment": sys.executable,
        },
    )
    project = controller.current_project
    assert project is not None
    source_id = str(initial["job_id"])
    source = project.repository.get_run(source_id)
    last = project.runs_dir / source_id / "training" / "model" / "weights" / "last.pt"
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"last")
    project.repository.update_run(
        source_id,
        artifacts={"last": str(last)},
        checkpoint_path=str(last),
    )
    controller.handle_process_finished(source_id, success=False, exit_code=1)

    resumed_launch = controller.resume_training(
        {"run_id": source_id, "ml_environment": sys.executable}
    )
    resumed = project.repository.get_run(str(resumed_launch["job_id"]))
    manifest = json.loads(
        (project.runs_dir / resumed.id / "job.json").read_text(encoding="utf-8")
    )
    assert resumed.id != source_id
    assert resumed.snapshot_path == source.snapshot_path
    assert manifest["resume_from_run_id"] == source_id
    assert manifest["checkpoint_source"] == str(last)
    assert manifest["config"]["resume"] == str(last)
    assert manifest["data_yaml"] == str(Path(source.snapshot_path) / "data.yaml")


def test_reopen_reconciles_interrupted_training_and_recovers_last_checkpoint(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    _seed_verified_project(controller, tmp_path / "project")
    launch = controller.start_training(
        "YOLO26n",
        {
            "epochs": 3,
            "patience": 0,
            "batch": 1,
            "device": 0,
            "workers": 0,
            "ml_environment": sys.executable,
        },
    )
    project = controller.current_project
    assert project is not None
    run_id = str(launch["job_id"])
    before = project.repository.get_run(run_id)
    assert before.status.value == "training"
    assert before.snapshot_path
    last = project.runs_dir / run_id / "training" / "model" / "weights" / "last.pt"
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"recoverable-last")
    metrics = Path(str(before.metrics_jsonl_path))
    metrics.write_text(
        ProtocolEvent.create(
            job_id=run_id,
            seq=4,
            event_type="metrics",
            payload={"epoch": 1, "epochs": 3, "box_loss": 0.25, "mAP50": 0.4},
        ).to_json()
        + "\n",
        encoding="utf-8",
    )
    console = project.runs_dir / run_id / "console.log"
    console.write_text("trainer survived until epoch 1\n", encoding="utf-8")
    preview = project.runs_dir / run_id / "training" / "model" / "results.png"
    preview.write_bytes(b"historical-preview")
    project.repository.update_run(
        run_id,
        artifacts={"training_visual": str(preview)},
    )
    snapshot_path = before.snapshot_path
    controller.close_project()

    reopened_controller = _controller(tmp_path)
    reopened = reopened_controller.open_project(tmp_path / "project")
    recovered = reopened.repository.get_run(run_id)
    assert recovered.status.value == "failed"
    assert recovered.snapshot_path == snapshot_path
    assert recovered.metrics_jsonl_path == str(metrics)
    assert recovered.artifacts["last"] == str(last.resolve())
    assert recovered.checkpoint_path == str(last.resolve())
    assert "可从原不可变快照恢复训练" in (recovered.error or "")
    assert console.read_text(encoding="utf-8") == "trainer survived until epoch 1\n"
    assert reopened_controller.last_reconciled_run_ids == (run_id,)

    history = reopened_controller.load_training_run_history(run_id)
    assert history["events"][0]["payload"]["box_loss"] == 0.25
    assert "trainer survived until epoch 1" in history["console_log"]
    assert history["preview_path"] == str(preview.resolve())

    resumed = reopened_controller.resume_training(
        {"run_id": run_id, "ml_environment": sys.executable}
    )
    assert str(resumed["job_id"]) != run_id


def test_reconcile_does_not_fail_controller_owned_active_run(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "active")
    run = project.repository.create_run("train", "YOLO26n", run_id="active-run")
    project.repository.update_run(run.id, status="training")
    controller._activate_job(run.id, "train")

    assert controller.reconcile_interrupted_runs() == ()
    assert project.repository.get_run(run.id).status.value == "training"


def test_controller_backup_cleanup_is_verified_scoped_and_recoverable(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "backup cleanup")
    for index in range(5):
        project._create_annotation_backup(f"controller-cleanup-{index}")

    listed_before = controller.list_annotation_backups()
    kept_names = {Path(item["path"]).name for item in listed_before[:2]}
    preview = controller.preview_backup_cleanup(keep_latest=2)
    candidate_paths = {Path(item["path"]) for item in preview["backups"]}
    assert preview["backup_count"] == 3
    assert preview["keep_latest"] == 2
    assert preview["total_bytes"] > 0

    for unverified in (False, None, 1, "yes"):
        with pytest.raises(PermissionError, match="部署已在设备上验证成功"):
            controller.cleanup_old_backups(
                keep_latest=2,
                deployment_verified=unverified,
            )
    assert all(path.is_file() for path in candidate_paths)

    report = controller.cleanup_old_backups(
        keep_latest=2,
        deployment_verified=True,
    )
    recovery_directory = Path(report["recovery_directory"])
    assert report["backup_count"] == 3
    assert report["keep_latest"] == 2
    # SQLite validation may create WAL/SHM sidecars; cleanup moves those too so
    # the active backup directory does not retain hidden disk usage.
    assert report["moved_count"] >= 6
    assert recovery_directory.parent == project.backups_dir / ".trash"
    assert recovery_directory.is_dir()
    assert all(not path.exists() for path in candidate_paths)
    assert all((recovery_directory / path.name).is_file() for path in candidate_paths)
    assert all(
        (recovery_directory / path.with_suffix(".json").name).is_file()
        for path in candidate_paths
    )

    listed_after = controller.list_annotation_backups()
    assert len(listed_after) == 2
    assert {Path(item["path"]).name for item in listed_after} == kept_names

    recovered_database = next(recovery_directory.glob("*.db"))
    restore = controller.restore_annotation_backup(recovered_database)
    assert Path(restore["restored_backup"]["path"]) == recovered_database
