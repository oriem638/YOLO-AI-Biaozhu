from __future__ import annotations

import json
import subprocess

from ai_biaozhu.ml.environment import (
    EnvironmentCandidate,
    discover_environments,
    inspect_environment,
    inspect_legacy_yolov5_repository,
)


def test_discovery_prefers_named_yolo_environment(tmp_path) -> None:
    current = tmp_path / "current"
    yolo = tmp_path / "yolo"
    current.mkdir()
    yolo.mkdir()
    (current / "python.exe").write_bytes(b"")
    (yolo / "python.exe").write_bytes(b"")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"envs": [str(current), str(yolo)]}),
            "",
        )

    environments = discover_environments(
        runner=runner,
        current_executable=current / "python.exe",
        environ={"USERPROFILE": str(tmp_path / "user")},
    )
    assert environments[0].prefix == yolo


def test_environment_baseline_and_yolo_config_dir(tmp_path) -> None:
    prefix = tmp_path / "yolo"
    prefix.mkdir()
    python = prefix / "python.exe"
    python.write_bytes(b"")
    captured = {}

    def runner(command, **kwargs):
        captured.update(kwargs)
        payload = {
            "python": "3.11.9",
            "executable": str(python),
            "torch": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "ultralytics": "8.4.82",
            "cuda_available": True,
            "cuda_version": "12.8",
            "device_name": "NVIDIA GeForce RTX 5060",
            "errors": [],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    report = inspect_environment(EnvironmentCandidate(prefix, python, "manual"), runner=runner)
    assert report.valid
    assert report.gpu_ready
    assert captured["env"]["YOLO_CONFIG_DIR"]


def test_environment_rejects_wrong_cuda_wheel(tmp_path) -> None:
    prefix = tmp_path / "wrong"
    prefix.mkdir()
    python = prefix / "python.exe"
    python.write_bytes(b"")

    def runner(command, **kwargs):
        payload = {
            "python": "3.11.9",
            "torch": "2.11.0+cpu",
            "torchvision": "0.26.0+cpu",
            "ultralytics": "8.5.0",
            "cuda_available": False,
            "cuda_version": None,
            "errors": [],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    report = inspect_environment(EnvironmentCandidate(prefix, python, "manual"), runner=runner)
    assert not report.valid
    assert len(report.compatibility_errors) >= 3


def test_pinned_environment_can_be_valid_without_available_gpu(tmp_path) -> None:
    prefix = tmp_path / "cpu-only"
    prefix.mkdir()
    python = prefix / "python.exe"
    python.write_bytes(b"")

    def runner(command, **kwargs):
        del kwargs
        payload = {
            "python": "3.11.9",
            "torch": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "ultralytics": "8.4.82",
            "cuda_available": False,
            "cuda_version": "12.8",
            "device_name": None,
            "errors": [],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    report = inspect_environment(
        EnvironmentCandidate(prefix, python, "manual"),
        runner=runner,
    )
    assert report.valid
    assert not report.gpu_ready
    assert report.compatibility_errors == ()


def test_legacy_yolov5_repository_requires_scripts_and_tag_lock(tmp_path) -> None:
    for name in ("train.py", "detect.py", "export.py"):
        (tmp_path / name).write_text("# placeholder", encoding="utf-8")
    (tmp_path / ".ai-biaozhu-yolov5-tag").write_text("v7.0\n", encoding="utf-8")
    report = inspect_legacy_yolov5_repository(tmp_path)
    assert report.valid
    assert report.tag == "v7.0"
    (tmp_path / ".ai-biaozhu-yolov5-tag").write_text("master\n", encoding="utf-8")
    assert not inspect_legacy_yolov5_repository(tmp_path).valid
