from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_biaozhu.workers.main as worker_main
from ai_biaozhu.deploy.docker_import import (
    DockerImageImportCancelled,
    DockerImageImportResult,
    import_docker_image_archive,
)
from ai_biaozhu.deploy.environment import DockerImageIdentity
from ai_biaozhu.workers.commands import build_worker_command


class _FakeInput:
    def __init__(self, process: _FakeProcess, *, delay: float = 0.0) -> None:
        self.process = process
        self.delay = delay
        self.data = bytearray()
        self.closed = False

    def write(self, data) -> int:
        if self.delay:
            time.sleep(self.delay)
        if self.process.returncode is not None:
            raise BrokenPipeError("terminated")
        value = bytes(data)
        self.data.extend(value)
        return len(value)

    def close(self) -> None:
        self.closed = True
        if self.process.returncode is None:
            self.process.returncode = 0
        self.process.finished.set()


class _FakeProcess:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.returncode: int | None = None
        self.finished = threading.Event()
        self.stdin = _FakeInput(self, delay=delay)
        self.stdout = io.BytesIO(b"Loaded image: pulsar2:6.0\n")
        self.stderr = io.BytesIO()
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.finished.wait(timeout):
            raise TimeoutError
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.finished.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.finished.set()


def _inspect_success(command, **_kwargs):
    name = command[-1]
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            [
                {
                    "Id": "sha256:abc123",
                    "RepoDigests": [f"{name.split(':')[0]}@sha256:digest"],
                }
            ]
        ),
        stderr="",
    )


def test_streamed_docker_import_reports_bytes_heartbeat_and_identity(tmp_path: Path) -> None:
    archive = tmp_path / "pulsar2.tar"
    archive.write_bytes(b"abcdefgh" * 16)
    process = _FakeProcess(delay=0.006)
    progress = []

    result = import_docker_image_archive(
        archive,
        "docker.exe",
        expected_images=("pulsar2:6.0",),
        progress_callback=progress.append,
        popen_factory=lambda command, **_kwargs: (
            process if command == ["docker.exe", "load"] else None
        ),
        inspect_runner=_inspect_success,
        chunk_size=8,
        heartbeat_interval=0.01,
        poll_interval=0.001,
    )

    assert bytes(process.stdin.data) == archive.read_bytes()
    assert result.bytes_read == result.total_bytes == archive.stat().st_size
    assert result.image_tags == ("pulsar2:6.0",)
    assert result.images[0].image_id == "sha256:abc123"
    assert result.images[0].repo_digests == ("pulsar2@sha256:digest",)
    assert any(item.heartbeat for item in progress)
    assert progress[-1].completed
    assert progress[-1].percent == 100.0
    assert all(item.elapsed_seconds >= 0 for item in progress)


def test_streamed_docker_import_cancellation_terminates_child(tmp_path: Path) -> None:
    archive = tmp_path / "large.tar"
    archive.write_bytes(b"x" * 4096)
    process = _FakeProcess(delay=0.02)
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(DockerImageImportCancelled):
        import_docker_image_archive(
            archive,
            "docker.exe",
            cancel_check=cancelled,
            popen_factory=lambda _command, **_kwargs: process,
            inspect_runner=_inspect_success,
            chunk_size=64,
            heartbeat_interval=0.01,
            poll_interval=0.001,
        )
    assert process.terminated


def test_streamed_docker_import_emits_stall_diagnostic_before_cancel(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "stalled.tar"
    archive.write_bytes(b"x" * 4096)
    process = _FakeProcess(delay=0.05)
    progress = []
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 30

    with pytest.raises(DockerImageImportCancelled):
        import_docker_image_archive(
            archive,
            "docker.exe",
            progress_callback=progress.append,
            cancel_check=cancelled,
            popen_factory=lambda _command, **_kwargs: process,
            inspect_runner=_inspect_success,
            chunk_size=64,
            heartbeat_interval=0.003,
            stall_diagnostic_after=0.005,
            poll_interval=0.001,
        )

    diagnostics = [item for item in progress if item.diagnostic]
    assert diagnostics
    assert diagnostics[0].stalled_seconds >= 0.005
    assert "docker load" in diagnostics[0].diagnostic_message


def test_worker_docker_import_emits_machine_readable_result(monkeypatch, tmp_path) -> None:
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"archive")
    identity = DockerImageIdentity(
        "pulsar2:6.0",
        True,
        "sha256:abc123",
        ("pulsar2@sha256:digest",),
    )

    def fake_import(_archive, _docker, **kwargs):
        callback = kwargs["progress_callback"]
        callback(
            SimpleNamespace(
                to_dict=lambda: {
                    "stage": "docker_image_import",
                    "bytes_read": 7,
                    "total_bytes": 7,
                    "percent": 100.0,
                    "elapsed_seconds": 0.5,
                    "bytes_per_second": 14.0,
                    "heartbeat": False,
                    "completed": True,
                }
            )
        )
        return DockerImageImportResult(
            archive,
            7,
            7,
            0.5,
            ("pulsar2:6.0",),
            (),
            (identity,),
            "Loaded image: pulsar2:6.0",
        )

    monkeypatch.setattr(worker_main, "import_docker_image_archive", fake_import)
    output = io.StringIO()
    code = worker_main.run_docker_import(
        {
            "job_id": "load-1",
            "archive_path": str(archive),
            "docker_executable": "docker.exe",
            "expected_images": ["pulsar2:6.0"],
        },
        stream=output,
        input_stream=io.StringIO(""),
    )
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert code == 0
    assert [item["type"] for item in events] == [
        "status",
        "progress",
        "artifact",
        "completed",
    ]
    assert events[1]["payload"]["bytes_read"] == 7
    assert events[2]["payload"]["repo_digests"] == ["pulsar2@sha256:digest"]
    assert events[-1]["payload"]["image_tags"] == ["pulsar2:6.0"]


def test_worker_command_supports_manifest_driven_docker_import(tmp_path: Path) -> None:
    command = build_worker_command(
        "worker.exe",
        "docker-import",
        manifest=tmp_path / "load.json",
    )
    assert command.arguments == (
        "-m",
        "ai_biaozhu.workers.main",
        "docker-import",
        "--manifest",
        str(tmp_path / "load.json"),
    )
