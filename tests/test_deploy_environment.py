from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ai_biaozhu.deploy.environment import (
    DockerDesktopState,
    assess_docker_desktop_recovery,
    build_docker_desktop_start_command,
    find_docker_desktop_executable,
    inspect_docker_environment,
)


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_docker_environment_reports_versions_images_and_mount(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "ai_biaozhu.deploy.environment.shutil.which",
        lambda _value: "C:/Program Files/Docker/docker.exe",
    )
    monkeypatch.setattr(
        "ai_biaozhu.deploy.environment._inspect_wsl2",
        lambda _runner: True,
    )

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert _kwargs["stdin"] is subprocess.DEVNULL
        calls.append(command)
        if command[1:3] == ["version", "--format"]:
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "Client": {"Version": "28.3.2"},
                        "Server": {"Version": "28.3.2"},
                    }
                ),
            )
        if command[1:3] == ["image", "inspect"]:
            name = command[3]
            return _completed(
                command,
                stdout=json.dumps(
                    [
                        {
                            "Id": "sha256:abc123",
                            "RepoDigests": [f"{name.split(':')[0]}@sha256:def456"],
                        }
                    ]
                ),
            )
        if command[1] == "run":
            assert command[3:5] == ["--entrypoint", "sh"]
            assert ":/ai_biaozhu_probe:ro" in command[6]
            assert command[-2:] == [
                "-c",
                "test -f /ai_biaozhu_probe/probe.txt",
            ]
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    report = inspect_docker_environment(
        required_images=("sipeed/pulsar2:6.0",),
        runner=runner,
        mount_root=tmp_path,
    )

    assert report.ready
    assert report.client_version == "28.3.2"
    assert report.server_version == "28.3.2"
    assert report.wsl2_ready is True
    assert report.mount_ready is True
    assert report.images[0].available
    assert report.images[0].image_id == "sha256:abc123"
    assert any(command[1] == "run" for command in calls)
    assert report.to_dict()["ready"] is True


def test_docker_environment_missing_cli_is_read_only(
    monkeypatch: Any,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("ai_biaozhu.deploy.environment.shutil.which", lambda _value: None)
    monkeypatch.setattr(
        "ai_biaozhu.deploy.environment._inspect_wsl2",
        lambda _runner: True,
    )

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        raise AssertionError(f"unexpected command: {command}")

    report = inspect_docker_environment(
        required_images=("converter:test",),
        runner=runner,
    )

    assert not report.ready
    assert report.executable is None
    assert report.wsl2_ready is True
    assert report.images[0].available is None
    assert report.images[0].status == "unchecked"
    assert report.images[0].error == "Docker 不可用"
    assert any("未找到 Docker CLI" in error for error in report.errors)
    assert calls == []


def test_docker_environment_does_not_report_missing_images_when_daemon_is_down(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "ai_biaozhu.deploy.environment.shutil.which",
        lambda _value: "docker",
    )
    monkeypatch.setattr(
        "ai_biaozhu.deploy.environment._inspect_wsl2",
        lambda _runner: False,
    )

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[1] == "version":
            return _completed(command, returncode=1, stderr="daemon stopped")
        raise AssertionError(f"unexpected command: {command}")

    report = inspect_docker_environment(
        required_images=("converter:test",),
        runner=runner,
    )

    assert not report.ready
    assert not report.daemon_ready
    assert report.mount_ready is None
    assert report.wsl2_ready is False
    assert report.images[0].available is None
    assert report.images[0].status == "unchecked"
    assert "尚未检查" in str(report.images[0].error)
    assert any("daemon stopped" in error for error in report.errors)
    assert not any("尚未加载转换镜像" in warning for warning in report.warnings)
    assert report.to_dict()["images"][0]["status"] == "unchecked"


def test_docker_desktop_discovery_start_command_and_poll_states(tmp_path: Path) -> None:
    executable = tmp_path / "Docker" / "Docker" / "Docker Desktop.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"exe")
    found = find_docker_desktop_executable(
        environ={"ProgramFiles": str(tmp_path)},
        which=lambda _name: None,
    )
    assert found == executable.resolve()
    assert build_docker_desktop_start_command(found) == (str(found),)

    stopped = inspect_docker_environment(
        docker_executable="definitely-not-a-real-docker-cli",
        required_images=("converter:test",),
    )
    recovery = assess_docker_desktop_recovery(
        stopped,
        desktop_executable=found,
    )
    assert recovery.state is DockerDesktopState.STOPPED
    assert recovery.can_start
    assert not recovery.should_poll

    starting = assess_docker_desktop_recovery(
        stopped,
        desktop_executable=found,
        launch_requested=True,
        elapsed_seconds=3,
        timeout_seconds=10,
    )
    assert starting.state is DockerDesktopState.STARTING
    assert starting.should_poll
    assert starting.to_dict()["state"] == "starting"

    timed_out = assess_docker_desktop_recovery(
        stopped,
        desktop_executable=found,
        launch_requested=True,
        elapsed_seconds=10,
        timeout_seconds=10,
    )
    assert timed_out.state is DockerDesktopState.TIMED_OUT
    assert not timed_out.should_poll
