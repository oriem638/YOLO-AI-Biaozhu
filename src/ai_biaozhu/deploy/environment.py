"""Read-only Docker Desktop, WSL2 and converter-image diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .maix import MAIXCAM2_IMAGE, MAIXCAM_PRO_IMAGE

Runner = Callable[..., subprocess.CompletedProcess[str]]

# Docker Desktop emits UTF-8 JSON and diagnostics even when the Windows
# console code page is GBK.  Decode its captured output explicitly so an
# unrelated localized diagnostic cannot crash a subprocess reader thread.
_DOCKER_TEXT_KWARGS = {"encoding": "utf-8", "errors": "replace"}

# The packaged desktop application is built without a console window.  On
# Windows that can leave the inherited standard-input handle invalid, causing
# ``subprocess.run`` to fail before Docker or WSL has a chance to start with
# ``[WinError 6] The handle is invalid``.  Every read-only diagnostic command
# therefore receives an explicit harmless input handle.
_DOCKER_PROCESS_KWARGS = {"stdin": subprocess.DEVNULL, **_DOCKER_TEXT_KWARGS}


@dataclass(frozen=True, slots=True)
class DockerImageIdentity:
    name: str
    # ``None`` deliberately means "not inspected".  It must not be folded
    # into ``False`` because a stopped daemon says nothing about whether an
    # image is present in Docker Desktop's local store.
    available: bool | None
    image_id: str | None = None
    repo_digests: tuple[str, ...] = ()
    error: str | None = None

    @property
    def status(self) -> str:
        if self.available is None:
            return "unchecked"
        return "available" if self.available else "missing"


class DockerDesktopState(str, Enum):
    """UI-independent Docker Desktop recovery states."""

    READY = "ready"
    STOPPED = "stopped"
    STARTING = "starting"
    TIMED_OUT = "timed_out"
    NOT_INSTALLED = "not_installed"


@dataclass(frozen=True, slots=True)
class DockerDesktopRecoveryStatus:
    """Pure state returned to a GUI while it starts/polls Docker Desktop."""

    state: DockerDesktopState
    desktop_executable: str | None
    start_command: tuple[str, ...]
    can_start: bool
    should_poll: bool
    poll_interval_seconds: float
    elapsed_seconds: float
    timeout_seconds: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value


@dataclass(frozen=True, slots=True)
class DockerEnvironmentReport:
    executable: str | None
    client_version: str | None
    server_version: str | None
    daemon_ready: bool
    wsl2_ready: bool | None
    mount_ready: bool | None
    images: tuple[DockerImageIdentity, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.daemon_ready
            and not self.errors
            and all(image.available for image in self.images)
            and self.mount_ready is not False
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["images"] = [
            {**asdict(image), "status": image.status} for image in self.images
        ]
        value["ready"] = self.ready
        return value


def find_docker_desktop_executable(
    configured: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Path | None:
    """Find Docker Desktop without starting it or mutating the machine.

    ``configured`` is tried first, followed by PATH and the normal per-user
    and machine-wide Windows installation locations.  Injected ``environ``
    and ``which`` keep the decision logic deterministic in tests and usable
    by a Qt controller without importing Qt here.
    """

    environment = os.environ if environ is None else environ
    candidates: list[str | Path] = []
    if configured:
        candidates.append(configured)
    located = which("Docker Desktop") or which("Docker Desktop.exe")
    if located:
        candidates.append(located)
    for variable in ("ProgramFiles", "ProgramW6432", "LOCALAPPDATA"):
        root = environment.get(variable)
        if root:
            candidates.append(Path(root) / "Docker" / "Docker" / "Docker Desktop.exe")
            if variable == "LOCALAPPDATA":
                candidates.append(Path(root) / "Docker" / "Docker Desktop.exe")
    program_files_x86 = environment.get("ProgramFiles(x86)")
    if program_files_x86:
        candidates.append(
            Path(program_files_x86) / "Docker" / "Docker" / "Docker Desktop.exe"
        )
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    return None


def build_docker_desktop_start_command(
    executable: str | Path,
) -> tuple[str, ...]:
    """Build the detached start command consumed by the GUI/controller."""

    value = str(executable).strip()
    if not value:
        raise ValueError("Docker Desktop 可执行文件路径不能为空")
    return (str(Path(value)),)


def assess_docker_desktop_recovery(
    report: DockerEnvironmentReport,
    *,
    desktop_executable: str | Path | None,
    launch_requested: bool = False,
    elapsed_seconds: float = 0.0,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 2.0,
) -> DockerDesktopRecoveryStatus:
    """Describe the next recovery action without sleeping or launching GUI apps.

    A Qt timer can repeatedly inspect the environment and call this function;
    tests can exercise the complete state machine without a real Docker daemon.
    """

    elapsed = max(0.0, float(elapsed_seconds))
    timeout = max(0.1, float(timeout_seconds))
    interval = max(0.1, float(poll_interval_seconds))
    executable = str(desktop_executable) if desktop_executable else None
    command = build_docker_desktop_start_command(executable) if executable else ()
    if report.daemon_ready:
        state = DockerDesktopState.READY
        message = "Docker daemon 已就绪。"
        should_poll = False
    elif executable is None:
        state = DockerDesktopState.NOT_INSTALLED
        message = "未找到 Docker Desktop，无法自动启动。"
        should_poll = False
    elif launch_requested and elapsed >= timeout:
        state = DockerDesktopState.TIMED_OUT
        message = f"等待 Docker daemon 超时（{timeout:.0f} 秒）。"
        should_poll = False
    elif launch_requested:
        state = DockerDesktopState.STARTING
        message = "Docker Desktop 正在启动，等待 daemon 就绪。"
        should_poll = True
    else:
        state = DockerDesktopState.STOPPED
        message = "Docker Desktop 已安装，但 daemon 尚未就绪。"
        should_poll = False
    return DockerDesktopRecoveryStatus(
        state=state,
        desktop_executable=executable,
        start_command=command,
        can_start=bool(executable) and not report.daemon_ready,
        should_poll=should_poll,
        poll_interval_seconds=interval,
        elapsed_seconds=elapsed,
        timeout_seconds=timeout,
        message=message,
    )


def inspect_docker_environment(
    *,
    docker_executable: str | Path | None = None,
    required_images: Sequence[str] = (MAIXCAM_PRO_IMAGE, MAIXCAM2_IMAGE),
    runner: Runner = subprocess.run,
    check_mount: bool = True,
    mount_root: str | Path | None = None,
) -> DockerEnvironmentReport:
    """Inspect conversion prerequisites without installing or pulling anything."""

    configured = str(docker_executable or "docker")
    located = shutil.which(configured)
    executable = located or (
        configured if Path(configured).is_file() else None
    )
    if executable is None:
        return DockerEnvironmentReport(
            executable=None,
            client_version=None,
            server_version=None,
            daemon_ready=False,
            wsl2_ready=_inspect_wsl2(runner),
            mount_ready=None,
            images=tuple(
                DockerImageIdentity(name=image, available=None, error="Docker 不可用")
                for image in required_images
            ),
            errors=("未找到 Docker CLI；请安装并启动 Docker Desktop。",),
            warnings=(),
        )

    errors: list[str] = []
    warnings: list[str] = []
    client_version: str | None = None
    server_version: str | None = None
    daemon_ready = False
    try:
        completed = runner(
            [executable, "version", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            **_DOCKER_PROCESS_KWARGS,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"Docker 版本检测失败：{exc}")
    else:
        if completed.returncode:
            errors.append(
                "Docker daemon 不可用："
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        else:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                errors.append(f"Docker version 输出不是有效 JSON：{exc}")
            else:
                client_version = _nested_version(payload, "Client")
                server_version = _nested_version(payload, "Server")
                daemon_ready = bool(server_version)
                if not daemon_ready:
                    errors.append("Docker CLI 可用，但 daemon 尚未启动。")

    if daemon_ready:
        images = tuple(
            _inspect_image(executable, name, runner) for name in required_images
        )
        missing = [image.name for image in images if image.available is False]
        if missing:
            warnings.append("尚未加载转换镜像：" + ", ".join(missing))
    else:
        images = tuple(
            DockerImageIdentity(
                name=image,
                available=None,
                error="Docker daemon 未就绪，镜像尚未检查",
            )
            for image in required_images
        )
    wsl2_ready = _inspect_wsl2(runner)
    if os.name == "nt" and wsl2_ready is False:
        warnings.append("未检测到可用 WSL2；Docker Desktop Linux 容器可能无法运行。")

    mount_ready: bool | None = None
    usable_image = next((item.name for item in images if item.available), None)
    if daemon_ready and check_mount and usable_image:
        mount_ready = _inspect_mount(
            executable,
            usable_image,
            runner,
            root=mount_root,
        )
        if mount_ready is False:
            errors.append("Docker 无法挂载所选本地目录；请检查文件共享权限。")
    return DockerEnvironmentReport(
        executable=executable,
        client_version=client_version,
        server_version=server_version,
        daemon_ready=daemon_ready,
        wsl2_ready=wsl2_ready,
        mount_ready=mount_ready,
        images=images,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def inspect_docker_images(
    docker_executable: str | Path,
    image_names: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> tuple[DockerImageIdentity, ...]:
    """Inspect explicit local image references without pulling or installing.

    This small public wrapper is shared by the environment probe and the
    streamed ``docker load`` worker.  It intentionally performs only local,
    read-only ``docker image inspect`` calls so the frozen application never
    downloads dependencies as a side effect of validation.
    """

    executable = str(docker_executable).strip()
    if not executable:
        raise ValueError("Docker CLI 路径不能为空")
    names = tuple(dict.fromkeys(str(item).strip() for item in image_names if str(item).strip()))
    return tuple(_inspect_image(executable, name, runner) for name in names)


def _inspect_image(executable: str, name: str, runner: Runner) -> DockerImageIdentity:
    try:
        completed = runner(
            [executable, "image", "inspect", name],
            capture_output=True,
            text=True,
            **_DOCKER_PROCESS_KWARGS,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DockerImageIdentity(name, False, error=str(exc))
    if completed.returncode:
        return DockerImageIdentity(
            name,
            False,
            error=completed.stderr.strip() or completed.stdout.strip() or "镜像不存在",
        )
    try:
        payload = json.loads(completed.stdout)
        record = payload[0]
        image_id = str(record.get("Id") or "") or None
        repo_digests = tuple(str(item) for item in record.get("RepoDigests") or ())
    except (json.JSONDecodeError, IndexError, TypeError, AttributeError) as exc:
        return DockerImageIdentity(name, False, error=f"镜像信息无效：{exc}")
    return DockerImageIdentity(name, True, image_id, repo_digests)


def _inspect_wsl2(runner: Runner) -> bool | None:
    if os.name != "nt":
        return None
    try:
        completed = runner(
            ["wsl.exe", "--status"],
            capture_output=True,
            text=True,
            **_DOCKER_PROCESS_KWARGS,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _inspect_mount(
    executable: str,
    image: str,
    runner: Runner,
    *,
    root: str | Path | None,
) -> bool:
    parent = Path(root) if root is not None else Path(tempfile.gettempdir())
    if not str(parent).isascii():
        return False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ai-biaozhu-mount-", dir=parent) as raw:
            probe = Path(raw)
            marker = probe / "probe.txt"
            marker.write_text("mount-ok", encoding="ascii")
            completed = runner(
                [
                    executable,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "sh",
                    "-v",
                    f"{probe.resolve()}:/ai_biaozhu_probe:ro",
                    image,
                    "-c",
                    "test -f /ai_biaozhu_probe/probe.txt",
                ],
                capture_output=True,
                text=True,
                **_DOCKER_PROCESS_KWARGS,
                check=False,
                timeout=30,
            )
            return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _nested_version(payload: Any, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    version = value.get("Version")
    return str(version) if version else None
