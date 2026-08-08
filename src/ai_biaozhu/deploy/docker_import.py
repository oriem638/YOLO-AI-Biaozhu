"""Cancellable, streaming import of an offline Docker image archive."""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .environment import DockerImageIdentity, Runner, inspect_docker_images

ProgressCallback = Callable[["DockerImageImportProgress"], None]
CancelCheck = Callable[[], bool]
PopenFactory = Callable[..., Any]


class DockerImageImportError(RuntimeError):
    """Raised when ``docker load`` or the post-load identity check fails."""


class DockerImageImportCancelled(DockerImageImportError):
    """Raised after the Docker CLI child has been stopped on user cancellation."""


@dataclass(frozen=True, slots=True)
class DockerImageImportProgress:
    stage: str
    bytes_read: int
    total_bytes: int
    percent: float
    elapsed_seconds: float
    bytes_per_second: float
    heartbeat: bool
    completed: bool = False
    diagnostic: bool = False
    stalled_seconds: float = 0.0
    diagnostic_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DockerImageImportResult:
    archive_path: Path
    bytes_read: int
    total_bytes: int
    elapsed_seconds: float
    image_tags: tuple[str, ...]
    loaded_image_ids: tuple[str, ...]
    images: tuple[DockerImageIdentity, ...]
    docker_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": str(self.archive_path),
            "bytes_read": self.bytes_read,
            "total_bytes": self.total_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "image_tags": list(self.image_tags),
            "loaded_image_ids": list(self.loaded_image_ids),
            "images": [
                {
                    **asdict(image),
                    "repo_digests": list(image.repo_digests),
                    "status": image.status,
                }
                for image in self.images
            ],
            "docker_output": self.docker_output,
        }


def import_docker_image_archive(
    archive_path: str | Path,
    docker_executable: str | Path = "docker",
    *,
    expected_images: Sequence[str] = (),
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
    inspect_runner: Runner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    chunk_size: int = 1024 * 1024,
    heartbeat_interval: float = 1.0,
    poll_interval: float = 0.05,
    stall_diagnostic_after: float = 120.0,
) -> DockerImageImportResult:
    """Pipe an archive into ``docker load`` while reporting byte progress.

    A writer thread owns the potentially blocking pipe write.  The calling
    thread remains free to emit a heartbeat at least once per configured
    interval and to terminate ``docker load`` promptly when cancellation is
    requested.  No package installation, image pull or network access occurs.
    """

    archive = _validate_archive(archive_path)
    executable = str(docker_executable).strip()
    if not executable:
        raise ValueError("Docker CLI 路径不能为空")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if heartbeat_interval <= 0 or poll_interval <= 0 or stall_diagnostic_after <= 0:
        raise ValueError(
            "heartbeat_interval、poll_interval 和 stall_diagnostic_after 必须大于 0"
        )

    total_bytes = archive.stat().st_size
    started_at = monotonic()
    state_lock = threading.Lock()
    bytes_read = 0
    writer_error: BaseException | None = None
    cancel_event = threading.Event()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    try:
        process = popen_factory(
            [executable, "load"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DockerImageImportError(f"无法启动 docker load：{exc}") from exc
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise DockerImageImportError("docker load 未提供完整的标准输入输出管道")

    def read_output(pipe: BinaryIO, destination: list[bytes]) -> None:
        try:
            while chunk := pipe.read(64 * 1024):
                destination.append(bytes(chunk))
        except (OSError, ValueError):
            return

    def write_archive() -> None:
        nonlocal bytes_read, writer_error
        try:
            with archive.open("rb") as source:
                while not cancel_event.is_set():
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view and not cancel_event.is_set():
                        written = process.stdin.write(view)
                        if written is None:
                            written = len(view)
                        if written <= 0:
                            raise BrokenPipeError("docker load 标准输入已关闭")
                        view = view[written:]
                        with state_lock:
                            bytes_read += written
        except BaseException as exc:  # propagate worker-thread failures to caller
            if not cancel_event.is_set():
                writer_error = exc
        finally:
            with contextlib.suppress(OSError, ValueError):
                process.stdin.close()

    stdout_thread = threading.Thread(
        target=read_output,
        args=(process.stdout, stdout_chunks),
        name="docker-load-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_output,
        args=(process.stderr, stderr_chunks),
        name="docker-load-stderr",
        daemon=True,
    )
    writer_thread = threading.Thread(
        target=write_archive,
        name="docker-load-stdin",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    writer_thread.start()

    last_reported = -1
    last_heartbeat = started_at
    last_progress_at = started_at
    last_diagnostic_at = started_at
    cancelled = False

    def emit_progress(
        *,
        heartbeat: bool,
        completed: bool = False,
        diagnostic: bool = False,
        stalled_seconds: float = 0.0,
    ) -> None:
        if progress_callback is None:
            return
        with state_lock:
            current = bytes_read
        elapsed = max(0.0, monotonic() - started_at)
        percent = min(100.0, current * 100.0 / total_bytes)
        progress_callback(
            DockerImageImportProgress(
                stage="docker_image_import",
                bytes_read=current,
                total_bytes=total_bytes,
                percent=percent,
                elapsed_seconds=elapsed,
                bytes_per_second=current / elapsed if elapsed > 0 else 0.0,
                heartbeat=heartbeat,
                completed=completed,
                diagnostic=diagnostic,
                stalled_seconds=stalled_seconds,
                diagnostic_message=(
                    "docker load 长时间没有新的归档字节进度；进程仍在运行。"
                    "可继续等待或取消，并检查 Docker Desktop 资源与下方输出。"
                    if diagnostic
                    else ""
                ),
            )
        )

    try:
        emit_progress(heartbeat=False)
        while writer_thread.is_alive() or process.poll() is None:
            if cancel_check is not None and cancel_check():
                cancelled = True
                cancel_event.set()
                _terminate_process(process)
                break
            if writer_error is not None:
                cancel_event.set()
                _terminate_process(process)
                break
            now = monotonic()
            with state_lock:
                current = bytes_read
            if current != last_reported:
                emit_progress(heartbeat=False)
                last_reported = current
                last_progress_at = now
            if now - last_heartbeat >= heartbeat_interval:
                emit_progress(heartbeat=True)
                last_heartbeat = now
            stalled_seconds = max(0.0, now - last_progress_at)
            if (
                stalled_seconds >= stall_diagnostic_after
                and now - last_diagnostic_at >= stall_diagnostic_after
            ):
                emit_progress(
                    heartbeat=True,
                    diagnostic=True,
                    stalled_seconds=stalled_seconds,
                )
                last_diagnostic_at = now
            sleep(poll_interval)
        writer_thread.join(timeout=5.0)
        if process.poll() is None:
            process.wait(timeout=30)
    except BaseException:
        cancel_event.set()
        _terminate_process(process)
        raise
    finally:
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

    if cancelled:
        raise DockerImageImportCancelled("Docker 镜像导入已由用户取消")
    if writer_error is not None:
        raise DockerImageImportError(f"读取或传输镜像归档失败：{writer_error}") from writer_error

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    output = "\n".join(item.strip() for item in (stdout, stderr) if item.strip())
    return_code = int(process.returncode or 0)
    if return_code != 0:
        raise DockerImageImportError(
            f"docker load 失败（退出码 {return_code}）：{output or '无诊断输出'}"
        )

    tags, image_ids = _parse_loaded_images(output)
    expected = tuple(dict.fromkeys(str(item).strip() for item in expected_images if str(item).strip()))
    inspect_targets = tuple(dict.fromkeys((*expected, *tags, *image_ids)))
    images = inspect_docker_images(executable, inspect_targets, runner=inspect_runner)
    by_name = {image.name: image for image in images}
    missing_expected = [name for name in expected if not by_name.get(name) or not by_name[name].available]
    if missing_expected:
        raise DockerImageImportError(
            "镜像归档已加载，但未找到预期镜像：" + ", ".join(missing_expected)
        )
    emit_progress(heartbeat=False, completed=True)
    with state_lock:
        final_bytes = bytes_read
    return DockerImageImportResult(
        archive_path=archive,
        bytes_read=final_bytes,
        total_bytes=total_bytes,
        elapsed_seconds=max(0.0, monotonic() - started_at),
        image_tags=tags,
        loaded_image_ids=image_ids,
        images=images,
        docker_output=output,
    )


def _validate_archive(path: str | Path) -> Path:
    archive = Path(path).expanduser()
    if not archive.is_file():
        raise FileNotFoundError(f"Docker 镜像归档不存在：{archive}")
    if archive.is_symlink():
        raise ValueError(f"拒绝从符号链接导入 Docker 镜像：{archive}")
    attributes = int(getattr(archive.lstat(), "st_file_attributes", 0))
    if attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
        raise ValueError(f"拒绝从 reparse point 导入 Docker 镜像：{archive}")
    if archive.stat().st_size <= 0:
        raise ValueError(f"Docker 镜像归档为空：{archive}")
    return archive.resolve()


def _parse_loaded_images(output: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tags: list[str] = []
    image_ids: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Loaded image: "):
            tags.append(line.removeprefix("Loaded image: ").strip())
        elif line.startswith("Loaded image ID: "):
            image_ids.append(line.removeprefix("Loaded image ID: ").strip())
    return tuple(dict.fromkeys(tags)), tuple(dict.fromkeys(image_ids))


def _terminate_process(process: Any) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=3.0)
    except (subprocess.TimeoutExpired, TimeoutError):
        with contextlib.suppress(OSError, ProcessLookupError):
            process.kill()
        with contextlib.suppress(OSError, ProcessLookupError, subprocess.TimeoutExpired):
            process.wait(timeout=3.0)
