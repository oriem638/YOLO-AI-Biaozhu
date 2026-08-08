"""Allowlisted, checksum-locked pretrained weight cache.

Downloads use one stable partial file per weight.  A short/failed transfer is
therefore resumable, while a complete file is not published until its byte
count (when known) and SHA-256 both match the lock record.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

import psutil

from .model_registry import MODEL_REGISTRY, get_model

ProgressCallback = Callable[[int, int | None], None]
CancelCheck = Callable[[], None]
Request = str | urllib.request.Request
Opener = Callable[[Request], AbstractContextManager[BinaryIO]]
Sleeper = Callable[[float], None]


class WeightIntegrityError(RuntimeError):
    pass


class WeightUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WeightRecord:
    model_key: str
    filename: str
    url: str
    sha256: str
    size: int | None = None


def default_lock_path() -> Path:
    return Path(__file__).with_name("weights.lock.json")


def default_models_dir() -> Path:
    configured = os.environ.get("AI_BIAOZHU_MODELS_DIR")
    if configured:
        return Path(configured)
    from ai_biaozhu.app_paths import AppPaths

    return AppPaths.discover().models


def default_seed_dirs() -> tuple[Path, ...]:
    """Return trusted *candidate* directories; every seed is still hashed."""

    values: list[Path] = []
    configured = os.environ.get("AI_BIAOZHU_MODEL_SEED_DIR")
    if configured:
        values.append(Path(configured))
    values.append(Path(sys.executable).resolve().parent / "model-seed")
    # Helpful for editable/source execution and deterministic packaging tests.
    values.append(Path(__file__).resolve().parents[3] / "bundled_models")
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = os.path.normcase(os.path.abspath(value))
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def load_weight_lock(path: str | Path | None = None) -> Mapping[str, WeightRecord]:
    lock_path = Path(path or default_lock_path())
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightIntegrityError(f"无法读取权重锁文件：{lock_path}: {exc}") from exc
    values = raw.get("weights") if isinstance(raw, Mapping) else None
    if not isinstance(values, Mapping):
        raise WeightIntegrityError("weights.lock.json 缺少 weights 对象")
    if set(values) != set(MODEL_REGISTRY):
        raise WeightIntegrityError("权重锁必须且只能包含注册表中的 8 个模型")
    records: dict[str, WeightRecord] = {}
    for model_key, value in values.items():
        if not isinstance(value, Mapping):
            raise WeightIntegrityError(f"{model_key} 权重锁记录无效")
        spec = get_model(model_key)
        filename = str(value.get("filename", ""))
        url = str(value.get("url", ""))
        digest = str(value.get("sha256", "")).casefold()
        if filename != spec.weight:
            raise WeightIntegrityError(
                f"{model_key} 锁定文件名 {filename!r} 与注册表 {spec.weight!r} 不一致"
            )
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise WeightIntegrityError(f"{model_key} 缺少固定 SHA-256，禁止首次信任下载")
        if not url.startswith("https://github.com/ultralytics/"):
            raise WeightIntegrityError(f"{model_key} 权重 URL 不在官方 allowlist")
        raw_size = value.get("size")
        try:
            size = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError) as exc:
            raise WeightIntegrityError(f"{model_key} 权重字节数无效") from exc
        if size is not None and size <= 0:
            raise WeightIntegrityError(f"{model_key} 权重字节数必须大于 0")
        records[model_key] = WeightRecord(
            model_key,
            filename,
            url,
            digest,
            size,
        )
    return records


class WeightManager:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        lock_path: str | Path | None = None,
        opener: Opener | None = None,
        seed_dirs: Sequence[str | Path] | None = None,
        retry_attempts: int = 3,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if retry_attempts < 1:
            raise ValueError("retry_attempts 必须大于 0")
        self.cache_dir = Path(cache_dir or default_models_dir())
        self.records = load_weight_lock(lock_path)
        self.opener = opener or _open_url
        self.seed_dirs = tuple(
            Path(value) for value in (seed_dirs if seed_dirs is not None else default_seed_dirs())
        )
        self.retry_attempts = retry_attempts
        self.sleeper = sleeper

    def ensure(
        self,
        model_key: str,
        *,
        offline: bool = False,
        progress: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Path:
        spec = get_model(model_key)
        record = self.records[spec.key]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / record.filename
        lock = destination.with_name(f".{destination.name}.download.lock")
        with _exclusive_download_lock(lock):
            cached = self._verified_cached(record, destination, progress)
            if cached is not None:
                return cached
            seeded = self._seed_verified(record, destination, progress, cancel_check)
            if seeded is not None:
                return seeded
            if offline:
                raise WeightUnavailableError(f"离线模式下未缓存 {record.filename}")
            self._clean_legacy_parts(destination)
            return self._download(
                record,
                destination,
                progress=progress,
                cancel_check=cancel_check,
            )

    def _verified_cached(
        self,
        record: WeightRecord,
        destination: Path,
        progress: ProgressCallback | None,
    ) -> Path | None:
        if not destination.is_file():
            return None
        actual = sha256_file(destination)
        size = destination.stat().st_size
        if actual == record.sha256 and (record.size is None or size == record.size):
            if progress is not None:
                progress(size, record.size or size)
            return destination
        quarantine = destination.with_name(f"{destination.name}.corrupt-{actual[:12]}")
        os.replace(destination, quarantine)
        return None

    def _seed_verified(
        self,
        record: WeightRecord,
        destination: Path,
        progress: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Path | None:
        for directory in self.seed_dirs:
            seed = directory / record.filename
            if not seed.is_file() or seed.resolve() == destination.resolve():
                continue
            _check_cancel(cancel_check)
            size = seed.stat().st_size
            if record.size is not None and size != record.size:
                continue
            if sha256_file(seed) != record.sha256:
                continue
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid4().hex}.seed"
            )
            try:
                with seed.open("rb") as source, temporary.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        _check_cancel(cancel_check)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                # Protect against a seed being replaced while it was copied.
                if sha256_file(temporary) != record.sha256:
                    raise WeightIntegrityError(
                        f"内置 {record.filename} 在复制期间发生变化"
                    )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            if progress is not None:
                progress(size, record.size or size)
            return destination
        return None

    def _download(
        self,
        record: WeightRecord,
        destination: Path,
        *,
        progress: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> Path:
        part = destination.with_name(f".{destination.name}.part")
        metadata_path = destination.with_name(f".{destination.name}.part.json")
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            _check_cancel(cancel_check)
            try:
                metadata = _load_partial_metadata(metadata_path)
                current = part.stat().st_size if part.is_file() else 0
                if not _partial_is_compatible(record, current, metadata):
                    _discard_partial(part, metadata_path)
                    current = 0
                    metadata = {}
                if current and record.size is not None and current == record.size:
                    if sha256_file(part) == record.sha256:
                        os.replace(part, destination)
                        metadata_path.unlink(missing_ok=True)
                        return destination
                    _discard_partial(part, metadata_path)
                    current = 0
                    metadata = {}

                headers: dict[str, str] = {}
                if current:
                    headers["Range"] = f"bytes={current}-"
                    etag = str(metadata.get("etag") or "")
                    if etag:
                        headers["If-Range"] = etag
                request = urllib.request.Request(record.url, headers=headers)
                with self.opener(request) as response:
                    status = _response_status(response)
                    response_etag = _response_header(response, "ETag")
                    content_range = _parse_content_range(
                        _response_header(response, "Content-Range")
                    )
                    if current and not (
                        status == 206
                        and content_range is not None
                        and content_range[0] == current
                        and (
                            not metadata.get("etag")
                            or not response_etag
                            or metadata.get("etag") == response_etag
                        )
                    ):
                        # Server ignored Range or the remote object changed.
                        _discard_partial(part, metadata_path)
                        current = 0
                        metadata = {}

                    response_length = _content_length(response)
                    advertised_total = (
                        content_range[2]
                        if current and content_range is not None
                        else response_length
                    )
                    expected_total = record.size or advertised_total
                    if (
                        record.size is not None
                        and advertised_total is not None
                        and advertised_total != record.size
                    ):
                        raise WeightIntegrityError(
                            f"{record.filename} 服务端字节数 {advertised_total} "
                            f"与锁定值 {record.size} 不一致"
                        )
                    _write_partial_metadata(
                        metadata_path,
                        {
                            "url": record.url,
                            "sha256": record.sha256,
                            "total": expected_total,
                            "etag": response_etag or metadata.get("etag"),
                        },
                    )
                    mode = "ab" if current else "wb"
                    downloaded = current
                    if progress is not None:
                        progress(downloaded, expected_total)
                    with part.open(mode) as handle:
                        while chunk := response.read(1024 * 1024):
                            _check_cancel(cancel_check)
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if expected_total is not None and downloaded > expected_total:
                                raise WeightIntegrityError(
                                    f"{record.filename} 下载字节数超过预期"
                                )
                            if progress is not None:
                                progress(downloaded, expected_total)
                        handle.flush()
                        os.fsync(handle.fileno())
                    _check_cancel(cancel_check)
                    if expected_total is not None and downloaded != expected_total:
                        raise WeightUnavailableError(
                            f"{record.filename} 下载中断：{downloaded}/{expected_total} 字节"
                        )
                actual = sha256_file(part)
                if actual != record.sha256:
                    _discard_partial(part, metadata_path)
                    raise WeightIntegrityError(
                        f"{record.filename} SHA-256 不匹配：{actual}"
                    )
                os.replace(part, destination)
                metadata_path.unlink(missing_ok=True)
                return destination
            except Exception as exc:
                # Cancellation callbacks deliberately raise the adapter's own
                # JobCancelled.  Preserve the valid partial and propagate it.
                if exc.__class__.__name__ == "JobCancelled":
                    raise
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                _check_cancel(cancel_check)
                self.sleeper(0.25 * attempt)
        assert last_error is not None
        raise last_error

    def _clean_legacy_parts(self, destination: Path) -> None:
        # 0.2.1 used random ``.<name>.<pid>.<uuid>.part`` names which cannot be
        # resumed and can survive a killed process.  The new stable part is not
        # matched because it contains no extra component before ``.part``.
        for candidate in destination.parent.glob(f".{destination.name}.*.part"):
            candidate.unlink(missing_ok=True)

    def verify(self, model_key: str) -> bool:
        spec = get_model(model_key)
        record = self.records[spec.key]
        path = self.cache_dir / record.filename
        return (
            path.is_file()
            and (record.size is None or path.stat().st_size == record.size)
            and sha256_file(path) == record.sha256
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_url(request: Request) -> AbstractContextManager[BinaryIO]:
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310 - strict allowlist


def _content_length(response: Any) -> int | None:
    value = _response_header(response, "Content-Length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _response_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getter = getattr(response, "getcode", None)
        status = getter() if callable(getter) else None
    try:
        return int(status) if status is not None else 200
    except (TypeError, ValueError):
        return 200


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value or not value.casefold().startswith("bytes ") or "/" not in value:
        return None
    interval, total_text = value[6:].split("/", 1)
    if "-" not in interval or total_text == "*":
        return None
    start_text, end_text = interval.split("-", 1)
    try:
        start, end, total = int(start_text), int(end_text), int(total_text)
    except ValueError:
        return None
    if start < 0 or end < start or total <= end:
        return None
    return start, end, total


def _partial_is_compatible(
    record: WeightRecord,
    current: int,
    metadata: Mapping[str, Any],
) -> bool:
    if current <= 0:
        return not metadata or metadata.get("url") in {None, record.url}
    if metadata.get("url") != record.url or metadata.get("sha256") != record.sha256:
        return False
    raw_total = metadata.get("total")
    try:
        total = int(raw_total) if raw_total is not None else record.size
    except (TypeError, ValueError):
        return False
    if record.size is not None and total not in {None, record.size}:
        return False
    return total is None or current <= total


def _load_partial_metadata(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _write_partial_metadata(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _discard_partial(part: Path, metadata: Path) -> None:
    part.unlink(missing_ok=True)
    metadata.unlink(missing_ok=True)


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()


@contextmanager
def _exclusive_download_lock(path: Path):
    descriptor: int | None = None
    for _attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if not _download_lock_is_stale(path):
                raise WeightUnavailableError(
                    "另一任务正在准备 "
                    f"{path.name.removeprefix('.').removesuffix('.download.lock')}"
                ) from exc
            with suppress(FileNotFoundError):
                path.unlink()
    if descriptor is None:
        raise WeightUnavailableError(
            f"无法获取模型准备锁：{path.name}"
        )
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _download_lock_is_stale(path: Path) -> bool:
    """Return true only when an old lock can safely be reclaimed.

    A newly created lock may be observed before its owner has written the PID,
    so malformed/empty locks receive a short grace period.  A syntactically
    valid PID can be reclaimed immediately once that process no longer exists.
    """

    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return False
    try:
        owner_pid = int(value)
    except ValueError:
        try:
            return time.time() - path.stat().st_mtime > 300
        except OSError:
            return False
    if owner_pid <= 0:
        return True
    return not psutil.pid_exists(owner_pid)
