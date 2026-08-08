from __future__ import annotations

import hashlib
import io
import json
from urllib.request import Request

import pytest

from ai_biaozhu.ml.model_registry import MODEL_REGISTRY
from ai_biaozhu.ml.weights import (
    WeightIntegrityError,
    WeightManager,
    load_weight_lock,
)


class Response(io.BytesIO):
    def __init__(
        self,
        value: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(value)
        self.status = status
        self.headers = {"Content-Length": str(len(value)), **(headers or {})}


def _lock(
    tmp_path,
    payload: bytes,
    *,
    bad_hash: bool = False,
    lock_size: bool = False,
):
    digest = hashlib.sha256(payload).hexdigest()
    values = {}
    for key, spec in MODEL_REGISTRY.items():
        values[key] = {
            "filename": spec.weight,
            "url": (
                "https://github.com/ultralytics/assets/releases/download/"
                f"test/{spec.weight}"
            ),
            "sha256": "0" * 64 if bad_hash else digest,
        }
        if lock_size:
            values[key]["size"] = len(payload)
    path = tmp_path / "weights.lock.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "weights": values}),
        encoding="utf-8",
    )
    return path


def test_download_is_sha_verified_atomic_and_cached(tmp_path) -> None:
    payload = b"official-weight-bytes"
    lock = _lock(tmp_path, payload)
    calls = []

    def opener(url):
        calls.append(url)
        return Response(payload)

    progress = []
    manager = WeightManager(
        tmp_path / "cache",
        lock_path=lock,
        opener=opener,
    )
    path = manager.ensure(
        "YOLO26n",
        progress=lambda current, total: progress.append((current, total)),
    )
    assert path.read_bytes() == payload
    assert not path.with_name(path.name + ".part").exists()
    assert manager.ensure("YOLO26n") == path
    assert len(calls) == 1
    assert progress[-1] == (len(payload), len(payload))


def test_corrupt_cache_is_quarantined_before_redownload(tmp_path) -> None:
    payload = b"correct"
    manager = WeightManager(
        tmp_path / "cache",
        lock_path=_lock(tmp_path, payload),
        opener=lambda _: Response(payload),
    )
    destination = manager.cache_dir / "yolov8n.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt")
    assert manager.ensure("YOLOv8n").read_bytes() == payload
    assert list(manager.cache_dir.glob("yolov8n.pt.corrupt-*"))


def test_hash_mismatch_never_publishes_part_file(tmp_path) -> None:
    expected = b"expected"
    manager = WeightManager(
        tmp_path / "cache",
        lock_path=_lock(tmp_path, expected),
        opener=lambda _: Response(b"tampered"),
    )
    with pytest.raises(WeightIntegrityError, match="SHA-256"):
        manager.ensure("YOLO11n")
    assert not (manager.cache_dir / "yolo11n.pt").exists()
    assert not (manager.cache_dir / "yolo11n.pt.part").exists()


def test_lock_without_fixed_hash_is_rejected(tmp_path) -> None:
    path = _lock(tmp_path, b"x")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["weights"]["YOLO26n"]["sha256"] = ""
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WeightIntegrityError, match="固定 SHA-256"):
        load_weight_lock(path)


def test_verified_bundled_seed_is_atomically_copied_without_network(tmp_path) -> None:
    payload = b"bundled-official-weight"
    seed_dir = tmp_path / "model-seed"
    seed_dir.mkdir()
    (seed_dir / "yolo26s.pt").write_bytes(payload)

    def no_network(_request: Request):
        raise AssertionError("verified seed must avoid the network")

    manager = WeightManager(
        tmp_path / "cache",
        lock_path=_lock(tmp_path, payload, lock_size=True),
        opener=no_network,
        seed_dirs=[seed_dir],
    )
    result = manager.ensure("YOLO26s", offline=True)
    assert result.read_bytes() == payload
    assert manager.verify("YOLO26s")


def test_truncated_download_retries_with_http_range_and_resumes(tmp_path) -> None:
    payload = b"0123456789abcdef"
    requests: list[Request] = []

    def opener(request: Request):
        requests.append(request)
        range_header = request.get_header("Range")
        if not range_header:
            return Response(
                payload[:6],
                headers={"Content-Length": str(len(payload)), "ETag": '"v1"'},
            )
        assert range_header == "bytes=6-"
        assert request.get_header("If-range") == '"v1"'
        return Response(
            payload[6:],
            status=206,
            headers={
                "Content-Range": f"bytes 6-{len(payload) - 1}/{len(payload)}",
                "ETag": '"v1"',
            },
        )

    manager = WeightManager(
        tmp_path / "cache",
        lock_path=_lock(tmp_path, payload, lock_size=True),
        opener=opener,
        seed_dirs=[],
        sleeper=lambda _seconds: None,
    )
    result = manager.ensure("YOLO26s")
    assert result.read_bytes() == payload
    assert len(requests) == 2
    assert not (manager.cache_dir / ".yolo26s.pt.part").exists()


def test_server_ignoring_range_restarts_from_zero(tmp_path) -> None:
    payload = b"complete-official-weight"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".yolo26s.pt.part").write_bytes(payload[:4])
    (cache / ".yolo26s.pt.part.json").write_text(
        json.dumps(
            {
                "url": "https://github.com/ultralytics/assets/releases/download/test/yolo26s.pt",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "total": len(payload),
            }
        ),
        encoding="utf-8",
    )
    seen_range: list[str | None] = []

    def opener(request: Request):
        seen_range.append(request.get_header("Range"))
        return Response(payload, status=200)

    manager = WeightManager(
        cache,
        lock_path=_lock(tmp_path, payload, lock_size=True),
        opener=opener,
        seed_dirs=[],
        sleeper=lambda _seconds: None,
    )
    assert manager.ensure("YOLO26s").read_bytes() == payload
    assert seen_range == ["bytes=4-"]


def test_cancel_preserves_valid_partial_for_future_resume(tmp_path) -> None:
    payload = b"0123456789abcdef"

    class JobCancelled(RuntimeError):
        pass

    manager = WeightManager(
        tmp_path / "cache",
        lock_path=_lock(tmp_path, payload, lock_size=True),
        opener=lambda _request: Response(payload),
        seed_dirs=[],
        sleeper=lambda _seconds: None,
    )
    cancelled = False

    def progress(current: int, _total: int | None) -> None:
        nonlocal cancelled
        cancelled = current > 0

    def cancel_check() -> None:
        if cancelled:
            raise JobCancelled("cancel")

    with pytest.raises(JobCancelled):
        manager.ensure(
            "YOLO26s",
            progress=progress,
            cancel_check=cancel_check,
        )
    assert (manager.cache_dir / ".yolo26s.pt.part").is_file()
    assert not (manager.cache_dir / ".yolo26s.pt.download.lock").exists()


def test_wrong_advertised_size_retries_then_fails_without_publishing(tmp_path) -> None:
    payload = b"correct-size"
    calls = 0

    def opener(_request: Request):
        nonlocal calls
        calls += 1
        return Response(payload, headers={"Content-Length": str(len(payload) + 1)})

    manager = WeightManager(
        tmp_path / "cache",
        lock_path=_lock(tmp_path, payload, lock_size=True),
        opener=opener,
        seed_dirs=[],
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(WeightIntegrityError, match="字节数"):
        manager.ensure("YOLO26s")
    assert calls == 3
    assert not (manager.cache_dir / "yolo26s.pt").exists()


def test_legacy_random_part_is_removed_before_new_download(tmp_path) -> None:
    payload = b"official"
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy = cache / ".yolo26s.pt.123.deadbeef.part"
    legacy.write_bytes(b"orphan")
    manager = WeightManager(
        cache,
        lock_path=_lock(tmp_path, payload),
        opener=lambda _request: Response(payload),
        seed_dirs=[],
    )
    manager.ensure("YOLO26s")
    assert not legacy.exists()


def test_stale_download_lock_is_reclaimed(tmp_path) -> None:
    payload = b"official"
    cache = tmp_path / "cache"
    cache.mkdir()
    lock = cache / ".yolo26s.pt.download.lock"
    lock.write_text("99999999", encoding="ascii")
    manager = WeightManager(
        cache,
        lock_path=_lock(tmp_path, payload),
        opener=lambda _request: Response(payload),
        seed_dirs=[],
    )
    assert manager.ensure("YOLO26s").read_bytes() == payload
    assert not lock.exists()


def test_live_download_lock_is_not_stolen(tmp_path) -> None:
    import os

    payload = b"official"
    cache = tmp_path / "cache"
    cache.mkdir()
    lock = cache / ".yolo26s.pt.download.lock"
    lock.write_text(str(os.getpid()), encoding="ascii")
    manager = WeightManager(
        cache,
        lock_path=_lock(tmp_path, payload),
        opener=lambda _request: Response(payload),
        seed_dirs=[],
    )
    with pytest.raises(Exception, match="另一任务"):
        manager.ensure("YOLO26s")
    assert lock.exists()
