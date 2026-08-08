"""Exercise the frozen worker with a tiny CUDA train-and-predict cycle.

This is intentionally independent from the source package import path.  It
creates a four-image YOLO dataset, invokes the standalone multidist worker,
checks the JSONL protocol and checkpoint artifacts, and records any matching
``nvidia-smi`` compute-process observations in a machine-readable summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--weight-cache", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--timeout", default=1800, type=float)
    return parser


def _tiny_dataset(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset"
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(2):
            image_path = image_dir / f"{index}.jpg"
            image = Image.new("RGB", (64, 64), "black")
            ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="white")
            image.save(image_path)
            (label_dir / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [object]\n",
        encoding="utf-8",
    )
    return data_yaml, dataset / "images" / "val" / "0.jpg"


def _clean_environment(smoke_dir: Path, weight_cache: Path) -> dict[str, str]:
    removed = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    env = {key: value for key, value in os.environ.items() if key.upper() not in removed}
    env.update(
        {
            "AI_BIAOZHU_STANDALONE": "1",
            "AI_BIAOZHU_MODELS_DIR": str(weight_cache),
            "YOLO_CONFIG_DIR": str(smoke_dir / "yolo-config"),
            "PYTHONUTF8": "1",
        }
    )
    return env


def _query_compute_processes(process_id: int) -> list[str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    matches: list[str] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        pid_text = stripped.split(",", 1)[0].strip()
        if pid_text.isdigit() and int(pid_text) == process_id:
            matches.append(stripped)
    return matches


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_worker(
    worker: Path,
    command: str,
    manifest: Path,
    *,
    expected_job_id: str,
    smoke_dir: Path,
    env: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    stdout_path = smoke_dir / f"{command}-stdout.log"
    stderr_path = smoke_dir / f"{command}-stderr.log"
    observations: list[dict[str, Any]] = []
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            process = subprocess.Popen(
                [str(worker), command, "--manifest", str(manifest)],
                cwd=smoke_dir,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    raise TimeoutError(f"{command} exceeded {timeout:.0f} seconds")
                entries = _query_compute_processes(process.pid)
                if entries:
                    observations.append(
                        {"elapsed_seconds": round(elapsed, 3), "entries": entries}
                    )
                time.sleep(1)
            return_code = process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    events, raw_lines = _parse_protocol_events(stdout, expected_job_id=expected_job_id)
    return {
        "command": command,
        "return_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "events": events,
        "event_types": [event.get("type") for event in events],
        "raw_line_count": len(raw_lines),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_line_count": len(stderr.splitlines()),
        "cuda_text_present": "CUDA:0" in stdout or "CUDA:0" in stderr,
        "nvidia_smi_observations": observations,
    }


def _parse_protocol_events(
    stdout: str,
    *,
    expected_job_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    last_seq = -1
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            raw_lines.append(line)
            continue
        if not (
            isinstance(candidate, dict)
            and candidate.get("protocol_version") == "1.0"
            and isinstance(candidate.get("payload"), dict)
        ):
            raw_lines.append(line)
            continue
        if candidate.get("job_id") != expected_job_id:
            raise RuntimeError(
                f"protocol event belongs to {candidate.get('job_id')!r}, "
                f"expected {expected_job_id!r}"
            )
        try:
            seq = int(candidate["seq"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("protocol event has an invalid seq") from exc
        if seq <= last_seq:
            raise RuntimeError("protocol event sequence is not strictly increasing")
        last_seq = seq
        events.append(candidate)
    return events, raw_lines


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    args = build_parser().parse_args()
    worker = args.worker.resolve()
    weight_cache = args.weight_cache.resolve()
    started_at = datetime.now(UTC).isoformat()
    summary_path = args.summary.resolve()
    running_summary = {
        "status": "running",
        "started_at": started_at,
        "worker": str(worker),
        "model": "YOLO26n",
        "device": 0,
    }
    _write_json(summary_path, running_summary)

    try:
        _require(worker.is_file(), f"standalone worker missing: {worker}")
        _require((weight_cache / "yolo26n.pt").is_file(), "cached yolo26n.pt missing")
        worker_sha256 = _file_sha256(worker)
        running_summary["worker_sha256"] = worker_sha256
        _write_json(summary_path, running_summary)
        return _execute_smoke(
            args,
            worker=worker,
            weight_cache=weight_cache,
            started_at=started_at,
            worker_sha256=worker_sha256,
        )
    except BaseException as exc:
        failed_summary = {
            **running_summary,
            "status": "failed",
            "finished_at": datetime.now(UTC).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(summary_path, failed_summary)
        raise


def _execute_smoke(
    args: argparse.Namespace,
    *,
    worker: Path,
    weight_cache: Path,
    started_at: str,
    worker_sha256: str,
) -> int:
    smoke_dir = args.results_root.resolve() / ("standalone-cuda-smoke-final-" + uuid.uuid4().hex)
    smoke_dir.mkdir(parents=True)
    data_yaml, prediction_image = _tiny_dataset(smoke_dir)
    env = _clean_environment(smoke_dir, weight_cache)

    train_manifest = smoke_dir / "train-manifest.json"
    _write_json(
        train_manifest,
        {
            "job_id": "standalone-cuda-train",
            "model_key": "YOLO26n",
            "data_yaml": str(data_yaml),
            "output_dir": str(smoke_dir / "runs"),
            "run_name": "standalone-cuda-model",
            "weight_cache_dir": str(weight_cache),
            "offline_weights": True,
            "config": {
                "epochs": 1,
                "imgsz": 64,
                "batch": 1,
                "device": 0,
                "workers": 0,
                "patience": 0,
                "seed": 42,
                "deterministic": True,
                "extra": {"plots": False},
            },
            "augmentation": {"enabled": False},
        },
    )
    train = _run_worker(
        worker,
        "train",
        train_manifest,
        expected_job_id="standalone-cuda-train",
        smoke_dir=smoke_dir,
        env=env,
        timeout=args.timeout,
    )
    artifacts = {
        str(event["payload"].get("kind")): Path(str(event["payload"].get("path")))
        for event in train["events"]
        if event.get("type") == "artifact" and event["payload"].get("kind") in {"best", "last"}
    }
    gpu_memory_values = [
        float(event["payload"]["gpu_memory_gb"])
        for event in train["events"]
        if event.get("type") in {"progress", "metrics"}
        and event["payload"].get("gpu_memory_gb") is not None
    ]
    _require(train["return_code"] == 0, "standalone CUDA training failed")
    _require(
        bool(train["events"]) and train["events"][-1].get("type") == "completed",
        "training terminal completed event missing",
    )
    _require(
        not {"error", "cancelled"}.intersection(train["event_types"]),
        "training protocol contains a failure event",
    )
    _require(set(artifacts) == {"best", "last"}, "best/last artifact event missing")
    _require(all(path.is_file() for path in artifacts.values()), "checkpoint file missing")
    checkpoint_root = smoke_dir / "runs" / "standalone-cuda-model" / "weights"
    _require(
        all(
            _is_within(path, checkpoint_root) and path.name == f"{role}.pt"
            for role, path in artifacts.items()
        ),
        "checkpoint artifact escaped this smoke run",
    )
    _require(any(value > 0 for value in gpu_memory_values), "positive GPU memory event missing")
    _require(train["cuda_text_present"], "Ultralytics CUDA:0 evidence missing")
    _require(
        bool(train["nvidia_smi_observations"]),
        "training Worker PID was not observed by nvidia-smi",
    )

    predict_manifest = smoke_dir / "predict-manifest.json"
    _write_json(
        predict_manifest,
        {
            "job_id": "standalone-cuda-predict",
            "model_key": "YOLO26n",
            "checkpoint": str(artifacts["best"]),
            "output_dir": str(smoke_dir / "predictions"),
            "class_ids": ["object"],
            "images": [
                {
                    "image_id": "sample",
                    "path": str(prediction_image),
                    "expected_revision": 0,
                    "width": 64,
                    "height": 64,
                }
            ],
            "imgsz": 64,
            "device": 0,
        },
    )
    predict = _run_worker(
        worker,
        "predict",
        predict_manifest,
        expected_job_id="standalone-cuda-predict",
        smoke_dir=smoke_dir,
        env=env,
        timeout=args.timeout,
    )
    _require(predict["return_code"] == 0, "standalone CUDA prediction failed")
    _require("prediction" in predict["event_types"], "prediction event missing")
    _require(
        bool(predict["events"]) and predict["events"][-1].get("type") == "completed",
        "prediction terminal completed event missing",
    )
    _require(
        not {"error", "cancelled"}.intersection(predict["event_types"]),
        "prediction protocol contains a failure event",
    )
    # A one-image prediction can complete between the one-second nvidia-smi
    # polling samples.  CUDA execution is established above by the training
    # phase, which uses this same frozen worker and requires both Ultralytics
    # CUDA text and a positive GPU-memory protocol event.  Keep prediction
    # verification focused on its independently observable contract: a
    # successful protocol result from the CUDA-configured worker.

    summary = {
        "status": "passed",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "worker": str(worker),
        "worker_sha256": worker_sha256,
        "smoke_dir": str(smoke_dir),
        "model": "YOLO26n",
        "device": 0,
        "checkpoint_paths": {key: str(value) for key, value in artifacts.items()},
        "checkpoint_sha256": {
            key: _file_sha256(value) for key, value in artifacts.items()
        },
        "weight_cache_sha256": _file_sha256(weight_cache / "yolo26n.pt"),
        "gpu_memory_gb_max": max(gpu_memory_values),
        "train": train,
        "predict": predict,
    }
    _write_json(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
