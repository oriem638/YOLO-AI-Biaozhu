"""Run each supported model's CUDA smoke test in an isolated pytest process.

The runner intentionally executes models sequentially. Each subprocess has its
own environment, temporary directory, and durable log so a failure cannot hide
the remaining matrix results and the final report remains auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODELS = (
    "YOLOv5n",
    "YOLOv5s",
    "YOLOv8n",
    "YOLOv8s",
    "YOLO11n",
    "YOLO11s",
    "YOLO26n",
    "YOLO26s",
)


def _deduplicated_windows_environment() -> dict[str, str]:
    """Return an environment without case-only duplicate variable names."""

    environment: dict[str, str] = {}
    spellings: dict[str, str] = {}
    for key, value in os.environ.items():
        normalized = key.casefold()
        previous = spellings.get(normalized)
        if previous is not None:
            environment.pop(previous, None)
        environment[key] = value
        spellings[normalized] = key
    return environment


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the Conda yolo environment.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result directory; defaults to build/test-results/gpu-matrix-<UTC>.",
    )
    parser.add_argument(
        "--augmentation-model",
        choices=MODELS[2:],
        help="Also run the real rotation/blur smoke test for this modern model.",
    )
    return parser.parse_args()


def _run_case(
    *,
    root: Path,
    python: Path,
    output: Path,
    name: str,
    environment_name: str,
    environment_value: str,
    test_node: str,
) -> dict[str, object]:
    case_dir = output / name
    case_dir.mkdir(parents=True, exist_ok=False)
    log_path = case_dir / "pytest.log"
    config_dir = output / "ultralytics-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    # Windows accepts PATH and Path in an inherited block, but constructing a
    # new block containing both can make Conda's python fail during DLL init.
    environment = _deduplicated_windows_environment()
    environment.update(
        {
            "PYTHONPATH": str(root / "src"),
            "YOLO_CONFIG_DIR": str(config_dir),
            "AI_BIAOZHU_MODELS_DIR": str(root / "build" / "model-cache"),
            "AI_BIAOZHU_YOLOV5_REPO": str(
                root / "third_party" / "runtime" / "yolov5"
            ),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            environment_name: environment_value,
        }
    )
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-s",
        "-m",
        "gpu",
        test_node,
        "--basetemp",
        str(case_dir / "pytest-tmp"),
    ]
    started_at = datetime.now(UTC)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"started_at={started_at.isoformat()}\n")
        log.write(f"command={subprocess.list2cmdline(command)}\n")
        log.write(f"YOLO_CONFIG_DIR={environment['YOLO_CONFIG_DIR']}\n")
        log.write(f"environment_path_keys={','.join(key for key in environment if key.casefold() == 'path')}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"pid={process.pid}\n")
        log.flush()
        return_code = process.wait()
        duration = time.monotonic() - started
        log.write(f"\nreturn_code={return_code}\n")
        log.write(f"duration_seconds={duration:.3f}\n")
    result = {
        "name": name,
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "duration_seconds": round(duration, 3),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "log_path": str(log_path.resolve()),
        "artifact_root": str((case_dir / "pytest-tmp").resolve()),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    args = _arguments()
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or root / "build" / "test-results" / f"gpu-matrix-{timestamp}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    python = Path(args.python).resolve()
    if not python.is_file():
        raise SystemExit(f"Python executable does not exist: {python}")

    results = [
        _run_case(
            root=root,
            python=python,
            output=output,
            name=model,
            environment_name="AI_BIAOZHU_GPU_SMOKE",
            environment_value=model,
            test_node=(
                "tests/test_ml_gpu_smoke.py::"
                "test_model_load_one_epoch_predict_and_artifact"
            ),
        )
        for model in args.models
    ]
    if args.augmentation_model:
        results.append(
            _run_case(
                root=root,
                python=python,
                output=output,
                name=f"{args.augmentation_model}-augmentation",
                environment_name="AI_BIAOZHU_MODERN_AUG_SMOKE",
                environment_value=args.augmentation_model,
                test_node=(
                    "tests/test_ml_gpu_smoke.py::"
                    "test_modern_rotation_and_blur_are_applied_and_persisted"
                ),
            )
        )

    summary = {
        "schema_version": 1,
        "python": str(python),
        "root": str(root),
        "gpu_matrix": results,
        "passed": sum(result["status"] == "passed" for result in results),
        "failed": sum(result["status"] != "passed" for result in results),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary={summary_path}", flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
