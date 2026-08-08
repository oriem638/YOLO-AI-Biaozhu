"""Run real YOLO checkpoint export and Maix ONNX gates without Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--model-key", default="YOLO26n")
    parser.add_argument("--timeout", default=1800, type=float)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _calibration_sample(root: Path) -> Path:
    path = root / "calibration-source.jpg"
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="white")
    image.save(path)
    return path


def _parse_events(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    raw: list[str] = []
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            raw.append(line)
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("protocol_version") == "1.0"
            and isinstance(candidate.get("payload"), dict)
        ):
            events.append(candidate)
        else:
            raw.append(line)
    return events, raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    summary_path = args.summary.resolve()
    started_at = datetime.now(UTC).isoformat()
    running = {
        "status": "running",
        "started_at": started_at,
        "checkpoint": str(checkpoint),
        "model_key": args.model_key,
    }
    _write_json(summary_path, running)
    try:
        _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
        running["checkpoint_sha256"] = _file_sha256(checkpoint)
        _write_json(summary_path, running)
        root = args.results_root.resolve() / ("real-onnx-gate-" + uuid4().hex)
        root.mkdir(parents=True)
        running["results_root"] = str(root)
        _write_json(summary_path, running)
        sample = _calibration_sample(root)
        calibration_images = [{"path": str(sample)} for _ in range(20)]
        env = dict(os.environ)
        project_root = Path(__file__).resolve().parents[1]
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(project_root / "src") + (
            os.pathsep + existing_path if existing_path else ""
        )
        env["YOLO_CONFIG_DIR"] = str(root / "yolo-config")
        env["PYTHONUTF8"] = "1"

        targets: dict[str, Any] = {}
        for target in ("maixcam_pro", "maixcam2"):
            target_root = root / target
            manifest_path = root / f"{target}-manifest.json"
            deployment_manifest: dict[str, Any] = {
                "job_id": f"real-onnx-gate-{target}",
                "execute": False,
                "checkpoint": str(checkpoint),
                "checkpoint_kind": "best",
                "source_run_id": "real-onnx-gate-source",
                "model_key": args.model_key,
                "target": target,
                "input_height": 64,
                "input_width": 64,
                "class_names": ["object"],
                "calibration_images": calibration_images,
                "calibration_count": 20,
                "cam2_npu_mode": "both",
                "package_outputs": [],
                "output_dir": str(target_root),
                "audit_dir": str(root / "audits"),
            }
            if args.model_key.casefold().startswith("yolov5"):
                deployment_manifest["legacy_yolov5_repo"] = str(
                    project_root / "third_party" / "runtime" / "yolov5"
                )
            _write_json(
                manifest_path,
                deployment_manifest,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai_biaozhu.workers.main",
                    "deploy",
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
                check=False,
            )
            stdout_path = root / f"{target}-stdout.log"
            stderr_path = root / f"{target}-stderr.log"
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            events, raw_lines = _parse_events(completed.stdout)
            expected_job_id = f"real-onnx-gate-{target}"
            _require(
                all(event.get("job_id") == expected_job_id for event in events),
                f"{target} protocol contains a foreign job_id",
            )
            sequences = [int(event.get("seq", -1)) for event in events]
            _require(
                sequences == sorted(set(sequences)),
                f"{target} protocol sequence is invalid",
            )
            event_types = [event.get("type") for event in events]
            validations = [
                event["payload"].get("validation")
                for event in events
                if event.get("type") == "artifact"
                and event["payload"].get("kind")
                in {"onnx_numeric_gate", "converter_onnx_numeric_gate"}
            ]
            completed_event = (
                events[-1]
                if events and events[-1].get("type") == "completed"
                else None
            )
            _require(completed.returncode == 0, f"{target} gate worker failed")
            _require(completed_event is not None, f"{target} completed event missing")
            _require(
                not {"error", "cancelled"}.intersection(event_types),
                f"{target} protocol contains a failure event",
            )
            _require(
                completed_event["payload"].get("device_validation") == "required",
                f"{target} must remain pending device validation",
            )
            _require(len(validations) == 2, f"{target} numeric gates missing")
            _require(
                all(isinstance(item, dict) and item.get("ok") for item in validations),
                f"{target} numeric gate failed",
            )
            targets[target] = {
                "return_code": completed.returncode,
                "event_types": event_types,
                "resolved_output_nodes": completed_event["payload"].get(
                    "resolved_output_nodes"
                ),
                "numeric_validations": validations,
                "raw_line_count": len(raw_lines),
                "stderr_line_count": len(completed.stderr.splitlines()),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "audit_path": completed_event["payload"].get(
                    "conversion_audit_path"
                ),
                "device_validation": completed_event["payload"].get(
                    "device_validation"
                ),
            }

        summary = {
            **running,
            "status": "passed",
            "finished_at": datetime.now(UTC).isoformat(),
            "results_root": str(root),
            "targets": targets,
        }
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        failed = {
            **running,
            "status": "failed",
            "finished_at": datetime.now(UTC).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(summary_path, failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
