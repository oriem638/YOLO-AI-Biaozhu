"""Machine-readable deployment report generation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .onnx_gate import file_sha256

DEPLOYMENT_REPORT_SCHEMA = "1.0"


def artifact_record(path: str | Path, archive_name: str) -> dict[str, Any]:
    artifact = Path(path)
    return {
        "path": archive_name,
        "size": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }


def deployment_report(
    *,
    target: str,
    model_key: str,
    artifacts: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    source_run_id: str | None = None,
    checkpoint_role: str | None = None,
    source_checkpoint: str | Path | None = None,
    source_onnx: str | Path | None = None,
    class_names: Sequence[str] = (),
    input_shape: Sequence[int] | None = None,
    opset: int = 17,
    output_tensors: Sequence[str] = (),
    quantization: str = "INT8",
    calibration_count: int | None = None,
    converter_image: str | None = None,
    converter_config: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, Any] | None = None,
    maixpy_min_version: str | None = None,
    maixpy_version: str | None = None,
    maixcdk_commit: str | None = None,
    package_files: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if source_run_id:
        source["run_id"] = source_run_id
    if checkpoint_role:
        source["checkpoint_role"] = checkpoint_role
    for key, value in (("checkpoint", source_checkpoint), ("onnx", source_onnx)):
        if value is None:
            continue
        path = Path(value)
        record: dict[str, Any] = {"path": str(path)}
        if path.is_file():
            record.update({"size": path.stat().st_size, "sha256": file_sha256(path)})
        source[key] = record
    converter: dict[str, Any] = {
        "image": converter_image,
        "config": dict(converter_config or {}),
    }
    if tool_versions:
        converter["tool_versions"] = dict(tool_versions)
    runtime = {
        key: value
        for key, value in (
            ("maixpy_min_version", maixpy_min_version),
            ("maixpy_version", maixpy_version),
            ("maixcdk_commit", maixcdk_commit),
        )
        if value
    }
    return {
        "schema_version": DEPLOYMENT_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "target": target,
        "model_key": model_key,
        "class_names": list(class_names),
        "source": source,
        "onnx": {
            "opset": opset,
            "input_shape": list(input_shape) if input_shape is not None else None,
            "output_tensors": list(output_tensors),
        },
        "quantization": {
            "mode": quantization,
            "calibration_count": calibration_count,
        },
        "converter": converter,
        "runtime": runtime,
        "payload": [dict(item) for item in artifacts],
        "package_files": {
            kind: [dict(item) for item in files]
            for kind, files in (package_files or {}).items()
        },
        "warnings": list(warnings),
    }


def write_deployment_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination
