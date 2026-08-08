"""Unified JSONL worker entry point for train, predict, deploy and environment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import queue
import runpy
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from ai_biaozhu.deploy.dependencies import inspect_deployment_dependencies
from ai_biaozhu.deploy.docker_import import (
    DockerImageImportCancelled,
    DockerImageImportProgress,
    import_docker_image_archive,
)
from ai_biaozhu.deploy.export import (
    build_legacy_yolov5_export_command,
    export_modern_onnx,
    extract_legacy_yolov5_anchors,
    run_checkpoint_forward,
    validate_checkpoint_class_names,
)
from ai_biaozhu.deploy.maix import (
    Cam2NpuMode,
    MaixConversionRequest,
    MaixTarget,
    build_conversion_plan,
    execute_conversion_plan,
)
from ai_biaozhu.deploy.onnx_gate import (
    file_sha256,
    inspect_onnx_numerics,
    load_rgb_nchw,
)
from ai_biaozhu.deploy.package import (
    DeploymentArtifact,
    DeploymentCancelled,
    PackageSizeWarning,
    build_deployment_package,
    validate_deployment_class_names,
)
from ai_biaozhu.ml.adapters import Adapter, JobCancelled, resolve_adapter
from ai_biaozhu.ml.environment import (
    ensure_writable_yolo_config_dir,
    inspect_environment,
    inspect_legacy_yolov5_repository,
)
from ai_biaozhu.ml.jobs import PredictionJob, TrainingJob, load_manifest
from ai_biaozhu.ml.legacy_process import (
    LEGACY_SCRIPT_NAMES,
    install_pillow_legacy_compatibility,
    legacy_subprocess_environment,
    legacy_torch_onnx_export_compatibility,
)
from ai_biaozhu.ml.model_registry import ModelBackend, get_model
from ai_biaozhu.ml.protocol import JsonlEmitter

AdapterFactory = Callable[[str], Adapter]


def _deployment_target_and_cam2_mode(
    manifest: Mapping[str, Any],
) -> tuple[MaixTarget, Cam2NpuMode]:
    """Resolve the deployment target without applying CAM2 modes to CAM-Pro.

    Older controller manifests record ``"cv181x"`` in ``cam2_npu_mode`` for
    MaixCAM-Pro so that the deployment database can describe the selected
    accelerator.  ``cv181x`` is not a :class:`Cam2NpuMode`, however, and must
    never be parsed as one.  The CAM-Pro conversion path does not use this
    field, so a valid neutral enum value is supplied internally.
    """

    target = MaixTarget(str(manifest["target"]))
    if target is MaixTarget.MAIXCAM_PRO:
        return target, Cam2NpuMode.BOTH
    return target, Cam2NpuMode(
        str(manifest.get("cam2_npu_mode", Cam2NpuMode.BOTH.value))
    )


def _deployment_class_names(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve original checkpoint labels and independently editable aliases."""

    original_value = manifest.get("checkpoint_class_names", manifest.get("class_names"))
    if not isinstance(original_value, Sequence) or isinstance(original_value, str | bytes):
        raise ValueError("deploy manifest 缺少 checkpoint_class_names/class_names")
    original = tuple(str(item) for item in original_value)
    aliases_value = manifest.get("deployment_class_names")
    aliases = (
        None
        if aliases_value is None
        else tuple(str(item) for item in aliases_value)
        if isinstance(aliases_value, Sequence)
        and not isinstance(aliases_value, str | bytes)
        else ()
    )
    deployed = validate_deployment_class_names(original, aliases)
    # validate_deployment_class_names also normalizes/strips the original list.
    checkpoint = validate_deployment_class_names(original, None)
    return checkpoint, deployed


def _ignore_polars_in_pyside_feature_probe() -> None:
    """Avoid a PySide signature-loader recursion in the frozen ML worker.

    A Nuitka multidist process contains the GUI's PySide runtime even when its
    selected entry point is the headless worker.  PySide's import hook tries to
    inspect every newly imported module for feature directives.  Inspecting
    Polars' lazy module ``__getattr__`` recursively re-enters that hook, which
    can otherwise abort Ultralytics while it saves ``best.pt``/``last.pt``.

    PySide's own feature table uses ``-1`` for modules that do not use PySide.
    Marking Polars explicitly is narrowly scoped and leaves normal GUI feature
    handling untouched because the GUI entry point never calls this function.
    """

    feature_module = sys.modules.get("shibokensupport.feature") or sys.modules.get(
        "PySide6.support.feature"
    )
    if feature_module is None:
        try:
            # Importing QtCore initializes PySide's embedded signature support
            # without creating a QApplication or requiring a display server.
            from PySide6 import QtCore as _qt_core  # noqa: F401
        except ImportError:
            return
        feature_module = sys.modules.get(
            "shibokensupport.feature"
        ) or sys.modules.get("PySide6.support.feature")
    feature_table = getattr(feature_module, "pyside_feature_dict", None)
    if isinstance(feature_table, dict):
        feature_table.setdefault("polars", -1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-biaozhu-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "predict", "deploy", "docker-import"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True, type=Path)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--python", required=True, type=Path)
    environment.add_argument("--job-id", default="environment-probe")
    legacy = subparsers.add_parser("legacy-script")
    legacy.add_argument("--repository", required=True, type=Path)
    legacy.add_argument("script", choices=sorted(LEGACY_SCRIPT_NAMES))
    legacy.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def run_job(
    command: str,
    manifest: Mapping[str, Any],
    *,
    stream: TextIO,
    adapter_factory: AdapterFactory = resolve_adapter,
) -> int:
    if command == "train":
        job = TrainingJob.from_mapping(manifest)
        emitter = JsonlEmitter(job.job_id, stream, metrics_path=job.metrics_path)
    elif command == "predict":
        job = PredictionJob.from_mapping(manifest)
        emitter = JsonlEmitter(job.job_id, stream)
    else:
        raise ValueError(f"未知 worker 命令：{command}")
    emitter.emit("status", {"stage": "started", "command": command})
    try:
        adapter = adapter_factory(job.model_key)
        if isinstance(job, TrainingJob):
            result = adapter.train(job, emitter)
        else:
            result = adapter.predict(job, emitter)
        emitter.emit("completed", {"command": command, "result": dict(result)})
        return 0
    except JobCancelled as exc:
        emitter.emit("cancelled", {"command": command, "message": str(exc)})
        return 130
    except Exception as exc:
        _emit_job_error(emitter, command, exc)
        return 1


def run_docker_import(
    manifest: Mapping[str, Any],
    *,
    stream: TextIO,
    input_stream: TextIO | None = None,
) -> int:
    """Load an offline Docker archive and expose progress over JSONL."""

    job_id = str(manifest.get("job_id") or "docker-import")
    emitter = JsonlEmitter(job_id, stream)
    emitter.emit(
        "status",
        {"stage": "started", "command": "docker-import"},
    )
    cancel_event = _start_worker_cancel_listener(input_stream, job_id=job_id)
    cancel_file = _optional_manifest_path(manifest, "cancel_file")
    raw_expected = manifest.get("expected_images", ())
    if isinstance(raw_expected, str):
        expected_images = (raw_expected,)
    elif isinstance(raw_expected, Sequence):
        expected_images = tuple(str(item) for item in raw_expected)
    else:
        expected_images = ()

    def cancelled() -> bool:
        return cancel_event.is_set() or (
            cancel_file is not None and cancel_file.exists()
        )

    def report(progress: DockerImageImportProgress) -> None:
        emitter.emit(
            "progress",
            {
                "command": "docker-import",
                **progress.to_dict(),
            },
        )

    try:
        result = import_docker_image_archive(
            Path(str(manifest["archive_path"])),
            str(manifest.get("docker_executable") or "docker"),
            expected_images=expected_images,
            progress_callback=report,
            cancel_check=cancelled,
        )
        for image in result.images:
            emitter.emit(
                "artifact",
                {
                    "kind": "docker_image",
                    "name": image.name,
                    "status": image.status,
                    "available": image.available,
                    "image_id": image.image_id,
                    "repo_digests": list(image.repo_digests),
                    "error": image.error,
                },
            )
        emitter.emit(
            "completed",
            {
                "command": "docker-import",
                "stage": "docker_image_import",
                **result.to_dict(),
            },
        )
        return 0
    except DockerImageImportCancelled as exc:
        emitter.emit(
            "cancelled",
            {"command": "docker-import", "message": str(exc)},
        )
        return 130
    except Exception as exc:
        _emit_job_error(emitter, "docker-import", exc)
        return 1


def _start_worker_cancel_listener(
    input_stream: TextIO | None,
    *,
    job_id: str,
) -> threading.Event:
    """Accept ``{"type":"cancel"}`` while a streaming import is active."""

    event = threading.Event()
    if input_stream is None:
        return event

    def listen() -> None:
        while not event.is_set():
            try:
                line = input_stream.readline()
            except Exception:
                return
            if not line:
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            message_type = str(payload.get("type") or payload.get("command") or "")
            target = str(payload.get("job_id") or job_id)
            if message_type.casefold() == "cancel" and target == job_id:
                event.set()
                return

    threading.Thread(
        target=listen,
        name=f"docker-import-cancel-{job_id}",
        daemon=True,
    ).start()
    return event


def run_deploy(
    manifest: Mapping[str, Any],
    *,
    stream: TextIO,
    input_stream: TextIO | None = None,
) -> int:
    job_id = str(manifest.get("job_id") or "deploy")
    emitter = JsonlEmitter(job_id, stream)
    emitter.emit("status", {"stage": "started", "command": "deploy"})
    output_dir: Path | None = None
    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "job_id": job_id,
        "command": "deploy",
        "started_at": time.time(),
    }
    audit_path: Path | None = None
    execute = bool(manifest.get("execute", True))
    try:
        dependency_report = inspect_deployment_dependencies().require_ready()
        emitter.emit(
            "artifact",
            {
                "kind": "deployment_dependency_preflight",
                "validation": dependency_report.to_dict(),
            },
        )
        audit["dependency_preflight"] = dependency_report.to_dict()
        target, cam2_npu_mode = _deployment_target_and_cam2_mode(manifest)
        checkpoint_class_names, deployment_class_names = _deployment_class_names(
            manifest
        )
        output_dir = _validated_conversion_output_dir(manifest, job_id)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"转换工作目录非空，拒绝混入旧产物：{output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = _deployment_audit_path(manifest, output_dir, job_id)
        _raise_if_cancelled(manifest)
        onnx_path, source_checkpoint = _export_deployment_onnx(
            manifest,
            output_dir=output_dir,
            emitter=emitter,
        )
        _raise_if_cancelled(manifest)
        calibration_dir = _prepare_calibration_directory(
            manifest,
            output_dir=output_dir,
            emitter=emitter,
        )
        model_key = str(manifest["model_key"])
        spec = get_model(model_key)
        anchors = (
            tuple(float(item) for item in manifest["anchors"])
            if manifest.get("anchors") is not None
            else None
        )
        if spec.backend is ModelBackend.LEGACY_YOLOV5 and anchors is None:
            repository_value = manifest.get("legacy_yolov5_repo")
            if not repository_value or source_checkpoint is None:
                raise ValueError("传统 YOLOv5 部署缺少仓库或 checkpoint，无法读取 anchors")
            emitter.emit("status", {"stage": "extracting_yolov5_anchors"})
            anchors = extract_legacy_yolov5_anchors(
                source_checkpoint,
                Path(str(repository_value)),
            )
        sample_path = _first_calibration_image(calibration_dir)
        sample = load_rgb_nchw(
            sample_path,
            height=int(manifest["input_height"]),
            width=int(manifest["input_width"]),
        )
        pytorch_outputs = None
        if source_checkpoint is not None:
            emitter.emit("status", {"stage": "validating_checkpoint_parity"})
            checkpoint_forward = run_checkpoint_forward(
                source_checkpoint,
                model_key=model_key,
                input_array=sample,
                legacy_repository=manifest.get("legacy_yolov5_repo"),
            )
            validate_checkpoint_class_names(
                checkpoint_forward.class_names,
                checkpoint_class_names,
            )
            pytorch_outputs = checkpoint_forward.outputs
        numeric_report = inspect_onnx_numerics(
            onnx_path,
            input_array=sample,
            pytorch_outputs=pytorch_outputs,
        ).require_ok()
        emitter.emit(
            "artifact",
            {
                "kind": "onnx_numeric_gate",
                "validation": numeric_report.to_dict(),
            },
        )
        request = MaixConversionRequest(
            target=target,
            model_key=model_key,
            onnx_path=onnx_path,
            output_dir=output_dir,
            calibration_dir=calibration_dir,
            class_names=checkpoint_class_names,
            input_height=int(manifest["input_height"]),
            input_width=int(manifest["input_width"]),
            calibration_count=int(manifest.get("calibration_count", 100)),
            docker_executable=str(manifest.get("docker_executable", "docker")),
            converter_image=(
                str(manifest["converter_image"])
                if manifest.get("converter_image")
                else None
            ),
            cam2_npu_mode=cam2_npu_mode,
            output_nodes=(
                tuple(str(item) for item in manifest["output_nodes"])
                if manifest.get("output_nodes")
                else None
            ),
            anchors=anchors,
        )
        emitter.emit("status", {"stage": "validating_onnx"})
        plan = build_conversion_plan(request)
        plan.materialize()
        converter_onnx = output_dir / "export.onnx"
        converter_numeric_report = inspect_onnx_numerics(
            converter_onnx,
            input_array=sample,
        ).require_ok()
        emitter.emit(
            "artifact",
            {
                "kind": "converter_onnx_numeric_gate",
                "validation": converter_numeric_report.to_dict(),
            },
        )
        calibration_audit = _calibration_audit(
            calibration_dir,
            request.calibration_count,
        )
        image_identity = (
            _inspect_converter_image(request.docker_executable, plan.image)
            if execute
            else {"name": plan.image, "inspection_skipped": True}
        )
        audit.update(
            {
                "model_key": model_key,
                "target": request.target.value,
                "source_checkpoint": (
                    {
                        "path": str(source_checkpoint),
                        "sha256": file_sha256(source_checkpoint),
                    }
                    if source_checkpoint is not None
                    else None
                ),
                "onnx_gate": (
                    plan.gate_report.to_dict() if plan.gate_report is not None else None
                ),
                "onnx_numeric_gate": numeric_report.to_dict(),
                "converter_onnx_numeric_gate": converter_numeric_report.to_dict(),
                "calibration": calibration_audit,
                "converter": {
                    "image": plan.image,
                    "identity": image_identity,
                    "commands": [list(command) for command in plan.commands],
                },
                "resolved_output_nodes": list(plan.expected_tensors),
                "checkpoint_class_names": list(checkpoint_class_names),
                "deployment_class_names": list(deployment_class_names),
            }
        )
        _write_deployment_audit(audit_path, audit)
        emitter.emit(
            "artifact",
            {
                "kind": "conversion_plan",
                "commands": [list(command) for command in plan.commands],
                "generated_files": [
                    item.relative_path for item in plan.generated_files
                ],
            },
        )
        conversion_logs: list[dict[str, Any]] = []
        if execute:
            emitter.emit("status", {"stage": "converting", "steps": len(plan.commands)})
            cancel_file = _optional_manifest_path(manifest, "cancel_file")

            def step_callback(
                state: str,
                index: int,
                total: int,
                command: tuple[str, ...],
                completed: subprocess.CompletedProcess[str] | None,
            ) -> None:
                stage = _conversion_stage(request.target, index, state)
                emitter.emit(
                    "status",
                    {
                        "stage": stage,
                        "step": index,
                        "steps": total,
                        "command": list(command),
                    },
                )
                if completed is None:
                    return
                record = {
                    "step": index,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
                conversion_logs.append(record)
                for stream_name, content in (
                    ("stdout", completed.stdout),
                    ("stderr", completed.stderr),
                ):
                    for line in str(content or "").splitlines():
                        emitter.emit(
                            "log",
                            {
                                "stream": stream_name,
                                "stage": stage,
                                "step": index,
                                "message": line,
                            },
                        )
                audit["converter"]["steps"] = list(conversion_logs)
                _write_deployment_audit(audit_path, audit)

            execute_conversion_plan(
                plan,
                materialize=False,
                runner=lambda command, **kwargs: _run_cancellable_command(
                    command,
                    cancel_file=cancel_file,
                    **kwargs,
                ),
                step_callback=step_callback,
            )
            for name in plan.output_models:
                emitter.emit(
                    "artifact",
                    {"kind": "runtime_model", "path": str(request.output_dir / name)},
                )
        raw_package_outputs = manifest.get(
            "package_outputs", ["model_only", "full_app"]
        )
        if not isinstance(raw_package_outputs, Sequence) or isinstance(
            raw_package_outputs, str | bytes
        ):
            raise ValueError("package_outputs 必须是数组")
        package_outputs = tuple(str(item) for item in raw_package_outputs)
        package_path = manifest.get("package_path")
        if not package_path and package_outputs:
            package_root = (
                output_dir.parent if bool(manifest.get("cleanup_workdir")) else output_dir
            )
            package_path = package_root / (
                f"{request.model_key}-{request.target.value}-app.maixapp"
            )
        package_result = None
        if package_path:
            if not execute:
                raise ValueError("execute=false 时不能立即创建部署包")
            package_destination = Path(str(package_path)).resolve()
            if bool(manifest.get("cleanup_workdir", False)) and (
                package_destination == output_dir
                or output_dir in package_destination.parents
            ):
                raise ValueError("部署包不能发布到即将自动清理的转换工作目录")
            editable_value = manifest.get("editable_project_path")
            if editable_value and bool(manifest.get("cleanup_workdir", False)):
                editable_destination = Path(str(editable_value)).resolve()
                if (
                    editable_destination == output_dir
                    or output_dir in editable_destination.parents
                ):
                    raise ValueError(
                        "可编辑工程不能发布到即将自动清理的转换工作目录"
                    )
            emitter.emit("status", {"stage": "packaging"})
            manifest_tool_versions = manifest.get(
                "tool_versions",
                manifest.get("converter_tool_versions"),
            )
            tool_versions = (
                dict(manifest_tool_versions)
                if isinstance(manifest_tool_versions, Mapping)
                else {}
            )
            if image_identity.get("docker_version"):
                tool_versions["docker"] = image_identity["docker_version"]
            package_result = build_deployment_package(
                package_path=package_destination,
                target=request.target,
                model_key=request.model_key,
                model_artifacts=[
                    DeploymentArtifact(request.output_dir / name, name)
                    for name in plan.output_models
                ],
                class_names=request.class_names,
                deployment_class_names=deployment_class_names,
                input_height=request.input_height,
                input_width=request.input_width,
                camera_height=int(
                    manifest.get("camera_height", request.input_height)
                ),
                camera_width=int(
                    manifest.get("camera_width", request.input_width)
                ),
                source_run_id=(
                    str(manifest["source_run_id"])
                    if manifest.get("source_run_id")
                    else None
                ),
                checkpoint_role=(
                    str(manifest["checkpoint_kind"])
                    if manifest.get("checkpoint_kind")
                    else None
                ),
                source_checkpoint=source_checkpoint,
                source_onnx=request.onnx_path,
                output_tensors=plan.expected_tensors,
                calibration_count=request.calibration_count,
                converter_image=plan.image,
                tool_versions=tool_versions,
                maixpy_version=(
                    str(manifest["maixpy_version"])
                    if manifest.get("maixpy_version")
                    else None
                ),
                maixcdk_commit=(
                    str(manifest["maixcdk_commit"])
                    if manifest.get("maixcdk_commit")
                    else None
                ),
                converter_config={
                    "image_identity": image_identity,
                    "commands": [list(command) for command in plan.commands],
                    "onnx_gate": {
                        "sha256": (
                            plan.gate_report.sha256
                            if plan.gate_report is not None
                            else None
                        ),
                        "opset": (
                            plan.gate_report.opset
                            if plan.gate_report is not None
                            else None
                        ),
                        "input_shape": (
                            list(plan.gate_report.input_shape)
                            if plan.gate_report is not None
                            and plan.gate_report.input_shape is not None
                            else None
                        ),
                        "resolved_output_nodes": list(plan.expected_tensors),
                    },
                    "numeric_validation": numeric_report.to_dict(),
                    "converter_onnx_numeric_validation": (
                        converter_numeric_report.to_dict()
                    ),
                    "calibration_set_sha256": calibration_audit["set_sha256"],
                },
                cam2_npu_mode=request.cam2_npu_mode,
                anchors=request.anchors,
                confidence=float(manifest.get("confidence", 0.35)),
                iou=float(manifest.get("iou", 0.45)),
                max_det=int(manifest.get("max_det", 100)),
                dual_buff=bool(manifest.get("dual_buff", True)),
                package_outputs=package_outputs,
                editable_project_path=(
                    str(manifest["editable_project_path"])
                    if manifest.get("editable_project_path")
                    else None
                ),
                allow_oversize=bool(manifest.get("allow_oversize", False)),
                oversize_confirmation=lambda warnings: _confirm_oversize(
                    warnings,
                    emitter=emitter,
                    input_stream=input_stream or sys.stdin,
                    timeout=float(manifest.get("confirmation_timeout", 300)),
                ),
            )
            for artifact in package_result.artifacts:
                emitter.emit(
                    "artifact",
                    {
                        "kind": {
                            "model-only": "maix_model_package",
                            "maixapp": "maix_app_package",
                            "editable-project": "maix_editable_project",
                            "deployment-report": "deployment_report",
                            "sha256-manifest": "package_sha256",
                        }[artifact.kind],
                        "path": str(artifact.path),
                        "is_directory": artifact.is_directory,
                        "sha256": artifact.sha256,
                        "files": list(artifact.files),
                    },
                )
        audit["finished_at"] = time.time()
        audit["status"] = "needs_device_validation"
        audit["packages"] = (
            {
                "model": (
                    str(package_result.model_package_path)
                    if package_result.model_package_path
                    else None
                ),
                "app": (
                    str(package_result.app_package_path)
                    if package_result.app_package_path
                    else None
                ),
                "editable_project": (
                    str(package_result.editable_project_path)
                    if package_result.editable_project_path
                    else None
                ),
                "report": str(package_result.report_path),
                "sha256": str(package_result.sha256_path),
                "artifacts": [
                    {
                        "kind": item.kind,
                        "path": str(item.path),
                        "is_directory": item.is_directory,
                        "sha256": item.sha256,
                    }
                    for item in package_result.artifacts
                ],
            }
            if package_result is not None
            else None
        )
        _write_deployment_audit(audit_path, audit)
        emitter.emit(
            "artifact",
            {"kind": "conversion_audit", "path": str(audit_path)},
        )
        if execute and bool(manifest.get("cleanup_workdir", False)):
            emitter.emit(
                "status",
                {"stage": "cleaning_conversion_workspace", "path": str(output_dir)},
            )
            _cleanup_conversion_workspace(output_dir, job_id)
            output_dir = None
        emitter.emit(
            "completed",
            {
                "command": "deploy",
                "target": request.target.value,
                "output_models": list(plan.output_models),
                "model_package_path": (
                    str(package_result.model_package_path) if package_result else None
                ),
                "app_package_path": (
                    str(package_result.app_package_path) if package_result else None
                ),
                "editable_project_path": (
                    str(package_result.editable_project_path)
                    if package_result
                    else None
                ),
                "deployment_artifacts": (
                    [
                        {
                            "kind": item.kind,
                            "path": str(item.path),
                            "is_directory": item.is_directory,
                            "sha256": item.sha256,
                        }
                        for item in package_result.artifacts
                    ]
                    if package_result
                    else []
                ),
                "resolved_output_nodes": list(plan.expected_tensors),
                "conversion_audit_path": str(audit_path),
                "device_validation": "required",
            },
        )
        return 0
    except DeploymentCancelled as exc:
        audit.update(
            {
                "finished_at": time.time(),
                "status": "cancelled",
                "error": str(exc),
            }
        )
        if audit_path is not None:
            _write_deployment_audit(audit_path, audit)
        emitter.emit("cancelled", {"command": "deploy", "message": str(exc)})
        return 130
    except Exception as exc:
        audit.update(
            {
                "finished_at": time.time(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        if audit_path is not None:
            _write_deployment_audit(audit_path, audit)
        _emit_job_error(emitter, "deploy", exc)
        return 1
    finally:
        if (
            output_dir is not None
            and execute
            and bool(manifest.get("cleanup_workdir", False))
            and output_dir.exists()
        ):
            try:
                _cleanup_conversion_workspace(output_dir, job_id)
            except Exception as exc:
                emitter.emit(
                    "warning",
                    {
                        "code": "conversion_workspace_cleanup_failed",
                        "message": str(exc),
                        "path": str(output_dir),
                    },
                )


def _validated_conversion_output_dir(
    manifest: Mapping[str, Any],
    job_id: str,
) -> Path:
    value = str(manifest.get("output_dir") or "").strip()
    if not value:
        raise ValueError("deploy manifest 缺少 output_dir")
    path = Path(value).resolve()
    if not str(path).isascii():
        raise ValueError("Docker 转换工作目录必须使用 ASCII 路径")
    if len(str(path)) > 180:
        raise ValueError("Docker 转换工作目录过长；请选择更短的 ASCII 路径")
    if path.parent == path:
        raise ValueError("拒绝把磁盘根目录作为转换工作目录")
    if bool(manifest.get("cleanup_workdir", False)) and path.name != job_id:
        raise ValueError("自动清理只允许用于以 job_id 命名的转换工作目录")
    return path


def _deployment_audit_path(
    manifest: Mapping[str, Any],
    output_dir: Path,
    job_id: str,
) -> Path:
    configured = manifest.get("audit_dir")
    root = Path(str(configured)).resolve() if configured else output_dir
    if bool(manifest.get("cleanup_workdir", False)) and (
        root == output_dir or output_dir in root.parents
    ):
        raise ValueError("自动清理转换目录时，audit_dir 必须位于该目录之外")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{job_id}-conversion-audit.json"


def _write_deployment_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _first_calibration_image(directory: Path) -> Path:
    images = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not images:
        raise ValueError("校准目录没有受支持的图片")
    return images[0]


def _calibration_audit(directory: Path, count: int) -> dict[str, Any]:
    images = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if len(images) < count:
        raise ValueError(f"校准图片不足：需要 {count} 张，实际 {len(images)} 张")
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in images[:count]:
        sha256 = file_sha256(path)
        record = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256,
        }
        records.append(record)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    result: dict[str, Any] = {
        "count": count,
        "set_sha256": digest.hexdigest(),
        "images": records,
    }
    snapshot_manifest = directory / "calibration-snapshot.json"
    if snapshot_manifest.is_file() and not snapshot_manifest.is_symlink():
        try:
            snapshot_payload = json.loads(
                snapshot_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            pass
        else:
            if isinstance(snapshot_payload, Mapping):
                result["snapshot_manifest"] = str(snapshot_manifest)
                result["used"] = list(snapshot_payload.get("used") or ())
                result["rejected"] = list(snapshot_payload.get("rejected") or ())
    return result


def _inspect_converter_image(docker_executable: str, image: str) -> dict[str, Any]:
    try:
        version = subprocess.run(
            [
                docker_executable,
                "version",
                "--format",
                "{{json .}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Docker 不可用：{exc}") from exc
    if version.returncode:
        message = version.stderr.strip() or version.stdout.strip()
        raise RuntimeError(f"Docker daemon 不可用：{message}")
    try:
        inspected = subprocess.run(
            [docker_executable, "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"转换镜像检查失败：{exc}") from exc
    if inspected.returncode:
        message = inspected.stderr.strip() or inspected.stdout.strip()
        raise RuntimeError(f"未找到转换镜像 {image}：{message}")
    try:
        raw = json.loads(inspected.stdout)
        record = raw[0]
        docker_version = json.loads(version.stdout)
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Docker 镜像审计信息无效：{exc}") from exc
    config = record.get("Config") if isinstance(record, Mapping) else None
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    return {
        "name": image,
        "id": str(record.get("Id") or ""),
        "repo_digests": [str(item) for item in record.get("RepoDigests") or ()],
        "created": str(record.get("Created") or ""),
        "labels": dict(labels) if isinstance(labels, Mapping) else {},
        "docker_version": docker_version,
    }


def _optional_manifest_path(
    manifest: Mapping[str, Any],
    key: str,
) -> Path | None:
    value = manifest.get(key)
    return Path(str(value)) if value not in (None, "") else None


def _raise_if_cancelled(manifest: Mapping[str, Any]) -> None:
    path = _optional_manifest_path(manifest, "cancel_file")
    if path is not None and path.exists():
        raise DeploymentCancelled("部署任务已由用户取消")


def _run_cancellable_command(
    command: Sequence[str],
    *,
    cancel_file: Path | None,
    cwd: str | None = None,
    capture_output: bool = True,
    text: bool = True,
    check: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    del capture_output, check
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        **kwargs,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            return subprocess.CompletedProcess(
                list(command),
                int(process.returncode or 0),
                stdout,
                stderr,
            )
        except subprocess.TimeoutExpired:
            if cancel_file is None or not cancel_file.exists():
                continue
            with contextlib.suppress(OSError, ProcessLookupError):
                process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError, ProcessLookupError):
                    process.kill()
                process.communicate()
            raise DeploymentCancelled("部署转换已由用户取消") from None


def _conversion_stage(target: MaixTarget, index: int, state: str) -> str:
    if target is MaixTarget.MAIXCAM_PRO:
        stages = {
            1: "compiling_mlir",
            2: "quantizing_int8",
            3: "compiling_cvimodel",
        }
        return stages.get(index, "converting")
    return "quantizing_int8" if state == "started" else "compiling_axmodel"


def _cleanup_conversion_workspace(path: Path, job_id: str) -> None:
    resolved = path.resolve()
    if resolved.name != job_id or resolved.parent == resolved:
        raise RuntimeError(f"拒绝清理非任务转换目录：{resolved}")
    if path.is_symlink():
        raise RuntimeError(f"拒绝清理符号链接转换目录：{path}")
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    if attributes & 0x400:
        raise RuntimeError(f"拒绝清理 reparse point 转换目录：{path}")
    shutil.rmtree(resolved)


def _prepare_calibration_directory(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    emitter: JsonlEmitter,
) -> Path:
    configured = manifest.get("calibration_dir")
    if configured:
        directory = Path(str(configured))
        if not directory.is_dir():
            raise ValueError(f"校准目录不存在：{directory}")
        return directory
    raw_images = manifest.get("calibration_images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("deploy manifest 必须提供 calibration_images 或 calibration_dir")
    raw_candidates = manifest.get("calibration_candidate_images", ())
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, str | bytes
    ):
        raise ValueError("calibration_candidate_images 必须是数组")
    required_count = int(manifest.get("calibration_count", len(raw_images)))
    if required_count <= 0:
        raise ValueError("calibration_count 必须大于 0")
    directory = output_dir / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [
        *(("selected", item) for item in raw_images),
        *(("fallback", item) for item in raw_candidates),
    ]
    emitter.emit(
        "status",
        {
            "stage": "preparing_calibration",
            "total": required_count,
            "candidate_total": len(candidates),
        },
    )
    used: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for candidate_index, (role, item) in enumerate(candidates, start=1):
        if len(used) >= required_count:
            break
        if not isinstance(item, Mapping):
            reason = "候选项不是对象"
            rejected.append({"candidate_index": candidate_index, "reason": reason})
            emitter.emit(
                "warning",
                {
                    "code": "calibration_candidate_rejected",
                    "candidate_index": candidate_index,
                    "role": role,
                    "message": reason,
                },
            )
            continue
        source = Path(str(item.get("path") or ""))
        source_key = str(source.resolve(strict=False)).casefold()
        if source_key in seen_sources:
            reason = f"重复候选：{source}"
            rejected.append(
                {"candidate_index": candidate_index, "path": str(source), "reason": reason}
            )
            continue
        seen_sources.add(source_key)
        try:
            record = _copy_verified_calibration_image(
                item,
                directory=directory,
                output_index=len(used) + 1,
            )
        except (OSError, ValueError) as exc:
            reason = str(exc)
            rejected.append(
                {"candidate_index": candidate_index, "path": str(source), "reason": reason}
            )
            emitter.emit(
                "warning",
                {
                    "code": "calibration_candidate_rejected",
                    "candidate_index": candidate_index,
                    "role": role,
                    "path": str(source),
                    "message": reason,
                },
            )
            continue
        record.update(
            {
                "candidate_index": candidate_index,
                "role": role,
                "image_id": item.get("image_id"),
            }
        )
        used.append(record)
        emitter.emit(
            "progress",
            {
                "stage": "preparing_calibration",
                "current": len(used),
                "total": required_count,
                "candidate_index": candidate_index,
                "replacement": role == "fallback",
            },
        )
    if len(used) < required_count:
        details = "; ".join(item["reason"] for item in rejected[-3:])
        raise ValueError(
            f"可用校准图片不足：需要 {required_count} 张，成功 {len(used)} 张"
            + (f"；最近失败：{details}" if details else "")
        )
    snapshot_manifest = {
        "schema_version": "1.0",
        "required_count": required_count,
        "used": used,
        "rejected": rejected,
    }
    (directory / "calibration-snapshot.json").write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    emitter.emit(
        "artifact",
        {
            "kind": "calibration_snapshot",
            "path": str(directory),
            "manifest": str(directory / "calibration-snapshot.json"),
            "used": used,
            "replacement_count": sum(item["role"] == "fallback" for item in used),
        },
    )
    return directory


def _copy_verified_calibration_image(
    item: Mapping[str, Any],
    *,
    directory: Path,
    output_index: int,
) -> dict[str, Any]:
    """Copy one calibration image and verify both sides after the copy."""

    source = Path(str(item.get("path") or ""))
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"校准图片不存在或不安全：{source}")
    suffix = source.suffix.casefold()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError(f"不支持的校准图片格式：{source}")
    expected_hash = str(item.get("sha256") or "").strip().casefold()
    before_hash = file_sha256(source)
    if expected_hash and expected_hash != before_hash:
        raise ValueError(f"校准图片复制前 SHA-256 不匹配：{source}")
    temporary = directory / f".{output_index:04d}-{before_hash[:12]}.copying"
    destination = directory / f"{output_index:04d}-{before_hash[:12]}{suffix}"
    try:
        shutil.copy2(source, temporary)
        copied_hash = file_sha256(temporary)
        after_hash = file_sha256(source)
        if copied_hash != before_hash or after_hash != before_hash:
            raise ValueError(f"校准图片在复制期间发生变化：{source}")
        if expected_hash and copied_hash != expected_hash:
            raise ValueError(f"校准图片复制后 SHA-256 不匹配：{source}")
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()
    published_hash = file_sha256(destination)
    if published_hash != before_hash:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise ValueError(f"校准快照发布后 SHA-256 不匹配：{source}")
    return {
        "source": str(source),
        "snapshot": str(destination),
        "name": destination.name,
        "sha256": published_hash,
        "size": destination.stat().st_size,
    }


def _export_deployment_onnx(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    emitter: JsonlEmitter,
) -> tuple[Path, Path | None]:
    """Export checkpoint inside the ML worker; an existing ONNX is debug-only."""

    checkpoint_value = manifest.get("checkpoint")
    if checkpoint_value is None and manifest.get("onnx_path"):
        existing = Path(str(manifest["onnx_path"]))
        if not existing.is_file():
            raise ValueError(f"ONNX 文件不存在：{existing}")
        emitter.emit(
            "warning",
            {
                "code": "preexported_onnx",
                "message": "使用了高级预导出 ONNX；常规 UI 流程应只提交 checkpoint",
            },
        )
        return existing, None
    if checkpoint_value is None:
        raise ValueError("deploy manifest 必须提供 checkpoint")
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint 不存在：{checkpoint}")
    model_key = str(manifest["model_key"])
    spec = get_model(model_key)
    height = int(manifest["input_height"])
    width = int(manifest["input_width"])
    export_dir = output_dir / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    working_checkpoint = export_dir / f"source-{file_sha256(checkpoint)[:12]}.pt"
    shutil.copy2(checkpoint, working_checkpoint)
    emitter.emit(
        "status",
        {
            "stage": "exporting_onnx",
            "checkpoint": str(checkpoint),
            "backend": spec.backend.value,
        },
    )
    if spec.backend is ModelBackend.ULTRALYTICS:
        generated = export_modern_onnx(working_checkpoint, imgsz=(height, width))
    else:
        repository_value = manifest.get("legacy_yolov5_repo")
        if not repository_value:
            raise ValueError("传统 YOLOv5 部署需要 legacy_yolov5_repo")
        repository = Path(str(repository_value))
        report = inspect_legacy_yolov5_repository(repository)
        if not report.valid:
            raise ValueError("传统 YOLOv5 仓库校验失败：" + "；".join(report.errors))
        command = build_legacy_yolov5_export_command(
            python_executable=sys.executable,
            repository=repository,
            checkpoint=working_checkpoint,
            imgsz=(height, width),
        )
        completed = subprocess.run(
            command,
            cwd=str(repository),
            capture_output=True,
            text=True,
            check=False,
            env=legacy_subprocess_environment(),
        )
        for line in completed.stdout.splitlines():
            emitter.emit("log", {"stream": "stdout", "message": line})
        for line in completed.stderr.splitlines():
            emitter.emit("log", {"stream": "stderr", "message": line})
        if completed.returncode != 0:
            raise RuntimeError(f"传统 YOLOv5 ONNX 导出失败：{completed.returncode}")
        generated = working_checkpoint.with_suffix(".onnx")
        if not generated.is_file():
            raise RuntimeError(f"YOLOv5 export.py 未生成预期文件：{generated}")
    destination = export_dir / "model.onnx"
    if generated.resolve() != destination.resolve():
        shutil.copy2(generated, destination)
    else:
        destination = generated
    emitter.emit("artifact", {"kind": "onnx", "path": str(destination)})
    return destination, checkpoint


def _confirm_oversize(
    warnings: tuple[PackageSizeWarning, ...],
    *,
    emitter: JsonlEmitter,
    input_stream: TextIO,
    timeout: float,
) -> bool:
    emitter.emit(
        "warning",
        {
            "code": "package_size_warning",
            "message": "部署包超过 30,000,000 字节建议值，等待用户确认",
            "packages": [warning.to_payload() for warning in warnings],
        },
    )
    line = _readline_with_timeout(input_stream, timeout)
    if not line:
        return False
    stripped = line.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped.casefold() in {"yes", "y", "true", "1", "accept"}
    return bool(payload.get("accepted", False)) if isinstance(payload, Mapping) else False


def _readline_with_timeout(stream: TextIO, timeout: float) -> str:
    if timeout <= 0:
        return ""
    results: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            results.put(stream.readline())
        except Exception:
            results.put("")

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        return results.get(timeout=timeout)
    except queue.Empty:
        return ""


def _run_environment(path: Path, job_id: str, stream: TextIO) -> int:
    emitter = JsonlEmitter(job_id, stream)
    emitter.emit("status", {"stage": "probing_environment", "python": str(path)})
    report = inspect_environment(path)
    payload = {
        "prefix": str(report.candidate.prefix),
        "python": str(report.candidate.python),
        "valid": report.valid,
        "python_version": report.python_version,
        "torch_version": report.torch_version,
        "torchvision_version": report.torchvision_version,
        "ultralytics_version": report.ultralytics_version,
        "cuda_available": report.cuda_available,
        "cuda_version": report.cuda_version,
        "device_name": report.device_name,
        "gpu_ready": report.gpu_ready,
        "errors": list(report.errors),
        "compatibility_errors": list(report.compatibility_errors),
    }
    if report.valid:
        emitter.emit("completed", payload)
        return 0
    emitter.emit("error", {"scope": "environment", **payload})
    return 1


def _run_legacy_script(
    repository: Path,
    script: str,
    script_args: Sequence[str],
) -> int:
    """Execute one allowlisted v7.0 script from a standalone worker bundle."""

    report = inspect_legacy_yolov5_repository(repository)
    if not report.valid:
        raise ValueError("传统 YOLOv5 仓库校验失败：" + "；".join(report.errors))
    normalized = script.casefold().removesuffix(".py")
    if normalized not in LEGACY_SCRIPT_NAMES:
        raise ValueError(f"不允许执行传统 YOLOv5 脚本：{script}")
    path = report.path / f"{normalized}.py"
    arguments = list(script_args)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    inserted = str(report.path) not in sys.path
    if inserted:
        sys.path.insert(0, str(report.path))
    os.environ.update(legacy_subprocess_environment())
    install_pillow_legacy_compatibility()
    try:
        os.chdir(report.path)
        sys.argv = [str(path), *arguments]
        with legacy_torch_onnx_export_compatibility(
            enabled=normalized == "export"
        ):
            runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(str(exc.code), file=sys.stderr)
        return 1
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(str(report.path))


def _emit_job_error(emitter: JsonlEmitter, command: str, exc: Exception) -> None:
    emitter.emit(
        "error",
        {
            "scope": "job",
            "command": command,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stream: TextIO | None = None,
    input_stream: TextIO | None = None,
    adapter_factory: AdapterFactory = resolve_adapter,
) -> int:
    _ignore_polars_in_pyside_feature_probe()
    ensure_writable_yolo_config_dir()
    args = build_parser().parse_args(argv)
    output = stream or sys.stdout
    if args.command == "environment":
        return _run_environment(args.python, args.job_id, output)
    if args.command == "legacy-script":
        return _run_legacy_script(
            args.repository,
            args.script,
            args.script_args,
        )
    manifest = load_manifest(args.manifest)
    if args.command == "deploy":
        return run_deploy(
            manifest,
            stream=output,
            input_stream=input_stream or sys.stdin,
        )
    if args.command == "docker-import":
        return run_docker_import(
            manifest,
            stream=output,
            input_stream=input_stream or sys.stdin,
        )
    return run_job(
        args.command,
        manifest,
        stream=output,
        adapter_factory=adapter_factory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
