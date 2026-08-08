"""Create audited source, standalone, installer, and checksum deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

ROOT_SOURCE_FILES = (
    ".editorconfig",
    ".gitignore",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "constraints-ml.txt",
    "environment.yml",
    "pyproject.toml",
)
SOURCE_DIRECTORIES = (
    "bundled_models",
    "docs",
    "locks",
    "packaging",
    "scripts",
    "src",
    "tests",
    "third_party",
)
SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
        ".idea",
        ".vscode",
    }
)
SKIPPED_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db"})
FORBIDDEN_SOURCE_SUFFIXES = frozenset(
    {
        ".dll",
        ".engine",
        ".exe",
        ".onnx",
        ".pt",
        ".pyc",
        ".pyo",
        ".tar",
        ".tmp",
        ".zip",
    }
)
FORBIDDEN_RUNTIME_SUFFIXES = frozenset({".engine", ".onnx", ".pt", ".pyc", ".pyo"})
MODEL_SEED_FILENAME = "yolo26s.pt"
MODEL_SEED_SIZE = 20_422_725
MODEL_SEED_SHA256 = (
    "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
)
SOURCE_MODEL_SEED = PurePosixPath("bundled_models", MODEL_SEED_FILENAME)
RUNTIME_MODEL_SEED = PurePosixPath("model-seed", MODEL_SEED_FILENAME)
RELEASE_STEM = "AI-Biaozhu-Maintenance"
RELEASE_APPLICATION = "AI Biaozhu Maintenance 0.2"
SOURCE_MIRROR_APPLICATION = "AI Biaozhu Maintenance 0.2 source mirror"
RELEASE_REPORT_NAME = f"{RELEASE_STEM}-release-report.json"
RELEASE_CHECKSUM_NAME = f"{RELEASE_STEM}-SHA256SUMS"
LOCK_CHECKSUM_NAME = "SHA256SUMS"
MIRROR_MARKER = ".ai-biaozhu-maintenance-source-mirror.json"
ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    default_project = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-mirror", type=Path)
    parser.add_argument("--replace-source-mirror", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--skip-standalone-zip", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--allow-partial-installer-validation", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve(strict=True)
    output_input = (
        args.output.absolute()
        if args.output is not None
        else (project.parent / f"{project.name}-outputs").absolute()
    )
    if output_input.exists():
        _reject_reparse(output_input)
    output = output_input.resolve()
    _assert_safe_output(output, project)
    output.mkdir(parents=True, exist_ok=True)
    if args.source_mirror is not None:
        mirror_input = args.source_mirror.absolute()
        if mirror_input.exists():
            _reject_reparse(mirror_input)
        mirror = mirror_input.resolve()
        _assert_safe_mirror(mirror)
        if _is_within(mirror, project) or _is_within(project, mirror):
            raise ReleaseError(
                f"source mirror must not contain or be contained by the project: {mirror}"
            )
        if _is_within(mirror, output) or _is_within(output, mirror):
            raise ReleaseError(
                f"source mirror and release output must be disjoint: {mirror}"
            )
    # A new invocation must never leave the previous successful report/checksum
    # looking current if any later preflight or packaging step fails.
    _invalidate_release_metadata(output)
    version = _project_version(project / "pyproject.toml")
    validation_report = project / "docs" / "VALIDATION_REPORT.md"
    if not validation_report.is_file():
        raise ReleaseError(f"missing validation report: {validation_report}")

    standalone: Path | None = None
    installer: Path | None = None
    build_report: Path | None = None
    gpu_validation: Path | None = None
    onnx_validations: dict[str, Path] = {}
    installer_validation: Path | None = None
    validation_evidence: dict[str, Any] | None = None
    if not args.source_only:
        standalone = project / "build" / "windows" / "AI-Biaozhu.dist"
        installer = (
            project
            / "dist"
            / f"AI-Biaozhu-Maintenance-Setup-{version}-x64.exe"
        )
        build_report = project / "build" / "windows" / "nuitka-report.xml"
        gpu_validation = (
            project / "build" / "test-results" / "final-standalone-cuda-summary.json"
        )
        onnx_validations = {
            "YOLO26n": (
                project
                / "build"
                / "test-results"
                / "final-real-onnx-gate-summary.json"
            ),
            "YOLOv8n": (
                project
                / "build"
                / "test-results"
                / "real-onnx-gate-yolov8-summary.json"
            ),
            "YOLO11n": (
                project
                / "build"
                / "test-results"
                / "real-onnx-gate-yolo11-summary.json"
            ),
            "YOLOv5n": (
                project
                / "build"
                / "test-results"
                / "real-onnx-gate-yolov5-summary.json"
            ),
        }
        installer_validation = (
            project / "build" / "test-results" / "final-installer-smoke-summary.json"
        )
        # Validate all irreplaceable build inputs before removing older outputs.
        _verify_standalone(standalone)
        _verify_pe_file(installer, minimum_size=100_000)
        _verify_nuitka_report(build_report)
        for required_validation in (
            gpu_validation,
            installer_validation,
            *onnx_validations.values(),
        ):
            if not required_validation.is_file():
                raise ReleaseError(f"missing validation evidence: {required_validation}")
        validation_evidence = _verify_release_evidence(
            standalone=standalone,
            installer=installer,
            gpu_validation=gpu_validation,
            onnx_validations=onnx_validations,
            installer_validation=installer_validation,
            allow_partial_installer_validation=(
                args.allow_partial_installer_validation
            ),
        )

    if args.clean_output:
        _clean_generated_output(output)

    build_root = project / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-", dir=build_root) as temporary:
        staging = Path(temporary)
        source_tree = staging / f"{RELEASE_STEM}-{version}-source"
        source_entries = _copy_source_tree(project, source_tree)

        artifacts: list[dict[str, Any]] = []
        source_archive = output / f"{RELEASE_STEM}-{version}-source.zip"
        _write_zip_atomic(source_tree, source_archive, include_root=True)
        _verify_source_archive(source_archive, version)
        artifacts.append(_artifact_record(source_archive, "source_archive"))

        validation_output = output / f"{RELEASE_STEM}-{version}-validation-report.md"
        _copy_file_atomic(validation_report, validation_output)
        artifacts.append(_artifact_record(validation_output, "validation_report"))

        if not args.source_only:
            assert standalone is not None
            assert installer is not None
            assert gpu_validation is not None
            assert installer_validation is not None

            installer_output = output / installer.name
            assert validation_evidence is not None
            _copy_verified_file(
                installer,
                installer_output,
                str(
                    validation_evidence["installer_smoke"]["installer_sha256"]
                ),
            )
            artifacts.append(_artifact_record(installer_output, "windows_installer"))

            for source, filename, role, expected_hash in (
                (
                    gpu_validation,
                    f"{RELEASE_STEM}-{version}-gpu-validation.json",
                    "gpu_validation",
                    str(validation_evidence["standalone_cuda"]["sha256"]),
                ),
                (
                    installer_validation,
                    f"{RELEASE_STEM}-{version}-installer-validation.json",
                    "installer_validation",
                    str(validation_evidence["installer_smoke"]["sha256"]),
                ),
            ):
                validation_artifact = output / filename
                _copy_verified_file(source, validation_artifact, expected_hash)
                artifacts.append(_artifact_record(validation_artifact, role))

            for model_key, source in onnx_validations.items():
                family = model_key.casefold()
                validation_artifact = (
                    output
                    / f"{RELEASE_STEM}-{version}-onnx-gate-{family}-validation.json"
                )
                _copy_verified_file(
                    source,
                    validation_artifact,
                    str(
                        validation_evidence["real_onnx_gates"][model_key]["sha256"]
                    ),
                )
                artifacts.append(
                    _artifact_record(
                        validation_artifact,
                        f"onnx_gate_validation_{model_key}",
                    )
                )

            if not args.skip_standalone_zip:
                current_tree_hash, _ = _runtime_tree_digest(standalone)
                if (
                    current_tree_hash
                    != validation_evidence["installer_smoke"][
                        "standalone_tree_sha256"
                    ]
                ):
                    raise ReleaseError(
                        "standalone changed after installer/evidence validation"
                    )
                standalone_archive = (
                    output / f"{RELEASE_STEM}-{version}-standalone-win64.zip"
                )
                _write_zip_atomic(standalone, standalone_archive, include_root=True)
                _verify_standalone_archive(standalone_archive, version)
                artifacts.append(
                    _artifact_record(standalone_archive, "standalone_archive")
                )

        source_manifest = output / f"{RELEASE_STEM}-{version}-source-manifest.json"
        _write_json_atomic(
            source_manifest,
            {
                "schema_version": 1,
                "source_archive": source_archive.name,
                "source_archive_sha256": _file_sha256(source_archive),
                "entry_count": len(source_entries),
                "entries": source_entries,
            },
        )
        artifacts.append(_artifact_record(source_manifest, "source_manifest"))

        if args.source_mirror is not None:
            _publish_source_mirror(
                source_tree,
                args.source_mirror,
                replace=args.replace_source_mirror,
                source_archive=source_archive,
                version=version,
            )

        (
            release_status,
            complete_release,
            partial_installer_validation,
        ) = _release_completion_state(
            source_only=args.source_only,
            validation_evidence=validation_evidence,
        )
        partial_requirements: list[str] = []
        if partial_installer_validation:
            partial_requirements.append(
                "Windows uninstall registry integration "
                "(explicit sandbox validation mode)"
            )
            if (
                validation_evidence
                and validation_evidence["installer_smoke"]["gui_status"]
                != "passed"
            ):
                partial_requirements.append("installed GUI startup")
        report = {
            "schema_version": 1,
            "status": release_status,
            "application": RELEASE_APPLICATION,
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "complete_release": complete_release,
            "source_mirror": (
                str(args.source_mirror.resolve())
                if args.source_mirror is not None
                else None
            ),
            "nuitka_report": (
                {
                    "path": str(build_report),
                    "sha256": _file_sha256(build_report),
                }
                if build_report is not None
                else None
            ),
            "environment_lock_sha256": _file_sha256(
                project / "locks" / LOCK_CHECKSUM_NAME
            ),
            "validation_evidence": validation_evidence,
            "code_signing": "unsigned",
            "artifacts": artifacts,
            "known_unverified_external_requirements": [
                "clean Windows machine without Conda",
                "Docker converter images",
                "MaixCAM-Pro physical device",
                "MaixCAM2 physical device",
                *partial_requirements,
            ],
        }
        report_path = output / RELEASE_REPORT_NAME
        _write_json_atomic(report_path, report)
        checksum_artifacts = [*artifacts, _artifact_record(report_path, "release_report")]
        _write_checksums(output / RELEASE_CHECKSUM_NAME, checksum_artifacts)

    print(f"Release deliverables prepared: {output}")
    return 0


class ReleaseError(RuntimeError):
    """A release gate failed."""


def _release_completion_state(
    *,
    source_only: bool,
    validation_evidence: dict[str, Any] | None,
) -> tuple[str, bool, bool]:
    partial_installer = bool(
        validation_evidence
        and validation_evidence["installer_smoke"]["status"]
        == "partial_sandbox_validation"
    )
    return (
        "partial" if partial_installer else "passed",
        not source_only and not partial_installer,
        partial_installer,
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid {label} evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} evidence must be a JSON object: {path}")
    return value


def _verify_nuitka_report(path: Path) -> None:
    if not path.is_file():
        raise ReleaseError(f"missing final Nuitka report: {path}")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ReleaseError(f"invalid final Nuitka report {path}: {exc}") from exc
    if (
        root.tag != "nuitka-compilation-report"
        or root.attrib.get("completion") != "yes"
        or root.attrib.get("mode") != "standalone"
    ):
        raise ReleaseError("Nuitka report is not a successful standalone build")


def _verify_release_evidence(
    *,
    standalone: Path,
    installer: Path,
    gpu_validation: Path,
    onnx_validations: dict[str, Path],
    installer_validation: Path,
    allow_partial_installer_validation: bool = False,
) -> dict[str, Any]:
    worker = (standalone / "AI-Biaozhu-Worker.exe").resolve()
    gui = (standalone / "AI-Biaozhu.exe").resolve()
    worker_hash = _file_sha256(worker)
    gui_hash = _file_sha256(gui)

    gpu = _load_json_object(gpu_validation, "standalone CUDA")
    if gpu.get("status") != "passed":
        raise ReleaseError("standalone CUDA validation did not pass")
    if Path(str(gpu.get("worker") or "")).resolve() != worker:
        raise ReleaseError("CUDA validation belongs to a different frozen Worker")
    if gpu.get("worker_sha256") != worker_hash:
        raise ReleaseError("CUDA validation Worker hash does not match the release")
    if gpu.get("device") not in {0, "0"}:
        raise ReleaseError("CUDA validation did not explicitly exercise device 0")
    if float(gpu.get("gpu_memory_gb_max") or 0) <= 0:
        raise ReleaseError("CUDA validation has no positive GPU-memory evidence")
    for phase in ("train", "predict"):
        details = gpu.get(phase)
        if not isinstance(details, dict) or details.get("return_code") != 0:
            raise ReleaseError(f"CUDA {phase} evidence is incomplete")
        event_types = details.get("event_types")
        if not isinstance(event_types, list) or not event_types or event_types[-1] != "completed":
            raise ReleaseError(f"CUDA {phase} did not end with completed")
        if {"error", "cancelled"}.intersection(str(item) for item in event_types):
            raise ReleaseError(f"CUDA {phase} contains a failure event")
        if not details.get("nvidia_smi_observations"):
            raise ReleaseError(f"CUDA {phase} lacks nvidia-smi process evidence")

    smoke_dir = Path(str(gpu.get("smoke_dir") or "")).resolve()
    checkpoint_paths = gpu.get("checkpoint_paths")
    checkpoint_hashes = gpu.get("checkpoint_sha256")
    if not isinstance(checkpoint_paths, dict) or not isinstance(checkpoint_hashes, dict):
        raise ReleaseError("CUDA validation lacks checkpoint paths/hashes")
    verified_checkpoints: dict[str, tuple[Path, str]] = {}
    for role in ("best", "last"):
        checkpoint = Path(str(checkpoint_paths.get(role) or "")).resolve()
        expected_hash = str(checkpoint_hashes.get(role) or "")
        if (
            checkpoint.name.casefold() != f"{role}.pt"
            or not checkpoint.is_file()
            or not _is_within(checkpoint, smoke_dir)
            or _file_sha256(checkpoint) != expected_hash
        ):
            raise ReleaseError(f"CUDA {role}.pt evidence is stale or unsafe")
        verified_checkpoints[role] = checkpoint, expected_hash

    best_path, best_hash = verified_checkpoints["best"]
    expected_models = {"YOLO26n", "YOLOv8n", "YOLO11n", "YOLOv5n"}
    if set(onnx_validations) != expected_models:
        raise ReleaseError("real ONNX evidence must cover all four model families")
    verified_onnx: dict[str, dict[str, str]] = {}
    for model_key, path in onnx_validations.items():
        onnx = _load_json_object(path, f"{model_key} real ONNX gate")
        if onnx.get("status") != "passed" or onnx.get("model_key") != model_key:
            raise ReleaseError(f"{model_key} real ONNX gate validation did not pass")
        checkpoint = Path(str(onnx.get("checkpoint") or "")).resolve()
        checkpoint_hash = str(onnx.get("checkpoint_sha256") or "")
        if not checkpoint.is_file() or _file_sha256(checkpoint) != checkpoint_hash:
            raise ReleaseError(f"{model_key} ONNX checkpoint evidence is stale")
        if model_key == "YOLO26n" and (
            checkpoint != best_path or checkpoint_hash != best_hash
        ):
            raise ReleaseError(
                "YOLO26n ONNX validation is not bound to the CUDA-tested best.pt"
            )
        targets = onnx.get("targets")
        if not isinstance(targets, dict) or set(targets) != {
            "maixcam_pro",
            "maixcam2",
        }:
            raise ReleaseError(
                f"{model_key} ONNX validation does not cover both Maix targets"
            )
        for target, details in targets.items():
            if not isinstance(details, dict) or details.get("return_code") != 0:
                raise ReleaseError(f"{model_key}/{target} ONNX gate did not complete")
            validations = details.get("numeric_validations")
            if (
                not isinstance(validations, list)
                or len(validations) != 2
                or not all(
                    isinstance(item, dict) and item.get("ok")
                    for item in validations
                )
            ):
                raise ReleaseError(
                    f"{model_key}/{target} ONNX numeric validation is incomplete"
                )
            if details.get("device_validation") != "required":
                raise ReleaseError(
                    f"{model_key}/{target} evidence lost the physical-device gate"
                )
        verified_onnx[model_key] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "checkpoint_sha256": checkpoint_hash,
        }

    runtime_tree_hash, runtime_file_count = _runtime_tree_digest(standalone)
    installed = _load_json_object(installer_validation, "installer smoke")
    installer_status = str(installed.get("status") or "")
    partial_installer = installer_status == "partial_sandbox_validation"
    if installer_status != "passed" and not (
        partial_installer and allow_partial_installer_validation
    ):
        raise ReleaseError(
            "installer smoke validation did not fully pass; "
            "explicit partial acceptance was not enabled"
        )
    if (
        Path(str(installed.get("installer") or "")).resolve() != installer.resolve()
        or installed.get("installer_sha256") != _file_sha256(installer)
    ):
        raise ReleaseError("installer smoke is not bound to the release installer")
    if Path(str(installed.get("standalone_root") or "")).resolve() != standalone.resolve():
        raise ReleaseError("installer smoke used a different standalone directory")
    if installed.get("standalone_tree_sha256") != runtime_tree_hash:
        raise ReleaseError("installer smoke standalone tree hash is stale")
    if installed.get("installed_tree_sha256") != runtime_tree_hash:
        raise ReleaseError("installed runtime differs from the frozen directory")
    if installed.get("standalone_file_count") != runtime_file_count:
        raise ReleaseError("installer smoke standalone file count is stale")
    if installed.get("installed_file_count") != runtime_file_count:
        raise ReleaseError("installed runtime file count differs from standalone")
    expected_entries = {
        "AI-Biaozhu.exe": gui_hash,
        "AI-Biaozhu-Worker.exe": worker_hash,
    }
    if installed.get("standalone_entry_hashes") != expected_entries:
        raise ReleaseError("installer smoke standalone entry hashes are stale")
    if installed.get("installed_entry_hashes") != expected_entries:
        raise ReleaseError("installed entry-point hashes differ from standalone")
    for phase in ("install", "worker_help", "uninstall"):
        details = installed.get(phase)
        if not isinstance(details, dict) or details.get("return_code") != 0:
            raise ReleaseError(f"installer smoke {phase} step did not pass")
    environment = installed.get("environment_completed")
    if not (
        isinstance(environment, dict)
        and environment.get("type") == "completed"
        and isinstance(environment.get("payload"), dict)
        and environment["payload"].get("gpu_ready")
    ):
        raise ReleaseError("installed environment/GPU probe is incomplete")
    gui_details = installed.get("gui")
    gui_status = (
        str(gui_details.get("status") or "")
        if isinstance(gui_details, dict)
        else ""
    )
    if partial_installer:
        if not (
            installed.get("sandbox_no_uninstall_registry") is True
            and installed.get("registry_validation")
            == "not_verified_sandbox_mode"
            and gui_status in {"passed", "sandbox_blocked"}
        ):
            raise ReleaseError("partial installer evidence is not explicit sandbox mode")
    elif not (
        installed.get("sandbox_no_uninstall_registry") is False
        and installed.get("registry_validation") == "verified"
        and gui_status == "passed"
    ):
        raise ReleaseError("installed GUI/registry was not fully verified")

    registrations_before = installed.get("registrations_before")
    registrations_during = installed.get("registrations_during")
    registrations_after = installed.get("registrations_after")
    if not all(
        isinstance(value, list)
        for value in (
            registrations_before,
            registrations_during,
            registrations_after,
        )
    ):
        raise ReleaseError("installer registry evidence is malformed")
    registry_items = [
        *registrations_before,
        *registrations_during,
        *registrations_after,
    ]
    if not all(isinstance(item, dict) for item in registry_items):
        raise ReleaseError("installer registry evidence contains a non-object entry")
    if not partial_installer and any(
        item.get("probe_error") for item in registry_items
    ):
        raise ReleaseError("normal installer registry probe was incomplete")
    actual_before = [
        item
        for item in registrations_before
        if isinstance(item, dict) and not item.get("probe_error")
    ]
    actual_during = [
        item
        for item in registrations_during
        if isinstance(item, dict) and not item.get("probe_error")
    ]
    actual_after = [
        item
        for item in registrations_after
        if isinstance(item, dict) and not item.get("probe_error")
    ]
    if actual_before or actual_after:
        raise ReleaseError("installer smoke touched an existing/remaining registration")
    if partial_installer:
        if actual_during:
            raise ReleaseError("sandbox installer unexpectedly created a registration")
    elif not actual_during:
        raise ReleaseError("normal installer registration creation was not verified")
    if (
        installed.get("failure")
        or installed.get("cleanup_failure")
        or not installed.get("install_root_removed")
        or installed.get("shortcuts_unchanged") is not True
        or int(installed.get("forbidden_file_count") or 0) != 0
        or int(installed.get("forbidden_directory_count") or 0) != 0
    ):
        raise ReleaseError("installer smoke cleanup or runtime audit failed")

    return {
        "standalone_cuda": {
            "path": str(gpu_validation),
            "sha256": _file_sha256(gpu_validation),
            "worker_sha256": worker_hash,
            "best_checkpoint_sha256": best_hash,
        },
        "real_onnx_gates": verified_onnx,
        "installer_smoke": {
            "path": str(installer_validation),
            "sha256": _file_sha256(installer_validation),
            "installer_sha256": _file_sha256(installer),
            "standalone_tree_sha256": runtime_tree_hash,
            "status": installer_status,
            "registry_validation": str(installed.get("registry_validation") or ""),
            "gui_status": gui_status,
        },
    }


def _runtime_tree_digest(root: Path) -> tuple[str, int]:
    _reject_reparse(root)
    entries: dict[str, str] = {}
    for path in root.rglob("*"):
        _reject_reparse(path)
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().casefold()
        if relative in entries:
            raise ReleaseError(f"case-insensitive runtime collision: {relative}")
        entries[relative] = _file_sha256(path)
    digest = hashlib.sha256()
    for relative, file_hash in sorted(entries.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(entries)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _project_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip().strip("\"'")
            if value and all(character.isdigit() or character == "." for character in value):
                return value
    raise ReleaseError(f"could not read a numeric release version from {path}")


def _copy_source_tree(project: Path, destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ROOT_SOURCE_FILES:
        source = project / name
        if not source.is_file():
            raise ReleaseError(f"required source file is missing: {source}")
        _reject_reparse(source)
        shutil.copy2(source, destination / name)
    for name in SOURCE_DIRECTORIES:
        source = project / name
        if not source.is_dir():
            raise ReleaseError(f"required source directory is missing: {source}")
        _copy_allowed_directory(source, destination / name)

    required = (
        destination / SOURCE_MODEL_SEED,
        destination / "src" / "ai_biaozhu" / "app.py",
        destination / "scripts" / "build_windows.ps1",
        destination / "scripts" / "build_installer.ps1",
        destination / "third_party" / "runtime" / "yolov5" / "train.py",
        destination / "third_party" / "runtime" / "yolov5" / "LICENSE",
        destination
        / "third_party"
        / "runtime"
        / "yolov5"
        / ".ai-biaozhu-yolov5-tag",
    )
    for path in required:
        if not path.is_file():
            raise ReleaseError(f"source archive would be incomplete: {path}")
    return _tree_manifest(destination)


def _copy_allowed_directory(source: Path, destination: Path) -> None:
    for current, directory_names, file_names in os.walk(source, topdown=True):
        current_path = Path(current)
        _reject_reparse(current_path)
        accepted_directories: list[str] = []
        for name in sorted(directory_names, key=str.casefold):
            child = current_path / name
            _reject_reparse(child)
            if name in SKIPPED_DIRECTORY_NAMES or name.endswith(".egg-info"):
                continue
            accepted_directories.append(name)
        directory_names[:] = accepted_directories

        relative = current_path.relative_to(source)
        target_directory = destination / relative
        target_directory.mkdir(parents=True, exist_ok=True)
        for name in sorted(file_names, key=str.casefold):
            child = current_path / name
            _reject_reparse(child)
            if name in SKIPPED_FILE_NAMES:
                continue
            suffix = child.suffix.casefold()
            source_relative = PurePosixPath(
                source.name,
                *relative.parts,
                name,
            )
            is_seed = source_relative == SOURCE_MODEL_SEED
            if suffix in FORBIDDEN_SOURCE_SUFFIXES and not is_seed:
                raise ReleaseError(f"forbidden generated/binary file in source tree: {child}")
            if is_seed:
                _verify_model_seed_file(child)
            target = target_directory / name
            shutil.copy2(child, target)


def _verify_standalone(root: Path) -> None:
    if not root.is_dir():
        raise ReleaseError(f"missing standalone directory: {root}")
    _reject_reparse(root)
    required = (
        root / "AI-Biaozhu.exe",
        root / "AI-Biaozhu-Worker.exe",
        root / "LICENSE",
        root / "THIRD_PARTY_NOTICES.md",
        root / "THIRD_PARTY_LICENSES" / "index.json",
        root / RUNTIME_MODEL_SEED,
        root / "third_party" / "yolov5" / "train.py",
        root / "third_party" / "yolov5" / ".ai-biaozhu-yolov5-tag",
    )
    for path in required:
        if not path.is_file():
            raise ReleaseError(f"standalone is incomplete: {path}")
    _verify_pe_file(root / "AI-Biaozhu.exe", minimum_size=10_000)
    _verify_pe_file(root / "AI-Biaozhu-Worker.exe", minimum_size=10_000)
    seen_casefold: set[str] = set()
    for path in sorted(root.rglob("*")):
        _reject_reparse(path)
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__"} for part in relative.parts):
            raise ReleaseError(f"forbidden directory in standalone: {relative}")
        if (
            path.is_file()
            and path.suffix.casefold() in FORBIDDEN_RUNTIME_SUFFIXES
            and PurePosixPath(relative.as_posix()) != RUNTIME_MODEL_SEED
        ):
            raise ReleaseError(f"model artifact was bundled in standalone: {relative}")
        key = relative.as_posix().casefold()
        if key in seen_casefold:
            raise ReleaseError(f"case-insensitive path collision in standalone: {relative}")
        seen_casefold.add(key)
    _verify_model_seed_file(root / RUNTIME_MODEL_SEED)


def _verify_model_seed_file(path: Path) -> None:
    if not path.is_file():
        raise ReleaseError(f"missing verified model seed: {path}")
    size = path.stat().st_size
    digest = _file_sha256(path)
    if size != MODEL_SEED_SIZE or digest != MODEL_SEED_SHA256:
        raise ReleaseError(
            "bundled yolo26s.pt does not match its locked size/SHA-256: "
            f"{path}"
        )


def _verify_pe_file(path: Path, *, minimum_size: int) -> None:
    if not path.is_file() or path.stat().st_size < minimum_size:
        raise ReleaseError(f"missing or implausibly small Windows executable: {path}")
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ReleaseError(f"file does not have a PE MZ header: {path}")


def _write_zip_atomic(source: Path, destination: Path, *, include_root: bool) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            prefix = source.name if include_root else ""
            for path in sorted(item for item in source.rglob("*") if item.is_file()):
                _reject_reparse(path)
                relative = path.relative_to(source).as_posix()
                archive_name = f"{prefix}/{relative}" if prefix else relative
                info = zipfile.ZipInfo(archive_name, ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                with (
                    path.open("rb") as input_stream,
                    archive.open(info, "w", force_zip64=True) as output_stream,
                ):
                    shutil.copyfileobj(
                        input_stream,
                        output_stream,
                        length=4 * 1024 * 1024,
                    )
        _replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_source_archive(path: Path, version: str) -> None:
    prefix = f"{RELEASE_STEM}-{version}-source/"
    required = {
        prefix + "LICENSE",
        prefix + "README.md",
        prefix + SOURCE_MODEL_SEED.as_posix(),
        prefix + "src/ai_biaozhu/app.py",
        prefix + "third_party/runtime/yolov5/train.py",
        prefix + "third_party/runtime/yolov5/LICENSE",
        prefix
        + "third_party/runtime/yolov5/.ai-biaozhu-yolov5-tag",
    }
    names = _verify_zip_entries(path)
    missing = sorted(required.difference(names))
    if missing:
        raise ReleaseError("source ZIP is incomplete: " + ", ".join(missing))
    for name in names:
        suffix = PurePosixPath(name).suffix.casefold()
        if (
            suffix in FORBIDDEN_SOURCE_SUFFIXES
            and name != prefix + SOURCE_MODEL_SEED.as_posix()
        ):
            raise ReleaseError(f"forbidden file in source ZIP: {name}")
    _verify_zip_model_seed(path, prefix + SOURCE_MODEL_SEED.as_posix())


def _verify_standalone_archive(path: Path, version: str) -> None:
    expected_filename = f"{RELEASE_STEM}-{version}-standalone-win64.zip"
    if path.name != expected_filename:
        raise ReleaseError(f"unexpected standalone ZIP filename: {path.name}")
    # ``_write_zip_atomic`` uses the real standalone directory name.
    expected_prefix = "AI-Biaozhu.dist/"
    names = _verify_zip_entries(path)
    required = {
        expected_prefix + "AI-Biaozhu.exe",
        expected_prefix + "AI-Biaozhu-Worker.exe",
        expected_prefix + "LICENSE",
        expected_prefix + RUNTIME_MODEL_SEED.as_posix(),
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ReleaseError("standalone ZIP is incomplete: " + ", ".join(missing))
    for name in names:
        if (
            PurePosixPath(name).suffix.casefold() in FORBIDDEN_RUNTIME_SUFFIXES
            and name != expected_prefix + RUNTIME_MODEL_SEED.as_posix()
        ):
            raise ReleaseError(f"model artifact in standalone ZIP: {name}")
    _verify_zip_model_seed(path, expected_prefix + RUNTIME_MODEL_SEED.as_posix())


def _verify_zip_model_seed(path: Path, member: str) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise ReleaseError(f"missing verified model seed in {path}: {member}") from exc
        digest = hashlib.sha256()
        with archive.open(info, "r") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    if info.file_size != MODEL_SEED_SIZE or digest.hexdigest() != MODEL_SEED_SHA256:
        raise ReleaseError(
            "bundled yolo26s.pt in ZIP does not match its locked "
            f"size/SHA-256: {member}"
        )


def _verify_zip_entries(path: Path) -> set[str]:
    seen: set[str] = set()
    casefolded: set[str] = set()
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise ReleaseError(f"corrupt ZIP entry in {path}")
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ReleaseError(f"unsafe ZIP path in {path}: {name}")
            if name in seen:
                raise ReleaseError(f"duplicate ZIP entry in {path}: {name}")
            folded = name.casefold()
            if folded in casefolded:
                raise ReleaseError(f"case-insensitive ZIP collision in {path}: {name}")
            seen.add(name)
            casefolded.add(folded)
    return seen


def _publish_source_mirror(
    source: Path,
    destination: Path,
    *,
    replace: bool,
    source_archive: Path,
    version: str,
) -> None:
    unresolved_target = destination.absolute()
    if unresolved_target.exists():
        _reject_reparse(unresolved_target)
    target = unresolved_target.resolve()
    _assert_safe_mirror(target)
    if target.exists():
        entries = list(target.iterdir())
        marker = target / MIRROR_MARKER
        if entries:
            if not replace:
                raise ReleaseError(
                    f"source mirror is non-empty; pass --replace-source-mirror: {target}"
                )
            if not marker.is_file():
                raise ReleaseError(
                    f"refusing to replace an unowned directory without {MIRROR_MARKER}: "
                    f"{target}"
                )
            _reject_reparse(marker)
            marker_value = _load_json_object(marker, "source mirror marker")
            if marker_value.get("application") != SOURCE_MIRROR_APPLICATION:
                raise ReleaseError(f"source mirror ownership marker is invalid: {marker}")
            for child in target.rglob("*"):
                _reject_reparse(child)
            shutil.rmtree(target)
    shutil.copytree(source, target, dirs_exist_ok=True)
    _write_json_atomic(
        target / MIRROR_MARKER,
        {
            "schema_version": 1,
            "application": SOURCE_MIRROR_APPLICATION,
            "version": version,
            "source_archive_sha256": _file_sha256(source_archive),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def _assert_safe_output(output: Path, project: Path) -> None:
    forbidden = {Path(output.anchor), Path.home().resolve(), project}
    if output in forbidden or _is_within(output, project) or len(output.parts) < 3:
        raise ReleaseError(f"unsafe output directory: {output}")


def _assert_safe_mirror(path: Path) -> None:
    if path == Path(path.anchor) or path == Path.home().resolve() or len(path.parts) < 3:
        raise ReleaseError(f"unsafe source mirror path: {path}")


def _clean_generated_output(output: Path) -> None:
    for path in output.iterdir():
        if not path.is_file():
            continue
        if (
            path.name.startswith(f"{RELEASE_STEM}-")
            or path.name in {RELEASE_REPORT_NAME, RELEASE_CHECKSUM_NAME}
        ):
            path.unlink()


def _invalidate_release_metadata(output: Path) -> None:
    for name in (RELEASE_REPORT_NAME, RELEASE_CHECKSUM_NAME):
        path = output / name
        if path.is_file():
            path.unlink()


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _copy_file_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        _replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_verified_file(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    if _file_sha256(source) != expected_sha256:
        raise ReleaseError(f"validated source changed before publication: {source}")
    _copy_file_atomic(source, destination)
    if _file_sha256(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ReleaseError(f"published artifact hash mismatch: {destination}")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checksums(path: Path, artifacts: list[dict[str, Any]]) -> None:
    lines = [
        f"{item['sha256']}  {item['filename']}"
        for item in sorted(artifacts, key=lambda value: str(value["filename"]).casefold())
    ]
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _reject_reparse(path: Path) -> None:
    if path.is_symlink():
        raise ReleaseError(f"symbolic link is not allowed in release inputs: {path}")
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except AttributeError:
        return
    if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ReleaseError(f"reparse point is not allowed in release inputs: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
