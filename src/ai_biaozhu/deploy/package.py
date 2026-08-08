"""Build separate model-only and full MaixPy packages from strict allowlists."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ai_biaozhu.ml.model_registry import get_model

from .maix import Cam2NpuMode, MaixTarget
from .mud import build_mud
from .onnx_gate import file_sha256, validate_class_names
from .report import artifact_record, deployment_report, write_deployment_report
from .templates import app_yaml, maixpy_main

SIZE_WARNING_BYTES = 30_000_000
_MAIXPY_MINIMUMS = {
    "yolov8": "4.3.0",
    "yolo11": "4.7.0",
    "yolo26": "4.12.5",
}


class DeploymentCancelled(RuntimeError):
    """Raised when the caller declines an oversized staged package."""


class OversizeConfirmationRequired(RuntimeError):
    def __init__(self, warnings: Sequence[PackageSizeWarning]) -> None:
        super().__init__("部署包超过建议大小，需要用户确认")
        self.warnings = tuple(warnings)


@dataclass(frozen=True, slots=True)
class DeploymentArtifact:
    source: Path
    archive_name: str


@dataclass(frozen=True, slots=True)
class PackageSizeWarning:
    package_kind: str
    zip_size: int
    unpacked_size: int
    largest_files: tuple[tuple[str, int], ...]

    def message(self) -> str:
        if self.zip_size == 0 and self.package_kind == "editable-project":
            return (
                f"{self.package_kind}：目录总计 {self.unpacked_size} 字节，超过 "
                f"{SIZE_WARNING_BYTES} 字节建议值"
            )
        return (
            f"{self.package_kind}：压缩后 {self.zip_size} 字节，"
            f"解压后 {self.unpacked_size} 字节，超过 "
            f"{SIZE_WARNING_BYTES} 字节建议值"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "package_kind": self.package_kind,
            "zip_size": self.zip_size,
            "unpacked_size": self.unpacked_size,
            "largest_files": [
                {"path": name, "size": size} for name, size in self.largest_files
            ],
            "threshold": SIZE_WARNING_BYTES,
        }


@dataclass(frozen=True, slots=True)
class DeploymentPackageResult:
    model_package_path: Path | None
    app_package_path: Path | None
    editable_project_path: Path | None
    report_path: Path
    sha256_path: Path
    warnings: tuple[str, ...]
    size_warnings: tuple[PackageSizeWarning, ...]
    model_files: tuple[str, ...]
    app_files: tuple[str, ...]
    report: Mapping[str, Any]
    artifacts: tuple[PublishedDeploymentArtifact, ...] = ()

    @property
    def package_path(self) -> Path:
        """Compatibility alias: the primary package is the full application."""

        selected = (
            self.app_package_path
            or self.model_package_path
            or self.editable_project_path
        )
        if selected is None:  # pragma: no cover - guarded by output validation
            raise RuntimeError("部署结果中没有可用产物")
        return selected

    @property
    def files(self) -> tuple[str, ...]:
        return self.app_files


@dataclass(frozen=True, slots=True)
class PublishedDeploymentArtifact:
    """A package or editable directory permanently published for the user."""

    kind: str
    path: Path
    is_directory: bool
    files: tuple[str, ...]
    sha256: str


OversizeConfirmation = Callable[[tuple[PackageSizeWarning, ...]], bool]

_PACKAGE_OUTPUT_ALIASES = {
    "model_only": "model_only",
    "model-only": "model_only",
    "full_app": "maixapp",
    "full-app": "maixapp",
    "maixapp": "maixapp",
    "editable_project": "editable_project",
    "editable-project": "editable_project",
    "maixvision_project": "editable_project",
}


def validate_deployment_class_names(
    checkpoint_class_names: Sequence[str],
    deployment_class_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate editable display aliases without changing checkpoint labels."""

    checkpoint = validate_class_names(checkpoint_class_names)
    deployed = validate_class_names(
        checkpoint if deployment_class_names is None else deployment_class_names
    )
    if len(deployed) != len(checkpoint):
        raise ValueError(
            "部署显示类别数量必须与 checkpoint 原始类别数量一致"
        )
    return deployed


def _normalize_package_outputs(values: Sequence[str] | None) -> tuple[str, ...]:
    raw = ("model_only", "maixapp") if values is None else tuple(values)
    selected: list[str] = []
    for value in raw:
        key = str(value).strip().casefold()
        normalized = _PACKAGE_OUTPUT_ALIASES.get(key)
        if normalized is None:
            raise ValueError(f"未知部署输出类型：{value}")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("至少选择一种部署输出：maixapp 或可编辑工程")
    return tuple(selected)


def build_deployment_package(
    *,
    package_path: str | Path,
    target: MaixTarget,
    model_key: str,
    model_artifacts: Sequence[DeploymentArtifact | tuple[str | Path, str]],
    class_names: Sequence[str],
    deployment_class_names: Sequence[str] | None = None,
    input_height: int,
    input_width: int,
    camera_height: int | None = None,
    camera_width: int | None = None,
    source_run_id: str | None = None,
    checkpoint_role: str | None = None,
    source_checkpoint: str | Path | None = None,
    source_onnx: str | Path | None = None,
    output_tensors: Sequence[str] = (),
    calibration_count: int | None = 100,
    converter_image: str | None = None,
    converter_config: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, Any] | None = None,
    maixpy_version: str | None = None,
    maixcdk_commit: str | None = None,
    cam2_npu_mode: Cam2NpuMode = Cam2NpuMode.BOTH,
    anchors: Sequence[float] | None = None,
    confidence: float = 0.35,
    iou: float = 0.45,
    max_det: int = 100,
    dual_buff: bool = True,
    model_package_path: str | Path | None = None,
    editable_project_path: str | Path | None = None,
    package_outputs: Sequence[str] | None = None,
    report_path: str | Path | None = None,
    sha256_path: str | Path | None = None,
    allow_oversize: bool = True,
    oversize_confirmation: OversizeConfirmation | None = None,
) -> DeploymentPackageResult:
    """Stage, verify and publish selected package forms.

    ``package_outputs=None`` preserves the historical behavior (model-only
    archive plus full ``.maixapp``).  New callers may request ``maixapp``,
    ``editable_project`` or both; the legacy spelling ``full_app`` remains an
    alias for ``maixapp``.
    """

    labels = validate_class_names(class_names)
    display_labels = validate_deployment_class_names(
        labels,
        deployment_class_names,
    )
    selected_outputs = _normalize_package_outputs(package_outputs)
    mode = Cam2NpuMode(cam2_npu_mode)
    app_destination = _package_destination(package_path)
    model_destination = _package_destination(
        model_package_path
        or app_destination.with_name(
            f"{app_destination.stem}-model{app_destination.suffix}"
        )
    )
    editable_destination = Path(
        editable_project_path
        or app_destination.with_name(f"{app_destination.stem}-maixvision-project")
    )
    if editable_destination.parent == editable_destination:
        raise ValueError("可编辑 MaixVision 工程不能发布到磁盘根目录")
    report_destination = Path(
        report_path
        or app_destination.with_name(f"{app_destination.stem}-deployment-report.json")
    )
    sha_destination = Path(
        sha256_path
        or app_destination.with_name(f"{app_destination.stem}-SHA256SUMS.txt")
    )
    destinations = [report_destination.resolve(), sha_destination.resolve()]
    if "maixapp" in selected_outputs:
        destinations.append(app_destination.resolve())
    if "model_only" in selected_outputs:
        destinations.append(model_destination.resolve())
    if "editable_project" in selected_outputs:
        destinations.append(editable_destination.resolve())
    if len(set(destinations)) != len(destinations):
        raise ValueError("部署产物、报告和 SHA 文件路径必须互不相同")
    normalized = tuple(_normalize_artifact(item) for item in model_artifacts)
    _validate_target_artifacts(target, normalized, mode)
    for artifact in normalized:
        if _is_reparse_point(artifact.source):
            raise ValueError(f"拒绝打包符号链接或 reparse point：{artifact.source}")
        if not artifact.source.is_file():
            raise ValueError(f"部署产物不存在：{artifact.source}")
    if input_height <= 0 or input_width <= 0:
        raise ValueError("模型输入尺寸必须大于 0")
    camera_height = int(camera_height or input_height)
    camera_width = int(camera_width or input_width)
    if camera_height <= 0 or camera_width <= 0:
        raise ValueError("相机尺寸必须大于 0")
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("confidence/iou 必须在 0 到 1 之间")
    if max_det <= 0:
        raise ValueError("max_det 必须大于 0")
    app_destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{app_destination.stem}-deploy-",
        dir=app_destination.parent,
    ) as temporary:
        staging = Path(temporary)
        model_root = staging / "model-only"
        app_root = staging / "full-app"
        model_root.mkdir()
        (app_root / "models").mkdir(parents=True)

        for artifact in normalized:
            shutil.copy2(artifact.source, model_root / artifact.archive_name)
            shutil.copy2(artifact.source, app_root / "models" / artifact.archive_name)
        mud_files = _mud_model_files(target, mode)
        mud_text = build_mud(
            target=target,
            model_key=model_key,
            class_names=display_labels,
            input_height=input_height,
            input_width=input_width,
            model_files=mud_files,
            cam2_npu_mode=mode,
            anchors=anchors,
        )
        (model_root / "model.mud").write_text(mud_text, encoding="utf-8", newline="\n")
        (app_root / "models" / "model.mud").write_text(
            mud_text,
            encoding="utf-8",
            newline="\n",
        )
        config = {
            "schema_version": "1.0",
            "checkpoint_class_names": list(labels),
            "class_names": list(display_labels),
            "class_name_mapping": [
                {"checkpoint": source, "display": display}
                for source, display in zip(labels, display_labels, strict=True)
            ],
            "camera": {"width": camera_width, "height": camera_height},
            "confidence": confidence,
            "iou": iou,
            "max_det": max_det,
            "dual_buff": dual_buff,
            "target": target.value,
            "cam2_npu_mode": mode.value if target is MaixTarget.MAIXCAM2 else None,
            # AI-ISP is a MaixCAM2 system setting selected before the camera
            # pipeline starts.  The generated app verifies the required mode
            # before loading a single-mode model instead of silently using an
            # incompatible NPU partition.
            "ai_isp_mode": (
                {
                    Cam2NpuMode.NPU2: "disabled",
                    Cam2NpuMode.VNPU: "required",
                    Cam2NpuMode.BOTH: "system",
                }[mode]
                if target is MaixTarget.MAIXCAM2
                else "not_applicable"
            ),
        }
        (app_root / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (app_root / "main.py").write_text(
            maixpy_main(model_key),
            encoding="utf-8",
            newline="\n",
        )
        model_allowlist = [
            *(artifact.archive_name for artifact in normalized),
            "model.mud",
        ]
        app_allowlist = [
            "app.yaml",
            "main.py",
            "config.json",
            "models/model.mud",
            *(f"models/{artifact.archive_name}" for artifact in normalized),
        ]
        (app_root / "app.yaml").write_text(
            app_yaml(app_allowlist),
            encoding="utf-8",
            newline="\n",
        )
        staged_model = staging / model_destination.name
        staged_app = staging / app_destination.name
        if "model_only" in selected_outputs:
            _write_deterministic_zip(model_root, staged_model, model_allowlist)
            _verify_zip(staged_model, model_root, model_allowlist)
        if "maixapp" in selected_outputs:
            _write_deterministic_zip(app_root, staged_app, app_allowlist)
            _verify_zip(staged_app, app_root, app_allowlist)
        if "editable_project" in selected_outputs:
            _verify_directory(app_root, app_allowlist)
        size_warnings = tuple(
            warning
            for warning in [
                (
                    _size_warning(
                        "model-only", staged_model, model_root, model_allowlist
                    )
                    if "model_only" in selected_outputs
                    else None
                ),
                (
                    _size_warning("maixapp", staged_app, app_root, app_allowlist)
                    if "maixapp" in selected_outputs
                    else None
                ),
                (
                    _directory_size_warning(
                        "editable-project", app_root, app_allowlist
                    )
                    if "editable_project" in selected_outputs
                    else None
                ),
            ]
            if warning is not None
        )
        if size_warnings and not allow_oversize:
            if oversize_confirmation is None:
                raise OversizeConfirmationRequired(size_warnings)
            if not oversize_confirmation(size_warnings):
                raise DeploymentCancelled("用户取消发布超出建议大小的部署包")
        warning_messages = tuple(item.message() for item in size_warnings)
        runtime_records = [
            artifact_record(model_root / artifact.archive_name, artifact.archive_name)
            for artifact in normalized
        ]
        model_file_records = [
            artifact_record(model_root / name, name) for name in model_allowlist
        ]
        app_file_records = [
            artifact_record(app_root / name, name) for name in app_allowlist
        ]
        resolved_checkpoint_role = checkpoint_role
        if (
            not resolved_checkpoint_role
            and source_checkpoint is not None
            and Path(source_checkpoint).stem.casefold() in {"best", "last"}
        ):
            resolved_checkpoint_role = Path(source_checkpoint).stem.casefold()
        report = deployment_report(
            target=target.value,
            model_key=model_key,
            artifacts=runtime_records,
            warnings=warning_messages,
            source_run_id=source_run_id,
            checkpoint_role=resolved_checkpoint_role,
            source_checkpoint=source_checkpoint,
            source_onnx=source_onnx,
            class_names=labels,
            input_shape=(1, 3, input_height, input_width),
            output_tensors=output_tensors,
            calibration_count=calibration_count,
            converter_image=converter_image,
            converter_config={
                **dict(converter_config or {}),
                "cam2_npu_mode": (
                    mode.value if target is MaixTarget.MAIXCAM2 else None
                ),
            },
            tool_versions=tool_versions,
            maixpy_min_version=_MAIXPY_MINIMUMS.get(get_model(model_key).family),
            maixpy_version=maixpy_version,
            maixcdk_commit=maixcdk_commit,
            package_files={
                **(
                    {"model-only": model_file_records}
                    if "model_only" in selected_outputs
                    else {}
                ),
                **(
                    {"full-app": app_file_records}
                    if "maixapp" in selected_outputs
                    else {}
                ),
                **(
                    {"editable-project": app_file_records}
                    if "editable_project" in selected_outputs
                    else {}
                ),
            },
        )
        report["checkpoint_class_names"] = list(labels)
        report["deployment_class_names"] = list(display_labels)
        report["class_name_mapping"] = [
            {"checkpoint": source, "display": display}
            for source, display in zip(labels, display_labels, strict=True)
        ]
        package_records: list[dict[str, Any]] = []
        if "model_only" in selected_outputs:
            package_records.append(
                {
                    "kind": "model-only",
                    "filename": model_destination.name,
                    "zip_size": staged_model.stat().st_size,
                    "unpacked_size": _unpacked_size(model_root, model_allowlist),
                    "sha256": file_sha256(staged_model),
                    "files": model_file_records,
                }
            )
        if "maixapp" in selected_outputs:
            package_records.append(
                {
                    "kind": "full-app",
                    "filename": app_destination.name,
                    "zip_size": staged_app.stat().st_size,
                    "unpacked_size": _unpacked_size(app_root, app_allowlist),
                    "sha256": file_sha256(staged_app),
                    "files": app_file_records,
                }
            )
        if "editable_project" in selected_outputs:
            package_records.append(
                {
                    "kind": "editable-project",
                    "directory": editable_destination.name,
                    "unpacked_size": _unpacked_size(app_root, app_allowlist),
                    "sha256": _directory_tree_sha256(app_root, app_allowlist),
                    "files": app_file_records,
                }
            )
        report["packages"] = package_records
        staged_report = staging / report_destination.name
        write_deployment_report(staged_report, report)
        staged_sha = staging / sha_destination.name
        staged_sha.write_text(
            "\n".join(
                f"{item['sha256']}  {item.get('filename') or item.get('directory')}"
                for item in report["packages"]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if "model_only" in selected_outputs:
            _atomic_publish(staged_model, model_destination)
        if "maixapp" in selected_outputs:
            _atomic_publish(staged_app, app_destination)
        if "editable_project" in selected_outputs:
            _atomic_publish_directory(app_root, editable_destination)
        _atomic_publish(staged_report, report_destination)
        _atomic_publish(staged_sha, sha_destination)
    published_artifacts: list[PublishedDeploymentArtifact] = []
    if "model_only" in selected_outputs:
        published_artifacts.append(
            PublishedDeploymentArtifact(
                "model-only",
                model_destination,
                False,
                tuple(model_allowlist),
                file_sha256(model_destination),
            )
        )
    if "maixapp" in selected_outputs:
        published_artifacts.append(
            PublishedDeploymentArtifact(
                "maixapp",
                app_destination,
                False,
                tuple(app_allowlist),
                file_sha256(app_destination),
            )
        )
    if "editable_project" in selected_outputs:
        published_artifacts.append(
            PublishedDeploymentArtifact(
                "editable-project",
                editable_destination,
                True,
                tuple(app_allowlist),
                _directory_tree_sha256(editable_destination, app_allowlist),
            )
        )
    published_artifacts.extend(
        (
            PublishedDeploymentArtifact(
                "deployment-report",
                report_destination,
                False,
                (report_destination.name,),
                file_sha256(report_destination),
            ),
            PublishedDeploymentArtifact(
                "sha256-manifest",
                sha_destination,
                False,
                (sha_destination.name,),
                file_sha256(sha_destination),
            ),
        )
    )
    return DeploymentPackageResult(
        model_package_path=(
            model_destination if "model_only" in selected_outputs else None
        ),
        app_package_path=(app_destination if "maixapp" in selected_outputs else None),
        editable_project_path=(
            editable_destination if "editable_project" in selected_outputs else None
        ),
        report_path=report_destination,
        sha256_path=sha_destination,
        warnings=warning_messages,
        size_warnings=size_warnings,
        model_files=tuple(model_allowlist),
        app_files=tuple(app_allowlist),
        report=report,
        artifacts=tuple(published_artifacts),
    )


def _package_destination(value: str | Path) -> Path:
    result = Path(value)
    if result.suffix.casefold() not in {".zip", ".maixapp"}:
        raise ValueError("部署包扩展名必须是 .zip 或 .maixapp")
    return result


def _normalize_artifact(
    value: DeploymentArtifact | tuple[str | Path, str],
) -> DeploymentArtifact:
    artifact = (
        value
        if isinstance(value, DeploymentArtifact)
        else DeploymentArtifact(Path(value[0]), value[1])
    )
    name = artifact.archive_name.replace("\\", "/")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in name
    ):
        raise ValueError(f"模型产物必须使用安全的根目录文件名：{artifact.archive_name!r}")
    return DeploymentArtifact(Path(artifact.source), name)


def _validate_target_artifacts(
    target: MaixTarget,
    artifacts: Sequence[DeploymentArtifact],
    mode: Cam2NpuMode,
) -> None:
    names = {item.archive_name for item in artifacts}
    if len(names) != len(artifacts):
        raise ValueError("部署产物包内名称不能重复")
    if target is MaixTarget.MAIXCAM_PRO:
        expected = {"model.cvimodel"}
    else:
        expected = {
            Cam2NpuMode.NPU2: {"model_npu.axmodel"},
            Cam2NpuMode.VNPU: {"model_vnpu.axmodel"},
            Cam2NpuMode.BOTH: {"model_npu.axmodel", "model_vnpu.axmodel"},
        }[mode]
    if names != expected:
        missing = expected - names
        unexpected = names - expected
        details = []
        if missing:
            details.append(f"缺少 {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"包含非运行时文件 {', '.join(sorted(unexpected))}")
        raise ValueError(f"{target.value} 产物白名单不匹配：" + "；".join(details))


def _mud_model_files(target: MaixTarget, mode: Cam2NpuMode) -> dict[str, str]:
    if target is MaixTarget.MAIXCAM_PRO:
        return {"model": "model.cvimodel"}
    result: dict[str, str] = {}
    if mode in {Cam2NpuMode.NPU2, Cam2NpuMode.BOTH}:
        result["model_npu"] = "model_npu.axmodel"
    if mode in {Cam2NpuMode.VNPU, Cam2NpuMode.BOTH}:
        result["model_vnpu"] = "model_vnpu.axmodel"
    return result


def _size_warning(
    kind: str,
    package_path: Path,
    root: Path,
    allowlist: Sequence[str],
) -> PackageSizeWarning | None:
    zip_size = package_path.stat().st_size
    unpacked = _unpacked_size(root, allowlist)
    if zip_size <= SIZE_WARNING_BYTES and unpacked <= SIZE_WARNING_BYTES:
        return None
    largest = sorted(
        ((name, (root / PurePosixPath(name)).stat().st_size) for name in allowlist),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return PackageSizeWarning(kind, zip_size, unpacked, tuple(largest))


def _directory_size_warning(
    kind: str,
    root: Path,
    allowlist: Sequence[str],
) -> PackageSizeWarning | None:
    unpacked = _unpacked_size(root, allowlist)
    if unpacked <= SIZE_WARNING_BYTES:
        return None
    largest = sorted(
        ((name, (root / PurePosixPath(name)).stat().st_size) for name in allowlist),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return PackageSizeWarning(kind, 0, unpacked, tuple(largest))


def _unpacked_size(root: Path, allowlist: Sequence[str]) -> int:
    return sum((root / PurePosixPath(name)).stat().st_size for name in allowlist)


def _write_deterministic_zip(root: Path, destination: Path, allowlist: Sequence[str]) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(set(allowlist)):
            path = root / PurePosixPath(name)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"allowlist 文件缺失或不安全：{name}")
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _verify_zip(destination: Path, root: Path, allowlist: Sequence[str]) -> None:
    """Reopen a staged archive and verify its exact roles and payload hashes."""

    expected = sorted(set(allowlist))
    with zipfile.ZipFile(destination, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if sorted(names) != expected or len(names) != len(expected):
            raise ValueError(f"部署 ZIP 条目与白名单不一致：{destination}")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError(f"部署 ZIP 包含大小写冲突条目：{destination}")
        for info in infos:
            path = PurePosixPath(info.filename)
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or ":" in info.filename
            ):
                raise ValueError(f"部署 ZIP 包含不安全条目：{info.filename!r}")
            payload = archive.read(info)
            source = root / path
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"部署 ZIP 白名单源文件缺失或不安全：{info.filename}")
            if file_sha256(source) != _bytes_sha256(payload):
                raise ValueError(f"部署 ZIP 条目哈希校验失败：{info.filename}")
        # testzip() also forces CRC verification and reports the first corrupt
        # member.  It should be redundant after archive.read, but keeps this
        # gate explicit for future implementation changes.
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"部署 ZIP CRC 校验失败：{corrupt}")


def _verify_directory(root: Path, allowlist: Sequence[str]) -> None:
    expected = sorted(set(allowlist))
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    if actual != expected:
        raise ValueError(f"可编辑工程文件与白名单不一致：{root}")
    if any(_is_reparse_point(root / PurePosixPath(name)) for name in expected):
        raise ValueError(f"可编辑工程包含符号链接或 reparse point：{root}")


def _directory_tree_sha256(root: Path, allowlist: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(set(allowlist)):
        path = root / PurePosixPath(name)
        if not path.is_file() or _is_reparse_point(path):
            raise ValueError(f"可编辑工程文件缺失或不安全：{name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _atomic_publish(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)


def _atomic_publish_directory(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"可编辑 MaixVision 工程目录已存在：{destination}")
    # ``staged`` is created below a TemporaryDirectory on the destination's
    # filesystem, so rename publishes the complete tree in one operation.
    os.replace(staged, destination)
