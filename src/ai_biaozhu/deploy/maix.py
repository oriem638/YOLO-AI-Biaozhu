"""Declarative Docker conversion plans for MaixCAM Pro and MaixCAM2."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations
from pathlib import Path
from typing import Any

from ai_biaozhu.ml.model_registry import get_model

from .onnx_gate import OnnxGateReport, inspect_onnx, validate_class_names

MAIXCAM_PRO_IMAGE = "sophgo/tpuc_dev:latest"
MAIXCAM2_IMAGE = "pulsar2:6.0"

_PRO_OUTPUTS: Mapping[str, tuple[str, ...]] = {
    "yolov5": (
        "/model.24/m.0/Conv_output_0",
        "/model.24/m.1/Conv_output_0",
        "/model.24/m.2/Conv_output_0",
    ),
    "yolov8": (
        "/model.22/dfl/conv/Conv_output_0",
        "/model.22/Sigmoid_output_0",
    ),
    "yolo11": (
        "/model.23/dfl/conv/Conv_output_0",
        "/model.23/Sigmoid_output_0",
    ),
    "yolo26": (
        "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
        "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
        "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
        "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
        "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
        "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
    ),
}

_CAM2_OUTPUTS: Mapping[str, tuple[str, ...]] = {
    "yolov5": _PRO_OUTPUTS["yolov5"],
    "yolov8": (
        "/model.22/Concat_output_0",
        "/model.22/Concat_1_output_0",
        "/model.22/Concat_2_output_0",
    ),
    "yolo11": (
        "/model.23/Concat_output_0",
        "/model.23/Concat_1_output_0",
        "/model.23/Concat_2_output_0",
    ),
    "yolo26": _PRO_OUTPUTS["yolo26"],
}


class MaixTarget(StrEnum):
    MAIXCAM_PRO = "maixcam_pro"
    MAIXCAM2 = "maixcam2"


class Cam2NpuMode(StrEnum):
    NPU2 = "npu2"
    VNPU = "vnpu"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class MaixConversionRequest:
    target: MaixTarget
    model_key: str
    onnx_path: Path
    output_dir: Path
    calibration_dir: Path
    class_names: tuple[str, ...]
    input_height: int
    input_width: int
    calibration_count: int = 100
    docker_executable: str = "docker"
    converter_image: str | None = None
    cam2_npu_mode: Cam2NpuMode = Cam2NpuMode.BOTH
    output_nodes: tuple[str, ...] | None = None
    anchors: tuple[float, ...] | None = None

    def validate(self) -> None:
        spec = get_model(self.model_key)
        validate_class_names(self.class_names)
        if not self.onnx_path.is_file():
            raise ValueError(f"ONNX 文件不存在：{self.onnx_path}")
        if not self.calibration_dir.is_dir():
            raise ValueError(f"校准图片目录不存在：{self.calibration_dir}")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("输入高宽必须大于 0")
        if self.input_height % 32 or self.input_width % 32:
            raise ValueError("输入高宽必须是 32 的整数倍")
        maximum = 200 if self.target is MaixTarget.MAIXCAM_PRO else 100
        if not 20 <= self.calibration_count <= maximum:
            raise ValueError(f"校准图片数量必须在 20 到 {maximum} 之间")
        Cam2NpuMode(self.cam2_npu_mode)
        if self.output_nodes is not None:
            if not self.output_nodes or len(set(self.output_nodes)) != len(self.output_nodes):
                raise ValueError("output_nodes 不能为空或重复")
            if any(not str(name).strip() for name in self.output_nodes):
                raise ValueError("output_nodes 不能含空名称")
        if spec.family == "yolov5":
            validate_yolov5_anchors(self.anchors)


@dataclass(frozen=True, slots=True)
class GeneratedFile:
    relative_path: str
    content: str


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    request: MaixConversionRequest
    image: str
    expected_tensors: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    generated_files: tuple[GeneratedFile, ...]
    output_models: tuple[str, ...]
    gate_report: OnnxGateReport | None = None

    def materialize(self, *, prepare_inputs: bool = True) -> tuple[Path, ...]:
        self.request.output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for item in self.generated_files:
            path = self.request.output_dir / item.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item.content, encoding="utf-8", newline="\n")
            paths.append(path)
        if prepare_inputs and self.gate_report is not None:
            paths.extend(_prepare_converter_inputs(self))
        return tuple(paths)


def expected_output_tensors(target: MaixTarget, model_key: str) -> tuple[str, ...]:
    family = get_model(model_key).family
    table = _PRO_OUTPUTS if target is MaixTarget.MAIXCAM_PRO else _CAM2_OUTPUTS
    return table[family]


def build_conversion_plan(
    request: MaixConversionRequest,
    *,
    gate_loader: Callable[[str], Any] | None = None,
    gate_checker: Callable[[Any], None] | None = None,
    perform_gate: bool = True,
) -> ConversionPlan:
    request.validate()
    preferred = expected_output_tensors(request.target, request.model_key)
    gate: OnnxGateReport | None = None
    if perform_gate:
        gate = inspect_onnx(
            request.onnx_path,
            loader=gate_loader,
            checker=gate_checker,
        ).require_ok()
        if gate.input_shape is not None:
            _, _, height, width = gate.input_shape
            if (height, width) != (request.input_height, request.input_width):
                raise ValueError(
                    "转换输入尺寸与 ONNX 静态输入不一致："
                    f"{request.input_height}x{request.input_width} vs {height}x{width}"
                )
        expected = resolve_output_tensors(
            gate,
            target=request.target,
            model_key=request.model_key,
            class_names=request.class_names,
            preferred=preferred,
            override=request.output_nodes,
        )
    else:
        expected = tuple(request.output_nodes or preferred)
    if request.target is MaixTarget.MAIXCAM_PRO:
        return _build_pro_plan(request, expected, gate)
    return _build_cam2_plan(request, expected, gate)


def resolve_output_tensors(
    report: OnnxGateReport,
    *,
    target: MaixTarget,
    model_key: str,
    class_names: Sequence[str],
    preferred: Sequence[str] | None = None,
    override: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve converter outputs, failing closed when graph matching is ambiguous."""

    family = get_model(model_key).family
    count = _expected_output_count(target, family)
    available = set(report.available_tensors)
    discovered = False
    if override is not None:
        selected = tuple(str(name) for name in override)
        if len(selected) != count:
            raise ValueError(
                f"{model_key}/{target.value} 需要 {count} 个输出节点，"
                f"高级设置提供了 {len(selected)} 个"
            )
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(f"高级输出节点不在 ONNX 图中：{', '.join(missing)}")
    else:
        official = tuple(preferred or expected_output_tensors(target, model_key))
        if len(official) == count and set(official).issubset(available):
            selected = official
        else:
            patterns = _family_patterns(target, family)
            scored: list[tuple[int, str]] = []
            for name in report.available_tensors:
                lowered = name.casefold()
                pattern_score = sum(
                    4 for pattern in patterns if re.search(pattern, lowered)
                )
                if pattern_score == 0:
                    continue
                shape = report.tensor_shapes.get(name)
                shape_score = 0
                if shape is not None:
                    if len(shape) == 4:
                        shape_score += 2
                    elif len(shape) == 3:
                        shape_score += 1
                    if len(shape) >= 2 and all(
                        isinstance(value, int) and value > 0 for value in shape[-2:]
                    ):
                        shape_score += 1
                operation_score = int(
                    any(
                        token in lowered
                        for token in ("conv", "concat", "sigmoid", "detect")
                    )
                )
                scored.append((pattern_score + shape_score + operation_score, name))
            scored.sort(key=lambda item: (-item[0], item[1]))
            if len(scored) < count:
                raise _output_resolution_error(model_key, target, count, len(scored))
            if len(scored) > count and scored[count - 1][0] == scored[count][0]:
                raise _output_resolution_error(model_key, target, count, len(scored))
            selected = tuple(name for _, name in scored[:count])
            discovered = True
    if discovered:
        selected = _unique_semantic_order(
            report,
            target=target,
            model_key=model_key,
            class_names=class_names,
            candidates=selected,
        )
    validate_output_tensor_semantics(
        report,
        target=target,
        model_key=model_key,
        class_names=class_names,
        output_tensors=selected,
    )
    return selected


def _unique_semantic_order(
    report: OnnxGateReport,
    *,
    target: MaixTarget,
    model_key: str,
    class_names: Sequence[str],
    candidates: Sequence[str],
) -> tuple[str, ...]:
    matches: list[tuple[str, ...]] = []
    for ordered in permutations(candidates):
        try:
            validate_output_tensor_semantics(
                report,
                target=target,
                model_key=model_key,
                class_names=class_names,
                output_tensors=ordered,
            )
        except ValueError:
            continue
        matches.append(tuple(ordered))
        if len(matches) > 1:
            break
    if len(matches) == 1:
        return matches[0]
    qualifier = "没有" if not matches else "存在多个"
    raise ValueError(
        f"{model_key}/{target.value} 自动发现的输出节点{qualifier}符合解码语义的"
        "唯一顺序，请用 Netron 核对后显式填写 output_nodes"
    )


def _expected_output_count(target: MaixTarget, family: str) -> int:
    if family == "yolo26":
        return 6
    if family == "yolov5":
        return 3
    return 2 if target is MaixTarget.MAIXCAM_PRO else 3


def _family_patterns(target: MaixTarget, family: str) -> tuple[str, ...]:
    if family == "yolov5":
        return (r"(?:^|/)m\.[012](?:/|$)", r"detect", r"conv_output")
    if family == "yolo26":
        return (r"one2one_cv[23]\.[012]",)
    if target is MaixTarget.MAIXCAM2:
        return (r"concat(?:_[123])?_output", r"detect.*concat")
    return (r"dfl", r"sigmoid", r"detect")


def validate_output_tensor_semantics(
    report: OnnxGateReport,
    *,
    target: MaixTarget,
    model_key: str,
    class_names: Sequence[str],
    output_tensors: Sequence[str],
) -> None:
    """Validate selected tensors against the decoder contract for the model family."""

    labels = validate_class_names(class_names)
    family = get_model(model_key).family
    expected_count = _expected_output_count(target, family)
    names = tuple(str(name) for name in output_tensors)
    if len(names) != expected_count:
        raise ValueError(
            f"{model_key}/{target.value} 需要 {expected_count} 个输出，"
            f"当前选择 {len(names)} 个"
        )
    if report.input_shape is None or len(report.input_shape) != 4:
        raise ValueError("无法根据 ONNX 输入形状验证检测输出语义")
    _, _, input_height, input_width = report.input_shape
    feature_scales = tuple(
        (input_height // stride, input_width // stride) for stride in (8, 16, 32)
    )
    shapes = tuple(
        _static_selected_shape(
            report,
            name,
            infer_static_batch=(
                target is MaixTarget.MAIXCAM2
                and family in {"yolov8", "yolo11"}
                and index == 2
            ),
        )
        for index, name in enumerate(names)
    )
    for name, shape in zip(names, shapes, strict=True):
        if shape[0] != 1:
            raise ValueError(f"检测输出 {name} 的 batch 必须为 1，当前形状为 {shape}")

    class_count = len(labels)
    if family == "yolov5":
        expected_channels = 3 * (class_count + 5)
        _validate_rank4_heads(
            names,
            shapes,
            expected_channels=(expected_channels,) * 3,
            expected_scales=feature_scales,
            role="YOLOv5 检测头",
        )
        return
    if family == "yolo26":
        _validate_rank4_heads(
            names[:3],
            shapes[:3],
            expected_channels=(4,) * 3,
            expected_scales=feature_scales,
            role="YOLO26 边框头",
        )
        _validate_rank4_heads(
            names[3:],
            shapes[3:],
            expected_channels=(class_count,) * 3,
            expected_scales=feature_scales,
            role="YOLO26 类别头",
        )
        return
    if target is MaixTarget.MAIXCAM2:
        total_locations = sum(height * width for height, width in feature_scales)
        expected_shapes = (
            (1, 64, total_locations),
            (1, class_count, total_locations),
            (1, 4, total_locations),
        )
        for name, shape, expected in zip(names, shapes, expected_shapes, strict=True):
            if shape != expected:
                raise ValueError(
                    f"{family}/{target.value} 输出 {name} 语义不匹配："
                    f"期望 {expected}，实际 {shape}"
                )
        return

    total_locations = sum(height * width for height, width in feature_scales)
    expected_shapes = (
        (1, 1, 4, total_locations),
        (1, class_count, total_locations),
    )
    for name, shape, expected in zip(names, shapes, expected_shapes, strict=True):
        if shape != expected:
            raise ValueError(
                f"{family}/{target.value} 输出 {name} 语义不匹配："
                f"期望 {expected}，实际 {shape}"
            )


def _static_selected_shape(
    report: OnnxGateReport,
    name: str,
    *,
    infer_static_batch: bool = False,
) -> tuple[int, ...]:
    shape = report.tensor_shapes.get(name)
    if (
        infer_static_batch
        and shape is not None
        and len(shape) == 3
        and shape[0] is None
        and report.input_shape is not None
        and report.input_shape[0] == 1
        and all(value is not None and value > 0 for value in shape[1:])
    ):
        # ONNX's built-in shape inference loses the batch dimension through
        # the decoded-box broadcast path. The extracted/simplified graph is
        # gated again below and must expose the fully static [1, 4, N] shape.
        shape = (1, *shape[1:])
    if shape is None or any(
        value is None or isinstance(value, bool) or value <= 0 for value in shape
    ):
        raise ValueError(f"检测输出 {name} 缺少完整静态形状：{shape}")
    return tuple(int(value) for value in shape)


def _validate_rank4_heads(
    names: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    *,
    expected_channels: Sequence[int],
    expected_scales: Sequence[tuple[int, int]],
    role: str,
) -> None:
    for name, shape, channels, scale in zip(
        names,
        shapes,
        expected_channels,
        expected_scales,
        strict=True,
    ):
        expected = (1, channels, *scale)
        if shape != expected:
            raise ValueError(
                f"{role} {name} 语义不匹配：期望 {expected}，实际 {shape}"
            )


def _output_resolution_error(
    model_key: str,
    target: MaixTarget,
    expected_count: int,
    candidates: int,
) -> ValueError:
    return ValueError(
        f"无法唯一识别 {model_key}/{target.value} 的 {expected_count} 个部署输出节点"
        f"（候选 {candidates} 个）。请用 Netron 检查 ONNX，并在高级部署设置"
        " output_nodes 中明确填写。"
    )


def _mounts(request: MaixConversionRequest) -> list[str]:
    return [
        "-v",
        f"{request.onnx_path.parent.resolve()}:/input:ro",
        "-v",
        f"{request.calibration_dir.resolve()}:/calibration:ro",
        "-v",
        f"{request.output_dir.resolve()}:/output",
    ]


def _docker_prefix(request: MaixConversionRequest, image: str) -> list[str]:
    privileges = ["--privileged"] if request.target is MaixTarget.MAIXCAM_PRO else []
    return [
        request.docker_executable,
        "run",
        "--rm",
        *privileges,
        *_mounts(request),
        image,
    ]


def _build_pro_plan(
    request: MaixConversionRequest,
    expected: tuple[str, ...],
    gate: OnnxGateReport | None,
) -> ConversionPlan:
    image = request.converter_image or MAIXCAM_PRO_IMAGE
    prefix = _docker_prefix(request, image)
    shape = f"[[1,3,{request.input_height},{request.input_width}]]"
    test_image = _calibration_images(request)[0]
    commands = (
        tuple(
            [
                *prefix,
                "model_transform.py",
                "--model_name",
                "detector",
                "--model_def",
                "/output/export.onnx",
                "--input_shapes",
                shape,
                "--pixel_format",
                "rgb",
                "--channel_format",
                "nchw",
                "--mean",
                "0,0,0",
                "--scale",
                "0.003921568627451,0.003921568627451,0.003921568627451",
                "--keep_aspect_ratio",
                "--output_names",
                ",".join(expected),
                "--test_input",
                f"/calibration/{test_image.name}",
                "--test_result",
                "/output/detector_top_outputs.npz",
                "--tolerance",
                "0.99,0.99",
                "--mlir",
                "/output/detector.mlir",
            ]
        ),
        tuple(
            [
                *prefix,
                "run_calibration.py",
                "/output/detector.mlir",
                "--dataset",
                "/calibration",
                "--input_num",
                str(request.calibration_count),
                "-o",
                "/output/calibration.table",
            ]
        ),
        tuple(
            [
                *prefix,
                "model_deploy.py",
                "--mlir",
                "/output/detector.mlir",
                "--quantize",
                "INT8",
                "--calibration_table",
                "/output/calibration.table",
                "--quant_input",
                "--processor",
                "cv181x",
                "--test_input",
                "/output/detector_in_f32.npz",
                "--test_reference",
                "/output/detector_top_outputs.npz",
                "--tolerance",
                "0.9,0.6",
                "--model",
                "/output/model.cvimodel",
            ]
        ),
    )
    metadata = {
        "target": request.target.value,
        "processor": "cv181x",
        "quantization": "INT8",
        "calibration_count": request.calibration_count,
        "output_tensors": list(expected),
        "test_image": test_image.name,
        "transform_tolerance": [0.99, 0.99],
        "deploy_tolerance": [0.9, 0.6],
    }
    return ConversionPlan(
        request=request,
        image=image,
        expected_tensors=expected,
        commands=commands,
        generated_files=(
            GeneratedFile(
                "conversion-config.json",
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            ),
        ),
        output_models=("model.cvimodel",),
        gate_report=gate,
    )


def _build_cam2_plan(
    request: MaixConversionRequest,
    expected: tuple[str, ...],
    gate: OnnxGateReport | None,
) -> ConversionPlan:
    image = request.converter_image or MAIXCAM2_IMAGE
    configs: list[GeneratedFile] = []
    commands: list[tuple[str, ...]] = []
    prefix = [
        request.docker_executable,
        "run",
        "--rm",
        "-v",
        f"{request.output_dir.resolve()}:/data",
        "-w",
        "/data",
        image,
    ]
    mode = Cam2NpuMode(request.cam2_npu_mode)
    variants = {
        Cam2NpuMode.NPU2: (("NPU2", "model_npu.axmodel"),),
        Cam2NpuMode.VNPU: (("NPU1", "model_vnpu.axmodel"),),
        Cam2NpuMode.BOTH: (
            ("NPU2", "model_npu.axmodel"),
            ("NPU1", "model_vnpu.axmodel"),
        ),
    }[mode]
    for npu_mode, _output_name in variants:
        config_name = (
            f"config/{request.model_key.casefold()}_{npu_mode.casefold()}.json"
        )
        config = _pulsar_config(
            request,
            expected,
            npu_mode,
            input_name=gate.input_name if gate and gate.input_name else "images",
        )
        configs.append(
            GeneratedFile(
                config_name,
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            )
        )
        commands.append(
            tuple(
                [
                    *prefix,
                    "pulsar2",
                    "build",
                    "--target_hardware",
                    "AX620E",
                    "--input",
                    "./export.onnx",
                    "--output_dir",
                    "./tmp",
                    "--config",
                    f"./{config_name}",
                ]
            )
        )
    return ConversionPlan(
        request=request,
        image=image,
        expected_tensors=expected,
        commands=tuple(commands),
        generated_files=tuple(configs),
        output_models=tuple(output_name for _, output_name in variants),
        gate_report=gate,
    )


def _pulsar_config(
    request: MaixConversionRequest,
    outputs: Sequence[str],
    npu_mode: str,
    *,
    input_name: str,
) -> dict[str, Any]:
    return {
        "model_type": "ONNX",
        "npu_mode": npu_mode,
        "quant": {
            "input_configs": [
                {
                    "tensor_name": input_name,
                    "calibration_dataset": "datasets/train.tar",
                    "calibration_size": request.calibration_count,
                    "calibration_mean": [0.0, 0.0, 0.0],
                    "calibration_std": [255.0, 255.0, 255.0],
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": True,
        },
        "input_processors": [
            {
                "tensor_name": input_name,
                "tensor_format": "RGB",
                "tensor_layout": "NCHW",
                "src_format": "RGB",
                "src_dtype": "U8",
                "src_layout": "NHWC",
                "csc_mode": "NoCSC",
            }
        ],
        "output_processors": [
            {"tensor_name": name, "dst_perm": [0, 2, 3, 1]} for name in outputs
        ],
        "compiler": {
            "check": 3,
            "check_mode": "CheckOutput",
            "check_cosine_simularity": 0.9,
        },
    }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def execute_conversion_plan(
    plan: ConversionPlan,
    *,
    runner: Runner = subprocess.run,
    materialize: bool = True,
    step_callback: Callable[
        [str, int, int, tuple[str, ...], subprocess.CompletedProcess[str] | None],
        None,
    ]
    | None = None,
) -> tuple[subprocess.CompletedProcess[str], ...]:
    """Materialize configs and run commands through an injectable runner."""

    if materialize:
        plan.materialize()
    results: list[subprocess.CompletedProcess[str]] = []
    total = len(plan.commands)
    for index, command in enumerate(plan.commands, start=1):
        if plan.request.target is MaixTarget.MAIXCAM2:
            temporary = plan.request.output_dir / "tmp"
            if temporary.exists():
                _safe_remove_directory(temporary, plan.request.output_dir)
        if step_callback is not None:
            step_callback("started", index, total, command, None)
        completed = runner(
            list(command),
            cwd=str(plan.request.output_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(completed)
        if step_callback is not None:
            step_callback("finished", index, total, command, completed)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"模型转换第 {index} 步失败：{message}")
        if plan.request.target is MaixTarget.MAIXCAM2:
            compiled = plan.request.output_dir / "tmp" / "compiled.axmodel"
            if not compiled.is_file():
                raise RuntimeError(
                    f"Pulsar2 第 {index} 步未生成 tmp/compiled.axmodel"
                )
            shutil.copy2(
                compiled,
                plan.request.output_dir / plan.output_models[index - 1],
            )
    missing = [
        name for name in plan.output_models if not (plan.request.output_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"转换命令完成但缺少产物：{', '.join(missing)}")
    return tuple(results)


def _prepare_converter_inputs(plan: ConversionPlan) -> tuple[Path, ...]:
    """Extract/simplify converter outputs and prepare target calibration data."""

    request = plan.request
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError("Maix 部署需要 onnx 和 onnxsim") from exc
    input_name = (
        plan.gate_report.input_name
        if plan.gate_report is not None and plan.gate_report.input_name
        else "images"
    )
    extracted = request.output_dir / "tmp_extract.onnx"
    simplified_path = request.output_dir / "export.onnx"
    onnx.utils.extract_model(
        str(request.onnx_path),
        str(extracted),
        [input_name],
        list(plan.expected_tensors),
    )
    simplified, check = simplify(onnx.load(str(extracted)))
    if not check:
        raise RuntimeError("onnxsim 未能证明 export.onnx 与裁剪模型等价")
    onnx.save(simplified, str(simplified_path))
    simplified_report = inspect_onnx(
        simplified_path,
        expected_tensors=plan.expected_tensors,
    ).require_ok()
    validate_output_tensor_semantics(
        simplified_report,
        target=request.target,
        model_key=request.model_key,
        class_names=request.class_names,
        output_tensors=plan.expected_tensors,
    )

    if request.target is MaixTarget.MAIXCAM_PRO:
        return extracted, simplified_path
    images = _calibration_images(request)
    dataset_dir = request.output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir / "train.tar"
    with tarfile.open(archive, "w") as handle:
        for index, image in enumerate(images[: request.calibration_count], start=1):
            handle.add(
                image,
                arcname=f"{index:04d}{image.suffix.casefold()}",
                recursive=False,
            )
    return extracted, simplified_path, archive


def _calibration_images(request: MaixConversionRequest) -> list[Path]:
    images = [
        path
        for path in sorted(request.calibration_dir.iterdir())
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if len(images) < request.calibration_count:
        raise ValueError(
            f"校准图片不足：需要 {request.calibration_count} 张，实际 {len(images)} 张"
        )
    return images[: request.calibration_count]


def _safe_remove_directory(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved.parent != resolved_root:
        raise RuntimeError(f"拒绝删除非部署工作目录：{resolved}")
    shutil.rmtree(resolved)


def validate_yolov5_anchors(anchors: Sequence[float] | None) -> tuple[float, ...]:
    if anchors is None or len(anchors) != 18:
        raise ValueError("传统 YOLOv5 部署必须提供训练模型的 18 个 anchors 数值")
    values = tuple(float(value) for value in anchors)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("YOLOv5 anchors 必须全部为有限正数")
    return values
