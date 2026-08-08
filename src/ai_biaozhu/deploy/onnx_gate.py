"""Fail-fast gates for ONNX models entering the Maix conversion pipeline."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_OPSET = 17


@dataclass(frozen=True, slots=True)
class GateIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OnnxGateReport:
    path: Path
    sha256: str | None
    opset: int | None
    input_name: str | None
    input_shape: tuple[int, ...] | None
    graph_outputs: tuple[str, ...]
    available_tensors: tuple[str, ...]
    tensor_shapes: Mapping[str, tuple[int | None, ...] | None]
    errors: tuple[GateIssue, ...]
    warnings: tuple[GateIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> OnnxGateReport:
        if self.errors:
            details = "；".join(issue.message for issue in self.errors)
            raise ValueError(f"ONNX 部署门禁未通过：{details}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "opset": self.opset,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "graph_outputs": list(self.graph_outputs),
            "available_tensors": list(self.available_tensors),
            "tensor_shapes": {
                name: list(shape) if shape is not None else None
                for name, shape in self.tensor_shapes.items()
            },
            "ok": self.ok,
            "errors": [
                {"code": issue.code, "message": issue.message} for issue in self.errors
            ],
            "warnings": [
                {"code": issue.code, "message": issue.message} for issue in self.warnings
            ],
        }


@dataclass(frozen=True, slots=True)
class NumericOutput:
    name: str
    shape: tuple[int, ...]
    dtype: str
    finite: bool
    minimum: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "finite": self.finite,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class ParityOutput:
    name: str
    pytorch_shape: tuple[int, ...]
    mean_absolute_error: float
    maximum_absolute_error: float
    cosine_similarity: float
    passed: bool
    comparison_mode: str = "raw_tensor"
    matched_detections: int | None = None
    minimum_box_iou: float | None = None
    maximum_confidence_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pytorch_shape": list(self.pytorch_shape),
            "mean_absolute_error": self.mean_absolute_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "cosine_similarity": self.cosine_similarity,
            "passed": self.passed,
            "comparison_mode": self.comparison_mode,
            "matched_detections": self.matched_detections,
            "minimum_box_iou": self.minimum_box_iou,
            "maximum_confidence_error": self.maximum_confidence_error,
        }


@dataclass(frozen=True, slots=True)
class OnnxNumericReport:
    path: Path
    input_name: str | None
    input_shape: tuple[int, ...] | None
    sample_sha256: str
    providers: tuple[str, ...]
    outputs: tuple[NumericOutput, ...]
    parity_outputs: tuple[ParityOutput, ...]
    errors: tuple[GateIssue, ...]
    warnings: tuple[GateIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> OnnxNumericReport:
        if self.errors:
            details = "；".join(issue.message for issue in self.errors)
            raise ValueError(f"ONNX 数值门禁未通过：{details}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "input_name": self.input_name,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "sample_sha256": self.sample_sha256,
            "providers": list(self.providers),
            "outputs": [item.to_dict() for item in self.outputs],
            "parity_outputs": [item.to_dict() for item in self.parity_outputs],
            "ok": self.ok,
            "errors": [
                {"code": issue.code, "message": issue.message}
                for issue in self.errors
            ],
            "warnings": [
                {"code": issue.code, "message": issue.message}
                for issue in self.warnings
            ],
        }


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_onnx(
    path: str | Path,
    *,
    expected_tensors: Iterable[str] = (),
    loader: Callable[[str], Any] | None = None,
    checker: Callable[[Any], None] | None = None,
    shape_inferer: Callable[[Any], Any] | None = None,
) -> OnnxGateReport:
    """Inspect a static detection ONNX without requiring ONNX in unit tests."""

    model_path = Path(path)
    errors: list[GateIssue] = []
    warnings: list[GateIssue] = []
    if not model_path.is_file():
        return OnnxGateReport(
            model_path,
            None,
            None,
            None,
            None,
            (),
            (),
            {},
            (GateIssue("file_missing", f"ONNX 文件不存在：{model_path}"),),
            (),
        )
    digest = file_sha256(model_path)
    if loader is None:
        try:
            import onnx
        except ImportError:
            return OnnxGateReport(
                model_path,
                digest,
                None,
                None,
                None,
                (),
                (),
                {},
                (
                    GateIssue(
                        "onnx_not_installed",
                        "部署环境未安装 onnx，无法验证模型结构",
                    ),
                ),
                (),
            )
        def load_onnx(value: str) -> Any:
            return onnx.load(value, load_external_data=False)

        loader = load_onnx
        checker = checker or onnx.checker.check_model
        shape_inferer = shape_inferer or onnx.shape_inference.infer_shapes
    try:
        model = loader(str(model_path))
    except Exception as exc:
        return OnnxGateReport(
            model_path,
            digest,
            None,
            None,
            None,
            (),
            (),
            {},
            (GateIssue("load_failed", f"ONNX 读取失败：{exc}"),),
            (),
        )
    if checker is not None:
        try:
            checker(model)
        except Exception as exc:
            errors.append(GateIssue("checker_failed", f"onnx.checker 校验失败：{exc}"))
    if shape_inferer is not None:
        try:
            model = shape_inferer(model)
        except Exception as exc:
            errors.append(
                GateIssue("shape_inference_failed", f"ONNX 形状推断失败：{exc}")
            )
    opsets = [
        int(item.version)
        for item in getattr(model, "opset_import", ())
        if getattr(item, "domain", "") in ("", "ai.onnx")
    ]
    opset = max(opsets) if opsets else None
    if opset != REQUIRED_OPSET:
        errors.append(
            GateIssue(
                "opset",
                f"需要 ONNX opset {REQUIRED_OPSET}，当前为 {opset}",
            )
        )
    graph = model.graph
    initializer_names = {item.name for item in getattr(graph, "initializer", ())}
    inputs = [
        item for item in getattr(graph, "input", ()) if item.name not in initializer_names
    ]
    input_name: str | None = None
    input_shape: tuple[int, ...] | None = None
    if len(inputs) != 1:
        errors.append(
            GateIssue("input_count", f"只支持单输入模型，当前输入数量：{len(inputs)}")
        )
        for value in inputs:
            shape = _tensor_shape(value)
            if shape is None or any(dimension is None for dimension in shape):
                errors.append(
                    GateIssue(
                        "dynamic_input_shape",
                        f"ONNX 输入 {value.name} 缺少完整静态形状：{shape}",
                    )
                )
    else:
        input_name = str(inputs[0].name)
        dimensions = inputs[0].type.tensor_type.shape.dim
        parsed: list[int] = []
        dynamic = False
        for dimension in dimensions:
            value = int(getattr(dimension, "dim_value", 0) or 0)
            parameter = str(getattr(dimension, "dim_param", "") or "")
            if value <= 0 or parameter:
                dynamic = True
            parsed.append(value)
        input_shape = tuple(parsed)
        if len(parsed) != 4:
            errors.append(GateIssue("input_rank", "模型输入必须是四维 NCHW"))
        elif dynamic:
            errors.append(GateIssue("dynamic_shape", "模型输入必须是静态形状"))
        else:
            batch, channels, height, width = parsed
            if batch != 1 or channels != 3:
                errors.append(
                    GateIssue(
                        "input_layout",
                        f"模型输入必须是 [1,3,H,W]，当前为 {parsed}",
                    )
                )
            if height % 32 or width % 32:
                errors.append(
                    GateIssue("input_stride", "输入高宽必须是 32 的整数倍")
                )
    output_values = tuple(getattr(graph, "output", ()))
    graph_outputs = tuple(str(item.name) for item in output_values)
    available = set(graph_outputs)
    for node in getattr(graph, "node", ()):
        available.update(str(item) for item in getattr(node, "output", ()) if item)
    tensor_shapes: dict[str, tuple[int | None, ...] | None] = {
        str(item.name): _tensor_shape(item)
        for collection in (
            getattr(graph, "input", ()),
            getattr(graph, "value_info", ()),
            getattr(graph, "output", ()),
        )
        for item in collection
    }
    for name in available:
        tensor_shapes.setdefault(name, None)
    expected = tuple(expected_tensors)
    missing = [name for name in expected if name not in available]
    if missing:
        errors.append(
            GateIssue(
                "output_tensors",
                f"模型缺少目标输出节点：{', '.join(missing)}",
            )
        )
    if not graph_outputs:
        errors.append(GateIssue("no_graph_outputs", "ONNX 图未声明输出"))
    for value in output_values:
        name = str(value.name)
        shape = tensor_shapes.get(name)
        if shape is None:
            errors.append(
                GateIssue(
                    "output_shape_missing",
                    f"ONNX 输出 {name} 无法推断静态形状",
                )
            )
        elif any(dimension is None or dimension <= 0 for dimension in shape):
            errors.append(
                GateIssue(
                    "dynamic_output_shape",
                    f"ONNX 输出 {name} 含动态维度：{shape}",
                )
            )
    return OnnxGateReport(
        path=model_path,
        sha256=digest,
        opset=opset,
        input_name=input_name,
        input_shape=input_shape,
        graph_outputs=graph_outputs,
        available_tensors=tuple(sorted(available)),
        tensor_shapes=tensor_shapes,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def load_rgb_nchw(
    image_path: str | Path,
    *,
    height: int,
    width: int,
) -> Any:
    """Load one deterministic RGB float32 sample for PyTorch/ORT parity."""

    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("ONNX 数值门禁需要 NumPy 和 Pillow") from exc
    if height <= 0 or width <= 0:
        raise ValueError("数值门禁输入高宽必须大于 0")
    with Image.open(image_path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def inspect_onnx_numerics(
    path: str | Path,
    *,
    input_array: Any,
    pytorch_outputs: Iterable[Any] | None = None,
    session_factory: Callable[..., Any] | None = None,
    mean_absolute_tolerance: float = 1e-3,
    maximum_absolute_tolerance: float = 5e-2,
    cosine_tolerance: float = 0.999,
) -> OnnxNumericReport:
    """Run ONNX Runtime and optionally compare raw PyTorch/ONNX tensors.

    Raw tensor parity is stricter than comparing only post-NMS boxes: it covers
    every class score and box-regression value emitted by the exported graph.
    """

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("ONNX 数值门禁需要 NumPy") from exc
    model_path = Path(path)
    errors: list[GateIssue] = []
    warnings: list[GateIssue] = []
    providers: tuple[str, ...] = ()
    input_name: str | None = None
    input_shape: tuple[int, ...] | None = None
    numeric_outputs: list[NumericOutput] = []
    parity_outputs: list[ParityOutput] = []
    sample = np.ascontiguousarray(np.asarray(input_array, dtype=np.float32))
    sample_digest = hashlib.sha256(sample.tobytes(order="C")).hexdigest()
    if sample.ndim != 4 or sample.shape[0] != 1 or sample.shape[1] != 3:
        errors.append(
            GateIssue(
                "numeric_input_shape",
                f"数值门禁样本必须为 [1,3,H,W]，当前为 {list(sample.shape)}",
            )
        )
    if not np.isfinite(sample).all():
        errors.append(GateIssue("numeric_input_finite", "数值门禁样本包含 NaN/Inf"))
    if session_factory is None:
        try:
            import onnxruntime as ort
        except ImportError:
            return OnnxNumericReport(
                model_path,
                None,
                None,
                sample_digest,
                (),
                (),
                (),
                (
                    GateIssue(
                        "onnxruntime_not_installed",
                        "部署环境未安装 onnxruntime，无法执行 ONNX 前向门禁",
                    ),
                ),
                (),
            )
        session_factory = ort.InferenceSession
    try:
        session = session_factory(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        providers = tuple(str(item) for item in session.get_providers())
        inputs = list(session.get_inputs())
        if len(inputs) != 1:
            errors.append(
                GateIssue(
                    "runtime_input_count",
                    f"ONNX Runtime 需要单输入，当前为 {len(inputs)} 个",
                )
            )
            runtime_values: list[Any] = []
            output_names: list[str] = []
        else:
            input_name = str(inputs[0].name)
            input_shape = tuple(int(value) for value in sample.shape)
            declared = tuple(_runtime_dimension(item) for item in inputs[0].shape)
            if declared != input_shape:
                errors.append(
                    GateIssue(
                        "runtime_input_shape",
                        f"ONNX Runtime 输入 {list(declared)} 与样本 {list(input_shape)} 不一致",
                    )
                )
            output_meta = list(session.get_outputs())
            output_names = [str(item.name) for item in output_meta]
            runtime_values = list(session.run(output_names, {input_name: sample}))
    except Exception as exc:
        errors.append(GateIssue("onnxruntime_failed", f"ONNX Runtime 前向失败：{exc}"))
        runtime_values = []
        output_names = []

    for name, value in zip(output_names, runtime_values, strict=False):
        array = np.asarray(value)
        finite = bool(np.isfinite(array).all())
        if not finite:
            errors.append(GateIssue("runtime_nonfinite", f"ONNX 输出 {name} 包含 NaN/Inf"))
        numeric_outputs.append(
            NumericOutput(
                name=name,
                shape=tuple(int(item) for item in array.shape),
                dtype=str(array.dtype),
                finite=finite,
                minimum=_finite_extreme(array, minimum=True),
                maximum=_finite_extreme(array, minimum=False),
            )
        )
    if not runtime_values and not any(
        issue.code == "onnxruntime_failed" for issue in errors
    ):
        errors.append(GateIssue("runtime_no_outputs", "ONNX Runtime 没有返回输出"))

    if pytorch_outputs is not None and runtime_values:
        torch_arrays = [
            np.asarray(value.detach().float().cpu().numpy())
            if all(hasattr(value, name) for name in ("detach", "float", "cpu"))
            else np.asarray(value)
            for value in pytorch_outputs
        ]
        unused = set(range(len(torch_arrays)))
        for name, onnx_value in zip(output_names, runtime_values, strict=False):
            onnx_array = np.asarray(onnx_value, dtype=np.float64)
            candidates = [
                index
                for index in unused
                if tuple(torch_arrays[index].shape) == tuple(onnx_array.shape)
            ]
            if not candidates:
                errors.append(
                    GateIssue(
                        "parity_shape",
                        f"找不到与 ONNX 输出 {name} 形状 {list(onnx_array.shape)} "
                        "对应的 PyTorch 张量",
                    )
                )
                continue
            index = candidates[0]
            unused.remove(index)
            torch_array = np.asarray(torch_arrays[index], dtype=np.float64)
            detection = _match_detection_output(torch_array, onnx_array)
            if detection is None:
                compared_torch = torch_array
                compared_onnx = onnx_array
                comparison_mode = "raw_tensor"
                matched_detections = None
                minimum_iou = None
                maximum_confidence_error = None
                detection_passed = True
            else:
                (
                    compared_torch,
                    compared_onnx,
                    matched_detections,
                    minimum_iou,
                    maximum_confidence_error,
                    detection_passed,
                ) = detection
                comparison_mode = "matched_detections"
            difference = np.abs(compared_torch - compared_onnx)
            mean_error = float(difference.mean()) if difference.size else 0.0
            maximum_error = float(difference.max()) if difference.size else 0.0
            cosine = _cosine_similarity(compared_torch, compared_onnx)
            passed = (
                mean_error <= mean_absolute_tolerance
                and maximum_error <= maximum_absolute_tolerance
                and cosine >= cosine_tolerance
                and detection_passed
            )
            parity_outputs.append(
                ParityOutput(
                    name=name,
                    pytorch_shape=tuple(int(item) for item in torch_array.shape),
                    mean_absolute_error=mean_error,
                    maximum_absolute_error=maximum_error,
                    cosine_similarity=cosine,
                    passed=passed,
                    comparison_mode=comparison_mode,
                    matched_detections=matched_detections,
                    minimum_box_iou=minimum_iou,
                    maximum_confidence_error=maximum_confidence_error,
                )
            )
            if not passed:
                errors.append(
                    GateIssue(
                        "pytorch_onnx_parity",
                        f"{name} 数值差异超限：mean={mean_error:.6g}, "
                        f"max={maximum_error:.6g}, cosine={cosine:.6g}",
                    )
                )
        if unused:
            warnings.append(
                GateIssue(
                    "unused_pytorch_outputs",
                    f"PyTorch 前向还有 {len(unused)} 个内部张量未参与 ONNX 输出比对",
                )
            )
    return OnnxNumericReport(
        path=model_path,
        input_name=input_name,
        input_shape=input_shape,
        sample_sha256=sample_digest,
        providers=providers,
        outputs=tuple(numeric_outputs),
        parity_outputs=tuple(parity_outputs),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_class_names(names: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(name).strip() for name in names)
    if not result:
        raise ValueError("至少需要一个类别")
    if any(not name for name in result):
        raise ValueError("类别名称不能为空")
    if len(set(result)) != len(result):
        raise ValueError("类别名称不能重复")
    if any("\n" in name or "\r" in name or "," in name for name in result):
        raise ValueError("类别名称不能包含逗号或换行")
    return result


def _tensor_shape(value_info: Any) -> tuple[int | None, ...] | None:
    try:
        dimensions = value_info.type.tensor_type.shape.dim
    except AttributeError:
        return None
    result: list[int | None] = []
    for dimension in dimensions:
        raw = int(getattr(dimension, "dim_value", 0) or 0)
        parameter = str(getattr(dimension, "dim_param", "") or "")
        result.append(raw if raw > 0 and not parameter else None)
    return tuple(result)


def _runtime_dimension(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _finite_extreme(value: Any, *, minimum: bool) -> float | None:
    try:
        import numpy as np

        array = np.asarray(value)
        finite = array[np.isfinite(array)]
        if not finite.size:
            return None
        result = finite.min() if minimum else finite.max()
        return float(result)
    except (TypeError, ValueError):
        return None


def _cosine_similarity(left: Any, right: Any) -> float:
    try:
        import numpy as np

        first = np.asarray(left, dtype=np.float64).reshape(-1)
        second = np.asarray(right, dtype=np.float64).reshape(-1)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-15:
            return 1.0 if np.allclose(first, second, atol=1e-12, rtol=0) else 0.0
        result = float(np.dot(first, second) / denominator)
        return max(-1.0, min(1.0, result)) if math.isfinite(result) else -1.0
    except (TypeError, ValueError):
        return -1.0


def _match_detection_output(
    pytorch_value: Any,
    onnx_value: Any,
) -> tuple[Any, Any, int, float, float, bool] | None:
    """Canonicalize end-to-end ``[..., 6]`` detections before comparison."""

    try:
        import numpy as np

        pytorch_rows = np.asarray(pytorch_value, dtype=np.float64)
        onnx_rows = np.asarray(onnx_value, dtype=np.float64)
        if (
            pytorch_rows.shape != onnx_rows.shape
            or pytorch_rows.ndim != 3
            or pytorch_rows.shape[0] != 1
            or pytorch_rows.shape[-1] != 6
        ):
            return None
        pytorch_rows = pytorch_rows[0]
        onnx_rows = onnx_rows[0]
        maximum_score = max(
            float(pytorch_rows[:, 4].max(initial=0.0)),
            float(onnx_rows[:, 4].max(initial=0.0)),
        )
        threshold = max(1e-5, maximum_score * 0.01)
        minimum_rows = min(20, len(onnx_rows))
        pytorch_selected = _selected_detection_rows(
            pytorch_rows,
            threshold,
            minimum_rows,
        )
        onnx_selected = _selected_detection_rows(
            onnx_rows,
            threshold,
            minimum_rows,
        )
        used: set[int] = set()
        matched_pytorch: list[Any] = []
        matched_onnx: list[Any] = []
        ious: list[float] = []
        confidence_errors: list[float] = []
        for onnx_row in sorted(onnx_selected, key=lambda row: -float(row[4])):
            candidates = [
                (index, row)
                for index, row in enumerate(pytorch_selected)
                if index not in used and int(round(row[5])) == int(round(onnx_row[5]))
            ]
            if not candidates:
                continue
            index, pytorch_row = max(
                candidates,
                key=lambda item: (
                    _box_iou(item[1][:4], onnx_row[:4]),
                    -abs(float(item[1][4] - onnx_row[4])),
                ),
            )
            used.add(index)
            matched_pytorch.append(pytorch_row)
            matched_onnx.append(onnx_row)
            ious.append(_box_iou(pytorch_row[:4], onnx_row[:4]))
            confidence_errors.append(abs(float(pytorch_row[4] - onnx_row[4])))
        matched_count = len(matched_pytorch)
        complete = (
            matched_count == len(pytorch_selected) == len(onnx_selected)
            and matched_count > 0
        )
        minimum_iou = min(ious) if ious else 0.0
        maximum_confidence_error = max(confidence_errors, default=float("inf"))
        passed = (
            complete
            and minimum_iou >= 0.99
            and maximum_confidence_error <= 0.01
        )
        return (
            np.asarray(matched_pytorch, dtype=np.float64),
            np.asarray(matched_onnx, dtype=np.float64),
            matched_count,
            minimum_iou,
            maximum_confidence_error,
            passed,
        )
    except (TypeError, ValueError):
        return None


def _selected_detection_rows(value: Any, threshold: float, minimum: int) -> Any:
    import numpy as np

    rows = np.asarray(value)
    selected = rows[rows[:, 4] >= threshold]
    if len(selected) >= minimum:
        return selected
    order = np.argsort(-rows[:, 4], kind="stable")
    return rows[order[:minimum]]


def _box_iou(left: Any, right: Any) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0,
        float(left[3]) - float(left[1]),
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0,
        float(right[3]) - float(right[1]),
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 1e-12 else 1.0
