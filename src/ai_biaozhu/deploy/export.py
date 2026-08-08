"""ONNX export helpers for modern and traditional model backends."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_biaozhu.ml.legacy_process import build_legacy_script_command
from ai_biaozhu.ml.model_registry import ModelBackend, get_model


@dataclass(frozen=True, slots=True)
class CheckpointForward:
    class_names: tuple[str, ...]
    outputs: tuple[Any, ...]


def export_modern_onnx(
    checkpoint: str | Path,
    *,
    imgsz: tuple[int, int] = (640, 640),
    model_factory: Callable[[str], Any] | None = None,
) -> Path:
    """Export static batch-1 opset-17 ONNX through a lazy Ultralytics import."""

    height, width = _validate_imgsz(imgsz)
    if model_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("当前 ML 环境未安装 ultralytics") from exc
        model_factory = YOLO
    checkpoint_path = Path(checkpoint).resolve()
    # In a frozen build the synthetic ``onnxslim.__file__`` lives below the
    # executable directory.  Importing while that directory is also the CWD
    # triggers onnxslim's "within its own root folder" source-tree warning even
    # though the bundled module is valid.  Export beside the isolated checkpoint
    # and always restore the caller's CWD.
    with (
        contextlib.chdir(checkpoint_path.parent),
        contextlib.redirect_stdout(sys.stderr),
    ):
        model = model_factory(str(checkpoint_path))
        output = model.export(
            format="onnx",
            imgsz=[height, width],
            batch=1,
            dynamic=False,
            simplify=True,
            opset=17,
        )
        output_path = Path(output).resolve()
    return output_path


def build_legacy_yolov5_export_command(
    *,
    python_executable: str | Path,
    repository: str | Path,
    checkpoint: str | Path,
    imgsz: tuple[int, int] = (640, 640),
) -> list[str]:
    """Build the official traditional YOLOv5 ``export.py`` invocation."""

    height, width = _validate_imgsz(imgsz)
    repo = Path(repository)
    if not (repo / "export.py").is_file():
        raise ValueError("传统 YOLOv5 仓库缺少 export.py")
    return build_legacy_script_command(
        repository=repo,
        script="export",
        python_executable=python_executable,
        arguments=[
        "--weights",
        str(Path(checkpoint)),
        "--include",
        "onnx",
        "--imgsz",
        str(height),
        str(width),
        "--batch-size",
        "1",
        "--opset",
        "17",
        "--simplify",
        ],
    )


def assert_export_backend(model_key: str, modern: bool) -> None:
    spec = get_model(model_key)
    expected = ModelBackend.ULTRALYTICS if modern else ModelBackend.LEGACY_YOLOV5
    if spec.backend is not expected:
        raise ValueError(f"{spec.key} 与所选 ONNX 导出后端不匹配")


def extract_legacy_yolov5_anchors(
    checkpoint: str | Path,
    repository: str | Path,
    *,
    torch_module: Any | None = None,
) -> tuple[float, ...]:
    """Read the actual auto-anchor result stored in a traditional v7 checkpoint."""

    repo = str(Path(repository).resolve())
    inserted = repo not in sys.path
    if inserted:
        sys.path.insert(0, repo)
    try:
        if torch_module is None:
            import torch as torch_module
        try:
            raw = torch_module.load(
                str(Path(checkpoint)),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            raw = torch_module.load(str(Path(checkpoint)), map_location="cpu")
        model = (
            (raw.get("ema") or raw.get("model"))
            if isinstance(raw, dict)
            else raw
        )
        if model is None:
            raise ValueError("checkpoint 不含 model/ema")
        detect = model.model[-1]
        anchors = detect.anchors.detach().cpu()
        strides = detect.stride.detach().cpu()
        values: list[float] = []
        for layer_index, layer in enumerate(anchors):
            stride = float(strides[layer_index])
            for pair in layer:
                values.extend((float(pair[0]) * stride, float(pair[1]) * stride))
        if len(values) != 18 or any(value <= 0 for value in values):
            raise ValueError("checkpoint Detect 层未提供 3×3 组有效 anchors")
        return tuple(values)
    except Exception as exc:
        raise RuntimeError(f"无法从传统 YOLOv5 checkpoint 读取 anchors：{exc}") from exc
    finally:
        if inserted:
            sys.path.remove(repo)


def run_checkpoint_forward(
    checkpoint: str | Path,
    *,
    model_key: str,
    input_array: Any,
    legacy_repository: str | Path | None = None,
    modern_model_factory: Callable[[str], Any] | None = None,
    torch_module: Any | None = None,
) -> CheckpointForward:
    """Run the source checkpoint on the same tensor used by ONNX Runtime."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("checkpoint 数值门禁需要 PyTorch") from exc
    spec = get_model(model_key)
    inserted: str | None = None
    try:
        if spec.backend is ModelBackend.ULTRALYTICS:
            if modern_model_factory is None:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:
                    raise RuntimeError("现代 checkpoint 门禁需要 ultralytics") from exc
                modern_model_factory = YOLO
            with contextlib.redirect_stdout(sys.stderr):
                wrapper = modern_model_factory(str(checkpoint))
            model = getattr(wrapper, "model", wrapper)
        else:
            if legacy_repository is None:
                raise ValueError("传统 YOLOv5 checkpoint 门禁需要 v7.0 仓库")
            repository = str(Path(legacy_repository).resolve())
            if repository not in sys.path:
                sys.path.insert(0, repository)
                inserted = repository
            try:
                raw = torch_module.load(
                    str(checkpoint),
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                raw = torch_module.load(str(checkpoint), map_location="cpu")
            model = (
                (raw.get("ema") or raw.get("model"))
                if isinstance(raw, dict)
                else raw
            )
            if model is None:
                raise ValueError("checkpoint 不含 model/ema")
        for method_name, arguments in (
            ("float", ()),
            ("cpu", ()),
            ("eval", ()),
        ):
            method = getattr(model, method_name, None)
            if callable(method):
                model = method(*arguments)
        tensor = torch_module.from_numpy(input_array).float()
        with torch_module.inference_mode():
            raw_output = model(tensor)
        outputs = tuple(_collect_tensor_outputs(raw_output))
        if not outputs:
            raise ValueError("PyTorch checkpoint 前向没有张量输出")
        return CheckpointForward(
            class_names=_checkpoint_class_names(model),
            outputs=outputs,
        )
    except Exception as exc:
        raise RuntimeError(f"checkpoint 前向门禁失败：{exc}") from exc
    finally:
        if inserted is not None:
            sys.path.remove(inserted)


def validate_checkpoint_class_names(
    actual: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    if not actual:
        raise ValueError("checkpoint 未保存类别名称，无法验证部署类别顺序")
    if actual != expected:
        raise ValueError(
            "checkpoint 类别顺序与当前项目不一致："
            f"checkpoint={list(actual)!r}, project={list(expected)!r}"
        )


def _validate_imgsz(value: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(item) for item in value)
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("ONNX 输入高宽必须为 32 的正整数倍")
    return height, width


def _collect_tensor_outputs(value: Any) -> list[Any]:
    if all(hasattr(value, name) for name in ("detach", "shape")):
        return [value]
    if isinstance(value, dict):
        result: list[Any] = []
        for key in sorted(value, key=str):
            result.extend(_collect_tensor_outputs(value[key]))
        return result
    if isinstance(value, list | tuple):
        result = []
        for item in value:
            result.extend(_collect_tensor_outputs(item))
        return result
    return []


def _checkpoint_class_names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "names", ())
    if isinstance(names, dict):
        try:
            return tuple(str(names[index]) for index in sorted(names))
        except (KeyError, TypeError):
            return tuple(str(value) for _, value in sorted(names.items(), key=str))
    if isinstance(names, list | tuple):
        return tuple(str(value) for value in names)
    return ()
