"""Generate Maix Unified Descriptor files for object detection models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from ai_biaozhu.ml.model_registry import get_model

from .maix import (
    Cam2NpuMode,
    MaixTarget,
    validate_yolov5_anchors,
)
from .onnx_gate import validate_class_names


def build_mud(
    *,
    target: MaixTarget,
    model_key: str,
    class_names: Iterable[str],
    input_height: int,
    input_width: int,
    model_files: Mapping[str, str] | None = None,
    cam2_npu_mode: Cam2NpuMode = Cam2NpuMode.BOTH,
    anchors: Iterable[float] | None = None,
) -> str:
    spec = get_model(model_key)
    names = validate_class_names(class_names)
    files = dict(model_files or {})
    if target is MaixTarget.MAIXCAM_PRO:
        basic = {
            "type": "cvimodel",
            "model": files.get("model", "model.cvimodel"),
        }
    else:
        mode = Cam2NpuMode(cam2_npu_mode)
        basic = {"type": "axmodel"}
        if mode in {Cam2NpuMode.NPU2, Cam2NpuMode.BOTH}:
            basic["model_npu"] = files.get("model_npu", "model_npu.axmodel")
        if mode in {Cam2NpuMode.VNPU, Cam2NpuMode.BOTH}:
            basic["model_vnpu"] = files.get("model_vnpu", "model_vnpu.axmodel")
    extra = {
        "type": "detector",
        "model_type": spec.deployment_decoder,
        "input_type": "rgb",
        "input_shape": f"1,3,{input_height},{input_width}",
        "mean": "0,0,0",
        "scale": "0.003921568627451,0.003921568627451,0.003921568627451",
        "labels": ",".join(names),
        "input_cache": "true",
        "output_cache": "true",
    }
    if target is MaixTarget.MAIXCAM2:
        extra.update(
            {
                "input_cache_flush": "false",
                "output_cache_inval": "true",
            }
        )
    if target is MaixTarget.MAIXCAM2:
        # Pulsar2 embeds the RGB input processor before the first NPU
        # operator, while the detector heads are NPU outputs.  These cache
        # controls match Sipeed's current MaixCAM2 MUD example and avoid stale
        # output buffers on device.
        extra.update(
            {
                "input_cache_flush": "false",
                "output_cache_inval": "true",
            }
        )
    if spec.family == "yolov5":
        values = validate_yolov5_anchors(
            tuple(float(value) for value in anchors) if anchors is not None else None
        )
        extra["anchors"] = ",".join(_number(value) for value in values)
    return _section("basic", basic) + "\n" + _section("extra", extra)


def write_mud(path: str | Path, **kwargs: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_mud(**kwargs), encoding="utf-8", newline="\n")  # type: ignore[arg-type]
    return destination


def _section(name: str, values: Mapping[str, str]) -> str:
    lines = [f"[{name}]"]
    lines.extend(f"{key} = {value}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
