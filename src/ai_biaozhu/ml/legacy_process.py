"""Safe command construction for the pinned traditional YOLOv5 scripts."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

LEGACY_SCRIPT_NAMES = frozenset({"train", "detect", "export"})


def legacy_subprocess_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a deterministic environment compatible with PyTorch 2.6+.

    YOLOv5 v7.0 predates PyTorch's ``weights_only=True`` default.  The
    documented environment override keeps official checkpoints loadable
    without patching the pinned upstream repository.
    """

    environment = dict(base or os.environ)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    environment["YOLOv5_AUTOINSTALL"] = "false"
    # Traditional v7 uses a differently named setting and otherwise writes
    # fonts under the user's roaming profile. Reuse the application's verified
    # writable Ultralytics config directory when one is available.
    if "YOLOV5_CONFIG_DIR" not in environment and environment.get("YOLO_CONFIG_DIR"):
        environment["YOLOV5_CONFIG_DIR"] = environment["YOLO_CONFIG_DIR"]
    config_dir = environment.get("YOLOV5_CONFIG_DIR")
    if config_dir and os.name == "nt":
        ensure_legacy_windows_fonts(
            config_dir,
            windows_dir=environment.get("WINDIR") or environment.get("SystemRoot"),
        )
    return environment


def ensure_legacy_windows_fonts(
    config_dir: str | Path,
    *,
    windows_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Seed YOLOv5 v7's font cache from the local Windows installation.

    The pinned upstream code otherwise downloads these files during every
    first-use environment. Copying installed system fonts into the user's
    writable application cache keeps training and prediction offline. These
    runtime copies are not included in source or deployment packages.
    """

    destination_root = Path(config_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    system_root = Path(
        windows_dir
        or os.environ.get("WINDIR")
        or os.environ.get("SYSTEMROOT")
        or r"C:\Windows"
    )
    font_root = system_root / "Fonts"
    candidates = {
        "Arial.ttf": ("arial.ttf", "segoeui.ttf"),
        "Arial.Unicode.ttf": (
            "arialuni.ttf",
            "simhei.ttf",
            "msyh.ttc",
            "arial.ttf",
            "segoeui.ttf",
        ),
    }
    prepared: list[Path] = []
    for destination_name, source_names in candidates.items():
        destination = destination_root / destination_name
        if destination.is_file() and destination.stat().st_size > 0:
            prepared.append(destination)
            continue
        source = next(
            (
                font_root / candidate
                for candidate in source_names
                if (font_root / candidate).is_file()
            ),
            None,
        )
        if source is None:
            raise RuntimeError(
                f"无法从 Windows 字体目录为传统 YOLOv5 准备 {destination_name}："
                f"{font_root}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_name}.",
            suffix=".part",
            dir=destination_root,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        prepared.append(destination)
    return tuple(prepared)


def install_pillow_legacy_compatibility(
    image_font_module: Any | None = None,
) -> bool:
    """Restore Pillow's removed ``FreeTypeFont.getsize`` for YOLOv5 v7.

    The shim is applied only inside a legacy worker/bootstrap process and
    leaves the pinned upstream repository untouched.
    """

    if image_font_module is None:
        from PIL import ImageFont as image_font_module

    font_class = image_font_module.FreeTypeFont
    if hasattr(font_class, "getsize"):
        return False

    def getsize(font: Any, text: str, *args: Any, **kwargs: Any) -> tuple[int, int]:
        left, top, right, bottom = font.getbbox(text, *args, **kwargs)
        return int(right - left), int(bottom - top)

    font_class.getsize = getsize
    return True


@contextmanager
def legacy_torch_onnx_export_compatibility(
    *,
    enabled: bool,
    torch_module: Any | None = None,
) -> Iterator[bool]:
    """Force the classic exporter only while YOLOv5 v7 ``export.py`` runs.

    PyTorch 2.11 defaults ``torch.onnx.export`` to the dynamo exporter, which
    requires ``onnxscript``.  The pinned upstream v7 script predates that
    change and is designed for the classic exporter.  This process-local,
    reversible wrapper leaves explicit upstream choices untouched and avoids
    editing the third-party repository or adding an unlocked dependency.
    """

    if not enabled:
        yield False
        return
    if torch_module is None:
        import torch as torch_module
    onnx_module = getattr(torch_module, "onnx", None)
    original = getattr(onnx_module, "export", None)
    if not callable(original):
        raise RuntimeError("当前 PyTorch 未提供 torch.onnx.export")
    if getattr(original, "_ai_biaozhu_legacy_export", False):
        yield False
        return

    @wraps(original)
    def classic_export(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("dynamo", False)
        return original(*args, **kwargs)

    classic_export._ai_biaozhu_legacy_export = True
    onnx_module.export = classic_export
    try:
        yield True
    finally:
        if onnx_module.export is classic_export:
            onnx_module.export = original


def build_legacy_script_command(
    *,
    repository: str | Path,
    script: str,
    arguments: Sequence[str],
    python_executable: str | Path | None = None,
    standalone: bool | None = None,
) -> list[str]:
    repo = Path(repository)
    normalized = str(script).casefold().removesuffix(".py")
    if normalized not in LEGACY_SCRIPT_NAMES:
        raise ValueError(f"不允许执行传统 YOLOv5 脚本：{script}")
    script_path = repo / f"{normalized}.py"
    if not script_path.is_file():
        raise ValueError(f"传统 YOLOv5 仓库缺少 {script_path.name}")
    executable = Path(python_executable or sys.executable)
    packaged = (
        _looks_like_standalone_worker(executable)
        if standalone is None
        else bool(standalone)
    )
    suffix = [str(item) for item in arguments]
    if packaged:
        return [
            str(executable),
            "legacy-script",
            "--repository",
            str(repo),
            normalized,
            "--",
            *suffix,
        ]
    return [
        str(executable),
        "-m",
        "ai_biaozhu.ml.legacy_bootstrap",
        "--repository",
        str(repo),
        normalized,
        "--",
        *suffix,
    ]


def _looks_like_standalone_worker(executable: Path) -> bool:
    configured = os.environ.get("AI_BIAOZHU_STANDALONE")
    if configured is not None:
        return configured.strip().casefold() in {"1", "true", "yes", "on"}
    stem = executable.stem.casefold()
    return not (stem == "py" or stem.startswith("python"))
