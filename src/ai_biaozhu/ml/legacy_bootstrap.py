"""Process-local compatibility bootstrap for the pinned YOLOv5 v7 scripts."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

from .legacy_process import (
    LEGACY_SCRIPT_NAMES,
    install_pillow_legacy_compatibility,
    legacy_torch_onnx_export_compatibility,
)


def install_ipython_legacy_compatibility() -> bool:
    """Provide the tiny IPython surface YOLOv5 uses outside notebooks.

    The pinned YOLOv5 v7 scripts import ``IPython`` unconditionally even
    though desktop training never uses notebook display helpers.  Keeping a
    compatibility module here lets the frozen Worker support legacy models
    without bundling the full interactive IPython dependency tree.
    """

    try:
        import IPython  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "IPython":
            raise
    else:
        return False

    def display(*_args: object, **_kwargs: object) -> None:
        return None

    display_module = ModuleType("IPython.display")
    display_module.display = display
    display_module.clear_output = display
    ipython_module = ModuleType("IPython")
    ipython_module.display = display_module
    ipython_module.get_ipython = lambda: None
    sys.modules["IPython"] = ipython_module
    sys.modules["IPython.display"] = display_module
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-biaozhu-yolov5-v7")
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("script", choices=sorted(LEGACY_SCRIPT_NAMES))
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def run_legacy_script(
    repository: str | Path,
    script: str,
    script_args: list[str],
) -> int:
    """Run one allowlisted upstream script without editing its source tree."""

    normalized = script.casefold().removesuffix(".py")
    if normalized not in LEGACY_SCRIPT_NAMES:
        raise ValueError(f"不允许执行传统 YOLOv5 脚本：{script}")
    root = Path(repository).resolve()
    path = root / f"{normalized}.py"
    if not path.is_file():
        raise ValueError(f"传统 YOLOv5 仓库缺少 {path.name}")
    arguments = list(script_args)
    if arguments and arguments[0] == "--":
        arguments.pop(0)

    install_pillow_legacy_compatibility()
    install_ipython_legacy_compatibility()
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        os.chdir(root)
        sys.argv = [str(path), *arguments]
        with legacy_torch_onnx_export_compatibility(
            enabled=normalized == "export"
        ):
            runpy.run_path(str(path), run_name="__main__")
        return 0
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(str(root))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_legacy_script(args.repository, args.script, args.script_args)


if __name__ == "__main__":
    raise SystemExit(main())
