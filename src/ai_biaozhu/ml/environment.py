"""Discover and validate candidate Conda/Python ML environments."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

PYTHON_BASELINE = (3, 11)
TORCH_BASELINE = "2.11.0+cu128"
TORCHVISION_BASELINE = "0.26.0+cu128"
ULTRALYTICS_MINIMUM = (8, 4, 82)
ULTRALYTICS_MAXIMUM = (8, 5, 0)
LEGACY_YOLOV5_TAG = "v7.0"
LEGACY_TAG_LOCK_FILE = ".ai-biaozhu-yolov5-tag"

_PROBE = r"""
import json, platform, sys
result = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "torch": None,
    "torchvision": None,
    "ultralytics": None,
    "cuda_available": False,
    "cuda_version": None,
    "device_name": None,
    "errors": [],
}
for package in ("torch", "torchvision", "ultralytics"):
    try:
        module = __import__(package)
        result[package] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        result["errors"].append(f"{package}: {exc}")
try:
    import torch
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_version"] = getattr(torch.version, "cuda", None)
    if result["cuda_available"]:
        result["device_name"] = torch.cuda.get_device_name(0)
except Exception:
    pass
print(json.dumps(result, ensure_ascii=False))
""".strip()


@dataclass(frozen=True, slots=True)
class EnvironmentCandidate:
    prefix: Path
    python: Path
    source: str


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    candidate: EnvironmentCandidate
    valid: bool
    python_version: str | None
    torch_version: str | None
    torchvision_version: str | None
    ultralytics_version: str | None
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    errors: tuple[str, ...]
    compatibility_errors: tuple[str, ...]
    gpu_ready: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LegacyYoloV5RepositoryReport:
    path: Path
    valid: bool
    tag: str | None
    expected_tag: str
    errors: tuple[str, ...]


def python_executable(prefix: str | Path) -> Path:
    prefix_path = Path(prefix)
    windows = prefix_path / "python.exe"
    return windows if windows.exists() else prefix_path / "bin" / "python"


def _prefix_from_manual(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.name.casefold() in {"python", "python.exe"}:
        return candidate.parent if candidate.parent.name != "bin" else candidate.parent.parent
    return candidate


def discover_environments(
    *,
    manual_paths: Iterable[str | Path] = (),
    environ: Mapping[str, str] | None = None,
    runner: CommandRunner = subprocess.run,
    current_executable: str | Path | None = None,
) -> list[EnvironmentCandidate]:
    """Find current, Conda-listed and common Windows environments."""

    env = dict(os.environ if environ is None else environ)
    discovered: list[tuple[Path, str]] = []
    current = Path(current_executable or sys.executable).resolve()
    current_prefix = current.parent if current.parent.name != "bin" else current.parent.parent
    discovered.append((current_prefix, "current"))
    if env.get("CONDA_PREFIX"):
        discovered.append((Path(env["CONDA_PREFIX"]), "CONDA_PREFIX"))
    for manual in manual_paths:
        discovered.append((_prefix_from_manual(manual), "manual"))
    try:
        completed = runner(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            raw = json.loads(completed.stdout)
            for prefix in raw.get("envs", []):
                discovered.append((Path(prefix), "conda"))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    user_profile = Path(env.get("USERPROFILE", Path.home()))
    for distribution in ("miniconda3", "anaconda3", "miniforge3"):
        root = user_profile / distribution
        discovered.append((root, "common"))
        envs_dir = root / "envs"
        if envs_dir.is_dir():
            for child in sorted(envs_dir.iterdir()):
                if child.is_dir():
                    discovered.append((child, "common"))
    result: list[EnvironmentCandidate] = []
    seen: set[str] = set()
    for prefix, source in discovered:
        try:
            normalized = str(prefix.resolve()).casefold()
        except OSError:
            normalized = str(prefix.absolute()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        executable = python_executable(prefix)
        if executable.is_file() or source in {"current", "manual", "CONDA_PREFIX"}:
            result.append(EnvironmentCandidate(prefix, executable, source))
    result.sort(key=lambda item: (item.prefix.name.casefold() != "yolo", item.source != "manual"))
    return result


def ensure_writable_yolo_config_dir(
    *,
    environ: Mapping[str, str] | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Set a writable Ultralytics config location outside user AppData."""

    mutable = os.environ if environ is None else environ
    configured = mutable.get("YOLO_CONFIG_DIR")
    if configured:
        destination = Path(configured)
    else:
        root = Path(base_dir or tempfile.gettempdir())
        destination = root / "ai_biaozhu" / "ultralytics"
        if hasattr(mutable, "__setitem__"):
            mutable["YOLO_CONFIG_DIR"] = str(destination)  # type: ignore[index]
    destination.mkdir(parents=True, exist_ok=True)
    probe = destination / ".write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"YOLO_CONFIG_DIR 不可写：{destination}: {exc}") from exc
    return destination


def inspect_environment(
    candidate: EnvironmentCandidate | str | Path,
    *,
    runner: CommandRunner = subprocess.run,
    timeout: float = 30,
) -> EnvironmentReport:
    if not isinstance(candidate, EnvironmentCandidate):
        prefix = _prefix_from_manual(candidate)
        candidate = EnvironmentCandidate(prefix, python_executable(prefix), "manual")
    if not candidate.python.is_file():
        return EnvironmentReport(
            candidate,
            False,
            None,
            None,
            None,
            None,
            False,
            None,
            None,
            ("未找到 Python 解释器",),
            (),
            False,
            {},
        )
    child_environment = dict(os.environ)
    try:
        child_environment["YOLO_CONFIG_DIR"] = str(ensure_writable_yolo_config_dir())
    except RuntimeError as exc:
        return EnvironmentReport(
            candidate,
            False,
            None,
            None,
            None,
            None,
            False,
            None,
            None,
            (str(exc),),
            (),
            False,
            {},
        )
    try:
        completed = runner(
            [str(candidate.python), "-c", _PROBE],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=child_environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return EnvironmentReport(
            candidate,
            False,
            None,
            None,
            None,
            None,
            False,
            None,
            None,
            (str(exc),),
            (),
            False,
            {},
        )
    if completed.returncode != 0:
        error = completed.stderr.strip() or f"探测进程退出码 {completed.returncode}"
        return EnvironmentReport(
            candidate,
            False,
            None,
            None,
            None,
            None,
            False,
            None,
            None,
            (error,),
            (),
            False,
            {},
        )
    try:
        raw = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return EnvironmentReport(
            candidate,
            False,
            None,
            None,
            None,
            None,
            False,
            None,
            None,
            (f"环境探测输出无效：{exc}",),
            (),
            False,
            {},
        )
    errors = tuple(str(item) for item in raw.get("errors", []))
    compatibility_errors = _compatibility_errors(raw)
    gpu_ready = bool(raw.get("cuda_available", False)) and str(
        raw.get("cuda_version") or ""
    ).startswith("12.8")
    valid = (
        not errors
        and not compatibility_errors
        and bool(raw.get("torch"))
        and bool(raw.get("torchvision"))
        and bool(raw.get("ultralytics"))
    )
    return EnvironmentReport(
        candidate=candidate,
        valid=valid,
        python_version=_optional(raw.get("python")),
        torch_version=_optional(raw.get("torch")),
        torchvision_version=_optional(raw.get("torchvision")),
        ultralytics_version=_optional(raw.get("ultralytics")),
        cuda_available=bool(raw.get("cuda_available", False)),
        cuda_version=_optional(raw.get("cuda_version")),
        device_name=_optional(raw.get("device_name")),
        errors=errors,
        compatibility_errors=tuple(compatibility_errors),
        gpu_ready=gpu_ready,
        raw=raw,
    )


def inspect_all(
    candidates: Sequence[EnvironmentCandidate],
    *,
    runner: CommandRunner = subprocess.run,
) -> list[EnvironmentReport]:
    return [inspect_environment(candidate, runner=runner) for candidate in candidates]


def inspect_legacy_yolov5_repository(
    path: str | Path,
    *,
    expected_tag: str = LEGACY_YOLOV5_TAG,
    runner: CommandRunner = subprocess.run,
) -> LegacyYoloV5RepositoryReport:
    """Validate official legacy scripts and an immutable repository tag lock."""

    repository = Path(path).resolve()
    errors: list[str] = []
    for filename in ("train.py", "detect.py", "export.py"):
        if not (repository / filename).is_file():
            errors.append(f"缺少 {filename}")
    tag: str | None = None
    lock_file = repository / LEGACY_TAG_LOCK_FILE
    if lock_file.is_file():
        tag = lock_file.read_text(encoding="utf-8").strip().splitlines()[0]
    elif (repository / ".git").exists():
        try:
            completed = runner(
                ["git", "describe", "--tags", "--exact-match", "HEAD"],
                cwd=str(repository),
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if completed.returncode == 0:
                tag = completed.stdout.strip()
            else:
                errors.append("仓库 HEAD 未锁定在发布 tag")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"无法读取 YOLOv5 git tag：{exc}")
    else:
        errors.append(
            f"缺少 {LEGACY_TAG_LOCK_FILE}，且目录不是可验证的 git 仓库"
        )
    if tag is not None and tag != expected_tag:
        errors.append(f"YOLOv5 tag 必须是 {expected_tag}，当前为 {tag}")
    return LegacyYoloV5RepositoryReport(
        path=repository,
        valid=not errors,
        tag=tag,
        expected_tag=expected_tag,
        errors=tuple(errors),
    )


def _compatibility_errors(raw: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    python_version = _numeric_version(raw.get("python"))
    if python_version[:2] != PYTHON_BASELINE:
        errors.append(f"Python 需要 3.11.x，当前为 {raw.get('python')}")
    if str(raw.get("torch") or "") != TORCH_BASELINE:
        errors.append(f"torch 需要 {TORCH_BASELINE}，当前为 {raw.get('torch')}")
    if str(raw.get("torchvision") or "") != TORCHVISION_BASELINE:
        errors.append(
            f"torchvision 需要 {TORCHVISION_BASELINE}，当前为 {raw.get('torchvision')}"
        )
    ultralytics = _numeric_version(raw.get("ultralytics"))
    if not (ULTRALYTICS_MINIMUM <= ultralytics < ULTRALYTICS_MAXIMUM):
        errors.append(
            "ultralytics 需要 >=8.4.82,<8.5.0，"
            f"当前为 {raw.get('ultralytics')}"
        )
    return errors


def _numeric_version(value: Any) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value or ""))
    if not match:
        return ()
    return tuple(int(item or 0) for item in match.groups())


def _optional(value: Any) -> str | None:
    return None if value is None else str(value)
