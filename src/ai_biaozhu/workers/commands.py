"""Pure command construction kept independent from Qt for easy testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    program: str
    arguments: tuple[str, ...]

    def as_list(self) -> list[str]:
        return [self.program, *self.arguments]


def build_worker_command(
    python_executable: str | Path,
    action: str,
    *,
    manifest: str | Path | None = None,
    inspect_python: str | Path | None = None,
) -> WorkerCommand:
    if action in {"train", "predict", "deploy", "docker-import"}:
        if manifest is None:
            raise ValueError("train/predict/deploy/docker-import 命令必须提供 manifest")
        arguments = (
            "-m",
            "ai_biaozhu.workers.main",
            action,
            "--manifest",
            str(Path(manifest)),
        )
    elif action == "environment":
        if inspect_python is None:
            raise ValueError("environment 命令必须提供 inspect_python")
        arguments = (
            "-m",
            "ai_biaozhu.workers.main",
            "environment",
            "--python",
            str(Path(inspect_python)),
        )
    else:
        raise ValueError(f"未知 worker action：{action}")
    return WorkerCommand(str(Path(python_executable)), arguments)
