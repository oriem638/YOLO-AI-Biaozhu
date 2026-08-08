"""Offline deployment dependency preflight for source and frozen workers."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

DEPLOYMENT_IMPORTS = ("onnx", "onnxruntime", "onnxsim", "onnxslim")


class DeploymentDependencyError(RuntimeError):
    """Raised before export when the bundled deployment runtime is incomplete."""


@dataclass(frozen=True, slots=True)
class DeploymentDependency:
    module: str
    available: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DeploymentDependencyReport:
    standalone: bool
    dependencies: tuple[DeploymentDependency, ...]

    @property
    def ready(self) -> bool:
        return all(item.available for item in self.dependencies)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(item.module for item in self.dependencies if not item.available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "standalone": self.standalone,
            "ready": self.ready,
            "dependencies": [asdict(item) for item in self.dependencies],
            "missing": list(self.missing),
        }

    def require_ready(self) -> DeploymentDependencyReport:
        if self.ready:
            return self
        names = ", ".join(self.missing)
        if self.standalone:
            raise DeploymentDependencyError(
                "独立安装版的部署组件不完整（缺少或无法导入："
                f"{names}）。程序不会在后台运行 pip；请使用安装程序的修复功能，"
                "或重新安装当前版本。"
            )
        raise DeploymentDependencyError(
            "部署环境缺少或无法导入依赖："
            f"{names}。请在开发环境中安装项目的 ml 可选依赖后重试。"
        )


def inspect_deployment_dependencies(
    modules: Sequence[str] = DEPLOYMENT_IMPORTS,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    standalone: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> DeploymentDependencyReport:
    """Import every required module without installing or changing anything."""

    environment = os.environ if environ is None else environ
    is_standalone = (
        _truthy(environment.get("AI_BIAOZHU_STANDALONE"))
        if standalone is None
        else bool(standalone)
    )
    results: list[DeploymentDependency] = []
    for name in modules:
        try:
            module = importer(str(name))
        except Exception as exc:  # an import-time DLL failure is also a hard failure
            results.append(
                DeploymentDependency(
                    str(name),
                    False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            version = getattr(module, "__version__", None)
            results.append(
                DeploymentDependency(
                    str(name),
                    True,
                    version=str(version) if version is not None else None,
                )
            )
    return DeploymentDependencyReport(is_standalone, tuple(results))


def _truthy(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}
