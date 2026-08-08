from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_biaozhu.deploy.dependencies import (
    DeploymentDependencyError,
    inspect_deployment_dependencies,
)


def test_dependency_preflight_reports_versions_without_installing() -> None:
    calls: list[str] = []

    def importer(name: str):
        calls.append(name)
        return SimpleNamespace(__version__=f"{name}-1")

    report = inspect_deployment_dependencies(
        ("onnx", "onnxruntime"),
        importer=importer,
        standalone=True,
    )
    assert report.ready
    assert report.missing == ()
    assert calls == ["onnx", "onnxruntime"]
    assert report.to_dict()["dependencies"][0]["version"] == "onnx-1"


def test_frozen_dependency_failure_requires_repair_and_never_pip() -> None:
    def importer(name: str):
        if name == "onnxsim":
            raise ImportError("missing DLL")
        return SimpleNamespace(__version__="1")

    report = inspect_deployment_dependencies(
        ("onnx", "onnxsim"),
        importer=importer,
        standalone=True,
    )
    assert not report.ready
    assert report.missing == ("onnxsim",)
    with pytest.raises(DeploymentDependencyError, match="不会在后台运行 pip"):
        report.require_ready()
