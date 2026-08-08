from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ai_biaozhu.deploy.export import (
    build_legacy_yolov5_export_command,
    export_modern_onnx,
    run_checkpoint_forward,
)


def test_modern_export_uses_static_batch1_opset17(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    output = checkpoint_dir / "model.onnx"
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    calls = {}

    class Model:
        def export(self, **kwargs):
            calls.update(kwargs)
            calls["export_cwd"] = Path.cwd()
            Path("model.onnx").write_bytes(b"onnx")
            return Path("model.onnx")

    def model_factory(checkpoint: str) -> Model:
        calls["factory_cwd"] = Path.cwd()
        calls["checkpoint"] = checkpoint
        return Model()

    result = export_modern_onnx(
        Path("..") / "checkpoint" / "best.pt",
        imgsz=(224, 320),
        model_factory=model_factory,
    )
    assert result == output
    assert Path.cwd() == caller_dir
    assert calls.pop("factory_cwd") == checkpoint_dir
    assert calls.pop("export_cwd") == checkpoint_dir
    assert calls.pop("checkpoint") == str(checkpoint_dir / "best.pt")
    assert calls == {
        "format": "onnx",
        "imgsz": [224, 320],
        "batch": 1,
        "dynamic": False,
        "simplify": True,
        "opset": 17,
    }


def test_legacy_export_uses_official_script(tmp_path) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    (repository / "export.py").write_text("# placeholder", encoding="utf-8")
    command = build_legacy_yolov5_export_command(
        python_executable=Path("C:/conda/envs/yolo/python.exe"),
        repository=repository,
        checkpoint=tmp_path / "best.pt",
        imgsz=(224, 320),
    )
    assert command[1:3] == ["-m", "ai_biaozhu.ml.legacy_bootstrap"]
    assert command[command.index("--repository") + 1] == str(repository)
    assert command[command.index(str(repository)) + 1] == "export"
    assert command[command.index("--opset") + 1] == "17"
    assert "--simplify" in command


def test_checkpoint_forward_reports_exact_class_order() -> None:
    class Model(torch.nn.Module):
        names = {0: "cat", 1: "dog"}

        def forward(self, value):
            return value * 2

    report = run_checkpoint_forward(
        "unused.pt",
        model_key="YOLO11n",
        input_array=np.ones((1, 3, 32, 32), dtype=np.float32),
        modern_model_factory=lambda _: SimpleNamespace(model=Model()),
        torch_module=torch,
    )
    assert report.class_names == ("cat", "dog")
    assert torch.equal(report.outputs[0], torch.full((1, 3, 32, 32), 2.0))
