from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ai_biaozhu.ml.adapters import AdapterError, LegacyYoloV5Adapter
from ai_biaozhu.ml.config import AugmentationOptions
from ai_biaozhu.ml.jobs import PredictionJob, TrainingJob
from ai_biaozhu.ml.legacy import (
    legacy_hyp_values,
    prepare_legacy_blur_snapshot,
    read_new_results_rows,
)
from ai_biaozhu.ml.legacy_bootstrap import (
    install_ipython_legacy_compatibility,
)
from ai_biaozhu.ml.legacy_bootstrap import (
    run_legacy_script as run_bootstrap_legacy_script,
)
from ai_biaozhu.ml.legacy_process import (
    build_legacy_script_command,
    install_pillow_legacy_compatibility,
    legacy_subprocess_environment,
    legacy_torch_onnx_export_compatibility,
)
from ai_biaozhu.ml.protocol import JsonlEmitter, read_jsonl_events


def test_legacy_hyp_maps_rotation_and_flips() -> None:
    values = legacy_hyp_values(
        AugmentationOptions(
            rotation_degrees=12,
            rotation_probability=0.5,
            fliplr=0.3,
            flipud=0.1,
        )
    )
    assert values["degrees"] == 12
    assert values["fliplr"] == 0.3
    assert values["flipud"] == 0.1
    assert values["lr0"] == 0.01


def test_pillow_legacy_compatibility_uses_getbbox() -> None:
    class Font:
        def getbbox(self, text, *args, **kwargs):
            assert text == "label"
            return 2, 3, 14, 10

    module = SimpleNamespace(FreeTypeFont=Font)
    assert install_pillow_legacy_compatibility(module) is True
    assert Font().getsize("label") == (12, 7)
    assert install_pillow_legacy_compatibility(module) is False


def test_legacy_torch_onnx_compatibility_defaults_and_restores_export() -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "exported"

    torch_module = SimpleNamespace(onnx=SimpleNamespace(export=original))
    with legacy_torch_onnx_export_compatibility(
        enabled=True,
        torch_module=torch_module,
    ) as installed:
        assert installed is True
        assert torch_module.onnx.export("model", "model.onnx") == "exported"
        torch_module.onnx.export("model", "dynamo.onnx", dynamo=True)
    assert calls[0][1] == {"dynamo": False}
    assert calls[1][1] == {"dynamo": True}
    assert torch_module.onnx.export is original


def test_ipython_compatibility_supports_yolov5_desktop_imports(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.delitem(sys.modules, "IPython.display", raising=False)

    assert install_ipython_legacy_compatibility() is True
    from IPython import display
    from IPython.display import clear_output

    assert display.display("notebook-only") is None
    assert clear_output() is None


def test_source_bootstrap_runs_yolov5_ipython_imports_without_dependency(tmp_path) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    (repository / "train.py").write_text(
        "from IPython.display import display\n"
        "import IPython\n"
        "assert IPython.get_ipython() is None\n"
        "assert display('desktop') is None\n",
        encoding="utf-8",
    )

    assert run_bootstrap_legacy_script(repository, "train", []) == 0


def test_source_bootstrap_applies_onnx_shim_only_to_export(
    tmp_path,
    monkeypatch,
) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    script = "import torch\ntorch.onnx.export('model', 'model.onnx')\n"
    (repository / "export.py").write_text(script, encoding="utf-8")
    (repository / "train.py").write_text(script, encoding="utf-8")
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))

    torch_module = SimpleNamespace(onnx=SimpleNamespace(export=original))
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert run_bootstrap_legacy_script(repository, "export", []) == 0
    assert calls[-1][1] == {"dynamo": False}
    assert torch_module.onnx.export is original

    assert run_bootstrap_legacy_script(repository, "train", []) == 0
    assert calls[-1][1] == {}
    assert torch_module.onnx.export is original


def test_blur_snapshot_is_train_only_and_keeps_labels(tmp_path) -> None:
    source = tmp_path / "snapshot"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32), "red").save(source / "images" / split / "one.jpg")
        (source / "labels" / split / "one.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    data_yaml = source / "data.yaml"
    data_yaml.write_text(
        f"path: {source.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [object]\n",
        encoding="utf-8",
    )
    generated = prepare_legacy_blur_snapshot(
        data_yaml,
        tmp_path / "blurred",
        probability=1.0,
        kernel=3,
        seed=42,
    )
    assert generated.is_file()
    assert (generated.parent / "images" / "train" / "one.jpg").is_file()
    assert (generated.parent / "labels" / "train" / "one.txt").read_text() != ""
    assert (generated.parent / "images" / "val" / "one.jpg").read_bytes() == (
        source / "images" / "val" / "one.jpg"
    ).read_bytes()


def test_results_csv_tail_returns_only_new_complete_epochs(tmp_path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "epoch,train/box_loss,metrics/mAP_0.5\n"
        "0,0.5,0.1\n"
        "1,0.4,0.2\n",
        encoding="utf-8",
    )
    rows = read_new_results_rows(path, after_epoch=0)
    assert rows == [
        (1, {"epoch": 1.0, "train/box_loss": 0.4, "metrics/mAP_0.5": 0.2})
    ]


def test_legacy_training_streams_epoch_metrics_and_retries_oom(tmp_path) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    for name in ("train.py", "detect.py", "export.py"):
        (repository / name).write_text("# placeholder", encoding="utf-8")
    (repository / ".ai-biaozhu-yolov5-tag").write_text("v7.0\n", encoding="utf-8")
    commands = []

    class Process:
        def __init__(self, lines, return_code):
            self.stdout = iter(lines)
            self.return_code = return_code

        def wait(self):
            return self.return_code

    def popen(command, **kwargs):
        commands.append(command)
        run_name = command[command.index("--name") + 1]
        save_dir = tmp_path / "runs" / run_name
        if len(commands) == 1:
            return Process(["RuntimeError: CUDA out of memory\n"], 1)
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"best")
        (save_dir / "results.csv").write_text(
            "epoch,train/box_loss,metrics/mAP_0.5\n0,0.5,0.2\n",
            encoding="utf-8",
        )
        return Process(["epoch 1/1 complete\n"], 0)

    job = TrainingJob.from_mapping(
        {
            "job_id": "legacy",
            "model_key": "YOLOv5n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "legacy_yolov5_repo": str(repository),
            "config": {"epochs": 1, "batch": 16},
            "augmentation": {"enabled": False},
            "checkpoint_source": str(tmp_path / "pretrained.pt"),
        }
    )
    stream = io.StringIO()
    result = LegacyYoloV5Adapter(popen_factory=popen).train(
        job, JsonlEmitter(job.job_id, stream)
    )
    assert len(commands) == 2
    assert commands[1][commands[1].index("--batch-size") + 1] == "8"
    assert result["artifacts"]["best"].endswith("best.pt")
    assert result["metrics"]["map50"] == 0.2
    assert result["metrics"]["dfl_loss"] is None
    assert result["training_end"] == {
        "reason": "max_epochs",
        "completed_epochs": 1,
        "requested_epochs": 1,
        "patience": 20,
        "monitor": "fitness",
        "evidence": ["completed_requested_epochs"],
    }
    events = read_jsonl_events(stream.getvalue().splitlines())
    assert any(event.type == "metrics" and event.payload["epoch"] == 1 for event in events)
    assert any(
        event.type == "metrics"
        and event.payload["metrics"]["map50"] == 0.2
        and event.payload["metrics"]["dfl_loss"] is None
        for event in events
    )
    assert any(
        event.type == "status"
        and event.payload.get("stage") == "training_finished"
        and event.payload["reason"] == "max_epochs"
        for event in events
    )
    assert any(
        event.type == "warning" and event.payload["code"] == "legacy_epoch_progress_only"
        for event in events
    )


def test_legacy_resume_uses_last_checkpoint_and_standalone_runner(tmp_path) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    for name in ("train.py", "detect.py", "export.py"):
        (repository / name).write_text("# placeholder", encoding="utf-8")
    resume = tmp_path / "last.pt"
    resume.write_bytes(b"checkpoint")
    job = TrainingJob.from_mapping(
        {
            "job_id": "legacy-resume",
            "model_key": "YOLOv5n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "config": {"resume": str(resume)},
            "checkpoint_source": str(tmp_path / "pretrained.pt"),
        }
    )
    command = LegacyYoloV5Adapter()._train_command(
        job,
        repo=repository,
        python=sys.executable,
        checkpoint=str(tmp_path / "pretrained.pt"),
        data_yaml=job.data_yaml,
        hyp_path=tmp_path / "hyp.yaml",
        batch=16,
        run_name="ignored",
    )
    assert command[-2:] == ["--resume", str(resume)]
    standalone = build_legacy_script_command(
        repository=repository,
        script="train",
        arguments=["--resume", str(resume)],
        python_executable=Path("AI-Biaozhu-Worker.exe"),
        standalone=True,
    )
    assert standalone[:5] == [
        "AI-Biaozhu-Worker.exe",
        "legacy-script",
        "--repository",
        str(repository),
        "train",
    ]
    windows_fonts = tmp_path / "Windows" / "Fonts"
    windows_fonts.mkdir(parents=True)
    (windows_fonts / "arial.ttf").write_bytes(b"ascii-font")
    (windows_fonts / "simhei.ttf").write_bytes(b"unicode-font")
    config_dir = tmp_path / "ultralytics-config"
    environment = legacy_subprocess_environment(
        {
            "YOLO_CONFIG_DIR": str(config_dir),
            "WINDIR": str(tmp_path / "Windows"),
        }
    )
    assert environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert environment["YOLOv5_AUTOINSTALL"] == "false"
    assert environment["YOLOV5_CONFIG_DIR"] == str(config_dir)
    assert (config_dir / "Arial.ttf").read_bytes() == b"ascii-font"
    assert (config_dir / "Arial.Unicode.ttf").read_bytes() == b"unicode-font"


def test_legacy_training_without_checkpoint_artifact_fails(tmp_path) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    for name in ("train.py", "detect.py", "export.py"):
        (repository / name).write_text("# placeholder", encoding="utf-8")
    (repository / ".ai-biaozhu-yolov5-tag").write_text(
        "v7.0\n", encoding="utf-8"
    )

    class Process:
        stdout = iter(())

        def wait(self):
            return 0

    job = TrainingJob.from_mapping(
        {
            "job_id": "legacy-no-artifact",
            "model_key": "YOLOv5n",
            "data_yaml": str(tmp_path / "data.yaml"),
            "output_dir": str(tmp_path / "runs"),
            "legacy_yolov5_repo": str(repository),
            "checkpoint_source": str(tmp_path / "source.pt"),
            "config": {"epochs": 1},
        }
    )
    with pytest.raises(AdapterError, match="best.pt/last.pt"):
        LegacyYoloV5Adapter(popen_factory=lambda *args, **kwargs: Process()).train(
            job,
            JsonlEmitter(job.job_id, io.StringIO()),
        )


def test_legacy_prediction_emits_image_running_before_result(tmp_path) -> None:
    repository = tmp_path / "yolov5"
    repository.mkdir()
    for name in ("train.py", "detect.py", "export.py"):
        (repository / name).write_text("# placeholder", encoding="utf-8")
    (repository / ".ai-biaozhu-yolov5-tag").write_text(
        "v7.0\n", encoding="utf-8"
    )
    source = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), "black").save(source)
    output = tmp_path / "predictions"

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def runner(command, **kwargs):
        run_name = command[command.index("--name") + 1]
        labels = output / run_name / "labels"
        labels.mkdir(parents=True)
        (labels / "sample.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.9\n",
            encoding="utf-8",
        )
        return Result()

    job = PredictionJob.from_mapping(
        {
            "job_id": "legacy-predict",
            "model_key": "YOLOv5n",
            "checkpoint": str(tmp_path / "best.pt"),
            "output_dir": str(output),
            "legacy_yolov5_repo": str(repository),
            "class_ids": ["object"],
            "images": [
                {
                    "image_id": "image-1",
                    "path": str(source),
                    "expected_revision": 7,
                    "width": 32,
                    "height": 32,
                }
            ],
        }
    )
    stream = io.StringIO()
    result = LegacyYoloV5Adapter(runner=runner).predict(
        job,
        JsonlEmitter(job.job_id, stream),
    )
    events = read_jsonl_events(stream.getvalue().splitlines())
    running_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "status" and event.payload.get("stage") == "image_running"
    )
    prediction_index = next(
        index for index, event in enumerate(events) if event.type == "prediction"
    )
    assert running_index < prediction_index
    assert events[running_index].payload["image_id"] == "image-1"
    assert events[running_index].payload["expected_revision"] == 7
    assert result == {"completed_images": 1, "failed_images": 0}
