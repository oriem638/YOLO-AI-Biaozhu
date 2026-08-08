from __future__ import annotations

from pathlib import Path

import pytest

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.errors import ValidationError
from ai_biaozhu.ml.protocol import ProtocolEvent
from ai_biaozhu.settings import SettingsStore


def _controller(root: Path) -> ApplicationController:
    paths = AppPaths(
        data=root / "data",
        cache=root / "cache",
        logs=root / "logs",
        models=root / "models",
        yolo_config=root / "yolo-config",
    )
    return ApplicationController(
        paths,
        settings=SettingsStore(root / "settings.json"),
        source_root=Path(__file__).resolve().parents[1],
    )


def test_completed_training_without_checkpoint_does_not_unlock_retraining(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "checkpoint gate")
    run = project.repository.create_run("train", "YOLO26n")
    project.repository.update_run(run.id, status="completed", progress=1.0)

    assert controller._successful_training_runs(project) == ()

    checkpoint = project.runs_dir / run.id / "training" / "model" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    project.repository.update_run(
        run.id,
        artifacts={"last": str(checkpoint)},
        checkpoint_path=str(checkpoint),
    )

    assert [item.id for item in controller._successful_training_runs(project)] == [run.id]


def test_checkpoint_role_never_aliases_last_as_best(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "checkpoint role")
    run = project.repository.create_run("train", "YOLO26n")
    checkpoint = project.runs_dir / run.id / "training" / "model" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"last")
    run = project.repository.update_run(
        run.id,
        status="completed",
        checkpoint_path=str(checkpoint),
    )

    assert controller._checkpoint_from_run(project, run, "last") == checkpoint.resolve()
    with pytest.raises(ValidationError, match="best.pt"):
        controller._checkpoint_from_run(project, run, "best")


def test_completed_event_without_training_artifact_is_failed(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "completed gate")
    run = project.repository.create_run("train", "YOLO26n")
    project.repository.update_run(run.id, status="training")

    controller.handle_job_event(
        ProtocolEvent.create(
            job_id=run.id,
            seq=0,
            event_type="completed",
            payload={"command": "train", "result": {"artifacts": {}}},
        ).to_dict()
    )

    failed = project.repository.get_run(run.id)
    assert failed.status.value == "failed"
    assert "best.pt" in str(failed.error)
