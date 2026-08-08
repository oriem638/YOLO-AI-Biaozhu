from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.core import AIPrediction, BoxInput
from ai_biaozhu.errors import ValidationError
from ai_biaozhu.ml.environment import EnvironmentCandidate, EnvironmentReport
from ai_biaozhu.settings import SettingsStore
from ai_biaozhu.ui.dialogs import TrainingPreflightDialog, TrainingSettingsDialog
from ai_biaozhu.ui.main_window import MainWindow, TrainingMonitorWidget


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class TrainingUiController:
    def __init__(self, root: Path) -> None:
        self.current_project = {"root": root, "project_id": "training-ui"}
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        self.images: list[dict[str, Any]] = []
        statuses = ("verified", "unreviewed", "draft", "verified", "verified")
        for index, status in enumerate(statuses, start=1):
            path = images_dir / f"{index}.png"
            image = QImage(48, 32, QImage.Format.Format_RGB32)
            image.fill(QColor("black"))
            assert image.save(str(path))
            self.images.append(
                {
                    "id": f"image-{index}",
                    "relative_path": str(path.relative_to(root)),
                    "original_name": path.name,
                    "review_status": status,
                    "origin": "ai" if status == "draft" else "manual",
                    "ai_status": "done" if status == "draft" else "none",
                    "training_selected": index != 5,
                }
            )
        self.seed_verified_count = 100
        self.started: list[tuple[str, dict[str, Any]]] = []
        self.finished_calls: list[dict[str, Any]] = []

    def list_images(self) -> list[dict[str, Any]]:
        return self.images

    @staticmethod
    def list_classes() -> list[dict[str, Any]]:
        return [{"id": "ball", "name": "BALL", "color": "#45c486"}]

    @staticmethod
    def list_runs() -> list[dict[str, Any]]:
        return []

    @staticmethod
    def get_boxes(_image_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "box",
                "class_id": "ball",
                "xmin": 2,
                "ymin": 2,
                "xmax": 20,
                "ymax": 20,
                "origin": "manual",
            }
        ]

    def training_preflight(self) -> dict[str, Any]:
        return _preflight_summary()

    def start_training(
        self,
        model_key: str,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        self.started.append((model_key, settings))
        summary = _preflight_summary()
        summary.update(
            {
                "split_counts": {"train": 1, "val": 1, "test": 0},
                "manifest_path": "C:/project/runs/run-1/snapshot/manifest.json",
                "current_selection_count": settings.get(
                    "current_selection_count",
                    0,
                ),
            }
        )
        return {"job_id": "run-1", "snapshot_summary": summary}

    def handle_process_finished(
        self,
        job_id: str,
        **values: Any,
    ) -> None:
        self.finished_calls.append({"job_id": job_id, **values})


def _preflight_summary() -> dict[str, Any]:
    return {
        "ok": True,
        "allowed": True,
        "counts": {
            "project_total": 5,
            "training_selected": 4,
            "trainable_total": 2,
            "trainable_verified": 1,
            "verified_negative": 1,
            "unlabeled": 1,
            "ai_unconfirmed": 1,
        },
        "split_counts": {"train": 1, "val": 1, "test": 0},
        "class_box_counts": {"BALL": 3},
        "training_member_fingerprint": "a" * 64,
        "manifest_path": None,
        "samples": {
            "unlabeled": [{"index": 2, "filename": "2.png"}],
            "ai_unconfirmed": [{"index": 3, "filename": "3.png"}],
        },
        "errors": [],
        "warnings": [],
    }


def test_global_indices_range_and_anchor_selection_are_filter_stable(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = TrainingUiController(tmp_path)
    window = MainWindow(controller)
    assert [
        window.image_list.item(row).data(Qt.ItemDataRole.UserRole + 2)
        for row in range(window.image_list.count())
    ] == [1, 2, 3, 4, 5]
    assert window.image_list.item(2).text().startswith("[3]")

    window.image_filter_combo.setCurrentIndex(
        window.image_filter_combo.findData("verified")
    )
    window.image_range_edit.setText("1-2，4, 2")
    assert window.apply_image_range_selection()
    assert {
        item.data(Qt.ItemDataRole.UserRole + 2)
        for item in window.image_list.selectedItems()
    } == {1, 2, 4}

    training_state = [item["training_selected"] for item in controller.images]
    selected_before = set(window.selected_image_ids())
    window.image_range_edit.setText("4-2")
    assert not window.apply_image_range_selection()
    assert set(window.selected_image_ids()) == selected_before
    assert [item["training_selected"] for item in controller.images] == training_state

    window.image_filter_combo.setCurrentIndex(
        window.image_filter_combo.findData("all")
    )
    window.show()
    QTest.qWait(5)
    start_item = window.image_list.item(1)
    end_item = window.image_list.item(4)
    window.image_list.scrollToItem(start_item)
    QTest.mouseClick(
        window.image_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=window.image_list.visualItemRect(start_item).center(),
    )
    QTest.mouseClick(window.select_to_here_button, Qt.MouseButton.LeftButton)
    window.image_list.scrollToItem(end_item)
    QTest.mouseClick(
        window.image_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=window.image_list.visualItemRect(end_item).center(),
    )
    QTest.qWait(5)
    assert not window.select_to_here_button.isChecked()
    assert {
        item.data(Qt.ItemDataRole.UserRole + 2)
        for item in window.image_list.selectedItems()
    }.issuperset({2, 3, 4, 5})
    window.close()
    app.processEvents()


def test_early_stopping_is_explicit_and_disabled_means_zero_patience(
    app: QApplication,
) -> None:
    dialog = TrainingSettingsDialog(
        {"early_stopping_enabled": False, "patience": 20}
    )
    assert not dialog.early_stopping_check.isChecked()
    assert not dialog.patience_spin.isEnabled()
    assert dialog.values()["patience"] == 0
    assert "fitness" in dialog.early_stopping_monitor_label.text()
    dialog.early_stopping_check.setChecked(True)
    dialog.patience_spin.setValue(7)
    assert dialog.values()["early_stopping_enabled"] is True
    assert dialog.values()["patience"] == 7
    dialog.close()
    app.processEvents()


def test_preflight_dialog_and_monitor_show_exact_membership_and_end_reason(
    app: QApplication,
) -> None:
    dialog = TrainingPreflightDialog(
        _preflight_summary(),
        has_unconfirmed=True,
    )
    text = dialog.summary_view.toPlainText()
    assert dialog.continue_button.text() == "仅训练已人工确认的样本"
    assert dialog.return_button.text() == "返回继续标注"
    assert "第 2 张：2.png" in text
    assert "第 3 张：3.png" in text
    assert "正样本：1" in text and "已确认空白负样本：1" in text
    dialog.close()

    monitor = TrainingMonitorWidget()
    monitor.handle_event(
        {
            "type": "metrics",
            "payload": {
                "train/box_loss": 0.25,
                "train/cls_loss": 0.1,
                "train/obj_loss": 0.05,
                "metrics/mAP_0.5": 0.75,
                "metrics/mAP_0.5:0.95": 0.5,
                "dfl_loss": None,
                "dfl_loss_status": "unavailable",
            },
        }
    )
    assert monitor.map_plot.history["map50"] == [0.75]
    assert monitor.map_plot.history["map50_95"] == [0.5]
    assert monitor.loss_plot.history["objectness_loss"] == [0.05]
    assert "obj 0.0500" in monitor.loss_label.text()
    assert "不可用" in monitor.loss_label.text()
    monitor.handle_event(
        {
            "type": "status",
            "payload": {
                "stage": "training_finished",
                "reason": "early_stopping",
                "completed_epochs": 36,
                "requested_epochs": 100,
                "patience": 20,
                "monitor": "fitness",
            },
        }
    )
    assert "触发早停" in monitor.state_label.text()
    assert "36 / 请求 100" in monitor.state_label.text()
    monitor.handle_event(
        {
            "type": "completed",
            "payload": {
                "result": {
                    "training_end": {
                        "reason": "max_epochs",
                        "completed_epochs": 100,
                        "requested_epochs": 100,
                        "patience": 20,
                        "monitor": "fitness",
                    }
                }
            },
        }
    )
    assert "达到最大轮数（max_epochs）" in monitor.state_label.text()
    monitor.handle_event(
        {
            "type": "cancelled",
            "payload": {"completed_epochs": 12, "requested_epochs": 100},
        }
    )
    assert "训练已取消（cancelled）" in monitor.state_label.text()
    assert "12 / 请求 100" in monitor.state_label.text()
    monitor.close()
    app.processEvents()


def test_start_training_replaces_preflight_with_created_snapshot_summary(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = TrainingUiController(tmp_path)
    window = MainWindow(controller)
    window._training_settings = {
        **window._training_settings,
        "early_stopping_enabled": False,
        "patience": 20,
    }
    assert window.start_training()
    assert controller.started[0][1]["patience"] == 0
    summary = window.training_snapshot_summary.toPlainText()
    assert "不可变训练快照已创建" in summary
    assert "当前图片列表多选：1" in summary
    assert "train 1 / val 1 / test 0" in summary
    assert "snapshot/manifest.json" in summary
    window.training_monitor.handle_event(
        {
            "type": "progress",
            "payload": {"stage": "training", "current": 2, "total": 100},
        }
    )
    window._on_process_event(
        {
            "job_id": "run-1",
            "seq": 99,
            "type": "process_finished",
            "payload": {
                "success": False,
                "cancelled": True,
                "exit_code": 130,
            },
            "_internal": True,
        }
    )
    assert controller.finished_calls == [
        {
            "job_id": "run-1",
            "success": False,
            "exit_code": 130,
            "cancelled": True,
            "completed_epochs": 2,
            "requested_epochs": 100,
        }
    ]
    assert "训练已取消（cancelled）" in window.training_monitor.state_label.text()
    window.close()
    app.processEvents()


def test_model_download_progress_never_impersonates_epoch_progress(
    app: QApplication,
) -> None:
    monitor = TrainingMonitorWidget()
    monitor.reset("正在准备训练…", requested_epochs=150)
    monitor.handle_event(
        {
            "type": "progress",
            "payload": {
                "stage": "downloading_weight",
                "filename": "yolo26s.pt",
                "current_bytes": 3_258_600,
                "total_bytes": 20_422_725,
                "progress": 3_258_600 / 20_422_725,
                "requested_epochs": 150,
            },
        }
    )
    assert monitor.preparation_progress.value() == 16
    assert "3.26 MB / 20.42 MB（16%）" in monitor.state_label.text()
    assert monitor.epoch_progress.value() == 0
    assert monitor.epoch_progress.maximum() == 150
    assert monitor._completed_epochs == 0

    # A generic worker percentage with no epoch fields must not move Epoch.
    monitor.handle_event(
        {"type": "progress", "payload": {"stage": "snapshot", "progress": 0.75}}
    )
    assert monitor.epoch_progress.value() == 0
    monitor.handle_event(
        {
            "type": "status",
            "payload": {
                "stage": "training",
                "current": 0,
                "total": 150,
                "requested_epochs": 150,
            },
        }
    )
    assert monitor.preparation_progress.value() == 100
    monitor.handle_event(
        {
            "type": "metrics",
            "payload": {"epoch": 1, "epochs": 150, "box_loss": 0.5},
        }
    )
    assert monitor.epoch_progress.value() == 1
    assert monitor._completed_epochs == 1
    monitor.close()
    app.processEvents()


def test_prepare_failure_is_idempotent_and_reports_zero_completed_epochs(
    app: QApplication,
) -> None:
    monitor = TrainingMonitorWidget()
    monitor.reset(requested_epochs=150)
    monitor.handle_event(
        {
            "type": "error",
            "payload": {"message": "下载中断", "requested_epochs": 150},
        }
    )
    monitor.handle_event(
        {
            "type": "process_finished",
            "payload": {"success": False, "exit_code": 1},
        }
    )
    assert monitor._completed_epochs == 0
    assert monitor.epoch_progress.value() == 0
    assert monitor.epoch_progress.maximum() == 150
    assert monitor.preparation_progress.format() == "模型准备失败"
    assert monitor.log_view.toPlainText().count("训练结束：") == 1
    monitor.close()
    app.processEvents()


def test_failure_during_first_epoch_batch_reports_zero_completed_epochs(
    app: QApplication,
) -> None:
    monitor = TrainingMonitorWidget()
    monitor.reset(requested_epochs=150)
    monitor.handle_event(
        {
            "type": "progress",
            "payload": {
                "stage": "training_batch",
                "epoch": 1,
                "epochs": 150,
                "current": 5,
                "total": 100,
            },
        }
    )
    # The UI may show that epoch 1 is currently running, but terminal
    # accounting still records zero *completed* epochs.
    assert monitor.epoch_progress.value() == 1
    assert monitor._completed_epochs == 0
    monitor.handle_event(
        {"type": "error", "payload": {"message": "worker failed"}}
    )
    assert monitor._completed_epochs == 0
    assert monitor.epoch_progress.value() == 0
    assert monitor.epoch_progress.maximum() == 150
    monitor.close()
    app.processEvents()


def test_training_start_is_single_flight_even_when_called_programmatically(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = TrainingUiController(tmp_path)
    window = MainWindow(controller)
    assert window.start_training()
    assert not window.start_training()
    assert len(controller.started) == 1
    assert "不能重复启动训练" in window.statusBar().currentMessage()
    window.close()
    app.processEvents()


def _paths(root: Path) -> AppPaths:
    return AppPaths(
        data=root / "app-data",
        cache=root / "cache",
        logs=root / "logs",
        models=root / "models",
        yolo_config=root / "ultralytics",
    )


def _valid_environment(_value: object) -> EnvironmentReport:
    python = Path(sys.executable).resolve()
    return EnvironmentReport(
        candidate=EnvironmentCandidate(python.parent, python, "test"),
        valid=True,
        python_version="3.11.15",
        torch_version="2.11.0+cu128",
        torchvision_version="0.26.0+cu128",
        ultralytics_version="8.4.82",
        cuda_available=False,
        cuda_version=None,
        device_name=None,
        errors=(),
        compatibility_errors=(),
        gpu_ready=False,
        raw={},
    )


def _application_controller(root: Path) -> ApplicationController:
    return ApplicationController(
        _paths(root),
        settings=SettingsStore(root / "settings.json"),
        environment_inspector=_valid_environment,
        source_root=Path(__file__).resolve().parents[1],
    )


def _add_image_file(project: Any, image_id: str, name: str) -> Any:
    destination = project.root / "images" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (20, 40, 60)).save(destination)
    return project.repository.add_image_record(
        image_id=image_id,
        relative_path=f"images/{name}",
        original_name=name,
        source_path=None,
        sha256=(image_id.encode("utf-8").hex() + "0" * 64)[:64],
        width=32,
        height=32,
    )


def _record_prior_success(project: Any) -> None:
    run = project.repository.create_run("train", "YOLO26n", run_id="prior")
    checkpoint = project.runs_dir / run.id / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"best")
    project.repository.update_run(
        run.id,
        status="completed",
        progress=1.0,
        artifacts={"best": str(checkpoint)},
        checkpoint_path=str(checkpoint),
    )


def test_controller_preflight_and_launch_report_exact_snapshot_membership(
    tmp_path: Path,
) -> None:
    controller = _application_controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "training")
    category = project.repository.list_categories()[0]
    positive = _add_image_file(project, "positive", "positive.png")
    negative = _add_image_file(project, "negative", "negative.png")
    unlabeled = _add_image_file(project, "unlabeled", "unlabeled.png")
    draft = _add_image_file(project, "draft", "draft.png")
    excluded = _add_image_file(project, "excluded", "excluded.png")
    project.repository.save_and_confirm(
        positive.id,
        [BoxInput(category.id, 2, 2, 20, 20)],
    )
    project.repository.confirm_image(negative.id, confirm_empty=True)
    project.repository.save_and_confirm(
        excluded.id,
        [BoxInput(category.id, 4, 4, 22, 22)],
    )
    project.set_training_selected((excluded.id,), False)
    predict = project.repository.create_run("predict", "YOLO26n")
    project.repository.import_ai_predictions(
        predict.id,
        draft.id,
        [AIPrediction(draft.id, category.id, 3, 3, 18, 18)],
    )
    _record_prior_success(project)

    settings = {
        "epochs": 3,
        "early_stopping_enabled": False,
        "patience": 99,
        "batch": 1,
        "device": "cpu",
        "workers": 0,
        "ml_environment": sys.executable,
        "split": {
            "train_ratio": 0.5,
            "val_ratio": 0.5,
            "test_ratio": 0.0,
            "seed": 7,
        },
    }
    preflight = controller.training_preflight(settings)
    assert preflight["counts"] == {
        "project_total": 5,
        "training_selected": 4,
        "unlabeled": 1,
        "ai_unconfirmed": 1,
        "trainable_verified": 1,
        "verified_negative": 1,
        "trainable_total": 2,
        "skipped_unconfirmed": 2,
        "excluded_not_selected": 1,
    }
    assert preflight["split_counts"] == {"train": 1, "val": 1, "test": 0}
    assert preflight["class_box_counts"] == {category.name: 1}
    assert len(preflight["training_member_fingerprint"]) == 64
    assert preflight["samples"]["ai_unconfirmed"][0]["filename"] == "draft.png"

    with pytest.raises(ValidationError, match="训练样本在确认后发生变化"):
        controller.start_training(
            "YOLO26n",
            {**settings, "expected_training_member_fingerprint": "0" * 64},
        )
    launch = controller.start_training("YOLO26n", settings)
    snapshot = launch["snapshot_summary"]
    assert snapshot["counts"]["trainable_total"] == 2
    assert snapshot["split_counts"] == {"train": 1, "val": 1, "test": 0}
    assert Path(snapshot["manifest_path"]).is_file()
    snapshot_manifest = json.loads(
        Path(snapshot["manifest_path"]).read_text(encoding="utf-8")
    )
    assert {item["image_id"] for item in snapshot_manifest["images"]} == {
        positive.id,
        negative.id,
    }
    job_manifest = project.runs_dir / str(launch["job_id"]) / "job.json"
    manifest = json.loads(job_manifest.read_text(encoding="utf-8"))
    assert manifest["config"]["patience"] == 0
    assert (
        manifest["training_member_fingerprint"]
        == snapshot["training_member_fingerprint"]
    )
    controller.handle_process_finished(
        str(launch["job_id"]),
        success=False,
        exit_code=130,
        cancelled=True,
        completed_epochs=1,
        requested_epochs=3,
    )
    cancelled_run = project.repository.get_run(str(launch["job_id"]))
    assert cancelled_run.status.value == "cancelled"
    assert cancelled_run.metrics["training_end"] == {
        "reason": "cancelled",
        "completed_epochs": 1,
        "requested_epochs": 3,
        "patience": 0,
        "monitor": "fitness",
        "evidence": ["cancelled"],
    }
    failed = project.repository.create_run(
        "train",
        "YOLO26n",
        parameters={"epochs": 5, "patience": 2},
    )
    project.repository.update_run(
        failed.id,
        status="training",
        metrics={"epoch": 2, "epochs": 5},
    )
    controller.handle_process_finished(
        failed.id,
        success=False,
        exit_code=1,
    )
    failed = project.repository.get_run(failed.id)
    assert failed.status.value == "failed"
    assert failed.metrics["training_end"]["reason"] == "failed"
    assert failed.metrics["training_end"]["completed_epochs"] == 2
    assert failed.metrics["training_end"]["requested_epochs"] == 5
    assert unlabeled.id != draft.id
