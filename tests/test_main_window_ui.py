from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QProcess, Qt, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
)

import ai_biaozhu.ui.main_window as main_window_module
from ai_biaozhu.ui.dialogs import (
    ImportReportDialog,
    MaixDeployDialog,
    MLEnvironmentDialog,
    TrainingSettingsDialog,
    training_setting_warnings,
    validate_maix_deployment,
    validate_training_settings,
)
from ai_biaozhu.ui.main_window import MODEL_OPTIONS, JsonlProcessBridge, MainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


class FakeController:
    def __init__(self, root: Path) -> None:
        self.current_project = {"root": root, "project_id": "project-1"}
        image_path = root / "images" / "示例 图片.png"
        image_path.parent.mkdir(parents=True)
        image = QImage(320, 200, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        assert image.save(str(image_path))
        self.images = [
            {
                "id": "image-1",
                "relative_path": "images/示例 图片.png",
                "original_name": "示例 图片.png",
                "review_status": "verified",
                "origin": "manual",
                "ai_status": "none",
            }
        ]
        self.categories = [{"id": "cat", "name": "猫", "color": "#ff5353"}]
        self.saved: list[tuple[str, list[dict[str, Any]]]] = []
        self.handled_events: list[dict[str, Any]] = []
        self.started_training: list[tuple[str, dict[str, Any]]] = []
        self.seed_verified_count = 100
        self.runs = [
            {
                "id": "run-1",
                "kind": "train",
                "status": "completed",
                "model_key": "YOLO26n",
                "created_at": "2026-07-24T12:00:00",
                "artifacts": {
                    "best": str(root / "runs" / "run-1" / "weights" / "best.pt"),
                    "last": str(root / "runs" / "run-1" / "weights" / "last.pt"),
                },
            }
        ]

    def list_images(self) -> list[dict[str, Any]]:
        return self.images

    def list_classes(self) -> list[dict[str, Any]]:
        return self.categories

    def list_runs(self) -> list[dict[str, Any]]:
        return self.runs

    def get_boxes(self, image_id: str) -> list[dict[str, Any]]:
        del image_id
        return [
            {
                "id": "box-1",
                "class_id": "cat",
                "xmin": 10,
                "ymin": 20,
                "xmax": 100,
                "ymax": 120,
                "origin": "manual",
            }
        ]

    def save_boxes(self, image_id: str, boxes: list[dict[str, Any]]) -> None:
        self.saved.append((image_id, boxes))

    def verify_and_next(self, image_id: str, boxes: list[dict[str, Any]]) -> None:
        self.saved.append((image_id, boxes))

    def training_preflight(self) -> dict[str, Any]:
        return {"ok": True, "errors": [], "warnings": []}

    def start_training(self, model_key: str, settings: dict[str, Any]) -> str:
        self.started_training.append((model_key, settings))
        return "job-1"

    def handle_job_event(self, event: dict[str, Any]) -> None:
        self.handled_events.append(event)


class FakeProcessBridge(QObject):
    eventReceived = Signal(dict)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.job_id = "deploy-job"
        self.is_running = True
        self.messages: list[dict[str, Any]] = []
        self.starts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def write_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def cancel(self) -> None:
        self.is_running = False

    def start(self, *args: Any, **kwargs: Any) -> None:
        self.starts.append((args, kwargs))


def _accept_voc_merge_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the VOC UI tests focused on the post-dialog import path."""

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: "C:/test-voc-dataset",
    )
    monkeypatch.setattr(
        main_window_module.VocImportDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        main_window_module.VocImportDialog,
        "payload",
        lambda _self: {
            "mode": "merge",
            "destination": None,
            "project_name": "",
            "category_mapping": {"BALL": "BALL"},
        },
    )


def _voc_preflight() -> dict[str, Any]:
    return {
        "image_count": 3,
        "box_count": 2,
        "annotated_image_count": 2,
        "verified_negative_count": 0,
        "unconfirmed_image_count": 1,
        "category_names": ["BALL"],
        "new_image_count": 3,
        "upgraded_image_count": 0,
        "conflict_count": 0,
        "preserved_unconfirmed_count": 0,
    }


def _voc_merge_result() -> dict[str, Any]:
    return {
        "imported_image_count": 3,
        "upgraded_image_count": 0,
        "conflict_image_count": 0,
        "source_annotated_image_count": 2,
        "source_verified_negative_count": 0,
        "source_unconfirmed_image_count": 1,
    }


def test_voc_merge_into_empty_project_does_not_require_current_image_save(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an empty native project must not silently abort VOC merge."""

    controller = FakeController(tmp_path)
    controller.images = []
    import_calls: list[dict[str, Any]] = []
    controller.inspect_voc_import = (  # type: ignore[attr-defined]
        lambda _source, **_kwargs: _voc_preflight()
    )

    def import_voc(_source: str, **kwargs: Any) -> dict[str, Any]:
        import_calls.append(kwargs)
        return _voc_merge_result()

    controller.import_voc_dataset = import_voc  # type: ignore[attr-defined]
    _accept_voc_merge_dialog(monkeypatch)
    window = MainWindow(controller)
    assert window._current_image is None
    save_calls: list[bool] = []

    def reject_save(*, silent: bool = False) -> bool:
        save_calls.append(silent)
        return False

    monkeypatch.setattr(window, "save_current_annotations", reject_save)
    window.import_voc_dataset()

    assert save_calls == []
    assert import_calls == [
        {
            "mode": "merge",
            "destination": None,
            "project_name": "",
            "category_mapping": {"BALL": "BALL"},
        }
    ]
    window.close()
    app.processEvents()


def test_voc_merge_stops_with_feedback_when_dirty_current_image_cannot_save(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never merge on top of an unsaved image after its save has failed."""

    controller = FakeController(tmp_path)
    controller.inspect_voc_import = (  # type: ignore[attr-defined]
        lambda _source, **_kwargs: _voc_preflight()
    )
    import_calls: list[dict[str, Any]] = []
    controller.import_voc_dataset = (  # type: ignore[attr-defined]
        lambda _source, **kwargs: import_calls.append(kwargs)
    )
    _accept_voc_merge_dialog(monkeypatch)
    window = MainWindow(controller)
    window._annotations_dirty = True
    monkeypatch.setattr(
        window,
        "save_current_annotations",
        lambda *, silent=False: False,
    )
    statuses: list[str] = []
    monkeypatch.setattr(
        window,
        "_set_status",
        lambda message: statuses.append(str(message)),
    )

    window.import_voc_dataset()

    assert import_calls == []
    assert statuses, "a failed pre-import save must not abort without visible feedback"
    assert any("保存" in message for message in statuses)
    window.close()
    app.processEvents()


def test_voc_merge_reports_recheck_and_import_stages_and_restores_button(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synchronous hash/import work must visibly enter and leave busy state."""

    controller = FakeController(tmp_path)
    inspect_button_states: list[bool] = []
    inspect_button_texts: list[str] = []
    import_button_states: list[bool] = []
    import_button_texts: list[str] = []
    window: MainWindow

    def inspect_voc(_source: str, **_kwargs: Any) -> dict[str, Any]:
        inspect_button_states.append(window.import_voc_button.isEnabled())
        inspect_button_texts.append(window.import_voc_button.text())
        return _voc_preflight()

    def import_voc(_source: str, **_kwargs: Any) -> dict[str, Any]:
        import_button_states.append(window.import_voc_button.isEnabled())
        import_button_texts.append(window.import_voc_button.text())
        return _voc_merge_result()

    controller.inspect_voc_import = inspect_voc  # type: ignore[attr-defined]
    controller.import_voc_dataset = import_voc  # type: ignore[attr-defined]
    _accept_voc_merge_dialog(monkeypatch)
    window = MainWindow(controller)
    original_button_text = window.import_voc_button.text()
    window.import_voc_dataset()

    assert len(inspect_button_states) == 2
    assert inspect_button_states == [False, False]
    assert import_button_states == [False]
    assert window.import_voc_button.isEnabled()
    assert window.import_voc_action.isEnabled()
    assert window.import_voc_button.text() == original_button_text
    assert "检查" in inspect_button_texts[0]
    assert "预检" in inspect_button_texts[1]
    assert "导入" in import_button_texts[0]
    window.close()
    app.processEvents()


def test_process_bridge_error_event_is_not_decoded_twice_during_modal_reentry(
    app: QApplication,
) -> None:
    bridge = JsonlProcessBridge()
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    reentered = False

    def finish_from_nested_event_loop(message: str) -> None:
        nonlocal reentered
        errors.append(message)
        if not reentered:
            reentered = True
            bridge._on_finished(1, QProcess.ExitStatus.NormalExit)

    bridge.error.connect(finish_from_nested_event_loop)
    bridge.eventReceived.connect(events.append)
    line = json.dumps(
        {
            "protocol_version": "1.0",
            "job_id": "deploy-job",
            "seq": 4,
            "type": "error",
            "payload": {"message": "calibration hash mismatch"},
        }
    )
    bridge._job_id = "deploy-job"
    bridge._append_process_output("_stdout_buffer", line + "\n", protocol=True)
    app.processEvents()

    assert errors == ["calibration hash mismatch"]
    assert not any("seq" in message for message in errors)
    assert sum(event.get("type") == "error" for event in events) == 1


def test_process_bridge_ignores_duplicate_and_out_of_order_business_events(
    app: QApplication,
) -> None:
    bridge = JsonlProcessBridge()
    errors: list[str] = []
    logs: list[str] = []
    events: list[dict[str, Any]] = []
    bridge.error.connect(errors.append)
    bridge.log.connect(logs.append)
    bridge.eventReceived.connect(events.append)
    bridge._job_id = "train-job"

    def decode(seq: int, event_type: str) -> None:
        bridge._decode_line(
            json.dumps(
                {
                    "protocol_version": "1.0",
                    "job_id": "train-job",
                    "seq": seq,
                    "type": event_type,
                    "payload": {"message": f"{event_type}-{seq}"},
                }
            )
        )

    decode(4, "status")
    decode(4, "error")
    decode(3, "error")
    decode(5, "progress")
    app.processEvents()

    assert errors == []
    business_events = [event for event in events if not event.get("_internal")]
    assert [(event["seq"], event["type"]) for event in business_events] == [
        (4, "status"),
        (5, "progress"),
    ]
    diagnostics = [
        event
        for event in events
        if event.get("_internal")
        and event.get("payload", {}).get("protocol_diagnostic")
    ]
    assert len(diagnostics) == 2
    assert all(event["type"] == "log" for event in diagnostics)
    assert any("已忽略重复" in message for message in logs)
    assert any("已忽略乱序" in message for message in logs)


def test_main_window_model_registry_tabs_and_best_last(
    app: QApplication, tmp_path: Path
) -> None:
    controller = FakeController(tmp_path)
    window = MainWindow(controller)
    assert [window.model_combo.itemText(index) for index in range(window.model_combo.count())] == [
        option[0] for option in MODEL_OPTIONS
    ]
    assert [window.model_combo.itemData(index)["weight"] for index in range(8)] == [
        "yolov5n.pt",
        "yolov5s.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo26n.pt",
        "yolo26s.pt",
    ]
    assert window.model_combo.currentText() == "YOLO26n"
    assert [window.right_tabs.tabText(index) for index in range(4)] == [
        "标注",
        "训练",
        "AI 标注",
        "Maix 部署",
    ]
    assert window.ai_model_combo.count() == 2
    assert {window.ai_model_combo.itemData(index)["checkpoint_kind"] for index in range(2)} == {
        "best",
        "last",
    }
    window.close()
    app.processEvents()


def test_v013_multi_select_shortcuts_navigation_and_split_plots(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = FakeController(tmp_path)
    second_path = tmp_path / "images" / "第二张.png"
    image = QImage(320, 200, QImage.Format.Format_RGB32)
    image.fill(QColor("gray"))
    assert image.save(str(second_path))
    controller.images.append(
        {
            "id": "image-2",
            "relative_path": "images/第二张.png",
            "original_name": "第二张.png",
            "review_status": "unreviewed",
            "origin": "none",
            "ai_status": "none",
            "training_selected": False,
        }
    )
    window = MainWindow(controller)
    assert (
        window.image_list.selectionMode()
        == QAbstractItemView.SelectionMode.ExtendedSelection
    )
    assert window.a_shortcut.key().toString() == "A"
    assert window.delete_shortcut.key().toString() == "S"
    assert "S" in window.delete_box_action.text()
    assert window.ai_dedup_check.isChecked()
    assert window.ai_dedup_iou_spin.value() == pytest.approx(0.8)
    assert {
        key for key, _label, _color in window.training_monitor.loss_plot._visible_series
    } == {"box_loss", "cls_loss", "objectness_loss", "dfl_loss"}
    assert {
        key for key, _label, _color in window.training_monitor.map_plot._visible_series
    } == {"map50", "map50_95"}
    window.refresh_images(select_image_id="image-2")
    assert window._current_image["id"] == "image-2"
    assert window.previous_image()
    assert window._current_image["id"] == "image-1"
    window.close()
    app.processEvents()


def test_closing_clean_verified_image_does_not_rewrite_annotations(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = FakeController(tmp_path)
    window = MainWindow(controller)
    assert not window._annotations_dirty
    window.close()
    app.processEvents()
    assert controller.saved == []


def test_training_visual_artifact_is_displayed(
    app: QApplication,
    tmp_path: Path,
) -> None:
    preview = tmp_path / "train_batch0.jpg"
    image = QImage(32, 24, QImage.Format.Format_RGB32)
    image.fill(QColor("green"))
    assert image.save(str(preview))
    window = MainWindow(FakeController(tmp_path))
    window.training_monitor.handle_event(
        {
            "type": "artifact",
            "payload": {
                "kind": "training_visual",
                "path": str(preview),
            },
        }
    )
    pixmap = window.training_monitor.preview_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    window.close()
    app.processEvents()


def test_letter_shortcuts_are_protected_while_editing(
    app: QApplication, tmp_path: Path
) -> None:
    window = MainWindow(FakeController(tmp_path))
    window.show()
    window.new_class_edit.setFocus()
    app.processEvents()
    assert not window.shortcut_allowed()
    tool_before = window.canvas.tool
    window._w_shortcut()
    assert window.canvas.tool == tool_before
    window.canvas.setFocus()
    app.processEvents()
    assert window.shortcut_allowed()
    window._w_shortcut()
    assert window.canvas.tool == window.canvas.TOOL_DRAW
    window.close()
    app.processEvents()


def test_training_payload_controls_and_busy_state(
    app: QApplication, tmp_path: Path
) -> None:
    controller = FakeController(tmp_path)
    window = MainWindow(controller)
    window._training_settings = {
        "imgsz": 960,
        "epochs": 12,
        "patience": 4,
        "batch": 8,
        "device": "0",
        "workers": 2,
        "seed": 7,
        "split": {
            "mode": "train_val_test",
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "seed": 9,
        },
        "augmentation": {
            "rotation_degrees": 15,
            "rotation_probability": 0.4,
            "blur_kernel": 5,
            "blur_probability": 0.2,
            "fliplr": 0.6,
            "flipud": 0.1,
        },
        "start_from": "official",
    }
    payload = window.training_payload(retrain=True)
    assert payload["split"]["test_ratio"] == 0.1
    assert payload["augmentation"]["rotation_probability"] == 0.4
    assert payload["retrain"] is True
    assert window.start_training(retrain=True)
    assert not window.train_button.isEnabled()
    assert not window.retrain_button.isEnabled()
    window.on_job_event(
        {
            "job_id": "job-1",
            "type": "completed",
            "payload": {"kind": "train", "success": True},
        }
    )
    assert window.train_button.isEnabled()
    window.close()
    app.processEvents()


def test_training_history_is_filtered_by_selected_model(
    app: QApplication, tmp_path: Path
) -> None:
    controller = FakeController(tmp_path)
    controller.runs.append(
        {
            "id": "run-8",
            "kind": "train",
            "status": "completed",
            "model_key": "YOLOv8n",
            "created_at": "2026-07-24T13:00:00",
            "artifacts": {
                "best": str(tmp_path / "runs" / "run-8" / "weights" / "best.pt")
            },
        }
    )
    window = MainWindow(controller)
    assert window.history_combo.count() == 1
    assert window.history_combo.currentData() == "run-1"
    window.model_combo.setCurrentText("YOLOv8n")
    app.processEvents()
    assert window.history_combo.count() == 1
    assert window.history_combo.currentData() == "run-8"
    assert any(
        window.deploy_checkpoint_combo.itemData(index)["model_key"] == "YOLOv8n"
        for index in range(window.deploy_checkpoint_combo.count())
    )
    window.close()
    app.processEvents()


def test_selecting_training_history_replays_persisted_metrics_and_logs(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = FakeController(tmp_path)

    def load_history(run_id: str) -> dict[str, Any]:
        assert run_id == "run-1"
        return {
            "run_id": run_id,
            "model_key": "YOLO26n",
            "status": "completed",
            "events": [
                {
                    "type": "metrics",
                    "payload": {
                        "epoch": 2,
                        "epochs": 5,
                        "box_loss": 0.375,
                        "mAP50": 0.625,
                    },
                }
            ],
            "console_log": "persisted trainer output",
            "warnings": (),
        }

    controller.load_training_run_history = load_history  # type: ignore[attr-defined]
    window = MainWindow(controller)
    assert window.training_monitor.epoch_progress.value() == 2
    assert window.training_monitor.plot.history["box_loss"] == [0.375]
    assert window.training_monitor.plot.history["map50"] == [0.625]
    assert "persisted trainer output" in window.training_monitor.log_view.toPlainText()
    assert "completed" in window.training_monitor.state_label.text()
    assert not window._active_job_id
    window.close()
    app.processEvents()


def test_verified_image_edited_to_empty_requires_negative_confirmation(
    app: QApplication, tmp_path: Path
) -> None:
    controller = FakeController(tmp_path)
    window = MainWindow(controller)
    window.canvas.select_box("box-1")
    assert window.canvas.delete_selected()
    assert not window.verify_and_next()
    assert controller.saved == []
    window.close()
    app.processEvents()


def test_training_settings_strict_ranges_and_nonblocking_large_image_warning(
    app: QApplication,
) -> None:
    dialog = TrainingSettingsDialog(
        devices=({"value": "0", "label": "CUDA 0 — RTX 5060"},)
    )
    values = dialog.values()
    assert validate_training_settings(values) == []
    assert dialog.device_combo.findData("auto") >= 0
    assert dialog.device_combo.findData("cpu") >= 0
    cuda_index = dialog.device_combo.findData("0")
    assert cuda_index >= 0
    dialog.device_combo.setCurrentIndex(cuda_index)
    assert dialog.values()["device"] == "0"
    dialog.imgsz_spin.setValue(1312)
    assert dialog.validation_errors() == []
    assert training_setting_warnings(dialog.values())
    dialog.epochs_spin.setValue(10)
    dialog.patience_spin.setValue(11)
    assert any("patience" in error for error in dialog.validation_errors())
    assert [dialog.blur_kernel_combo.itemData(index) for index in range(3)] == [3, 5, 7]
    dialog.close()
    app.processEvents()


def test_environment_creation_requires_confirmation_and_reports_result(
    app: QApplication, tmp_path: Path
) -> None:
    python_path = tmp_path / "envs" / "yolo" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    calls: list[dict[str, Any]] = []

    def creator(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {
            "success": True,
            "python_executable": str(python_path),
            "message": "创建完成",
        }

    dialog = MLEnvironmentDialog(
        discoverer=lambda: [{"name": "yolo", "python": str(python_path), "valid": True}],
        validator=lambda path: {"valid": path == str(python_path), "message": "通过"},
        creator=creator,
    )
    assert not dialog.request_create_environment()
    assert calls == []
    assert dialog.request_create_environment(confirmed=True)
    assert calls == [{"name": "yolo", "action": "create_or_repair", "confirmed": True}]
    assert dialog.selected_path() == str(python_path)
    assert "通过" in dialog.status_label.text()
    dialog.close()
    app.processEvents()


def test_maix_dialog_int8_project_calibration_and_cam2_modes(
    app: QApplication, tmp_path: Path
) -> None:
    images = [
        {"id": f"image-{index}", "original_name": f"{index}.jpg"}
        for index in range(25)
    ]
    dialog = MaixDeployDialog(
        [
            {
                "name": "run-1 best.pt",
                "run_id": "run-1",
                "model_key": "YOLO26n",
                "checkpoint_kind": "best",
                "checkpoint": str(tmp_path / "best.pt"),
            },
            {
                "name": "run-1 last.pt",
                "run_id": "run-1",
                "model_key": "YOLOv8n",
                "checkpoint_kind": "last",
                "checkpoint": str(tmp_path / "last.pt"),
            },
        ],
        calibration_images=images,
        recommended_calibration_ids=[f"image-{index}" for index in range(2, 22)],
        project_root=tmp_path,
        project_id="project-1",
        initial_target="maixcam2",
    )
    assert dialog.deployment_config()["quantization"] == "int8"
    assert not hasattr(dialog, "quantization_combo")
    assert dialog.calibration_count_spin.value() == 20
    assert {
        item.data(Qt.ItemDataRole.UserRole)
        for item in dialog.calibration_list.selectedItems()
    } == {f"image-{index}" for index in range(2, 22)}
    assert dialog.validation_errors() == []
    assert dialog.target_combo.currentData() == "maixcam2"
    dialog.cam2_mode_combo.setCurrentIndex(dialog.cam2_mode_combo.findData("vnpu"))
    config = dialog.deployment_config()
    assert config["target"] == "maixcam2"
    assert config["cam2_npu_mode"] == "vnpu"
    assert (config["input_width"], config["input_height"]) == (640, 480)
    assert (config["camera_width"], config["camera_height"]) == (1920, 1080)
    assert config["confidence"] == 0.35
    assert config["max_det"] == 100
    assert config["dual_buff"] is True
    assert config["iou_effective"] is False
    assert config["package_outputs"] == ["maixapp", "editable_project"]
    assert Path(config["output_directory"]).name == "AI-Biaozhu-Deployments"
    dialog.maixapp_output_check.setChecked(False)
    assert dialog.deployment_config()["package_outputs"] == ["editable_project"]
    assert dialog.validation_errors() == []
    dialog.editable_project_output_check.setChecked(False)
    assert "至少选择一种部署输出" in "\n".join(dialog.validation_errors())
    assert dialog.include_example_check.isEnabled()
    assert config["oversize_policy"] == "warn_and_confirm"
    assert validate_maix_deployment(config) == []
    dialog.checkpoint_combo.setCurrentIndex(1)
    switched = dialog.deployment_config()
    assert switched["model_key"] == "YOLOv8n"
    assert switched["iou_effective"] is True
    assert dialog.iou_spin.isEnabled()
    dialog.close()
    app.processEvents()


def test_oversize_confirmation_is_top_level_for_worker_stdin(
    app: QApplication, tmp_path: Path
) -> None:
    bridge = FakeProcessBridge()
    window = MainWindow(FakeController(tmp_path), process_bridge=bridge)
    window._active_job_id = "deploy-job"
    window._pending_deploy_confirmation = {
        "code": "package_size_warning",
        "packages": [{"package_kind": "full_app", "zip_size": 31_000_000}],
    }
    window.respond_to_deployment_size_warning(True)
    assert bridge.messages[-1]["protocol_version"] == "1.0"
    assert bridge.messages[-1]["accepted"] is True
    assert "payload" not in bridge.messages[-1]
    bridge.is_running = False
    window.close()
    app.processEvents()


def test_oversize_warning_shows_exact_excess_largest_files_and_advice(
    app: QApplication,
    tmp_path: Path,
) -> None:
    bridge = FakeProcessBridge()
    window = MainWindow(FakeController(tmp_path), process_bridge=bridge)
    window._handle_deploy_event(
        "warning",
        {
            "code": "package_size_warning",
            "message": "等待确认",
            "packages": [
                {
                    "package_kind": "full-app",
                    "zip_size": 31_000_001,
                    "unpacked_size": 32_000_000,
                    "threshold": 30_000_000,
                    "largest_files": [
                        {"path": "models/model_npu.axmodel", "size": 29_500_000},
                        {"path": "models/model_vnpu.axmodel", "size": 2_000_000},
                    ],
                }
            ],
        },
    )
    log = window.deploy_log.toPlainText()
    assert "ZIP：31,000,001 字节，超出 1,000,001 字节" in log
    assert "解压后：32,000,000 字节，超出 2,000,000 字节" in log
    assert "models/model_npu.axmodel: 29,500,000 字节" in log
    assert "n 型模型" in log
    assert "降低部署分辨率" in log
    assert "只保留一种 NPU 模式" in log
    assert "#ef6262" in window.deploy_status_label.styleSheet()

    window._handle_deploy_event(
        "completed",
        {"device_validation": "required", "app_package_path": "detector.maixapp"},
    )
    assert "待真机验证" in window.deploy_status_label.text()
    bridge.is_running = False
    window.close()
    app.processEvents()


def test_docker_environment_report_and_confirmed_pull_are_displayed(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = FakeController(tmp_path)
    inspections: list[str] = []
    pulls: list[dict[str, Any]] = []

    def inspect_environment(target: str) -> dict[str, Any]:
        inspections.append(target)
        return {
            "ready": True,
            "executable": "C:/Program Files/Docker/docker.exe",
            "client_version": "28.3.2",
            "server_version": "28.3.2",
            "daemon_ready": True,
            "wsl2_ready": True,
            "mount_ready": True,
            "images": [
                {
                    "name": "sophgo/tpuc_dev:latest",
                    "available": True,
                    "image_id": "sha256:abcdef",
                    "repo_digests": ("sophgo/tpuc_dev@sha256:123456",),
                }
            ],
            "warnings": (),
            "errors": (),
        }

    def pull_image(payload: dict[str, Any]) -> dict[str, Any]:
        pulls.append(payload)
        return {
            "job_id": "docker-pull-1",
            "program": "docker",
            "arguments": ["pull", "sophgo/tpuc_dev:latest"],
        }

    controller.inspect_conversion_environment = inspect_environment  # type: ignore[attr-defined]
    controller.pull_converter_image = pull_image  # type: ignore[attr-defined]
    bridge = FakeProcessBridge()
    bridge.is_running = False
    window = MainWindow(controller, process_bridge=bridge)
    window.docker_target_combo.setCurrentIndex(
        window.docker_target_combo.findData("maixcam_pro")
    )

    assert window.inspect_docker_environment()
    assert inspections == ["maixcam_pro"]
    assert "Docker daemon：可用" in window.docker_status_label.text()
    assert "sophgo/tpuc_dev@sha256:123456" in window.docker_status_label.text()

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    assert window.pull_docker_image()
    assert pulls == [{"target": "maixcam_pro", "confirmed": True}]
    assert bridge.starts
    assert bridge.starts[-1][0][:2] == (
        "docker",
        ["pull", "sophgo/tpuc_dev:latest"],
    )
    assert not window.docker_detect_button.isEnabled()

    bridge.is_running = False
    window.on_job_event(
        {
            "job_id": "docker-pull-1",
            "type": "process_finished",
            "payload": {"success": True, "exit_code": 0},
        }
    )
    assert inspections == ["maixcam_pro", "maixcam_pro"]
    assert window.docker_detect_button.isEnabled()
    assert "镜像 sophgo/tpuc_dev:latest：已加载" in window.docker_status_label.text()
    window.close()
    app.processEvents()


def test_maixcam2_pull_redirects_to_official_archive_import(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = FakeController(tmp_path)
    pulls: list[dict[str, Any]] = []
    messages: list[str] = []

    def pull_image(payload: dict[str, Any]) -> dict[str, Any]:
        pulls.append(payload)
        raise AssertionError("MaixCAM2 must not invoke docker pull")

    controller.pull_converter_image = pull_image  # type: ignore[attr-defined]
    window = MainWindow(controller, process_bridge=FakeProcessBridge())
    window.docker_target_combo.setCurrentIndex(
        window.docker_target_combo.findData("maixcam2")
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, message, *_args, **_kwargs: messages.append(message),
    )

    assert not window.pull_docker_image()
    assert pulls == []
    assert messages
    assert "tar" in messages[0]
    assert "导入镜像" in messages[0]
    assert "huggingface.co/AXERA-TECH/Pulsar2/tree/main/6.0" in messages[0]
    window.close()
    app.processEvents()


def test_docker_image_import_requires_explicit_confirmation(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = FakeController(tmp_path)
    imports: list[dict[str, Any]] = []
    archive = tmp_path / "maix-converter.tar"
    archive.write_bytes(b"docker archive")

    def import_image(payload: dict[str, Any]) -> dict[str, Any]:
        imports.append(payload)
        return {"job_id": "docker-load-1", "program": "docker", "arguments": ["load"]}

    controller.import_converter_image = import_image  # type: ignore[attr-defined]
    bridge = FakeProcessBridge()
    bridge.is_running = False
    window = MainWindow(controller, process_bridge=bridge)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(archive), "Docker 镜像归档 (*.tar)"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )
    assert not window.import_docker_image()
    assert imports == []

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    assert window.import_docker_image()
    assert imports == [
        {"path": str(archive), "target": "maixcam2", "confirmed": True}
    ]
    assert bridge.starts
    bridge.is_running = False
    window.close()
    app.processEvents()


def test_docker_unchecked_image_progress_and_cancellation_ui(
    app: QApplication,
    tmp_path: Path,
) -> None:
    window = MainWindow(FakeController(tmp_path), process_bridge=FakeProcessBridge())
    ready = window._show_docker_environment_report(
        {
            "ready": False,
            "executable": "docker.exe",
            "daemon_ready": False,
            "wsl2_ready": True,
            "mount_ready": None,
            "images": [
                {
                    "name": "pulsar2:6.0",
                    "available": None,
                    "error": "Docker daemon 未就绪，镜像尚未检查",
                }
            ],
            "warnings": (),
            "errors": ("Docker daemon 不可用",),
        }
    )
    assert not ready
    assert "镜像 pulsar2:6.0：未检查" in window.docker_status_label.text()
    assert "镜像 pulsar2:6.0：缺失" not in window.docker_status_label.text()

    window._handle_docker_environment_event(
        "progress",
        {
            "percent": 50.0,
            "bytes_read": 5 * 1024 * 1024,
            "total_bytes": 10 * 1024 * 1024,
            "elapsed_seconds": 12.5,
            "bytes_per_second": 512 * 1024,
            "heartbeat": True,
        },
    )
    assert window.docker_import_progress.value() == 50
    assert "5.0 MiB / 10.0 MiB" in window.docker_import_detail_label.text()
    assert "Docker 仍在运行" in window.docker_import_detail_label.text()
    window.close()
    app.processEvents()


def test_start_docker_desktop_begins_recovery_polling(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = FakeController(tmp_path)
    calls: list[dict[str, Any]] = []

    def start(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {
            "state": "starting",
            "can_start": True,
            "should_poll": True,
            "elapsed_seconds": 0,
            "timeout_seconds": 120,
            "message": "Docker Desktop 正在启动，等待 daemon 就绪。",
        }

    controller.start_docker_desktop = start  # type: ignore[attr-defined]
    window = MainWindow(controller, process_bridge=FakeProcessBridge())
    assert window.start_docker_desktop()
    assert calls == [{"target": "maixcam2", "confirmed": True}]
    assert window._docker_recovery_timer.isActive()
    assert "0/120 秒" in window.docker_status_label.text()
    window._docker_recovery_timer.stop()
    window.close()
    app.processEvents()


def test_deploy_completion_exposes_both_outputs_without_claiming_installation(
    app: QApplication,
    tmp_path: Path,
) -> None:
    publish = tmp_path / "deployments" / "run-2"
    publish.mkdir(parents=True)
    app_path = publish / "detector.maixapp"
    app_path.write_bytes(b"app")
    editable = publish / "detector-editable"
    editable.mkdir()
    (editable / "main.py").write_text("print('ok')", encoding="utf-8")

    window = MainWindow(FakeController(tmp_path), process_bridge=FakeProcessBridge())
    window._active_job_id = "deploy-run-2"
    window._handle_deploy_event(
        "completed",
        {
            "app_package_path": str(app_path),
            "editable_project_path": str(editable),
            "deployment_artifacts": [
                {"kind": "maixapp", "path": str(app_path)},
                {
                    "kind": "editable-project",
                    "path": str(editable),
                    "is_directory": True,
                },
            ],
            "device_validation": "required",
        },
    )
    assert "尚未安装到设备" in window.deploy_status_label.text()
    assert str(app_path) in window.deploy_status_label.text()
    assert str(editable) in window.deploy_status_label.text()
    assert window.open_deploy_output_button.isEnabled()
    assert window.open_maixapp_button.isEnabled()
    assert window.cleanup_backups_button.isEnabled()
    assert window._last_completed_deploy_run_id == "deploy-run-2"
    window.close()
    app.processEvents()


def test_backup_cleanup_requires_verified_deployment_and_uses_recovery_trash(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = FakeController(tmp_path)
    markers: list[str] = []
    cleanups: list[dict[str, Any]] = []

    controller.preview_backup_cleanup = (  # type: ignore[attr-defined]
        lambda *, keep_latest, include_recovery_trash: {
            "backup_count": 2,
            "total_bytes": 4096,
            "keep_latest": keep_latest,
            "include_recovery_trash": include_recovery_trash,
        }
    )

    def cleanup(
        *,
        keep_latest: int,
        deployment_verified: bool,
        permanently_delete: bool,
    ) -> dict[str, Any]:
        cleanups.append(
            {
                "keep_latest": keep_latest,
                "deployment_verified": deployment_verified,
                "permanently_delete": permanently_delete,
            }
        )
        return {"deleted_count": 4, "permanently_deleted": True}

    controller.cleanup_old_backups = cleanup  # type: ignore[attr-defined]
    controller.mark_deployment_verified = markers.append  # type: ignore[attr-defined]
    window = MainWindow(controller, process_bridge=FakeProcessBridge())
    window._last_completed_deploy_run_id = "deploy-run-3"
    assert window.cleanup_old_backups_after_deploy()
    assert markers == ["deploy-run-3"]
    assert cleanups == [
        {
            "keep_latest": 0,
            "deployment_verified": True,
            "permanently_delete": True,
        }
    ]
    window.close()
    app.processEvents()


def test_import_report_accepts_controller_failure_fields(
    app: QApplication,
) -> None:
    dialog = ImportReportDialog(
        {
            "imported": 2,
            "duplicates": 1,
            "failed": 1,
            "failures": [{"path": "坏图.jpg", "reason": "无法解码"}],
        }
    )
    assert "坏图.jpg：无法解码" in dialog.details.toPlainText()
    assert any("跳过 1 张" in label.text() for label in dialog.findChildren(QLabel))
    dialog.close()
    app.processEvents()


def test_worker_event_is_persisted_before_ui_update(
    app: QApplication, tmp_path: Path
) -> None:
    controller = FakeController(tmp_path)
    window = MainWindow(controller)
    event = {
        "protocol_version": 1,
        "job_id": "job-1",
        "seq": 7,
        "type": "metrics",
        "payload": {
            "kind": "train",
            "epoch": 2,
            "epochs": 10,
            "box_loss": 0.5,
            "map50": 0.75,
            "eta_seconds": 65,
            "gpu_utilization": 72,
            "gpu_memory_gb": 3.5,
            "metrics": {"lr/pg0": 0.001},
        },
    }
    window._on_process_event(event)
    assert controller.handled_events == [event]
    assert window.training_monitor.epoch_progress.value() == 2
    assert window.training_monitor.plot.history["box_loss"] == [0.5]
    assert "00:01:05" in window.training_monitor.timing_label.text()
    assert "0.0010" in window.training_monitor.timing_label.text()
    assert "72.0000%" in window.training_monitor.gpu_label.text()
    assert "3.5000 GB" in window.training_monitor.gpu_label.text()
    window.close()
    app.processEvents()
