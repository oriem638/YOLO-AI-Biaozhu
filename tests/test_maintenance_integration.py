from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.core import BoxInput
from ai_biaozhu.settings import SettingsStore


def _controller(tmp_path: Path) -> ApplicationController:
    paths = AppPaths(
        data=tmp_path / "app-data",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        models=tmp_path / "models",
        yolo_config=tmp_path / "yolo-config",
    )
    return ApplicationController(
        paths,
        settings=SettingsStore(tmp_path / "settings.json"),
        source_root=Path(__file__).resolve().parents[1],
    )


def _write_voc_annotation(path: Path, filename: str) -> None:
    path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<annotation><filename>ball.jpg</filename>"
        "<size><width>40</width><height>30</height><depth>3</depth></size>"
        "<object><name>小刚球</name><bndbox>"
        "<xmin>2</xmin><ymin>3</ymin><xmax>15</xmax><ymax>20</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )


def _make_voc_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "maixhub-voc"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    Image.new("RGB", (40, 30), (20, 30, 40)).save(root / "images" / "ball.jpg")
    _write_voc_annotation(root / "annotations" / "ball.xml", "ball.jpg")
    return root


def test_controller_voc_import_new_and_safe_duplicate_merge(tmp_path: Path) -> None:
    source = _make_voc_dataset(tmp_path)
    controller = _controller(tmp_path)

    preview = controller.inspect_voc_import(source)
    assert preview["image_count"] == 1
    assert preview["box_count"] == 1
    assert preview["category_names"] == ["小刚球"]

    destination = tmp_path / "native-project"
    created = controller.import_voc_dataset(
        source,
        mode="new",
        destination=destination,
        project_name="钢球",
        category_mapping={"小刚球": "小刚球"},
    )
    assert created["mode"] == "new"
    assert created["image_count"] == 1
    assert created["box_count"] == 1
    assert controller.current_project is not None

    merged = controller.import_voc_dataset(
        source,
        mode="merge",
        category_mapping={"小刚球": "小刚球"},
    )
    assert merged["mode"] == "merge"
    assert merged["imported_image_count"] == 0
    assert merged["upgraded_image_count"] == 0
    assert merged["conflict_image_count"] == 1
    controller.close_project()


def test_controller_clear_all_annotations_returns_backup_details(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "clear")
    category = project.repository.list_categories()[0]
    image = project.repository.add_image_record(
        image_id="image-1",
        relative_path="images/image-1.jpg",
        original_name="image-1.jpg",
        source_path=None,
        sha256="1" * 64,
        width=40,
        height=30,
    )
    project.save_boxes(image.id, [BoxInput(category.id, 1, 2, 10, 12)])

    preview = controller.preview_clear_all_annotations([image.id])
    assert preview["image_count"] == 1
    assert preview["box_count"] == 1
    report = controller.clear_all_annotations([image.id])
    assert report["box_count"] == 1
    assert Path(report["backup"]["path"]).is_file()
    assert project.list_boxes(image.id) == ()
    controller.close_project()


try:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import QApplication
except ImportError:
    pytest.skip("PySide6 is required for the UI maintenance tests", allow_module_level=True)

from ai_biaozhu.ui.canvas import AnnotationDisplayMode  # noqa: E402
from ai_biaozhu.ui.main_window import MainWindow, NullController  # noqa: E402
from ai_biaozhu.ui.maintenance_dialogs import (  # noqa: E402
    BulkAnnotationClearDialog,
    VocImportDialog,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


def test_maintenance_dialogs_select_records_and_preserve_mapping(app: QApplication) -> None:
    records = [
        {"id": "manual", "original_name": "manual.jpg", "review_status": "verified"},
        {"id": "draft", "original_name": "draft.jpg", "review_status": "draft"},
    ]
    clear_dialog = BulkAnnotationClearDialog(records, current_image_id="manual")
    assert clear_dialog.selected_image_ids() == ("manual",)
    clear_dialog.status_combo.setCurrentIndex(clear_dialog.status_combo.findData("draft"))
    clear_dialog._set_visible_checked(True)
    assert clear_dialog.selected_image_ids() == ("manual", "draft")
    clear_dialog.close()

    voc_dialog = VocImportDialog(
        "C:/dataset",
        {
            "image_count": 2,
            "box_count": 3,
            "negative_count": 0,
            "category_names": ["小刚球"],
        },
        existing_category_names=["钢球"],
        current_project_root="C:/project",
    )
    assert voc_dialog.mode == "new"
    assert voc_dialog.payload()["destination"] != "C:/dataset"
    assert voc_dialog.payload()["project_name"] == "dataset_标注项目"
    voc_dialog.merge_project_radio.setChecked(True)
    assert voc_dialog.mode == "merge"
    combo = voc_dialog._category_combos["小刚球"]
    combo.setCurrentIndex(combo.findData("钢球"))
    assert voc_dialog.payload()["category_mapping"] == {"小刚球": "钢球"}
    voc_dialog.close()
    app.processEvents()


def test_main_window_maintenance_actions_drive_canvas_without_data_mutation(
    app: QApplication,
) -> None:
    window = MainWindow(NullController())
    pixmap = QPixmap(100, 60)
    pixmap.fill(QColor("white"))
    assert window.canvas.set_image(pixmap)
    window.canvas.set_categories([{"id": "ball", "name": "小刚球", "color": "#22C55E"}])
    window.canvas.set_annotations(
        [
            {
                "id": "baseline",
                "class_id": "ball",
                "xmin": 5,
                "ymin": 6,
                "xmax": 35,
                "ymax": 26,
                "origin": "ai",
                "confidence": 0.85,
            }
        ]
    )
    before = window.canvas.annotations()

    window.annotation_display_boxes_action.trigger()
    assert window.canvas.annotation_display_mode is AnnotationDisplayMode.BOX_ONLY
    assert window.canvas.annotations() == before
    window.annotation_display_hidden_action.trigger()
    assert window.canvas.annotation_display_mode is AnnotationDisplayMode.HIDDEN
    assert window.canvas.annotations() == before
    window.annotation_display_full_action.trigger()
    assert window.canvas.annotation_display_mode is AnnotationDisplayMode.FULL

    window.canvas.add_box(QRectF(45, 10, 20, 20), "ball")
    assert len(window.canvas.annotations()) == 2
    window.undo_all_current_image()
    assert window.canvas.annotations() == before
    window.close()
    app.processEvents()
