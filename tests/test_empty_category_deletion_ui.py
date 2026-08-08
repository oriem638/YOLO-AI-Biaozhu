from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from ai_biaozhu.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


class _EmptyCategoryController:
    def __init__(self, root: Path) -> None:
        self.current_project = {"root": root, "project_id": "empty-category-ui"}
        self.categories = [
            {
                "id": "ball",
                "name": "BALL",
                "display_name": None,
                "color": "#22c55e",
            },
            {
                "id": "steel",
                "name": "钢球",
                "display_name": None,
                "color": "#45c486",
            },
        ]
        image_path = root / "images" / "sample.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(200, 120, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        assert image.save(str(image_path))
        self.images = [
            {
                "id": "image-1",
                "relative_path": "images/sample.png",
                "original_name": "sample.png",
                "review_status": "verified",
                "origin": "manual",
                "ai_status": "none",
                "training_selected": True,
            }
        ]
        self.delete_calls: list[str] = []

    def list_images(self) -> list[dict[str, Any]]:
        return self.images

    def list_classes(self) -> list[dict[str, Any]]:
        return self.categories

    @staticmethod
    def list_runs() -> list[object]:
        return []

    @staticmethod
    def get_boxes(_image_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "box-1",
                "class_id": "ball",
                "xmin": 20,
                "ymin": 20,
                "xmax": 60,
                "ymax": 60,
                "origin": "manual",
                "confidence": None,
            }
        ]

    @staticmethod
    def save_boxes(_image_id: str, _boxes: list[dict[str, Any]]) -> None:
        return None

    @staticmethod
    def verify_and_next(
        _image_id: str, _boxes: list[dict[str, Any]]
    ) -> None:
        return None

    def delete_empty_category(self, category_id: str) -> dict[str, Any]:
        self.delete_calls.append(category_id)
        category = next(item for item in self.categories if item["id"] == category_id)
        if category_id != "steel":
            raise RuntimeError("类别仍有 1 个标注框，不能删除")
        self.categories.remove(category)
        return {
            "category": category,
            "backup": {"path": "C:/project/backups/before-empty-category-delete.db"},
        }


def test_delete_empty_category_button_removes_only_selected_empty_category(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = _EmptyCategoryController(tmp_path)
    window = MainWindow(controller)
    try:
        assert window.delete_empty_category_button.text() == "删除空类别"
        window.class_list.setCurrentRow(1)

        assert window.delete_selected_empty_category(confirmed=True)

        assert controller.delete_calls == ["steel"]
        assert [item["name"] for item in controller.categories] == ["BALL"]
        assert window.class_list.count() == 1
        assert window.class_list.item(0).text() == "BALL"
        assert "BALL" in window.box_list.item(0).text()
        assert "before-empty-category-delete.db" in window.statusBar().currentMessage()
    finally:
        window.close()
        app.processEvents()


def test_delete_empty_category_cancel_and_nonempty_failure_leave_categories_intact(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = _EmptyCategoryController(tmp_path)
    window = MainWindow(controller)
    try:
        window.class_list.setCurrentRow(1)
        assert not window.delete_selected_empty_category(confirmed=False)
        assert controller.delete_calls == []
        assert [item["name"] for item in controller.categories] == ["BALL", "钢球"]

        window.class_list.setCurrentRow(0)
        assert not window.delete_selected_empty_category(confirmed=True)
        assert controller.delete_calls == ["ball"]
        assert [item["name"] for item in controller.categories] == ["BALL", "钢球"]
    finally:
        window.close()
        app.processEvents()
