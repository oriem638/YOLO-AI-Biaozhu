from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ai_biaozhu.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class RangeSelectionController:
    def __init__(self, root: Path, image_count: int = 407) -> None:
        self.current_project = {"root": root, "project_id": "range-selection"}
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        shared_path = images_dir / "shared.png"
        image = QImage(48, 32, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        assert image.save(str(shared_path))
        relative_path = str(shared_path.relative_to(root))
        self.images: list[dict[str, Any]] = [
            {
                "id": f"image-{index}",
                "relative_path": relative_path,
                "original_name": f"image-{index}.png",
                "review_status": "verified" if index % 2 else "unreviewed",
                "origin": "manual",
                "ai_status": "none",
                "training_selected": index % 3 != 0,
            }
            for index in range(1, image_count + 1)
        ]

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


def _selected_indices(window: MainWindow) -> set[int]:
    return {
        int(item.data(Qt.ItemDataRole.UserRole + 2))
        for item in window.image_list.selectedItems()
    }


def _click_index(
    window: MainWindow,
    index: int,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    item = window.image_list.item(index - 1)
    assert item is not None and not item.isHidden()
    window.image_list.scrollToItem(item)
    QTest.qWait(2)
    item_rect = window.image_list.visualItemRect(item)
    position = QPoint(8, item_rect.center().y())
    assert window.image_list.viewport().rect().contains(position)
    QTest.mouseClick(
        window.image_list.viewport(),
        Qt.MouseButton.LeftButton,
        modifiers,
        position,
    )
    QTest.qWait(5)


def _click_button(button: Any) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QTest.qWait(5)


def test_select_to_here_real_mouse_is_additive_global_and_filter_stable(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = RangeSelectionController(tmp_path)
    training_state = [record["training_selected"] for record in controller.images]
    window = MainWindow(controller)
    window.resize(1280, 760)
    window.show()
    QTest.qWait(10)

    # Forward selection includes both endpoints and exits the waiting state.
    _click_index(window, 403)
    _click_button(window.select_to_here_button)
    _click_index(window, 407)
    assert _selected_indices(window) == set(range(403, 408))
    assert not window.select_to_here_button.isChecked()
    assert window._selection_anchor_index is None
    assert window.select_to_here_button.text() == "选到这里"

    # Reverse selection produces the same inclusive interval.
    _click_button(window.clear_image_selection_button)
    _click_index(window, 407)
    _click_button(window.select_to_here_button)
    _click_index(window, 403)
    assert _selected_indices(window) == set(range(403, 408))

    # A selection made with Ctrl before arming the range is restored after the
    # endpoint's ordinary click and the range is added to it.
    _click_button(window.clear_image_selection_button)
    _click_index(window, 100, Qt.KeyboardModifier.ControlModifier)
    _click_index(window, 403, Qt.KeyboardModifier.ControlModifier)
    _click_button(window.select_to_here_button)
    _click_index(window, 407)
    assert _selected_indices(window) == {100, *range(403, 408)}

    # Equal endpoints select one image, and repeated operations do not retain
    # a stale anchor from the previous range.
    _click_button(window.clear_image_selection_button)
    _click_index(window, 403)
    _click_button(window.select_to_here_button)
    _click_index(window, 403)
    assert _selected_indices(window) == {403}
    assert window._selection_anchor_index is None

    # Clicking the checkable button again cancels waiting; clearing selection
    # also cancels and resets all anchor state.
    _click_button(window.select_to_here_button)
    _click_button(window.select_to_here_button)
    assert not window.select_to_here_button.isChecked()
    assert window._selection_anchor_index is None
    _click_button(window.select_to_here_button)
    _click_button(window.clear_image_selection_button)
    assert not window.select_to_here_button.isChecked()
    assert window._selection_anchor_index is None
    assert _selected_indices(window) == set()

    # The interval uses project-global numbers even when rows inside it are
    # hidden by search. Hidden interval members become visibly selected again
    # when the filter is cleared.
    _click_index(window, 403)
    _click_button(window.select_to_here_button)
    window.image_search_edit.setText("image-407.png")
    QTest.qWait(5)
    _click_index(window, 407)
    assert _selected_indices(window) == set(range(403, 408))
    window.image_search_edit.clear()
    QTest.qWait(5)
    assert _selected_indices(window) == set(range(403, 408))

    # Temporary blue selection never changes training membership or records.
    assert [record["training_selected"] for record in controller.images] == training_state

    window.close()
    app.processEvents()
