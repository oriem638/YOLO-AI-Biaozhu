"""Regression coverage for keyboard navigation and responsive annotation UI.

These tests deliberately send real Qt events instead of invoking the shortcut
slots directly.  The reported failures only occurred when focus moved away
from the canvas, so exercising the event dispatcher is important here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QSizePolicy

from ai_biaozhu.ui.canvas import AnnotationCanvas
from ai_biaozhu.ui.fonts import bundled_font_path, configure_application_font
from ai_biaozhu.ui.main_window import MainWindow
from ai_biaozhu.ui.theme import APP_STYLESHEET


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class NavigationController:
    """Small in-memory controller that follows the legacy ``None`` next API."""

    def __init__(self, root: Path, *, image_count: int = 3, long_name: str | None = None) -> None:
        self.current_project = {"root": root, "project_id": "ui-regression"}
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        self.images: list[dict[str, Any]] = []
        self.saved: list[tuple[str, list[dict[str, Any]]]] = []
        self.categories = [{"id": "vehicle", "name": "vehicle", "color": "#42a5f5"}]
        for index in range(image_count):
            path = images_dir / f"source-{index}.png"
            image = QImage(96, 64, QImage.Format.Format_RGB32)
            image.fill(QColor("black"))
            assert image.save(str(path))
            self.images.append(
                {
                    "id": f"image-{index}",
                    "relative_path": str(path.relative_to(root)),
                    "original_name": long_name if index == 0 and long_name else path.name,
                    "review_status": "unreviewed",
                    "origin": "none",
                    "ai_status": "none",
                }
            )

    def list_images(self) -> list[dict[str, Any]]:
        return self.images

    def list_classes(self) -> list[dict[str, Any]]:
        return self.categories

    def list_runs(self) -> list[dict[str, Any]]:
        return []

    def get_boxes(self, image_id: str) -> list[dict[str, Any]]:
        del image_id
        # A non-empty image avoids the negative-sample confirmation dialog
        # while testing D navigation.
        return [
            {
                "id": "base-box",
                "class_id": "vehicle",
                "xmin": 5,
                "ymin": 6,
                "xmax": 30,
                "ymax": 35,
                "origin": "manual",
            }
        ]

    def save_boxes(self, image_id: str, boxes: list[dict[str, Any]]) -> None:
        self.saved.append((image_id, boxes))

    def verify_and_next(self, image_id: str, boxes: list[dict[str, Any]]) -> None:
        """Exercise the compatibility path, whose old implementation double-loaded."""
        self.saved.append((image_id, boxes))
        for record in self.images:
            if record["id"] == image_id:
                record["review_status"] = "verified"
                break
        return None

    def training_preflight(self) -> dict[str, Any]:
        return {"ok": True, "errors": [], "warnings": []}


@pytest.fixture()
def window(app: QApplication, tmp_path: Path) -> MainWindow:
    result = MainWindow(NavigationController(tmp_path))
    result.show()
    QTest.qWait(10)
    yield result
    result.close()
    app.processEvents()


def _send_key(widget: Any, key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.NoModifier) -> None:
    widget.setFocus()
    QTest.qWait(1)
    QTest.keyClick(widget, key, modifiers)
    QTest.qWait(1)


def test_theme_does_not_make_every_label_an_opaque_window_coloured_widget() -> None:
    """A blanket QWidget background made compressed labels cover each other."""
    assert "QWidget {\n    background:" not in APP_STYLESHEET
    assert "QLabel" in APP_STYLESHEET and "background: transparent" in APP_STYLESHEET


def test_bundled_cjk_font_is_pinned_loadable_and_covers_required_glyphs(
    app: QApplication,
) -> None:
    path = bundled_font_path()
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a"
    )
    diagnostics = configure_application_font(app)
    assert diagnostics.bundled_font_loaded
    assert diagnostics.supports_required_glyphs
    assert app.font().pointSize() == 10


def test_long_image_status_is_elided_without_growing_the_window_minimum(
    app: QApplication, tmp_path: Path
) -> None:
    long_name = "roboflow-" + ("a" * 245) + ".jpg"
    controller = NavigationController(tmp_path, long_name=long_name)
    window = MainWindow(controller)
    window.show()
    app.processEvents()
    baseline = window.minimumSizeHint().width()

    assert window.open_image(controller.images[0])
    app.processEvents()

    assert window.minimumSizeHint().width() <= baseline + 40
    assert window.image_status_label.minimumWidth() == 0
    assert long_name in window.image_status_label.toolTip()
    assert (
        window.image_status_label.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Ignored
    )
    window.close()


def test_three_panes_stay_available_and_right_tabs_scroll_on_a_small_window(
    window: MainWindow, app: QApplication
) -> None:
    assert window.main_splitter.childrenCollapsible() is False
    assert all(not window.main_splitter.isCollapsible(index) for index in range(3))

    window.resize(1280, 720)
    app.processEvents()
    left, _, right = (window.main_splitter.widget(index) for index in range(3))
    assert left.isVisible() and right.isVisible()
    assert left.width() >= 220
    assert right.width() >= 340
    for index in range(window.right_tabs.count()):
        page = window.right_tabs.widget(index)
        assert isinstance(page, QScrollArea)
        assert page.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_w_v_and_history_shortcuts_are_real_window_shortcuts(
    window: MainWindow, app: QApplication
) -> None:
    canvas = window.canvas
    assert canvas.tool == AnnotationCanvas.TOOL_SELECT
    _send_key(canvas.viewport(), Qt.Key_W)
    assert canvas.tool == AnnotationCanvas.TOOL_DRAW
    _send_key(canvas.viewport(), Qt.Key_W)
    assert canvas.tool == AnnotationCanvas.TOOL_DRAW
    _send_key(canvas.viewport(), Qt.Key_V)
    assert canvas.tool == AnnotationCanvas.TOOL_SELECT

    for focus_target in (canvas.viewport(), window.image_list, window.undo_button):
        canvas.add_box((40, 10, 20, 20), "vehicle")
        assert len(canvas.annotations()) == 2
        assert window.undo_button.isEnabled()
        _send_key(focus_target, Qt.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert len(canvas.annotations()) == 1
        assert window.redo_button.isEnabled()
        _send_key(focus_target, Qt.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert len(canvas.annotations()) == 2
        canvas.undo()
        app.processEvents()


def test_text_inputs_keep_letters_and_undo_redo_for_themselves(window: MainWindow) -> None:
    canvas = window.canvas
    assert canvas.tool == AnnotationCanvas.TOOL_SELECT
    before = list(canvas.annotations())
    edit = window.new_class_edit
    edit.setText("abc")
    _send_key(edit, Qt.Key_W)
    assert canvas.tool == AnnotationCanvas.TOOL_SELECT
    assert edit.text().endswith("w")
    _send_key(edit, Qt.Key_V)
    assert canvas.tool == AnnotationCanvas.TOOL_SELECT
    assert edit.text().endswith("wv")

    _send_key(edit, Qt.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert canvas.annotations() == before
    assert edit.text() == "abc"


def test_d_and_list_navigation_load_each_target_once_via_the_unified_path(
    window: MainWindow, app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original = window.canvas.load_image

    def counted_load(path: object, *args: Any, **kwargs: Any) -> bool:
        calls.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(window.canvas, "load_image", counted_load)

    # The controller returns None, so this specifically verifies the formerly
    # double-loading fallback used by D.  The next image must be loaded once.
    _send_key(window.canvas.viewport(), Qt.Key_D)
    app.processEvents()
    assert len(calls) == 1
    assert window._current_image["id"] == "image-1"

    calls.clear()
    window.image_list.setCurrentRow(2)
    app.processEvents()
    assert len(calls) == 1
    assert window._current_image["id"] == "image-2"
