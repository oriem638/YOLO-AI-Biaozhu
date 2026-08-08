from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from ai_biaozhu.ui.canvas import HANDLE_NAMES, AnnotationCanvas


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


def test_canvas_create_edit_undo_and_delete(app: QApplication) -> None:
    canvas = AnnotationCanvas()
    pixmap = QPixmap(320, 200)
    pixmap.fill(QColor("black"))
    assert canvas.set_image(pixmap)
    canvas.set_categories(
        [
            {"id": "cat", "name": "猫", "color": "#ff0000"},
            {"id": "dog", "name": "狗", "color": "#00ff00"},
        ]
    )
    canvas.set_current_class("cat")

    annotation_id = canvas.add_box(QRectF(10, 20, 100, 80))
    assert annotation_id
    assert canvas.annotations()[0] == {
        "id": annotation_id,
        "class_id": "cat",
        "xmin": 10.0,
        "ymin": 20.0,
        "xmax": 110.0,
        "ymax": 100.0,
        "origin": "manual",
        "confidence": None,
        "model_run_id": None,
        "prediction_id": None,
    }

    canvas.select_box(annotation_id)
    assert canvas.set_selected_class("dog")
    assert canvas.annotations()[0]["class_id"] == "dog"
    canvas.undo()
    assert canvas.annotations()[0]["class_id"] == "cat"
    canvas.redo()
    assert canvas.annotations()[0]["class_id"] == "dog"

    assert canvas.delete_selected()
    assert canvas.annotations() == []
    canvas.undo()
    assert len(canvas.annotations()) == 1
    canvas.close()
    app.processEvents()


def test_canvas_clamps_boxes_zoom_and_exposes_eight_handles(app: QApplication) -> None:
    canvas = AnnotationCanvas()
    pixmap = QPixmap(100, 50)
    pixmap.fill(QColor("white"))
    canvas.set_image(pixmap)
    annotation_id = canvas.add_box(QRectF(-20, -10, 200, 100), "0")
    assert annotation_id
    box = canvas.annotations()[0]
    assert box["xmin"] == 0
    assert box["ymin"] == 0
    assert box["xmax"] == 100
    assert box["ymax"] == 50
    assert set(HANDLE_NAMES) == {"nw", "n", "ne", "e", "se", "s", "sw", "w"}

    canvas.reset_zoom()
    canvas.zoom_by(1000)
    assert canvas.zoom_factor == 30
    canvas.zoom_by(0.00001)
    assert canvas.zoom_factor == 0.05
    canvas.close()
    app.processEvents()


def test_canvas_tools_history_and_temporary_pan_are_explicit(app: QApplication) -> None:
    canvas = AnnotationCanvas()
    pixmap = QPixmap(100, 50)
    pixmap.fill(QColor("white"))
    assert canvas.set_image(pixmap)

    changes: list[str] = []
    undo_states: list[bool] = []
    redo_states: list[bool] = []
    canvas.toolChanged.connect(changes.append)
    canvas.undoAvailableChanged.connect(undo_states.append)
    canvas.redoAvailableChanged.connect(redo_states.append)

    canvas.set_tool(canvas.TOOL_DRAW)
    canvas.set_tool(canvas.TOOL_DRAW)
    assert canvas.tool == canvas.TOOL_DRAW
    assert changes == [canvas.TOOL_DRAW]

    canvas.begin_temporary_pan()
    assert canvas.tool == canvas.TOOL_PAN
    canvas.begin_temporary_pan()
    canvas.end_temporary_pan()
    assert canvas.tool == canvas.TOOL_DRAW
    assert changes == [canvas.TOOL_DRAW, canvas.TOOL_PAN, canvas.TOOL_DRAW]

    assert not canvas.can_undo
    assert not canvas.can_redo
    assert canvas.add_box(QRectF(1, 1, 20, 20), "0")
    assert canvas.can_undo
    canvas.undo()
    assert canvas.can_redo
    canvas.redo()
    assert canvas.can_undo
    assert undo_states
    assert redo_states

    # Loading a different image starts a new, isolated annotation-history session.
    assert canvas.set_image(pixmap)
    assert not canvas.can_undo
    assert not canvas.can_redo
    canvas.close()
    app.processEvents()
