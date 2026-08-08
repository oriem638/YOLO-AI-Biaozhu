from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ai_biaozhu.ui.canvas import AnnotationCanvas, AnnotationDisplayMode


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


def _canvas() -> AnnotationCanvas:
    canvas = AnnotationCanvas()
    canvas.resize(340, 260)
    canvas.show()
    pixmap = QPixmap(240, 180)
    pixmap.fill(QColor("black"))
    assert canvas.set_image(pixmap, fit=False)
    canvas.set_categories(
        [
            {"id": "red", "name": "红框", "color": "#ff0000"},
            {"id": "green", "name": "绿框", "color": "#00ff00"},
        ]
    )
    QApplication.processEvents()
    return canvas


def _add_box(
    canvas: AnnotationCanvas,
    annotation_id: str,
    rect: QRectF,
    class_id: str = "red",
) -> None:
    assert canvas.add_box(
        rect,
        class_id,
        annotation_id=annotation_id,
        record_history=False,
    ) == annotation_id


def _annotation(canvas: AnnotationCanvas, annotation_id: str) -> dict[str, Any]:
    return next(box for box in canvas.annotations() if box["id"] == annotation_id)


def _view_point(canvas: AnnotationCanvas, scene_point: QPointF):
    return canvas.mapFromScene(scene_point)


def _drag(
    canvas: AnnotationCanvas,
    start: QPointF,
    end: QPointF,
    app: QApplication,
    *,
    changes_during_move: list[list[dict[str, Any]]] | None = None,
) -> None:
    start_view = _view_point(canvas, start)
    end_view = _view_point(canvas, end)
    QTest.mousePress(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        start_view,
    )
    QTest.mouseMove(canvas.viewport(), end_view, delay=5)
    app.processEvents()
    if changes_during_move is not None:
        assert changes_during_move == []
    QTest.mouseRelease(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        end_view,
    )
    app.processEvents()


def _selected_id(canvas: AnnotationCanvas) -> str | None:
    selected = canvas.selected_box()
    return None if selected is None else str(selected["id"])


def _viewport_pixel(canvas: AnnotationCanvas, point: QPointF) -> QColor:
    grabbed = canvas.viewport().grab()
    image: QImage = grabbed.toImage()
    view_point = canvas.mapFromScene(point)
    ratio = grabbed.devicePixelRatio()
    return image.pixelColor(
        round(view_point.x() * ratio),
        round(view_point.y() * ratio),
    )


def _dispose(canvas: AnnotationCanvas, app: QApplication) -> None:
    canvas.close()
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_overlapped_selected_sw_handle_locks_resize_to_one_box(
    app: QApplication,
) -> None:
    canvas = _canvas()
    _add_box(canvas, "target", QRectF(50, 30, 120, 90), "red")
    _add_box(canvas, "covering", QRectF(35, 20, 100, 125), "green")
    covering_before = _annotation(canvas, "covering")

    # The target was created first and would normally be below the covering
    # box. Selecting it must raise its resize handles above the other box body.
    canvas.select_box("target")
    assert canvas._items["target"].zValue() > canvas._items["covering"].zValue()
    emitted: list[list[dict[str, Any]]] = []
    canvas.annotationsChanged.connect(emitted.append)

    _drag(
        canvas,
        QPointF(50, 120),
        QPointF(25, 155),
        app,
        changes_during_move=emitted,
    )

    target = _annotation(canvas, "target")
    assert target["xmin"] == 25
    assert target["ymin"] == 30
    assert target["xmax"] == 170
    assert target["ymax"] == 155
    assert _annotation(canvas, "covering") == covering_before
    assert _selected_id(canvas) == "target"
    assert canvas._active_interaction_item is None
    assert len(emitted) == 1
    assert emitted[0] == canvas.annotations()
    _dispose(canvas, app)


def test_alt_click_cycles_overlaps_in_stable_order(app: QApplication) -> None:
    canvas = _canvas()
    overlap = QRectF(45, 35, 100, 100)
    _add_box(canvas, "back", overlap)
    _add_box(canvas, "middle", overlap)
    _add_box(canvas, "front", overlap)
    center = _view_point(canvas, overlap.center())

    canvas.scene().clearSelection()
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=center)
    app.processEvents()
    assert _selected_id(canvas) == "front"

    expected = ["middle", "back", "front"]
    for annotation_id in expected:
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            center,
        )
        app.processEvents()
        assert _selected_id(canvas) == annotation_id

    _dispose(canvas, app)


def test_hidden_boxes_cannot_be_selected_and_box_only_remains_editable(
    app: QApplication,
) -> None:
    canvas = _canvas()
    _add_box(canvas, "box", QRectF(40, 40, 70, 60))
    center = _view_point(canvas, QPointF(75, 70))
    original = canvas.annotations()

    canvas.set_annotation_display_mode(AnnotationDisplayMode.HIDDEN)
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.AltModifier,
        center,
    )
    app.processEvents()
    assert _selected_id(canvas) is None
    assert canvas.annotations() == original

    canvas.set_annotation_display_mode(AnnotationDisplayMode.BOX_ONLY)
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=center)
    app.processEvents()
    assert _selected_id(canvas) == "box"
    _drag(canvas, QPointF(75, 70), QPointF(95, 80), app)
    moved = _annotation(canvas, "box")
    assert (moved["xmin"], moved["ymin"]) == (60, 50)
    assert (moved["xmax"], moved["ymax"]) == (130, 110)

    _dispose(canvas, app)


def test_release_repaints_old_geometry_and_emits_only_final_change(
    app: QApplication,
) -> None:
    canvas = _canvas()
    _add_box(canvas, "moving", QRectF(30, 30, 50, 50), "red")
    canvas.select_box("moving")
    app.processEvents()

    old_interior = QPointF(50, 50)
    background = _viewport_pixel(canvas, QPointF(10, 10))
    before = _viewport_pixel(canvas, old_interior)
    assert before != background

    emitted: list[list[dict[str, Any]]] = []
    canvas.annotationsChanged.connect(emitted.append)
    _drag(
        canvas,
        QPointF(55, 55),
        QPointF(145, 105),
        app,
        changes_during_move=emitted,
    )

    after = _viewport_pixel(canvas, old_interior)
    assert after == background
    assert len(emitted) == 1
    assert emitted[0] == canvas.annotations()
    moved = _annotation(canvas, "moving")
    assert (moved["xmin"], moved["ymin"]) == (120, 80)
    assert (moved["xmax"], moved["ymax"]) == (170, 130)

    _dispose(canvas, app)
