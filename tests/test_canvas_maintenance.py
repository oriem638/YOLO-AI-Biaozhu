from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsItem

from ai_biaozhu.ui.canvas import (
    AnnotationCanvas,
    AnnotationDisplayMode,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


def _canvas_with_box(
    *,
    class_name: str = "小刚球",
    confidence: float | None = 0.85,
) -> tuple[AnnotationCanvas, str]:
    canvas = AnnotationCanvas()
    pixmap = QPixmap(240, 160)
    pixmap.fill(QColor("white"))
    assert canvas.set_image(pixmap, fit=False)
    canvas.set_categories([{"id": "ball", "name": class_name, "color": "#00c96b"}])
    annotation_id = canvas.add_box(
        QRectF(30, 50, 50, 40),
        "ball",
        annotation_id="box-1",
        origin="ai",
        confidence=confidence,
        record_history=False,
    )
    assert annotation_id
    return canvas, annotation_id


def _paint_item(canvas: AnnotationCanvas, annotation_id: str) -> QImage:
    image = QImage(260, 180, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.translate(20, 60)
    canvas._items[annotation_id].paint(painter, None)
    painter.end()
    return image


def _opaque_pixels(image: QImage, rect: QRectF) -> int:
    clipped = rect.toAlignedRect().intersected(image.rect())
    return sum(
        1
        for y in range(clipped.top(), clipped.bottom() + 1)
        for x in range(clipped.left(), clipped.right() + 1)
        if image.pixelColor(x, y).alpha() > 0
    )


def test_long_label_geometry_is_inside_bounding_rect(app: QApplication) -> None:
    canvas, annotation_id = _canvas_with_box(
        class_name="这是一个很长的小刚球类别名称",
        confidence=0.876,
    )
    item = canvas._items[annotation_id]
    label_rect = item._label_rect()

    assert label_rect.width() > item.rect().width()
    assert item.boundingRect().contains(label_rect)

    canvas.set_annotation_display_mode(AnnotationDisplayMode.BOX_ONLY)
    assert item.boundingRect().top() > label_rect.top()
    assert item.boundingRect().right() < label_rect.right()
    canvas.close()
    app.processEvents()


def test_display_modes_are_view_only_and_hidden_items_do_not_interact(
    app: QApplication,
) -> None:
    canvas, annotation_id = _canvas_with_box()
    item = canvas._items[annotation_id]
    original = canvas.annotations()

    full_image = _paint_item(canvas, annotation_id)
    assert _opaque_pixels(full_image, QRectF(20, 35, 120, 24)) > 0

    emitted: list[str] = []
    canvas.annotationDisplayModeChanged.connect(emitted.append)
    canvas.set_annotation_display_mode("box_only")
    box_only_image = _paint_item(canvas, annotation_id)
    assert _opaque_pixels(box_only_image, QRectF(20, 35, 120, 24)) == 0
    assert _opaque_pixels(box_only_image, QRectF(18, 58, 60, 48)) > 0

    canvas.set_annotation_display_mode(AnnotationDisplayMode.HIDDEN)
    hidden_image = _paint_item(canvas, annotation_id)
    assert _opaque_pixels(hidden_image, QRectF(0, 0, 260, 180)) == 0
    assert not item.isVisible()
    assert not item.isEnabled()
    assert not bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
    assert not bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
    canvas.select_box(annotation_id)
    assert canvas.selected_box() is None

    canvas.set_annotation_display_mode(AnnotationDisplayMode.FULL)
    assert item.isVisible()
    assert item.isEnabled()
    assert bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
    assert bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
    assert canvas.annotations() == original
    assert emitted == ["box_only", "hidden", "full"]
    canvas.close()
    app.processEvents()


def test_hidden_mode_cannot_start_or_complete_an_invisible_draw(app: QApplication) -> None:
    canvas, _ = _canvas_with_box()
    original = canvas.annotations()
    canvas.set_tool(AnnotationCanvas.TOOL_DRAW)
    canvas.set_annotation_display_mode(AnnotationDisplayMode.HIDDEN)

    # The view may still be panned, but it must not start a hidden annotation.
    point = canvas.mapFromScene(QPointF(40, 40))
    global_point = canvas.mapToGlobal(point)
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(point),
        QPointF(global_point),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(event)

    assert not canvas._drawing
    assert canvas._preview_item is None
    assert canvas.annotations() == original
    canvas.close()
    app.processEvents()


def test_label_is_suppressed_during_interactive_edit(app: QApplication) -> None:
    canvas, annotation_id = _canvas_with_box()
    item = canvas._items[annotation_id]
    label_rect = item._label_rect()
    assert item.boundingRect().contains(label_rect)

    item._set_interactive_edit_active(True)
    assert item.boundingRect().top() > label_rect.top()
    assert item.boundingRect().right() < label_rect.right()
    moving_image = _paint_item(canvas, annotation_id)
    assert _opaque_pixels(moving_image, QRectF(20, 35, 120, 24)) == 0

    item._set_interactive_edit_active(False)
    assert item.boundingRect().contains(label_rect)
    canvas.close()
    app.processEvents()


def test_undo_all_restores_baseline_metadata_and_keeps_redo(
    app: QApplication,
) -> None:
    canvas = AnnotationCanvas()
    pixmap = QPixmap(200, 120)
    pixmap.fill(QColor("white"))
    assert canvas.set_image(pixmap, fit=False)
    canvas.set_categories(
        [
            {"id": "ball", "name": "小刚球"},
            {"id": "other", "name": "其他"},
        ]
    )
    baseline = {
        "id": "ai-1",
        "class_id": "ball",
        "xmin": 10.0,
        "ymin": 20.0,
        "xmax": 40.0,
        "ymax": 50.0,
        "origin": "ai",
        "confidence": 0.91,
        "model_run_id": "run-7",
        "prediction_id": "prediction-8",
    }
    canvas.set_annotations([baseline])

    assert canvas.add_box(
        QRectF(80, 20, 20, 20),
        "ball",
        annotation_id="manual-1",
    )
    canvas.select_box("ai-1")
    assert canvas.set_selected_class("other")
    assert canvas.can_undo_all
    assert canvas.annotations() != [baseline]

    assert canvas.undo_all()
    assert canvas.annotations() == [baseline]
    assert not canvas.can_undo
    assert canvas.can_redo

    canvas.redo()
    assert {box["id"] for box in canvas.annotations()} == {"ai-1", "manual-1"}
    assert canvas.undo_all()
    canvas.close()
    app.processEvents()
