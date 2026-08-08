"""Interactive image annotation canvas.

The scene coordinate system is deliberately identical to the source image
pixel coordinate system.  No normalized YOLO coordinates enter this module;
conversion belongs to the data/export layer.
"""

from __future__ import annotations

import math
import uuid
import weakref
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFontMetricsF,
    QImageReader,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QUndoCommand,
    QUndoStack,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
)

from .theme import COLORS, class_color

HANDLE_NAMES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
_HANDLE_CURSORS = {
    "nw": Qt.CursorShape.SizeFDiagCursor,
    "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor,
    "sw": Qt.CursorShape.SizeBDiagCursor,
    "n": Qt.CursorShape.SizeVerCursor,
    "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor,
    "w": Qt.CursorShape.SizeHorCursor,
}


class AnnotationDisplayMode(str, Enum):
    """View-only annotation presentation modes."""

    FULL = "full"
    BOX_ONLY = "box_only"
    HIDDEN = "hidden"


def _value(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _string_value(value: object, name: str, default: str = "") -> str:
    raw = _value(value, name, default)
    return default if raw is None else str(getattr(raw, "value", raw))


class _SnapshotCommand(QUndoCommand):
    def __init__(
        self,
        canvas: AnnotationCanvas,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        text: str,
    ) -> None:
        super().__init__(text)
        self._canvas_ref = weakref.ref(canvas)
        self._before = before
        self._after = after
        self._first_redo = True

    def undo(self) -> None:
        if canvas := self._canvas_ref():
            canvas._restore_snapshot(self._before)

    def redo(self) -> None:
        if canvas := self._canvas_ref():
            if self._first_redo:
                self._first_redo = False
                canvas.annotationsChanged.emit(canvas._snapshot())
            else:
                canvas._restore_snapshot(self._after)


class AnnotationRectItem(QGraphicsRectItem):
    """A movable bounding box with eight resize handles."""

    handle_size = 8.0
    min_size = 2.0

    def __init__(self, canvas: AnnotationCanvas, annotation: Mapping[str, Any]) -> None:
        x1 = float(annotation["xmin"])
        y1 = float(annotation["ymin"])
        x2 = float(annotation["xmax"])
        y2 = float(annotation["ymax"])
        super().__init__(0.0, 0.0, max(self.min_size, x2 - x1), max(self.min_size, y2 - y1))
        self.canvas = canvas
        self.annotation_id = str(annotation.get("id") or uuid.uuid4())
        self.class_id = str(annotation.get("class_id", ""))
        self.origin = str(annotation.get("origin", "manual"))
        self.confidence = annotation.get("confidence")
        self.model_run_id = annotation.get("model_run_id")
        self.prediction_id = annotation.get("prediction_id")
        self._resize_handle: str | None = None
        self._interaction_mode: str | None = None
        self._press_scene_pos = QPointF()
        self._initial_scene_rect = QRectF()
        self._press_snapshot: list[dict[str, Any]] | None = None
        self._interactive_edit_active = False
        self._applying_interactive_geometry = False
        self._base_z = canvas._allocate_annotation_z()
        self.setPos(x1, y1)
        self._set_interaction_enabled(
            canvas.annotation_display_mode is not AnnotationDisplayMode.HIDDEN
        )
        self.setZValue(self._base_z)

    @property
    def class_name(self) -> str:
        return self.canvas.class_style(self.class_id)[0]

    @property
    def color(self) -> QColor:
        return self.canvas.class_style(self.class_id)[1]

    def scene_rect(self) -> QRectF:
        rect = self.rect()
        return QRectF(self.pos().x(), self.pos().y(), rect.width(), rect.height())

    def to_dict(self) -> dict[str, Any]:
        rect = self.scene_rect()
        return {
            "id": self.annotation_id,
            "class_id": self.class_id,
            "xmin": round(rect.left(), 4),
            "ymin": round(rect.top(), 4),
            "xmax": round(rect.right(), 4),
            "ymax": round(rect.bottom(), 4),
            "origin": self.origin,
            "confidence": self.confidence,
            "model_run_id": self.model_run_id,
            "prediction_id": self.prediction_id,
        }

    def _label_text(self) -> str:
        label = self.class_name or self.class_id or "未分类"
        if self.confidence is not None:
            label = f"{label} {float(self.confidence):.2f}"
        return label

    def _label_rect(self) -> QRectF:
        metrics = QFontMetricsF(self.canvas.font())
        height = max(18.0, math.ceil(metrics.height() + 4.0))
        width = max(44.0, math.ceil(metrics.horizontalAdvance(self._label_text()) + 10.0))
        return QRectF(0.0, -height, width, height)

    def _shows_label(self) -> bool:
        return (
            self.canvas.annotation_display_mode is AnnotationDisplayMode.FULL
            and not self._interactive_edit_active
        )

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        margin = self.handle_size / 2 + 2
        bounds = self.rect().adjusted(-margin, -margin, margin, margin)
        if self._shows_label():
            bounds = bounds.united(self._label_rect().adjusted(-1, -1, 1, 1))
        return bounds

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        if self.canvas.annotation_display_mode is AnnotationDisplayMode.HIDDEN:
            return path
        # Handle rectangles overlap the box path. Winding fill keeps that
        # overlap hittable; the default odd/even rule would punch eight holes.
        path.setFillRule(Qt.FillRule.WindingFill)
        path.addRect(self.rect().adjusted(-3, -3, 3, 3))
        if self.isSelected():
            for handle_rect in self._handle_rects().values():
                path.addRect(handle_rect)
        return path

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:
        del option, widget
        if self.canvas.annotation_display_mode is AnnotationDisplayMode.HIDDEN:
            return
        color = self.color
        pen = QPen(color, 2.0 if self.isSelected() else 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(28 if self.isSelected() else 12)
        painter.setBrush(fill)
        painter.drawRect(self.rect())

        if self._shows_label():
            painter.setFont(self.canvas.font())
            label_rect = self._label_rect()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(label_rect, 2, 2)
            painter.setPen(QColor("white"))
            painter.drawText(
                label_rect.adjusted(5, 0, -3, 0),
                Qt.AlignmentFlag.AlignVCenter,
                self._label_text(),
            )

        if self.isSelected():
            painter.setPen(QPen(QColor("white"), 1))
            painter.setBrush(color)
            for handle_rect in self._handle_rects().values():
                painter.drawRect(handle_rect)

    def _set_interaction_enabled(self, enabled: bool) -> None:
        self.setEnabled(enabled)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, enabled)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(enabled)

    def _set_interactive_edit_active(self, active: bool) -> None:
        if self._interactive_edit_active == active:
            return
        self.prepareGeometryChange()
        self._interactive_edit_active = active
        self.update()

    def _handle_rects(self) -> dict[str, QRectF]:
        rect = self.rect()
        half = self.handle_size / 2

        def square(x: float, y: float) -> QRectF:
            return QRectF(x - half, y - half, self.handle_size, self.handle_size)

        cx, cy = rect.center().x(), rect.center().y()
        return {
            "nw": square(rect.left(), rect.top()),
            "n": square(cx, rect.top()),
            "ne": square(rect.right(), rect.top()),
            "e": square(rect.right(), cy),
            "se": square(rect.right(), rect.bottom()),
            "s": square(cx, rect.bottom()),
            "sw": square(rect.left(), rect.bottom()),
            "w": square(rect.left(), cy),
        }

    def _handle_at(self, point: QPointF) -> str | None:
        if not self.isSelected():
            return None
        for name, rect in self._handle_rects().items():
            if rect.contains(point):
                return name
        return None

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        handle = self._handle_at(event.pos())
        self.setCursor(QCursor(_HANDLE_CURSORS.get(handle, Qt.CursorShape.SizeAllCursor)))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self.canvas.annotation_display_mode is AnnotationDisplayMode.HIDDEN:
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            resize_handle = self._handle_at(event.pos())
            if not self.canvas._begin_item_interaction(self):
                event.ignore()
                return
            self._press_snapshot = self.canvas._snapshot()
            self._resize_handle = resize_handle
            self._interaction_mode = "resize" if resize_handle else "move"
            self._press_scene_pos = event.scenePos()
            self._initial_scene_rect = self.scene_rect()
            self._set_interactive_edit_active(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if (
            self._interaction_mode is None
            or self.canvas._active_interaction_item is not self
        ):
            super().mouseMoveEvent(event)
            return
        if self._interaction_mode == "move":
            delta = event.scenePos() - self._press_scene_pos
            rect = QRectF(self._initial_scene_rect)
            bounds = self.canvas.image_rect
            rect.moveLeft(
                min(
                    max(rect.left() + delta.x(), bounds.left()),
                    bounds.right() - rect.width(),
                )
            )
            rect.moveTop(
                min(
                    max(rect.top() + delta.y(), bounds.top()),
                    bounds.bottom() - rect.height(),
                )
            )
            self._set_scene_rect(rect)
            event.accept()
            return

        point = self.canvas.clamp_to_image(event.scenePos())
        rect = QRectF(self._initial_scene_rect)
        name = self._resize_handle
        if name is None:
            event.ignore()
            return
        if "w" in name:
            rect.setLeft(min(point.x(), rect.right() - self.min_size))
        if "e" in name:
            rect.setRight(max(point.x(), rect.left() + self.min_size))
        if "n" in name:
            rect.setTop(min(point.y(), rect.bottom() - self.min_size))
        if "s" in name:
            rect.setBottom(max(point.y(), rect.top() + self.min_size))
        self._set_scene_rect(rect)
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._interaction_mode is None:
            super().mouseReleaseEvent(event)
            return

        self._interaction_mode = None
        self._resize_handle = None
        self._set_interactive_edit_active(False)
        if self._press_snapshot is not None:
            self.canvas._finish_interactive_edit(self._press_snapshot, "调整标注框")
            self._press_snapshot = None
        self.canvas._end_item_interaction(self)
        event.accept()

    def _set_scene_rect(self, rect: QRectF) -> None:
        """Apply one drag geometry update without involving another item."""

        self._applying_interactive_geometry = True
        try:
            self.setPos(rect.left(), rect.top())
            # QGraphicsRectItem.setRect() performs its own geometry-change
            # notification; calling prepareGeometryChange() a second time can
            # leave the scene index inconsistent during rapid test teardown.
            self.setRect(0.0, 0.0, rect.width(), rect.height())
        finally:
            self._applying_interactive_geometry = False
        self.update()

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.scene()
            and self.canvas.has_image
            and not self._applying_interactive_geometry
        ):
            proposed = QPointF(value)
            bounds = self.canvas.image_rect
            width, height = self.rect().width(), self.rect().height()
            proposed.setX(min(max(proposed.x(), bounds.left()), bounds.right() - width))
            proposed.setY(min(max(proposed.y(), bounds.top()), bounds.bottom() - height))
            return proposed
        return super().itemChange(change, value)


class AnnotationCanvas(QGraphicsView):
    """Image view supporting box drawing, editing, history, zoom and pan."""

    annotationsChanged = Signal(list)
    selectionChanged = Signal(object)
    annotationDisplayModeChanged = Signal(str)
    zoomChanged = Signal(float)
    toolChanged = Signal(str)
    undoAvailableChanged = Signal(bool)
    redoAvailableChanged = Signal(bool)
    statusMessage = Signal(str)

    TOOL_SELECT = "select"
    TOOL_DRAW = "draw"
    TOOL_PAN = "pan"

    DISPLAY_FULL = AnnotationDisplayMode.FULL
    DISPLAY_BOX_ONLY = AnnotationDisplayMode.BOX_ONLY
    DISPLAY_HIDDEN = AnnotationDisplayMode.HIDDEN

    _ANNOTATION_Z_BASE = 10.0
    _ANNOTATION_Z_STEP = 0.001
    _SELECTED_ANNOTATION_Z = 100.0

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.setObjectName("annotationCanvas")
        self.setBackgroundBrush(QColor(COLORS["canvas"]))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.undo_stack = QUndoStack(self)
        self._image_item: QGraphicsPixmapItem | None = None
        self._image_path: Path | None = None
        self._image_rect = QRectF()
        self._items: dict[str, AnnotationRectItem] = {}
        self._class_styles: dict[str, tuple[str, QColor]] = {}
        self._current_class_id = ""
        self._annotation_display_mode = AnnotationDisplayMode.FULL
        self._tool = self.TOOL_SELECT
        self._temporary_pan_active = False
        self._tool_before_temporary_pan = self.TOOL_SELECT
        self._drawing = False
        self._draw_start = QPointF()
        self._preview_item: QGraphicsRectItem | None = None
        self._panning = False
        self._pan_last = QPoint()
        self._zoom = 1.0
        self._restoring = False
        self._annotation_z_serial = 0
        self._active_interaction_item: AnnotationRectItem | None = None
        self._cycle_click_active = False
        self.scene().selectionChanged.connect(self._on_scene_selection_changed)
        self.undo_stack.canUndoChanged.connect(self.undoAvailableChanged)
        self.undo_stack.canRedoChanged.connect(self.redoAvailableChanged)

    @property
    def has_image(self) -> bool:
        return self._image_item is not None

    @property
    def image_rect(self) -> QRectF:
        return QRectF(self._image_rect)

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def tool(self) -> str:
        return self._tool

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    @property
    def can_undo(self) -> bool:
        """Whether the current image session has an undoable annotation edit."""
        return self.undo_stack.canUndo()

    @property
    def can_redo(self) -> bool:
        """Whether the current image session has a redoable annotation edit."""
        return self.undo_stack.canRedo()

    @property
    def can_undo_all(self) -> bool:
        """Whether all edits in the current image session can be undone."""
        return self.undo_stack.index() > 0

    @property
    def annotation_display_mode(self) -> AnnotationDisplayMode:
        return self._annotation_display_mode

    def set_annotation_display_mode(
        self, mode: AnnotationDisplayMode | str
    ) -> None:
        """Change annotation presentation without touching annotation data."""
        try:
            normalized = mode if isinstance(mode, AnnotationDisplayMode) else AnnotationDisplayMode(mode)
        except ValueError as exc:
            raise ValueError(f"未知标注显示模式: {mode}") from exc
        if normalized is self._annotation_display_mode:
            return

        before = self._snapshot()
        if normalized is AnnotationDisplayMode.HIDDEN:
            self._cancel_drawing_preview()
        self.scene().clearSelection()
        for item in self._items.values():
            item.prepareGeometryChange()
        self._annotation_display_mode = normalized
        interactive = normalized is not AnnotationDisplayMode.HIDDEN
        for item in self._items.values():
            item._set_interaction_enabled(interactive)
            item.setVisible(interactive)
            item.update()
        self.scene().update()
        # Display state must never leak into the persisted annotation snapshot.
        assert self._snapshot() == before
        self.annotationDisplayModeChanged.emit(normalized.value)

    def class_style(self, class_id: str) -> tuple[str, QColor]:
        if class_id in self._class_styles:
            return self._class_styles[class_id]
        index = max(0, len(self._class_styles))
        return class_id, class_color(index)

    def set_categories(self, categories: Iterable[object]) -> None:
        styles: dict[str, tuple[str, QColor]] = {}
        for index, category in enumerate(categories):
            class_id = _string_value(category, "id", str(index))
            # ``name`` is the canonical class name persisted into training and
            # export data.  ``display_name`` is deliberately presentation-only
            # so an operator can see e.g. BALL without silently changing the
            # trained class from its original MaixHub/VOC spelling.
            canonical_name = _string_value(category, "name", class_id)
            name = (
                _string_value(category, "effective_display_name", "")
                or _string_value(category, "display_name", "")
                or canonical_name
            )
            raw_color = _string_value(category, "color", "")
            color = QColor(raw_color) if raw_color and QColor(raw_color).isValid() else class_color(index)
            styles[class_id] = (name, color)
        for item in self._items.values():
            item.prepareGeometryChange()
        self._class_styles = styles
        if not self._current_class_id and styles:
            self._current_class_id = next(iter(styles))
        for item in self._items.values():
            item.update()

    def set_current_class(self, class_id: str) -> None:
        self._current_class_id = str(class_id)

    def set_tool(self, tool: str) -> None:
        if tool not in {self.TOOL_SELECT, self.TOOL_DRAW, self.TOOL_PAN}:
            raise ValueError(f"未知画布工具: {tool}")
        self._temporary_pan_active = False
        if self._tool == tool:
            self.viewport().setCursor(self._cursor_for_tool(tool))
            return
        self._tool = tool
        self.viewport().setCursor(self._cursor_for_tool(tool))
        self.toolChanged.emit(tool)

    def toggle_draw_tool(self) -> None:
        """Compatibility alias for old callers; the window should set DRAW directly."""
        self.set_tool(self.TOOL_DRAW)

    def begin_temporary_pan(self) -> None:
        """Temporarily enter pan mode while Space is held."""
        if self._temporary_pan_active:
            return
        self._tool_before_temporary_pan = self._tool
        self._temporary_pan_active = True
        if self._tool != self.TOOL_PAN:
            self._tool = self.TOOL_PAN
            self.viewport().setCursor(self._cursor_for_tool(self.TOOL_PAN))
            self.toolChanged.emit(self.TOOL_PAN)

    def end_temporary_pan(self) -> None:
        """Restore the tool selected before a temporary Space-pan."""
        if not self._temporary_pan_active:
            return
        self._temporary_pan_active = False
        restore_tool = self._tool_before_temporary_pan
        if self._tool != restore_tool:
            self._tool = restore_tool
            self.viewport().setCursor(self._cursor_for_tool(restore_tool))
            self.toolChanged.emit(restore_tool)

    @staticmethod
    def _cursor_for_tool(tool: str) -> Qt.CursorShape:
        return {
            AnnotationCanvas.TOOL_SELECT: Qt.CursorShape.ArrowCursor,
            AnnotationCanvas.TOOL_DRAW: Qt.CursorShape.CrossCursor,
            AnnotationCanvas.TOOL_PAN: Qt.CursorShape.OpenHandCursor,
        }[tool]

    def set_image(self, source: str | Path | QPixmap | None, *, fit: bool = True) -> bool:
        self._active_interaction_item = None
        self._cycle_click_active = False
        self.scene().clear()
        self._items.clear()
        self._annotation_z_serial = 0
        self._drawing = False
        self._preview_item = None
        self._panning = False
        self._image_item = None
        self._image_path = None
        self._image_rect = QRectF()
        self.undo_stack.clear()
        if source is None:
            self.annotationsChanged.emit([])
            return False
        if isinstance(source, QPixmap):
            pixmap = source
        else:
            path = Path(source)
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                self.statusMessage.emit(f"无法读取图片：{reader.errorString()}")
                return False
            pixmap = QPixmap.fromImage(image)
            self._image_path = path
        if pixmap.isNull():
            return False
        self._image_item = self.scene().addPixmap(pixmap)
        self._image_item.setZValue(0)
        self._image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.scene().setSceneRect(self._image_rect)
        if fit:
            self.fit_image()
        return True

    def load_image(
        self,
        source: str | Path | QPixmap,
        annotations: Iterable[object] = (),
        *,
        fit: bool = True,
    ) -> bool:
        loaded = self.set_image(source, fit=fit)
        if loaded:
            self.set_annotations(annotations)
        return loaded

    def set_annotations(self, annotations: Iterable[object]) -> None:
        snapshot = [self._normalise_box(box) for box in annotations]
        self._restore_snapshot(snapshot, emit=False)
        self.undo_stack.clear()
        self.annotationsChanged.emit(self.annotations())

    def annotations(self) -> list[dict[str, Any]]:
        return self._snapshot()

    # Friendly aliases for controller implementations and tests.
    boxes = annotations
    set_boxes = set_annotations

    def add_box(
        self,
        rect: QRectF | tuple[float, float, float, float],
        class_id: str | None = None,
        *,
        annotation_id: str | None = None,
        origin: str = "manual",
        confidence: float | None = None,
        record_history: bool = True,
    ) -> str | None:
        if not self.has_image:
            return None
        if not isinstance(rect, QRectF):
            rect = QRectF(*rect)
        rect = rect.normalized().intersected(self._image_rect)
        if rect.width() < AnnotationRectItem.min_size or rect.height() < AnnotationRectItem.min_size:
            return None
        before = self._snapshot()
        annotation = {
            "id": annotation_id or str(uuid.uuid4()),
            "class_id": str(class_id if class_id is not None else self._current_class_id),
            "xmin": rect.left(),
            "ymin": rect.top(),
            "xmax": rect.right(),
            "ymax": rect.bottom(),
            "origin": origin,
            "confidence": confidence,
            "model_run_id": None,
            "prediction_id": None,
        }
        self._add_item(annotation)
        after = self._snapshot()
        if record_history:
            self.undo_stack.push(_SnapshotCommand(self, before, after, "新建标注框"))
        else:
            self.annotationsChanged.emit(after)
        return str(annotation["id"])

    def delete_selected(self) -> bool:
        selected = [item for item in self.scene().selectedItems() if isinstance(item, AnnotationRectItem)]
        if not selected:
            return False
        before = self._snapshot()
        for item in selected:
            self.scene().removeItem(item)
            self._items.pop(item.annotation_id, None)
        self.undo_stack.push(_SnapshotCommand(self, before, self._snapshot(), "删除标注框"))
        return True

    def select_box(self, annotation_id: str | None) -> None:
        self.scene().clearSelection()
        if (
            self._annotation_display_mode is not AnnotationDisplayMode.HIDDEN
            and annotation_id
            and annotation_id in self._items
        ):
            self._items[annotation_id].setSelected(True)
            self.centerOn(self._items[annotation_id])

    def selected_box(self) -> dict[str, Any] | None:
        for item in self.scene().selectedItems():
            if isinstance(item, AnnotationRectItem):
                return item.to_dict()
        return None

    def set_selected_class(self, class_id: str) -> bool:
        selected = [item for item in self.scene().selectedItems() if isinstance(item, AnnotationRectItem)]
        if not selected:
            self.set_current_class(class_id)
            return False
        before = self._snapshot()
        for item in selected:
            item.prepareGeometryChange()
            item.class_id = str(class_id)
            if item.origin == "ai":
                item.origin = "mixed"
            item.update()
        self.undo_stack.push(_SnapshotCommand(self, before, self._snapshot(), "更改类别"))
        return True

    def fit_image(self) -> None:
        if not self.has_image:
            return
        self.resetTransform()
        self.fitInView(self._image_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoomChanged.emit(self._zoom)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self.centerOn(self._image_rect.center())
        self.zoomChanged.emit(self._zoom)

    def zoom_by(self, factor: float) -> None:
        if not self.has_image or factor <= 0:
            return
        current = self.transform().m11()
        target = max(0.05, min(30.0, current * factor))
        if math.isclose(target, current):
            return
        actual = target / current
        self.scale(actual, actual)
        self._zoom = target
        self.zoomChanged.emit(target)

    def clamp_to_image(self, point: QPointF) -> QPointF:
        return QPointF(
            min(max(point.x(), self._image_rect.left()), self._image_rect.right()),
            min(max(point.y(), self._image_rect.top()), self._image_rect.bottom()),
        )

    def undo(self) -> None:
        self.undo_stack.undo()

    def undo_all(self) -> bool:
        """Undo every edit in the current image session while retaining redo."""
        if self.undo_stack.index() <= 0:
            return False
        self.undo_stack.setIndex(0)
        return True

    def redo(self) -> None:
        self.undo_stack.redo()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pan_button = event.button() == Qt.MouseButton.MiddleButton
        if pan_button or (
            self._tool == self.TOOL_PAN and event.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            self._tool == self.TOOL_SELECT
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
            and self._annotation_display_mode is not AnnotationDisplayMode.HIDDEN
            and self._cycle_annotation_selection(event.position().toPoint())
        ):
            self._cycle_click_active = True
            event.accept()
            return
        if (
            self._tool == self.TOOL_DRAW
            and event.button() == Qt.MouseButton.LeftButton
            and self.has_image
            and self._annotation_display_mode is not AnnotationDisplayMode.HIDDEN
        ):
            point = self.mapToScene(event.position().toPoint())
            if self._image_rect.contains(point):
                self._drawing = True
                self._draw_start = point
                self._preview_item = self.scene().addRect(
                    QRectF(point, point),
                    QPen(QColor(COLORS["accent"]), 1.5, Qt.PenStyle.DashLine),
                    QColor(76, 141, 255, 30),
                )
                self._preview_item.setZValue(self._SELECTED_ANNOTATION_Z + 100.0)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position().toPoint() - self._pan_last
            self._pan_last = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._drawing and self._preview_item:
            end = self.clamp_to_image(self.mapToScene(event.position().toPoint()))
            self._preview_item.setRect(QRectF(self._draw_start, end).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self.viewport().setCursor(self._cursor_for_tool(self._tool))
            event.accept()
            return
        if self._cycle_click_active and event.button() == Qt.MouseButton.LeftButton:
            self._cycle_click_active = False
            event.accept()
            return
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self._preview_item:
                rect = self._preview_item.rect()
                self.scene().removeItem(self._preview_item)
                self._preview_item = None
                annotation_id = self.add_box(rect)
                if annotation_id:
                    self.select_box(annotation_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _cancel_drawing_preview(self) -> None:
        """Discard an in-progress draw preview without changing annotations."""
        self._drawing = False
        if self._preview_item is not None:
            self.scene().removeItem(self._preview_item)
            self._preview_item = None

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.begin_temporary_pan()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.redo()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.end_temporary_pan()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _normalise_box(self, value: object) -> dict[str, Any]:
        xmin = _value(value, "xmin", _value(value, "x1", 0.0))
        ymin = _value(value, "ymin", _value(value, "y1", 0.0))
        xmax = _value(value, "xmax", _value(value, "x2", 0.0))
        ymax = _value(value, "ymax", _value(value, "y2", 0.0))
        return {
            "id": _string_value(value, "id", str(uuid.uuid4())),
            "class_id": _string_value(value, "class_id"),
            "xmin": float(xmin),
            "ymin": float(ymin),
            "xmax": float(xmax),
            "ymax": float(ymax),
            "origin": _string_value(value, "origin", "manual"),
            "confidence": _value(value, "confidence"),
            "model_run_id": _value(value, "model_run_id"),
            "prediction_id": _value(value, "prediction_id"),
        }

    def _add_item(self, annotation: Mapping[str, Any]) -> AnnotationRectItem:
        item = AnnotationRectItem(self, annotation)
        self.scene().addItem(item)
        if self._annotation_display_mode is AnnotationDisplayMode.HIDDEN:
            item.setVisible(False)
        self._items[item.annotation_id] = item
        return item

    def _allocate_annotation_z(self) -> float:
        self._annotation_z_serial += 1
        return self._ANNOTATION_Z_BASE + (
            self._annotation_z_serial * self._ANNOTATION_Z_STEP
        )

    def _begin_item_interaction(self, item: AnnotationRectItem) -> bool:
        """Lock a complete press/move/release gesture to exactly one box."""

        if (
            self._annotation_display_mode is AnnotationDisplayMode.HIDDEN
            or self._active_interaction_item not in {None, item}
        ):
            return False
        self._active_interaction_item = item
        # Editing is intentionally single-target even if a caller previously
        # constructed a multi-selection through the graphics scene API.
        for selected in self.scene().selectedItems():
            if selected is not item:
                selected.setSelected(False)
        item.setSelected(True)
        self._update_annotation_z_order()
        return True

    def _end_item_interaction(self, item: AnnotationRectItem) -> None:
        if self._active_interaction_item is item:
            self._active_interaction_item = None
        # Force a complete post-gesture repaint so labels, handles, and the old
        # geometry are redrawn together instead of leaving a backing-store trace.
        self.scene().invalidate(
            self.sceneRect(),
            QGraphicsScene.SceneLayer.AllLayers,
        )
        self.scene().update()
        self.viewport().update()

    def _cycle_annotation_selection(self, view_point: QPoint) -> bool:
        """Select the next overlapping box in stable base-Z order."""

        hits = sorted(
            {
                item
                for item in self.items(view_point)
                if isinstance(item, AnnotationRectItem)
                and item.isVisible()
                and item.isEnabled()
            },
            key=lambda item: item._base_z,
            reverse=True,
        )
        if not hits:
            return False
        selected = next(
            (
                item
                for item in self.scene().selectedItems()
                if isinstance(item, AnnotationRectItem) and item in hits
            ),
            None,
        )
        target = hits[0] if selected is None else hits[(hits.index(selected) + 1) % len(hits)]
        self.scene().clearSelection()
        target.setSelected(True)
        self._update_annotation_z_order()
        return True

    def _update_annotation_z_order(self) -> None:
        for item in self._items.values():
            z_value = (
                self._SELECTED_ANNOTATION_Z
                + (item._base_z * self._ANNOTATION_Z_STEP)
                if item.isSelected()
                else item._base_z
            )
            item.setZValue(z_value)
            item.update()

    def _snapshot(self) -> list[dict[str, Any]]:
        return sorted((item.to_dict() for item in self._items.values()), key=lambda box: box["id"])

    def _restore_snapshot(self, snapshot: list[dict[str, Any]], *, emit: bool = True) -> None:
        selected_ids = {
            item.annotation_id
            for item in self.scene().selectedItems()
            if isinstance(item, AnnotationRectItem)
        }
        self._restoring = True
        self._active_interaction_item = None
        try:
            for item in list(self._items.values()):
                self.scene().removeItem(item)
            self._items.clear()
            self._annotation_z_serial = 0
            for annotation in snapshot:
                item = self._add_item(annotation)
                if item.annotation_id in selected_ids:
                    item.setSelected(True)
        finally:
            self._restoring = False
        if emit:
            self.annotationsChanged.emit(self._snapshot())

    def _finish_interactive_edit(
        self, before: list[dict[str, Any]], description: str
    ) -> None:
        after = self._snapshot()
        if before != after:
            before_by_id = {box["id"]: box for box in before}
            for item in self._items.values():
                old = before_by_id.get(item.annotation_id)
                if old is None or item.origin != "ai":
                    continue
                new = item.to_dict()
                coordinates = ("xmin", "ymin", "xmax", "ymax")
                if any(old.get(key) != new.get(key) for key in coordinates):
                    item.origin = "mixed"
            after = self._snapshot()
            self.undo_stack.push(_SnapshotCommand(self, before, after, description))

    def _on_scene_selection_changed(self) -> None:
        self._update_annotation_z_order()
        selected = self.selected_box()
        self.selectionChanged.emit(selected)
