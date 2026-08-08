"""Main window and process bridge for the annotation application.

Only controller protocols live here: the UI does not import the SQLite store,
Ultralytics, or a worker implementation.  A production controller and compact
test doubles can therefore use the same window.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PySide6.QtCore import (
    QEvent,
    QObject,
    QProcess,
    QProcessEnvironment,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ai_biaozhu import __version__
from ai_biaozhu.core.annotation_quality import scan_annotation_quality
from ai_biaozhu.core.training_selection import (
    TrainingSelectionError,
    parse_image_index_expression,
    select_from_anchor,
)

from .canvas import AnnotationCanvas, AnnotationDisplayMode
from .dialogs import (
    ImportReportDialog,
    MaixDeployDialog,
    MLEnvironmentDialog,
    TrainingPreflightDialog,
    TrainingSettingsDialog,
    confirm_empty_annotation,
    merged_training_settings,
    training_preflight_text,
    training_setting_warnings,
    validate_training_settings,
)
from .maintenance_dialogs import (
    BulkAnnotationClearDialog,
    VocImportDialog,
    confirm_annotation_quality_warnings,
)
from .theme import COLORS, apply_theme, class_color

MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("YOLOv5n", "yolov5n.pt"),
    ("YOLOv5s", "yolov5s.pt"),
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLO11n", "yolo11n.pt"),
    ("YOLO11s", "yolo11s.pt"),
    ("YOLO26n", "yolo26n.pt"),
    ("YOLO26s", "yolo26s.pt"),
)
UI_PROTOCOL_VERSION = "1.0"
logger = logging.getLogger(__name__)


@runtime_checkable
class AnnotationController(Protocol):
    """Minimal duck-typed boundary consumed by :class:`MainWindow`."""

    def list_images(self) -> Sequence[object]: ...

    def list_classes(self) -> Sequence[object]: ...

    def get_boxes(self, image_id: object) -> Sequence[object]: ...

    def save_boxes(self, image_id: object, boxes: Sequence[Mapping[str, Any]]) -> object: ...

    def verify_and_next(
        self, image_id: object, boxes: Sequence[Mapping[str, Any]]
    ) -> object: ...

    def start_training(self, model_key: str, settings: Mapping[str, Any]) -> object: ...

    def start_autolabel(self, settings: Mapping[str, Any]) -> object: ...

    def cancel_job(self, job_id: str | None = None) -> object: ...

    def handle_job_event(self, event: Mapping[str, Any]) -> object: ...

    def load_training_run_history(self, run_id: str) -> Mapping[str, Any]: ...


class NullController:
    """Read-only empty controller used for UI previews and early integration."""

    current_project: object | None = None
    seed_verified_count = 0

    @staticmethod
    def list_images() -> list[object]:
        return []

    @staticmethod
    def list_classes() -> list[object]:
        return []

    @staticmethod
    def list_runs() -> list[object]:
        return []

    @staticmethod
    def get_boxes(image_id: object) -> list[object]:
        del image_id
        return []

    @staticmethod
    def handle_job_event(event: Mapping[str, Any]) -> None:
        del event


def _value(record: object, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _enum_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(getattr(value, "value", value))


def _as_records(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(
        value, str | bytes | bytearray | Mapping
    ):
        return list(value)
    return []


class ElidingStatusLabel(QLabel):
    """A status label whose full text never contributes to layout width."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setAccessibleName("当前图片状态")
        self.set_full_text(text)

    @property
    def full_text(self) -> str:
        return self._full_text

    def set_full_text(self, text: object) -> None:
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self.setAccessibleDescription(self._full_text)
        self._update_elided_text()

    def setText(self, text: str) -> None:  # noqa: N802
        self.set_full_text(text)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        hint.setWidth(0)
        return hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        width = max(0, self.contentsRect().width())
        shown = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideRight,
            width,
        )
        super().setText(shown)


class JsonlProcessBridge(QObject):
    """Run a worker with ``QProcess`` and decode its versioned JSONL protocol.

    Lines that are not JSON are preserved as log events rather than discarded.
    """

    message = Signal(dict)
    eventReceived = Signal(dict)
    log = Signal(str)
    logReceived = Signal(str)
    error = Signal(str)
    finished = Signal(dict)
    started = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.started.connect(self._on_started)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_finished)
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._stdin_payload: Mapping[str, Any] | None = None
        self._job_id = ""
        self._last_seq = -1
        self._last_protocol_seq = -1
        self._cancel_requested = False

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @property
    def job_id(self) -> str:
        return self._job_id

    def start(
        self,
        program: str,
        arguments: Sequence[str] = (),
        *,
        working_directory: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
        stdin_payload: Mapping[str, Any] | None = None,
        job_id: str = "",
    ) -> None:
        if self.is_running:
            raise RuntimeError("已有 ML 任务正在运行。")
        if not program:
            raise ValueError("缺少 worker 程序路径。")
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._stdin_payload = stdin_payload
        self._job_id = str(job_id or (stdin_payload or {}).get("job_id", ""))
        self._last_seq = -1
        self._last_protocol_seq = -1
        self._cancel_requested = False
        if working_directory:
            self.process.setWorkingDirectory(str(working_directory))
        if environment:
            process_environment = QProcessEnvironment.systemEnvironment()
            for key, value in environment.items():
                process_environment.insert(str(key), str(value))
            self.process.setProcessEnvironment(process_environment)
        self.process.start(str(program), [str(argument) for argument in arguments])

    start_command = start

    def write_message(self, message: Mapping[str, Any]) -> None:
        if not self.is_running:
            raise RuntimeError("worker 尚未运行。")
        raw = json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")) + "\n"
        self.process.write(raw.encode("utf-8"))

    def cancel(
        self,
        *,
        terminate_after_ms: int = 750,
        force_after_ms: int = 4000,
    ) -> None:
        if not self.is_running:
            return
        self._cancel_requested = True
        with suppress(RuntimeError):
            self.write_message(
                {
                    "protocol_version": UI_PROTOCOL_VERSION,
                    "job_id": self._job_id,
                    "seq": self._last_seq + 1,
                    "type": "cancel",
                    "payload": {},
                }
            )
        QTimer.singleShot(
            max(0, terminate_after_ms),
            lambda: self.process.terminate() if self.is_running else None,
        )
        QTimer.singleShot(
            max(terminate_after_ms, force_after_ms),
            lambda: self.process.kill() if self.is_running else None,
        )

    stop = cancel

    def _on_started(self) -> None:
        self.started.emit(self._job_id)
        if self._stdin_payload is not None:
            self.write_message(self._stdin_payload)

    def _read_stdout(self) -> None:
        self._append_process_output(
            "_stdout_buffer",
            bytes(self.process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            ),
            protocol=True,
        )

    def _read_stderr(self) -> None:
        self._append_process_output(
            "_stderr_buffer",
            bytes(self.process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            ),
            protocol=False,
        )

    def _append_process_output(
        self,
        buffer_name: str,
        text: str,
        *,
        protocol: bool,
    ) -> None:
        """Consume complete process lines without exposing them to re-entry.

        Emitting an error opens a modal message box in the main window.  Its
        nested Qt event loop can deliver ``QProcess.finished`` before the
        current ready-read callback returns.  Store only the incomplete tail
        *before* emitting any signals so the finished handler cannot decode the
        same worker event a second time.
        """

        buffer = str(getattr(self, buffer_name, "")) + text
        lines = buffer.splitlines(keepends=True)
        tail = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            tail = lines.pop()
        setattr(self, buffer_name, tail)
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if protocol:
                self._decode_line(line)
            else:
                self.log.emit(line)
                self.logReceived.emit(line)
                self._emit_event("log", {"message": line, "stream": "stderr"})

    def _decode_line(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            self.log.emit(line)
            self.logReceived.emit(line)
            self._emit_event("log", {"message": line, "stream": "stdout"})
            return
        if not isinstance(message, Mapping):
            self._emit_event("log", {"message": str(message), "stream": "stdout"})
            return
        envelope = dict(message)
        envelope.setdefault("protocol_version", UI_PROTOCOL_VERSION)
        envelope.setdefault("job_id", self._job_id)
        envelope.setdefault("seq", self._last_seq + 1)
        envelope.setdefault("type", "event")
        envelope.setdefault("payload", {})
        if str(envelope["protocol_version"]) != UI_PROTOCOL_VERSION:
            self._protocol_failure(
                f"worker 协议版本 {envelope['protocol_version']!r} 不受支持，"
                f"需要 {UI_PROTOCOL_VERSION!r}。"
            )
            return
        message_job_id = str(envelope.get("job_id", ""))
        if self._job_id and message_job_id and message_job_id != self._job_id:
            self._protocol_failure(
                f"worker job_id {message_job_id!r} 与当前任务 {self._job_id!r} 不一致。"
            )
            return
        try:
            protocol_seq = int(envelope["seq"])
        except (TypeError, ValueError):
            self._protocol_failure("worker 事件 seq 不是整数。")
            return
        if protocol_seq <= self._last_protocol_seq:
            relation = (
                "重复" if protocol_seq == self._last_protocol_seq else "乱序"
            )
            diagnostic = (
                f"已忽略{relation} worker 事件：job_id={message_job_id!r}, "
                f"seq={protocol_seq}, last_seq={self._last_protocol_seq}, "
                f"type={str(envelope.get('type', 'event'))!r}"
            )
            self.log.emit(diagnostic)
            self.logReceived.emit(diagnostic)
            self._emit_event(
                "log",
                {
                    "message": diagnostic,
                    "stream": "protocol",
                    "protocol_diagnostic": True,
                    "ignored_seq": protocol_seq,
                    "last_seq": self._last_protocol_seq,
                    "ignored_type": str(envelope.get("type", "event")),
                },
            )
            return
        self._last_protocol_seq = protocol_seq
        self._last_seq = max(self._last_seq, protocol_seq)
        self.message.emit(envelope)
        self.eventReceived.emit(envelope)
        if envelope["type"] == "log":
            payload = envelope.get("payload") or {}
            text = payload.get("message", payload) if isinstance(payload, Mapping) else payload
            self.log.emit(str(text))
            self.logReceived.emit(str(text))
        elif envelope["type"] == "error":
            payload = envelope.get("payload") or {}
            text = payload.get("message", payload) if isinstance(payload, Mapping) else payload
            self.error.emit(str(text))

    def _protocol_failure(self, message: str) -> None:
        self.error.emit(message)
        self._emit_event("error", {"message": message, "protocol_error": True})

    def _emit_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self._last_seq += 1
        envelope = {
            "protocol_version": UI_PROTOCOL_VERSION,
            "job_id": self._job_id,
            "seq": self._last_seq,
            "type": event_type,
            "payload": dict(payload),
            "_internal": True,
        }
        self.message.emit(envelope)
        self.eventReceived.emit(envelope)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        message = self.process.errorString() or str(error)
        self.error.emit(message)
        self._emit_event(
            "error",
            {"message": message, "process_error": int(getattr(error, "value", error))},
        )

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        # A platform may signal ``finished`` before queued ready-read callbacks.
        # Drain the native buffers first, then atomically take any incomplete
        # tails before dispatching signals that can enter a modal event loop.
        self._append_process_output(
            "_stdout_buffer",
            bytes(self.process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            ),
            protocol=True,
        )
        self._append_process_output(
            "_stderr_buffer",
            bytes(self.process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            ),
            protocol=False,
        )
        stdout_tail, self._stdout_buffer = self._stdout_buffer.strip(), ""
        stderr_tail, self._stderr_buffer = self._stderr_buffer.strip(), ""
        if stdout_tail:
            self._decode_line(stdout_tail)
        if stderr_tail:
            text = stderr_tail
            self.log.emit(text)
            self.logReceived.emit(text)
        payload = {
            "exit_code": exit_code,
            "exit_status": int(getattr(exit_status, "value", exit_status)),
            "cancelled": self._cancel_requested,
            "success": exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0,
        }
        self._emit_event("process_finished", payload)
        self.finished.emit(payload)


class MetricsPlotWidget(QWidget):
    """Small dependency-free training curve plot."""

    SERIES: tuple[tuple[str, str, QColor], ...] = (
        ("box_loss", "box", QColor("#ef6262")),
        ("cls_loss", "cls", QColor("#f3b64c")),
        ("objectness_loss", "obj", QColor("#55c5c2")),
        ("dfl_loss", "dfl", QColor("#ab77e8")),
        ("map50", "mAP50", QColor("#45c486")),
        ("map50_95", "mAP50-95", QColor("#4c8dff")),
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        visible_keys: Sequence[str] | None = None,
        fixed_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricsPlot")
        self.setMinimumHeight(145)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._history: dict[str, list[float]] = {key: [] for key, _, _ in self.SERIES}
        requested = set(visible_keys or (key for key, _, _ in self.SERIES))
        self._visible_series = tuple(
            item for item in self.SERIES if item[0] in requested
        )
        self._fixed_range = fixed_range

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(360, 170)

    @property
    def history(self) -> dict[str, list[float]]:
        return {key: list(values) for key, values in self._history.items()}

    def clear(self) -> None:
        for values in self._history.values():
            values.clear()
        self.update()

    def append_metrics(self, metrics: Mapping[str, Any]) -> None:
        loss_items = metrics.get("loss_items", metrics.get("loss"))
        loss_by_key: dict[str, Any] = {}
        if isinstance(loss_items, Sequence) and not isinstance(loss_items, str | bytes):
            loss_by_key = {
                key: loss_items[index]
                for index, key in enumerate(("box_loss", "cls_loss", "dfl_loss"))
                if index < len(loss_items)
            }
        for key, _, _ in self.SERIES:
            value = _metric(metrics, key)
            if value is None:
                value = loss_by_key.get(key)
            if value is None:
                continue
            try:
                self._history[key].append(float(value))
                self._history[key] = self._history[key][-1000:]
            except (TypeError, ValueError):
                continue
        self.update()

    append = append_metrics

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#151820"))
        plot = self.rect().adjusted(42, 12, -12, -30)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRect(plot)
        all_values = [
            value
            for key, _label, _color in self._visible_series
            for value in self._history[key]
        ]
        if not all_values:
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "等待训练指标…")
            return
        if self._fixed_range is None:
            minimum = min(0.0, min(all_values))
            maximum = max(all_values)
        else:
            minimum, maximum = self._fixed_range
        if abs(maximum - minimum) < 1e-9:
            maximum = minimum + 1.0
        max_points = max(
            (len(self._history[key]) for key, _, _ in self._visible_series),
            default=1,
        )
        for key, _label, color in self._visible_series:
            values = self._history[key]
            if not values:
                continue
            path = QPainterPath()
            for index, value in enumerate(values):
                x = plot.left() + (index / max(1, max_points - 1)) * plot.width()
                y = plot.bottom() - ((value - minimum) / (maximum - minimum)) * plot.height()
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(color, 1.5))
            painter.drawPath(path)
        painter.setPen(QColor(COLORS["muted"]))
        painter.drawText(3, plot.top() + 10, f"{maximum:.3g}")
        painter.drawText(3, plot.bottom(), f"{minimum:.3g}")
        legend_x = plot.left()
        legend_y = self.height() - 10
        for key, label, color in self._visible_series:
            if not self._history[key]:
                continue
            painter.setPen(QPen(color, 3))
            painter.drawLine(legend_x, legend_y - 4, legend_x + 14, legend_y - 4)
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(legend_x + 18, legend_y, label)
            legend_x += 72


def _metric(payload: Mapping[str, Any], *names: str) -> Any:
    aliases = {
        "box_loss": ("box_loss", "train/box_loss", "loss/box", "box"),
        "cls_loss": ("cls_loss", "train/cls_loss", "loss/cls", "cls"),
        "objectness_loss": (
            "objectness_loss",
            "train/obj_loss",
            "obj_loss",
            "loss/obj",
        ),
        "dfl_loss": ("dfl_loss", "train/dfl_loss", "loss/dfl", "dfl"),
        "precision": (
            "precision",
            "metrics/precision",
            "metrics/precision(B)",
            "p",
        ),
        "recall": ("recall", "metrics/recall", "metrics/recall(B)", "r"),
        "map50": (
            "map50",
            "mAP50",
            "metrics/mAP_0.5",
            "metrics/mAP50",
            "metrics/mAP50(B)",
        ),
        "map50_95": (
            "map50_95",
            "map50-95",
            "mAP50-95",
            "metrics/mAP_0.5:0.95",
            "metrics/mAP50-95",
            "metrics/mAP50-95(B)",
        ),
        "gpu_memory": ("gpu_memory", "gpu_mem", "gpu_memory_mb", "gpu_memory_gb"),
        "gpu_utilization": ("gpu_utilization", "gpu_usage", "gpu_percent"),
        "learning_rate": ("learning_rate", "lr", "lr/pg0", "lr0"),
        "eta_seconds": ("eta_seconds", "eta", "remaining_seconds"),
    }
    for name in names:
        candidates = aliases.get(name, (name,))
        for candidate in candidates:
            if candidate in payload:
                return payload[candidate]
    nested = payload.get("metrics")
    if isinstance(nested, Mapping) and nested is not payload:
        return _metric(nested, *names)
    return None


def _deployment_size_warning_detail(payload: Mapping[str, Any]) -> str:
    raw_packages = payload.get("packages")
    packages = (
        [item for item in raw_packages if isinstance(item, Mapping)]
        if isinstance(raw_packages, Sequence)
        and not isinstance(raw_packages, str | bytes)
        else []
    )
    if not packages:
        packages = [payload]
    lines = ["部署包超过十进制 30,000,000 字节警告值："]
    for item in packages:
        package_name = str(
            item.get("package_kind", item.get("package", "部署包"))
        )
        threshold = _byte_count(
            item.get("threshold", payload.get("threshold", 30_000_000))
        ) or 30_000_000
        zip_size = _byte_count(
            item.get(
                "zip_size",
                item.get(
                    "zip_bytes",
                    item.get("compressed_size", payload.get("compressed_size")),
                ),
            )
        )
        unpacked_size = _byte_count(
            item.get(
                "unpacked_size",
                item.get(
                    "payload_bytes",
                    item.get("total_size", payload.get("total_size")),
                ),
            )
        )
        lines.append(f"\n{package_name}")
        lines.append(_size_status_line("ZIP", zip_size, threshold))
        lines.append(_size_status_line("解压后", unpacked_size, threshold))
        largest = item.get("largest_files")
        if isinstance(largest, Sequence) and not isinstance(largest, str | bytes):
            file_lines = []
            for record in largest:
                if not isinstance(record, Mapping):
                    continue
                path = str(record.get("path", record.get("name", "未知文件")))
                size = _byte_count(record.get("size"))
                file_lines.append(
                    f"  - {path}: "
                    f"{size:,} 字节" if size is not None else f"  - {path}: 未知"
                )
            if file_lines:
                lines.append("最大文件：")
                lines.extend(file_lines)
    lines.extend(
        (
            "",
            "建议：改用 n 型模型、降低部署分辨率、MaixCAM2 只保留一种 "
            "NPU 模式，或移除图标等可选资源。",
            "必要模型文件不会被自动删除。是否仍然生成？",
        )
    )
    return "\n".join(lines)


def _size_status_line(label: str, size: int | None, threshold: int) -> str:
    if size is None:
        return f"{label}：未知"
    excess = max(0, size - threshold)
    suffix = f"，超出 {excess:,} 字节" if excess else "，未超出"
    return f"{label}：{size:,} 字节{suffix}"


def _byte_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _format_megabytes(value: int) -> str:
    return f"{max(0, int(value)) / 1_000_000:.2f} MB"


class TrainingMonitorWidget(QWidget):
    """Progress, metric cards, curves, and logs for external jobs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.state_label = QLabel("尚未启动训练")
        self.state_label.setObjectName("trainingStateLabel")
        layout.addWidget(self.state_label)

        self.preparation_progress = QProgressBar()
        self.preparation_progress.setObjectName("trainingPreparationProgress")
        self.preparation_progress.setFormat("模型准备 %p%")
        layout.addWidget(self.preparation_progress)

        self.epoch_progress = QProgressBar()
        self.epoch_progress.setObjectName("epochProgress")
        self.epoch_progress.setFormat("Epoch %v / %m")
        self.batch_progress = QProgressBar()
        self.batch_progress.setObjectName("batchProgress")
        self.batch_progress.setFormat("Batch %v / %m")
        layout.addWidget(self.epoch_progress)
        layout.addWidget(self.batch_progress)

        metric_frame = QFrame()
        metric_layout = QFormLayout(metric_frame)
        metric_layout.setContentsMargins(0, 3, 0, 3)
        self.loss_label = QLabel("box —  cls —  obj —  dfl —")
        self.quality_label = QLabel("P —  R —  mAP50 —  mAP50-95 —")
        self.timing_label = QLabel("ETA —  学习率 —")
        self.gpu_label = QLabel("GPU —%  显存 —")
        self._eta_seconds: float | None = None
        self._learning_rate: Any = None
        self._gpu_utilization: Any = None
        self._gpu_memory: Any = None
        self._training_end_text: str | None = None
        self._requested_epochs_known = False
        self._requested_epochs: int | None = None
        self._completed_epochs = 0
        metric_layout.addRow("损失", self.loss_label)
        metric_layout.addRow("验证", self.quality_label)
        metric_layout.addRow("训练", self.timing_label)
        metric_layout.addRow("资源", self.gpu_label)
        layout.addWidget(metric_frame)

        loss_title = QLabel("损失曲线（box / cls / obj / dfl）")
        loss_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(loss_title)
        self.loss_plot = MetricsPlotWidget(
            visible_keys=(
                "box_loss",
                "cls_loss",
                "objectness_loss",
                "dfl_loss",
            )
        )
        # Keep the historical public attribute for integrations and older tests.
        self.plot = self.loss_plot
        layout.addWidget(self.loss_plot)
        map_title = QLabel("验证精度曲线（mAP50 / mAP50-95）")
        map_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(map_title)
        self.map_plot = MetricsPlotWidget(
            visible_keys=("map50", "map50_95"),
            fixed_range=(0.0, 1.0),
        )
        layout.addWidget(self.map_plot)
        plot_row = QHBoxLayout()
        plot_row.addStretch(1)
        self.refresh_plot_button = QPushButton("刷新曲线")
        self.refresh_plot_button.clicked.connect(self.loss_plot.update)
        self.refresh_plot_button.clicked.connect(self.map_plot.update)
        plot_row.addWidget(self.refresh_plot_button)
        layout.addLayout(plot_row)

        self.preview_label = QLabel("训练预览图将在生成后显示")
        self.preview_label.setObjectName("trainingPreview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setMaximumHeight(220)
        self.preview_label.setStyleSheet(
            f"border: 1px solid {COLORS['border']}; color: {COLORS['muted']};"
        )
        layout.addWidget(self.preview_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("trainingLog")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setMinimumHeight(110)
        layout.addWidget(self.log_view)

    def reset(
        self,
        title: str = "正在准备训练…",
        *,
        requested_epochs: int | None = None,
    ) -> None:
        self.state_label.setText(title)
        self.preparation_progress.setRange(0, 100)
        self.preparation_progress.setValue(0)
        self.preparation_progress.setFormat("模型准备 %p%")
        requested = max(1, int(requested_epochs)) if requested_epochs else 100
        self.epoch_progress.setRange(0, requested)
        self.epoch_progress.setValue(0)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.loss_label.setText("box —  cls —  obj —  dfl —")
        self.quality_label.setText("P —  R —  mAP50 —  mAP50-95 —")
        self._eta_seconds: float | None = None
        self._learning_rate: Any = None
        self._gpu_utilization: Any = None
        self._gpu_memory: Any = None
        self._training_end_text = None
        self._requested_epochs_known = requested_epochs is not None
        self._requested_epochs = requested if requested_epochs is not None else None
        self._completed_epochs = 0
        self.timing_label.setText("ETA —  学习率 —")
        self.gpu_label.setText("GPU —%  显存 —")
        self.loss_plot.clear()
        self.map_plot.clear()
        self.log_view.clear()
        self.preview_label.setText("训练预览图将在生成后显示")
        self.preview_label.setPixmap(QPixmap())

    def append_log(self, text: str) -> None:
        if text:
            self.log_view.appendPlainText(str(text).rstrip())

    def handle_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", "event"))
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {"message": payload}
        training_end = _training_end_mapping(payload)
        if event_type in {"progress", "epoch", "batch", "metrics", "metric", "train_progress"}:
            self._update_progress(payload, event_type=event_type)
            self._update_metrics(payload)
        elif event_type in {"status", "stage", "started"}:
            self._update_progress(payload, event_type=event_type)
        if training_end is not None:
            self._apply_training_end(training_end)
        if event_type in {"status", "stage", "started"} and training_end is None:
            self.state_label.setText(
                str(payload.get("message") or payload.get("stage") or payload.get("status") or event_type)
            )
        elif event_type in {"log", "warning"}:
            self.append_log(str(payload.get("message", payload)))
        elif event_type == "error":
            if training_end is None:
                self._apply_training_end(
                    self._inferred_training_end("failed", payload)
                )
            self.append_log("错误：" + str(payload.get("message", payload)))
        elif event_type in {"cancelled", "canceled"}:
            if training_end is None:
                self._apply_training_end(
                    self._inferred_training_end("cancelled", payload)
                )
        elif event_type in {"finished", "completed", "process_finished"}:
            success = bool(payload.get("success", event_type != "process_finished"))
            if self._training_end_text is not None:
                self.state_label.setText(self._training_end_text)
            else:
                reason = (
                    "cancelled"
                    if bool(payload.get("cancelled"))
                    else "unknown" if success else "failed"
                )
                self._apply_training_end(self._inferred_training_end(reason, payload))
        elif event_type == "artifact":
            kind = str(payload.get("kind", ""))
            path = payload.get("path")
            if path and kind in {
                "preview",
                "results",
                "results_plot",
                "confusion_matrix",
                "pr_curve",
                "training_visual",
            }:
                self.show_preview(str(path))
        if (
            "loss" in payload
            or "loss_items" in payload
            or any(
                _metric(payload, key) is not None
                for key, _, _ in MetricsPlotWidget.SERIES
            )
        ):
            self.loss_plot.append_metrics(payload)
            self.map_plot.append_metrics(payload)

    def _inferred_training_end(
        self,
        reason: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "reason": reason,
            "completed_epochs": int(
                payload.get("completed_epochs", self._completed_epochs) or 0
            ),
            "requested_epochs": int(
                payload.get(
                    "requested_epochs",
                    self._requested_epochs or 0,
                )
                or 0
            ),
            "patience": int(payload.get("patience", 0) or 0),
            "monitor": str(payload.get("monitor", "fitness")),
        }

    def _apply_training_end(self, result: Mapping[str, Any]) -> None:
        reason = str(result.get("reason", "unknown")).casefold()
        completed = max(0, int(result.get("completed_epochs", 0) or 0))
        requested = max(0, int(result.get("requested_epochs", 0) or 0))
        labels = {
            "max_epochs": "达到最大轮数（max_epochs）",
            "early_stopping": "触发早停（early_stopping）",
            "cancelled": "训练已取消（cancelled）",
            "canceled": "训练已取消（cancelled）",
            "failed": "训练失败（failed）",
            "unknown": "训练已结束（unknown）",
        }
        title = labels.get(reason, f"训练已结束（{reason or 'unknown'}）")
        epoch_text = (
            f" · 已完成 {completed} / 请求 {requested} epochs"
            if requested > 0
            else f" · 已完成 {completed} epochs"
        )
        text = title + epoch_text
        if reason == "early_stopping":
            patience = int(result.get("patience", 0) or 0)
            monitor = str(result.get("monitor", "fitness"))
            text += f" · patience={patience} · monitor={monitor}"
        if reason in {"failed", "cancelled", "canceled"} and completed == 0:
            self.preparation_progress.setRange(0, 100)
            self.preparation_progress.setValue(0)
            self.preparation_progress.setFormat(
                "模型准备已取消" if reason in {"cancelled", "canceled"} else "模型准备失败"
            )
        if requested > 0:
            self._requested_epochs_known = True
            self._requested_epochs = requested
            self.epoch_progress.setMaximum(max(1, requested))
        self._completed_epochs = completed
        self.epoch_progress.setValue(completed)
        if text != self._training_end_text:
            self.append_log("训练结束：" + text)
        self._training_end_text = text
        self.state_label.setText(text)

    def show_preview(self, path: str | os.PathLike[str]) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        target = self.preview_label.size()
        self.preview_label.setPixmap(
            pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.preview_label.setToolTip(str(path))
        return True

    def _update_progress(
        self,
        payload: Mapping[str, Any],
        *,
        event_type: str = "progress",
    ) -> None:
        stage = str(payload.get("stage", ""))
        requested_epochs = payload.get("requested_epochs")
        if requested_epochs is not None:
            self._requested_epochs = max(1, int(requested_epochs))
            self._requested_epochs_known = True
            self.epoch_progress.setMaximum(self._requested_epochs)
        if stage in {
            "resolving_pretrained_weight",
            "resolving_amp_reference_weight",
            "loading_model",
        }:
            self.preparation_progress.setRange(0, 0)
            self.preparation_progress.setFormat("正在准备模型…")
        if stage in {"downloading_weight", "downloading_amp_reference_weight"}:
            current_bytes = _byte_count(payload.get("current_bytes")) or 0
            total_bytes = _byte_count(payload.get("total_bytes"))
            filename = str(payload.get("filename") or payload.get("model_key") or "模型")
            if total_bytes and total_bytes > 0:
                percent = min(100, round(current_bytes * 100 / total_bytes))
                self.preparation_progress.setRange(0, 100)
                self.preparation_progress.setValue(percent)
                self.preparation_progress.setFormat("模型准备 %p%")
                self.state_label.setText(
                    f"正在下载 {filename}：{_format_megabytes(current_bytes)} / "
                    f"{_format_megabytes(total_bytes)}（{percent}%）"
                )
            else:
                self.preparation_progress.setRange(0, 0)
                self.preparation_progress.setFormat("正在下载模型…")
                self.state_label.setText(
                    f"正在下载 {filename}：{_format_megabytes(current_bytes)}"
                )
            return
        epoch = payload.get("epoch", payload.get("current_epoch"))
        epochs = payload.get("epochs", payload.get("total_epochs"))
        batch = payload.get("batch", payload.get("current_batch"))
        batches = payload.get("batches", payload.get("total_batches"))
        if stage == "training_batch":
            batch = payload.get("current", batch)
            batches = payload.get("total", batches)
        elif stage in {"training", "evaluating"}:
            epoch = payload.get("current", epoch)
            epochs = payload.get("total", epochs)
            self.preparation_progress.setRange(0, 100)
            self.preparation_progress.setValue(100)
            self.preparation_progress.setFormat("模型准备完成")
        if epochs is not None:
            self._requested_epochs_known = True
            self._requested_epochs = max(1, int(epochs))
            self.epoch_progress.setMaximum(self._requested_epochs)
        if epoch is not None:
            displayed_epoch = max(0, int(epoch))
            self.epoch_progress.setValue(displayed_epoch)
            # ``training_batch`` reports the epoch currently in progress.  It
            # is not a completed epoch and must never leak into terminal task
            # accounting when a run fails during its first batch.  Only the
            # epoch-end progress callback or validated metrics proves that an
            # epoch completed.
            if stage == "training" or event_type in {"metrics", "metric"}:
                self._completed_epochs = displayed_epoch
        if batches is not None:
            self.batch_progress.setMaximum(max(1, int(batches)))
        if batch is not None:
            self.batch_progress.setValue(max(0, int(batch)))
        eta = _metric(payload, "eta_seconds")
        if eta is not None:
            try:
                self._eta_seconds = max(0.0, float(eta))
            except (TypeError, ValueError):
                self._eta_seconds = None
            self._refresh_runtime_labels()

    def _update_metrics(self, payload: Mapping[str, Any]) -> None:
        box = _metric(payload, "box_loss")
        cls = _metric(payload, "cls_loss")
        objectness = _metric(payload, "objectness_loss")
        dfl = _metric(payload, "dfl_loss")
        loss_items = payload.get("loss_items", payload.get("loss"))
        if isinstance(loss_items, Sequence) and not isinstance(loss_items, str | bytes):
            if box is None and len(loss_items) > 0:
                box = loss_items[0]
            if cls is None and len(loss_items) > 1:
                cls = loss_items[1]
            if dfl is None and len(loss_items) > 2:
                dfl = loss_items[2]
        precision = _metric(payload, "precision")
        recall = _metric(payload, "recall")
        map50 = _metric(payload, "map50")
        map95 = _metric(payload, "map50_95")
        gpu = _metric(payload, "gpu_memory")
        gpu_utilization = _metric(payload, "gpu_utilization")
        learning_rate = _metric(payload, "learning_rate")
        dfl_status = payload.get("dfl_loss_status")
        nested_metrics = payload.get("metrics")
        if dfl_status is None and isinstance(nested_metrics, Mapping):
            dfl_status = nested_metrics.get("dfl_loss_status")
        dfl_text = (
            "不可用（YOLOv5）"
            if str(dfl_status).casefold() == "unavailable"
            else _number(dfl)
        )
        if (
            any(value is not None for value in (box, cls, objectness, dfl))
            or dfl_status
        ):
            self.loss_label.setText(
                f"box {_number(box)}  cls {_number(cls)}  "
                f"obj {_number(objectness)}  dfl {dfl_text}"
            )
        if any(value is not None for value in (precision, recall, map50, map95)):
            self.quality_label.setText(
                f"P {_number(precision)}  R {_number(recall)}  "
                f"mAP50 {_number(map50)}  mAP50-95 {_number(map95)}"
            )
        if gpu is not None:
            self._gpu_memory = gpu
        if gpu_utilization is not None:
            self._gpu_utilization = gpu_utilization
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if any(value is not None for value in (gpu, gpu_utilization, learning_rate)):
            self._refresh_runtime_labels()

    def _refresh_runtime_labels(self) -> None:
        self.timing_label.setText(
            f"ETA {_duration(self._eta_seconds)}  学习率 {_number(self._learning_rate)}"
        )
        utilization = (
            "—"
            if self._gpu_utilization is None
            else f"{_number(self._gpu_utilization)}%"
        )
        memory = (
            "—"
            if self._gpu_memory is None
            else f"{_number(self._gpu_memory)} GB"
        )
        self.gpu_label.setText(f"GPU {utilization}  显存 {memory}")


def _number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _training_end_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[object] = [payload.get("training_end")]
    result = payload.get("result")
    if isinstance(result, Mapping):
        candidates.append(result.get("training_end"))
    candidates.append(payload)
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("reason") is not None:
            return candidate
    return None


class MainWindow(QMainWindow):
    """Three-column annotation, training and deployment workspace."""

    requestStatus = Signal(str)

    def __init__(
        self,
        controller: AnnotationController | object | None = None,
        *,
        process_bridge: JsonlProcessBridge | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller if controller is not None else NullController()
        self.setObjectName("mainWindow")
        self.setWindowTitle(f"AI 数据集标注与训练 维护版 {__version__}")
        self.setMinimumSize(1024, 640)
        application = QApplication.instance()
        if application is not None:
            apply_theme(application)

        self._image_records: list[object] = []
        self._selection_anchor_index: int | None = None
        self._selection_anchor_existing_indices: set[int] = set()
        self._last_training_preflight: Mapping[str, Any] | object | None = None
        self._category_records: list[object] = []
        self._current_image: object | None = None
        self._loading_image = False
        self._navigation_in_progress = False
        self._annotations_dirty = False
        self._annotation_session_baseline: dict[str, Any] | None = None
        self._restoring_annotation_session_baseline = False
        self._closing = False
        self._active_job_id = ""
        self._active_job_kind = ""
        self._ui_busy = False
        self._forwarding_controller_event = False
        self._ml_environment = ""
        self._environment_dialog: MLEnvironmentDialog | None = None
        self._pending_deploy_confirmation: dict[str, Any] | None = None
        self._docker_environment_action = ""
        self._docker_terminal_event_seen = False
        self._docker_recovery_started_at = 0.0
        self._last_deploy_output_dir: Path | None = None
        self._last_maixapp_path: Path | None = None
        self._last_completed_deploy_run_id = ""
        self._training_settings = merged_training_settings()
        self._training_presets: dict[str, dict[str, Any]] = {}
        self._successful_runs: list[object] = []
        self._resumable_runs: list[object] = []
        self._modified_image_ids: set[str] = set()
        self._last_annotation_quality_report: object | None = None

        self.process_bridge = process_bridge or JsonlProcessBridge(self)
        self.process_bridge.eventReceived.connect(self._on_process_event)
        self.process_bridge.error.connect(self._show_process_error)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(250)
        self._autosave_timer.timeout.connect(self.save_current_annotations)

        self._docker_recovery_timer = QTimer(self)
        self._docker_recovery_timer.setInterval(2000)
        self._docker_recovery_timer.timeout.connect(self._poll_docker_recovery)

        self._build_actions()
        self._build_ui()
        self._wire_signals()
        self._resize_for_available_screen()
        self._connect_controller_signals()
        self.refresh_project()
        self.statusBar().showMessage("就绪")

    # ---- UI construction -------------------------------------------------

    def _build_actions(self) -> None:
        self.new_project_action = QAction("新建项目…", self)
        self.open_project_action = QAction("打开项目…", self)
        self.import_images_action = QAction("导入图片…", self)
        self.import_voc_action = QAction("导入 MaixHub/VOC 混合数据集…", self)
        self.restore_annotation_backup_action = QAction("恢复标注备份…", self)
        self.export_yolo_action = QAction("导出 YOLO 数据集…", self)
        self.exit_action = QAction("退出", self)
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action.triggered.connect(self.open_project)
        self.import_images_action.triggered.connect(self.import_images)
        self.import_voc_action.triggered.connect(self.import_voc_dataset)
        self.restore_annotation_backup_action.triggered.connect(
            self.restore_annotation_backup
        )
        self.export_yolo_action.triggered.connect(self.export_yolo)
        self.exit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addActions(
            [
                self.new_project_action,
                self.open_project_action,
                self.import_images_action,
                self.import_voc_action,
                self.restore_annotation_backup_action,
                self.export_yolo_action,
            ]
        )
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)
        self.select_tool_action = QAction("选择  V", self)
        self.draw_tool_action = QAction("框选  W", self)
        self.pan_tool_action = QAction("平移  Space", self)
        for action in (
            self.select_tool_action,
            self.draw_tool_action,
            self.pan_tool_action,
        ):
            action.setCheckable(True)
            self.tool_action_group.addAction(action)
        self.select_tool_action.setChecked(True)

        self.undo_action = QAction("撤销  Ctrl+Z", self)
        self.redo_action = QAction("重做  Ctrl+Y", self)
        self.undo_all_action = QAction("全部撤销  Ctrl+Alt+Z", self)
        self.delete_box_action = QAction("删除框  S", self)
        self.delete_all_annotations_action = QAction(
            "删除所有标记  Ctrl+Shift+Delete", self
        )
        self.fit_action = QAction("适应窗口  F", self)
        self.previous_image_action = QAction("上一张  A", self)
        self.verify_next_action = QAction("确认并下一张  D", self)
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.undo_all_action.setEnabled(False)

        self.annotation_display_action_group = QActionGroup(self)
        self.annotation_display_action_group.setExclusive(True)
        self.annotation_display_full_action = QAction("完整显示", self)
        self.annotation_display_boxes_action = QAction("仅显示框", self)
        self.annotation_display_hidden_action = QAction("全部隐藏", self)
        for action, mode in (
            (self.annotation_display_full_action, AnnotationDisplayMode.FULL.value),
            (self.annotation_display_boxes_action, AnnotationDisplayMode.BOX_ONLY.value),
            (self.annotation_display_hidden_action, AnnotationDisplayMode.HIDDEN.value),
        ):
            action.setCheckable(True)
            action.setData(mode)
            self.annotation_display_action_group.addAction(action)
        self.annotation_display_full_action.setChecked(True)

    def _build_ui(self) -> None:
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("mainSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_center_panel())
        self.main_splitter.addWidget(self._build_right_panel())
        for index in range(3):
            self.main_splitter.setCollapsible(index, False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([260, 820, 400])
        self.setCentralWidget(self.main_splitter)

    def _resize_for_available_screen(self) -> None:
        application = QApplication.instance()
        screen = self.screen() or (application.primaryScreen() if application else None)
        if screen is None:
            self.resize(1280, 800)
            return
        available = screen.availableGeometry()
        width = min(1500, max(self.minimumWidth(), int(available.width() * 0.95)))
        height = min(900, max(self.minimumHeight(), int(available.height() * 0.95)))
        self.resize(width, height)

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        return panel

    def _build_left_panel(self) -> QWidget:
        panel = self._panel()
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        title = QLabel("数据项目")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title)
        self.project_label = QLabel("尚未打开项目")
        self.project_label.setObjectName("projectPathLabel")
        self.project_label.setWordWrap(True)
        self.project_label.setStyleSheet(f"color: {COLORS['muted']};")
        layout.addWidget(self.project_label)
        project_row = QHBoxLayout()
        self.new_project_button = QPushButton("新建")
        self.open_project_button = QPushButton("打开")
        self.import_images_button = QPushButton("导入图片")
        project_row.addWidget(self.new_project_button)
        project_row.addWidget(self.open_project_button)
        project_row.addWidget(self.import_images_button)
        layout.addLayout(project_row)
        self.import_voc_button = QPushButton("导入 MaixHub/VOC 混合数据")
        self.import_voc_button.setObjectName("importVocButton")
        layout.addWidget(self.import_voc_button)

        filter_group = QGroupBox("图片列表")
        filter_layout = QVBoxLayout(filter_group)
        self.image_filter_combo = QComboBox()
        self.image_filter_combo.setObjectName("imageStatusFilter")
        self.image_filter_combo.addItem("全部状态", "all")
        self.image_filter_combo.addItem("未标注", "unreviewed")
        self.image_filter_combo.addItem("AI 待复核", "draft")
        self.image_filter_combo.addItem("人工已确认", "verified")
        self.image_search_edit = QLineEdit()
        self.image_search_edit.setObjectName("imageSearchEdit")
        self.image_search_edit.setPlaceholderText("搜索文件名")
        filter_layout.addWidget(self.image_filter_combo)
        filter_layout.addWidget(self.image_search_edit)
        self.image_list = QListWidget()
        self.image_list.setObjectName("imageList")
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.setUniformItemSizes(True)
        filter_layout.addWidget(self.image_list, 1)
        range_row = QHBoxLayout()
        self.image_range_edit = QLineEdit()
        self.image_range_edit.setObjectName("imageRangeEdit")
        self.image_range_edit.setPlaceholderText("编号，如 1-50、88、900-999")
        self.image_range_edit.setToolTip(
            "支持英文逗号 ,、中文逗号 ，和顿号 、；范围包含首尾编号"
        )
        self.apply_image_range_button = QPushButton("应用选择")
        self.apply_image_range_button.setObjectName("applyImageRangeButton")
        range_row.addWidget(self.image_range_edit, 1)
        range_row.addWidget(self.apply_image_range_button)
        filter_layout.addLayout(range_row)
        selection_row = QHBoxLayout()
        self.select_all_visible_button = QPushButton("全选当前筛选")
        self.clear_image_selection_button = QPushButton("清除选择")
        self.select_to_here_button = QPushButton("选到这里")
        self.select_to_here_button.setObjectName("selectToHereButton")
        self.select_to_here_button.setCheckable(True)
        self.select_to_here_button.setToolTip(
            "先选中起点并点击此按钮，再点击终点；按项目全局编号连续多选"
        )
        selection_row.addWidget(self.select_all_visible_button)
        selection_row.addWidget(self.clear_image_selection_button)
        selection_row.addWidget(self.select_to_here_button)
        filter_layout.addLayout(selection_row)
        training_row = QHBoxLayout()
        self.training_only_selected_button = QPushButton("仅训练所选")
        self.training_include_button = QPushButton("加入训练")
        self.training_exclude_button = QPushButton("移出训练")
        training_row.addWidget(self.training_only_selected_button)
        training_row.addWidget(self.training_include_button)
        training_row.addWidget(self.training_exclude_button)
        filter_layout.addLayout(training_row)
        self.delete_selected_images_button = QPushButton("删除所选图片（可恢复）")
        self.delete_selected_images_button.setProperty("danger", True)
        filter_layout.addWidget(self.delete_selected_images_button)
        layout.addWidget(filter_group, 1)
        self.image_count_label = QLabel("0 张图片")
        self.image_count_label.setObjectName("imageCountLabel")
        self.image_count_label.setStyleSheet(f"color: {COLORS['muted']};")
        layout.addWidget(self.image_count_label)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(0)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 0, 5, 0)
        primary_toolbar = QHBoxLayout()
        secondary_toolbar = QHBoxLayout()
        self.select_tool_button = self._tool_button(self.select_tool_action)
        self.draw_tool_button = self._tool_button(self.draw_tool_action)
        self.pan_tool_button = self._tool_button(self.pan_tool_action)
        self.fit_button = self._tool_button(self.fit_action)
        self.previous_image_toolbar_button = self._tool_button(
            self.previous_image_action
        )
        self.undo_button = self._tool_button(self.undo_action)
        self.redo_button = self._tool_button(self.redo_action)
        self.undo_all_button = self._tool_button(self.undo_all_action)
        self.delete_box_button = self._tool_button(self.delete_box_action)
        self.delete_all_annotations_button = self._tool_button(
            self.delete_all_annotations_action
        )
        self.annotation_display_button = QToolButton()
        self.annotation_display_button.setObjectName("annotationDisplayButton")
        self.annotation_display_button.setText("标注显示")
        self.annotation_display_button.setToolTip("切换框、标签和置信度的显示方式")
        self.annotation_display_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.annotation_display_menu = QMenu(self.annotation_display_button)
        self.annotation_display_menu.addActions(
            [
                self.annotation_display_full_action,
                self.annotation_display_boxes_action,
                self.annotation_display_hidden_action,
            ]
        )
        self.annotation_display_button.setMenu(self.annotation_display_menu)
        for button in (
            self.select_tool_button,
            self.draw_tool_button,
            self.pan_tool_button,
            self.fit_button,
            self.previous_image_toolbar_button,
            self.annotation_display_button,
        ):
            primary_toolbar.addWidget(button)
        primary_toolbar.addStretch(1)
        for button in (
            self.undo_button,
            self.redo_button,
            self.undo_all_button,
            self.delete_box_button,
            self.delete_all_annotations_button,
        ):
            secondary_toolbar.addWidget(button)
        self.image_status_label = ElidingStatusLabel("请选择图片")
        self.image_status_label.setStyleSheet(f"color: {COLORS['muted']};")
        secondary_toolbar.addWidget(self.image_status_label, 1)
        layout.addLayout(primary_toolbar)
        layout.addLayout(secondary_toolbar)

        self.canvas = AnnotationCanvas()
        layout.addWidget(self.canvas, 1)
        footer = QHBoxLayout()
        self.zoom_label = QLabel("缩放 100%")
        self.coordinate_label = QLabel("原图像素坐标")
        self.shortcut_label = QLabel(
            "V 选择 · W 框选 · A 上一张 · S 删除框 · D 确认下一张"
        )
        self.shortcut_label.setStyleSheet(f"color: {COLORS['muted']};")
        footer.addWidget(self.zoom_label)
        footer.addStretch(1)
        footer.addWidget(self.coordinate_label)
        footer.addStretch(1)
        footer.addWidget(self.shortcut_label)
        layout.addLayout(footer)
        return panel

    @staticmethod
    def _tool_button(action: QAction) -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        return button

    def _build_right_panel(self) -> QWidget:
        panel = self._panel()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(560)
        layout = QVBoxLayout(panel)
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.addTab(self._scrollable_page(self._build_annotation_tab()), "标注")
        self.right_tabs.addTab(self._scrollable_page(self._build_training_tab()), "训练")
        self.right_tabs.addTab(self._scrollable_page(self._build_ai_tab()), "AI 标注")
        self.right_tabs.addTab(self._scrollable_page(self._build_deploy_tab()), "Maix 部署")
        layout.addWidget(self.right_tabs)
        return panel

    @staticmethod
    def _scrollable_page(page: QWidget) -> QScrollArea:
        page_layout = page.layout()
        if page_layout is not None:
            page_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        page.setMinimumWidth(0)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        scroll.setMinimumSize(0, 0)
        scroll.setWidget(page)
        return scroll

    def _build_annotation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        class_group = QGroupBox("类别")
        class_layout = QVBoxLayout(class_group)
        self.class_list = QListWidget()
        self.class_list.setObjectName("classList")
        self.class_list.setMaximumHeight(190)
        class_layout.addWidget(self.class_list)
        add_row = QHBoxLayout()
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setObjectName("newClassEdit")
        self.new_class_edit.setPlaceholderText("新类别名称")
        self.add_class_button = QPushButton("添加")
        self.delete_empty_category_button = QPushButton("删除空类别")
        self.delete_empty_category_button.setObjectName("deleteEmptyCategoryButton")
        self.delete_empty_category_button.setProperty("danger", True)
        self.delete_empty_category_button.setEnabled(False)
        add_row.addWidget(self.new_class_edit, 1)
        add_row.addWidget(self.add_class_button)
        add_row.addWidget(self.delete_empty_category_button)
        class_layout.addLayout(add_row)
        self.apply_class_button = QPushButton("将类别应用到选中框")
        class_layout.addWidget(self.apply_class_button)
        alias_row = QHBoxLayout()
        self.display_alias_edit = QLineEdit()
        self.display_alias_edit.setObjectName("categoryDisplayAliasEdit")
        self.display_alias_edit.setPlaceholderText("输入新的显示名称或规范类别名")
        self.set_display_alias_button = QPushButton("仅修改显示名称")
        self.set_display_alias_button.setObjectName("setCategoryDisplayAliasButton")
        self.rename_category_button = QPushButton("完整重命名类别")
        self.rename_category_button.setObjectName("renameCategoryCanonicalButton")
        self.clear_display_alias_button = QPushButton("恢复原名")
        self.clear_display_alias_button.setObjectName("clearCategoryDisplayAliasButton")
        self.display_alias_edit.setEnabled(False)
        self.set_display_alias_button.setEnabled(False)
        self.rename_category_button.setEnabled(False)
        self.clear_display_alias_button.setEnabled(False)
        alias_row.addWidget(self.display_alias_edit, 1)
        alias_row.addWidget(self.set_display_alias_button)
        alias_row.addWidget(self.rename_category_button)
        alias_row.addWidget(self.clear_display_alias_button)
        class_layout.addLayout(alias_row)
        alias_note = QLabel(
            "“仅修改显示名称”只影响当前界面；“完整重命名类别”会影响后续训练、"
            "导出和部署，但不会改写历史模型或已生成的部署包。"
        )
        alias_note.setWordWrap(True)
        alias_note.setStyleSheet(f"color: {COLORS['muted']};")
        class_layout.addWidget(alias_note)
        layout.addWidget(class_group)

        boxes_group = QGroupBox("当前图片的框")
        boxes_layout = QVBoxLayout(boxes_group)
        self.box_list = QListWidget()
        self.box_list.setObjectName("boxList")
        boxes_layout.addWidget(self.box_list, 1)
        box_actions = QHBoxLayout()
        self.delete_selected_box_button = QPushButton("删除选中框")
        self.delete_selected_box_button.setProperty("danger", True)
        self.save_button = QPushButton("保存")
        self.verify_next_button = QPushButton("确认并下一张  D")
        self.previous_image_button = QPushButton("上一张  A")
        self.verify_next_button.setProperty("primary", True)
        box_actions.addWidget(self.delete_selected_box_button)
        box_actions.addWidget(self.save_button)
        boxes_layout.addLayout(box_actions)
        boxes_layout.addWidget(self.previous_image_button)
        boxes_layout.addWidget(self.verify_next_button)
        layout.addWidget(boxes_group, 1)
        return page

    def _build_training_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        seed_group = QGroupBox("训练数据")
        seed_layout = QVBoxLayout(seed_group)
        self.seed_count_label = QLabel("已选入训练且人工确认：0 / 100")
        self.seed_count_label.setObjectName("seedCountLabel")
        self.seed_progress = QProgressBar()
        self.seed_progress.setRange(0, 100)
        self.seed_progress.setValue(0)
        seed_layout.addWidget(self.seed_count_label)
        seed_layout.addWidget(self.seed_progress)
        seed_note = QLabel(
            "训练只读取左侧已加入训练且人工确认的图片；AI 待复核草稿自动跳过，"
            "人工确认的空图片可作为负样本。"
        )
        seed_note.setWordWrap(True)
        seed_note.setStyleSheet(f"color: {COLORS['muted']};")
        seed_layout.addWidget(seed_note)
        self.training_snapshot_summary = QPlainTextEdit()
        self.training_snapshot_summary.setObjectName("trainingSnapshotSummary")
        self.training_snapshot_summary.setReadOnly(True)
        self.training_snapshot_summary.setMaximumBlockCount(500)
        self.training_snapshot_summary.setMinimumHeight(145)
        self.training_snapshot_summary.setPlainText(
            "训练前检查后将在这里显示选择成员与快照摘要。"
        )
        seed_layout.addWidget(self.training_snapshot_summary)
        layout.addWidget(seed_group)

        config_group = QGroupBox("模型与参数")
        config_layout = QFormLayout(config_group)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelCombo")
        for label, weight in MODEL_OPTIONS:
            self.model_combo.addItem(label, {"model_key": label, "weight": weight})
        self.model_combo.setCurrentIndex(
            next(index for index, item in enumerate(MODEL_OPTIONS) if item[0] == "YOLO26n")
        )
        self.settings_summary_label = QLabel()
        self.settings_summary_label.setWordWrap(True)
        self.advanced_settings_button = QPushButton("高级参数…")
        self.environment_button = QPushButton("ML 环境：自动检测")
        self.history_combo = QComboBox()
        self.history_combo.setObjectName("trainingRunCombo")
        self.history_combo.addItem("尚无成功训练", None)
        self.resume_combo = QComboBox()
        self.resume_combo.setObjectName("resumeRunCombo")
        self.resume_combo.addItem("尚无可恢复训练", None)
        config_layout.addRow("模型", self.model_combo)
        config_layout.addRow("当前参数", self.settings_summary_label)
        config_layout.addRow(self.advanced_settings_button)
        config_layout.addRow(self.environment_button)
        config_layout.addRow("历史模型", self.history_combo)
        config_layout.addRow("断点恢复", self.resume_combo)
        layout.addWidget(config_group)

        action_row = QHBoxLayout()
        self.train_button = QPushButton("一键训练")
        self.train_button.setObjectName("trainButton")
        self.train_button.setProperty("primary", True)
        self.retrain_button = QPushButton("重新训练")
        self.retrain_button.setObjectName("retrainButton")
        self.resume_training_button = QPushButton("恢复中断训练")
        self.resume_training_button.setObjectName("resumeTrainingButton")
        self.resume_training_button.setEnabled(False)
        self.cancel_job_button = QPushButton("取消任务")
        self.cancel_job_button.setEnabled(False)
        action_row.addWidget(self.train_button)
        action_row.addWidget(self.retrain_button)
        action_row.addWidget(self.resume_training_button)
        action_row.addWidget(self.cancel_job_button)
        layout.addLayout(action_row)

        self.training_monitor = TrainingMonitorWidget()
        layout.addWidget(self.training_monitor, 1)
        self._update_settings_summary()
        return page

    def _build_ai_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "先用至少 100 张人工确认图片训练模型。训练完成后，可选择历史 best.pt "
            "流式标注全部尚未人工确认的图片。结果会作为 AI 待复核草稿导入。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.ai_model_combo = QComboBox()
        self.ai_model_combo.setObjectName("aiRunCombo")
        self.ai_model_combo.addItem("请先训练模型", None)
        self.ai_confidence_spin = QDoubleSpinBox()
        self.ai_confidence_spin.setRange(0.01, 1.0)
        self.ai_confidence_spin.setSingleStep(0.05)
        self.ai_confidence_spin.setDecimals(2)
        self.ai_confidence_spin.setValue(0.25)
        self.ai_iou_spin = QDoubleSpinBox()
        self.ai_iou_spin.setRange(0.01, 1.0)
        self.ai_iou_spin.setSingleStep(0.05)
        self.ai_iou_spin.setDecimals(2)
        self.ai_iou_spin.setValue(0.7)
        self.ai_dedup_check = QCheckBox("自动清理同一目标的重框")
        self.ai_dedup_check.setChecked(True)
        self.ai_dedup_iou_spin = QDoubleSpinBox()
        self.ai_dedup_iou_spin.setRange(0.70, 0.95)
        self.ai_dedup_iou_spin.setSingleStep(0.05)
        self.ai_dedup_iou_spin.setDecimals(2)
        self.ai_dedup_iou_spin.setValue(0.80)
        self.ai_dedup_iou_spin.setToolTip(
            "仅比较同类别、未人工修改的 AI 框；达到阈值时保留置信度较高的框"
        )
        self.ai_resume_check = QCheckBox("跳过已完成图片，支持中断后继续")
        self.ai_resume_check.setChecked(True)
        form.addRow("历史成功模型", self.ai_model_combo)
        form.addRow("置信度阈值", self.ai_confidence_spin)
        form.addRow("NMS IoU", self.ai_iou_spin)
        form.addRow(self.ai_dedup_check)
        form.addRow("重框去重 IoU", self.ai_dedup_iou_spin)
        form.addRow(self.ai_resume_check)
        layout.addLayout(form)
        self.autolabel_button = QPushButton("AI 自动标注")
        self.autolabel_button.setObjectName("autolabelButton")
        self.autolabel_button.setProperty("primary", True)
        self.autolabel_button.setEnabled(False)
        layout.addWidget(self.autolabel_button)
        cleanup_group = QGroupBox("清理历史 AI 重框")
        cleanup_layout = QFormLayout(cleanup_group)
        self.historical_dedup_scope_combo = QComboBox()
        self.historical_dedup_scope_combo.setObjectName("historicalDedupScopeCombo")
        self.historical_dedup_scope_combo.addItem("当前多选图片", "selected")
        self.historical_dedup_scope_combo.addItem("当前筛选结果", "visible")
        self.historical_dedup_iou_spin = QDoubleSpinBox()
        self.historical_dedup_iou_spin.setObjectName("historicalDedupIouSpin")
        self.historical_dedup_iou_spin.setRange(0.70, 0.95)
        self.historical_dedup_iou_spin.setSingleStep(0.05)
        self.historical_dedup_iou_spin.setDecimals(2)
        self.historical_dedup_iou_spin.setValue(0.80)
        self.historical_dedup_button = QPushButton("预览并清理…")
        self.historical_dedup_button.setObjectName("historicalDedupButton")
        self.historical_dedup_button.setToolTip(
            "只比较同类别、未人工确认且未人工修改的纯 AI 草稿；执行前自动备份"
        )
        cleanup_layout.addRow("处理范围", self.historical_dedup_scope_combo)
        cleanup_layout.addRow("去重 IoU", self.historical_dedup_iou_spin)
        cleanup_layout.addRow(self.historical_dedup_button)
        layout.addWidget(cleanup_group)
        self.ai_progress = QProgressBar()
        self.ai_progress.setRange(0, 100)
        self.ai_status_label = QLabel("等待可用模型")
        self.ai_status_label.setWordWrap(True)
        layout.addWidget(self.ai_progress)
        layout.addWidget(self.ai_status_label)
        self.ai_log = QPlainTextEdit()
        self.ai_log.setReadOnly(True)
        self.ai_log.setMaximumBlockCount(3000)
        layout.addWidget(self.ai_log, 1)
        return page

    def _build_deploy_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("MaixCAM-Pro / MaixCAM2 部署")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title)
        intro = QLabel(
            "将训练成功的 best.pt 或 last.pt 转换为 MaixCAM-Pro 的 cvimodel，"
            "或 MaixCAM2 的 NPU2 / VNPU axmodel。首版统一使用项目内校准图做 INT8 量化。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        warning = QLabel(
            "部署包 ZIP 或解压总量超过 30,000,000 字节时会请求确认，但不会阻止继续生成。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {COLORS['warning']};")
        layout.addWidget(warning)

        docker_group = QGroupBox("转换环境（Docker Desktop）")
        docker_layout = QVBoxLayout(docker_group)
        docker_form = QFormLayout()
        self.docker_target_combo = QComboBox()
        self.docker_target_combo.addItem("MaixCAM-Pro（TPU-MLIR）", "maixcam_pro")
        self.docker_target_combo.addItem("MaixCAM2（Pulsar2）", "maixcam2")
        last_target = str(
            self._read("get_last_maix_target", "last_maix_target", default="maixcam2")
            or "maixcam2"
        )
        target_index = self.docker_target_combo.findData(last_target)
        self.docker_target_combo.setCurrentIndex(
            target_index
            if target_index >= 0
            else self.docker_target_combo.findData("maixcam2")
        )
        docker_form.addRow("检查目标", self.docker_target_combo)
        docker_layout.addLayout(docker_form)
        docker_buttons = QHBoxLayout()
        self.docker_detect_button = QPushButton("检测 Docker / WSL / 镜像")
        self.docker_start_button = QPushButton("启动 Docker Desktop 并重新检测")
        self.docker_import_button = QPushButton("导入镜像 tar…")
        self.docker_pull_button = QPushButton("拉取官方镜像…")
        docker_buttons.addWidget(self.docker_detect_button)
        docker_buttons.addWidget(self.docker_start_button)
        docker_buttons.addWidget(self.docker_import_button)
        docker_buttons.addWidget(self.docker_pull_button)
        docker_layout.addLayout(docker_buttons)
        self.docker_import_progress = QProgressBar()
        self.docker_import_progress.setRange(0, 100)
        self.docker_import_progress.setValue(0)
        self.docker_import_progress.setVisible(False)
        self.docker_import_detail_label = QLabel("")
        self.docker_import_detail_label.setWordWrap(True)
        self.docker_import_detail_label.setVisible(False)
        self.docker_cancel_button = QPushButton("取消镜像操作")
        self.docker_cancel_button.setEnabled(False)
        self.docker_cancel_button.setVisible(False)
        docker_layout.addWidget(self.docker_import_progress)
        docker_layout.addWidget(self.docker_import_detail_label)
        docker_layout.addWidget(self.docker_cancel_button)
        self.docker_status_label = QLabel(
            "尚未检测。转换前请确认 Docker daemon、WSL2、转换镜像和目录挂载均可用。"
        )
        self.docker_status_label.setWordWrap(True)
        self.docker_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        docker_layout.addWidget(self.docker_status_label)
        layout.addWidget(docker_group)

        self.deploy_model_label = QLabel("当前模型：YOLO26n")
        self.deploy_checkpoint_combo = QComboBox()
        self.deploy_checkpoint_combo.addItem("请先完成训练", None)
        self.deploy_wizard_button = QPushButton("打开部署向导…")
        self.deploy_wizard_button.setEnabled(False)
        self.deploy_wizard_button.setProperty("primary", True)
        layout.addWidget(self.deploy_model_label)
        layout.addWidget(self.deploy_checkpoint_combo)
        layout.addWidget(self.deploy_wizard_button)
        generation_note = QLabel(
            "本页负责转换模型并生成文件，不会自动把模型安装进设备。"
            "生成后可打开产物目录，再由 MaixVision 连接设备并安装。"
        )
        generation_note.setWordWrap(True)
        generation_note.setStyleSheet(f"color: {COLORS['warning']};")
        layout.addWidget(generation_note)
        deploy_output_buttons = QHBoxLayout()
        self.open_deploy_output_button = QPushButton("打开产物目录")
        self.open_deploy_output_button.setEnabled(False)
        self.open_maixapp_button = QPushButton("交给 MaixVision / 默认程序打开 .maixapp")
        self.open_maixapp_button.setEnabled(False)
        self.cleanup_backups_button = QPushButton("真机验证成功后清理旧备份…")
        self.cleanup_backups_button.setEnabled(False)
        deploy_output_buttons.addWidget(self.open_deploy_output_button)
        deploy_output_buttons.addWidget(self.open_maixapp_button)
        deploy_output_buttons.addWidget(self.cleanup_backups_button)
        layout.addLayout(deploy_output_buttons)
        self.deploy_progress = QProgressBar()
        self.deploy_progress.setRange(0, 100)
        self.deploy_status_label = QLabel("尚未开始转换")
        self.deploy_status_label.setWordWrap(True)
        self.deploy_log = QPlainTextEdit()
        self.deploy_log.setReadOnly(True)
        self.deploy_log.setMaximumBlockCount(3000)
        layout.addWidget(self.deploy_progress)
        layout.addWidget(self.deploy_status_label)
        layout.addWidget(self.deploy_log, 1)
        return page

    def _wire_signals(self) -> None:
        self.new_project_button.clicked.connect(self.new_project)
        self.open_project_button.clicked.connect(self.open_project)
        self.import_images_button.clicked.connect(self.import_images)
        self.import_voc_button.clicked.connect(self.import_voc_dataset)
        self.image_filter_combo.currentIndexChanged.connect(self.apply_image_filter)
        self.image_search_edit.textChanged.connect(self.apply_image_filter)
        self.image_list.currentItemChanged.connect(self._on_image_item_changed)
        self.image_list.itemClicked.connect(self._queue_select_to_here)
        self.select_all_visible_button.clicked.connect(
            self.select_all_visible_images
        )
        self.apply_image_range_button.clicked.connect(
            self.apply_image_range_selection
        )
        self.image_range_edit.returnPressed.connect(
            self.apply_image_range_selection
        )
        self.clear_image_selection_button.clicked.connect(
            self.clear_image_selection
        )
        self.select_to_here_button.toggled.connect(
            self._toggle_select_to_here
        )
        self.training_only_selected_button.clicked.connect(
            self.select_only_selected_for_training
        )
        self.training_include_button.clicked.connect(
            lambda: self.set_selected_images_training_state(True)
        )
        self.training_exclude_button.clicked.connect(
            lambda: self.set_selected_images_training_state(False)
        )
        self.delete_selected_images_button.clicked.connect(
            self.delete_selected_images
        )
        self.class_list.currentItemChanged.connect(self._on_class_changed)
        self.box_list.currentItemChanged.connect(self._on_box_list_changed)
        self.add_class_button.clicked.connect(self.add_class)
        self.new_class_edit.returnPressed.connect(self.add_class)
        self.delete_empty_category_button.clicked.connect(
            self.delete_selected_empty_category
        )
        self.apply_class_button.clicked.connect(self.apply_class_to_selected)
        self.set_display_alias_button.clicked.connect(self.set_category_display_alias)
        self.rename_category_button.clicked.connect(self.rename_category_canonical)
        self.clear_display_alias_button.clicked.connect(
            lambda: self.set_category_display_alias(clear=True)
        )
        self.display_alias_edit.returnPressed.connect(self.set_category_display_alias)
        self.delete_selected_box_button.clicked.connect(self.delete_box_action.trigger)
        self.save_button.clicked.connect(self.save_current_annotations)
        self.verify_next_button.clicked.connect(self.verify_next_action.trigger)
        self.previous_image_button.clicked.connect(
            self.previous_image_action.trigger
        )

        self.select_tool_action.triggered.connect(
            lambda _checked=False: self.canvas.set_tool(AnnotationCanvas.TOOL_SELECT)
        )
        self.draw_tool_action.triggered.connect(
            lambda _checked=False: self.canvas.set_tool(AnnotationCanvas.TOOL_DRAW)
        )
        self.pan_tool_action.triggered.connect(
            lambda _checked=False: self.canvas.set_tool(AnnotationCanvas.TOOL_PAN)
        )
        self.undo_action.triggered.connect(self.canvas.undo)
        self.redo_action.triggered.connect(self.canvas.redo)
        self.undo_all_action.triggered.connect(self.undo_all_current_image)
        self.delete_box_action.triggered.connect(self.canvas.delete_selected)
        self.delete_all_annotations_action.triggered.connect(self.clear_all_annotations)
        self.fit_action.triggered.connect(self.canvas.fit_image)
        self.previous_image_action.triggered.connect(self.previous_image)
        self.verify_next_action.triggered.connect(self.verify_and_next)
        self.annotation_display_action_group.triggered.connect(
            self._on_annotation_display_action
        )
        self.canvas.toolChanged.connect(self._on_canvas_tool_changed)
        self.canvas.undoAvailableChanged.connect(self.undo_action.setEnabled)
        self.canvas.undoAvailableChanged.connect(self.undo_all_action.setEnabled)
        self.canvas.redoAvailableChanged.connect(self.redo_action.setEnabled)
        self.canvas.annotationDisplayModeChanged.connect(
            self._on_annotation_display_mode_changed
        )
        self.canvas.zoomChanged.connect(
            lambda zoom: self.zoom_label.setText(f"缩放 {zoom * 100:.0f}%")
        )
        self.canvas.annotationsChanged.connect(self._on_annotations_changed)
        self.canvas.selectionChanged.connect(self._on_canvas_selection_changed)
        self.canvas.statusMessage.connect(self._set_status)
        self.historical_dedup_button.clicked.connect(
            self.cleanup_historical_ai_duplicates
        )

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.history_combo.currentIndexChanged.connect(
            self._on_training_history_selection_changed
        )
        self.resume_combo.currentIndexChanged.connect(
            self._on_training_history_selection_changed
        )
        self.advanced_settings_button.clicked.connect(self.open_training_settings)
        self.environment_button.clicked.connect(self.open_environment_dialog)
        self.train_button.clicked.connect(lambda: self.start_training(retrain=False))
        self.retrain_button.clicked.connect(lambda: self.start_training(retrain=True))
        self.resume_training_button.clicked.connect(self.resume_selected_training)
        self.cancel_job_button.clicked.connect(self.cancel_active_job)
        self.autolabel_button.clicked.connect(self.start_autolabel)
        self.deploy_checkpoint_combo.currentIndexChanged.connect(
            self._on_deploy_checkpoint_changed
        )
        self.deploy_wizard_button.clicked.connect(self.open_deploy_wizard)
        self.open_deploy_output_button.clicked.connect(self.open_deploy_output_directory)
        self.open_maixapp_button.clicked.connect(self.open_generated_maixapp)
        self.cleanup_backups_button.clicked.connect(self.cleanup_old_backups_after_deploy)
        self.docker_target_combo.currentIndexChanged.connect(
            self._persist_docker_target
        )
        self.docker_detect_button.clicked.connect(self.inspect_docker_environment)
        self.docker_start_button.clicked.connect(self.start_docker_desktop)
        self.docker_import_button.clicked.connect(self.import_docker_image)
        self.docker_pull_button.clicked.connect(self.pull_docker_image)
        self.docker_cancel_button.clicked.connect(self.cancel_active_job)

        self.v_shortcut = self._register_shortcut("V", self._v_shortcut)
        self.w_shortcut = self._register_shortcut("W", self._w_shortcut)
        self.d_shortcut = self._register_shortcut("D", self._d_shortcut)
        self.a_shortcut = self._register_shortcut("A", self._a_shortcut)
        self.fit_shortcut = self._register_shortcut("F", self._fit_shortcut)
        self.delete_shortcut = self._register_shortcut("S", self._delete_shortcut)
        self.undo_shortcut = self._register_shortcut(
            QKeySequence(QKeySequence.StandardKey.Undo),
            self._undo_shortcut,
        )
        self.redo_shortcut = self._register_shortcut(
            QKeySequence(QKeySequence.StandardKey.Redo),
            self._redo_shortcut,
        )
        self.undo_all_shortcut = self._register_shortcut(
            "Ctrl+Alt+Z",
            self._undo_all_shortcut,
        )
        self.delete_all_annotations_shortcut = self._register_shortcut(
            "Ctrl+Shift+Delete",
            self._delete_all_annotations_shortcut,
        )
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def _connect_controller_signals(self) -> None:
        if self.controller is None:
            return
        sources = [self.controller]
        job_manager = getattr(self.controller, "job_manager", None)
        if job_manager is not None:
            sources.append(job_manager)
        for source in sources:
            aggregate_connected = False
            for signal_name in ("event", "job_event", "eventReceived"):
                signal = getattr(source, signal_name, None)
                if signal is not None and hasattr(signal, "connect"):
                    signal.connect(self._on_controller_event)
                    aggregate_connected = True
                    break
            if aggregate_connected:
                continue
            for signal_name, event_type in (
                ("progress", "progress"),
                ("metrics", "metrics"),
                ("prediction", "prediction"),
                ("log", "log"),
                ("error", "error"),
                ("finished", "completed"),
            ):
                signal = getattr(source, signal_name, None)
                if signal is not None and hasattr(signal, "connect"):
                    signal.connect(
                        lambda payload, kind=event_type: self._on_controller_payload(
                            kind, payload
                        )
                    )

    @Slot(dict)
    def _on_process_event(self, event: dict[str, Any]) -> None:
        """Persist worker output through the controller before updating widgets."""

        handler = self._member("handle_job_event", "consume_job_event", "import_job_event")
        if (
            callable(handler)
            and not self._forwarding_controller_event
        ):
            self._forwarding_controller_event = True
            try:
                handler(event)
            except Exception as exc:  # persistence boundary; keep the worker/UI alive
                self.training_monitor.append_log(f"任务事件持久化失败：{exc}")
                self._set_status(f"任务事件持久化失败：{exc}")
            finally:
                self._forwarding_controller_event = False
        if str(event.get("type", "")) == "process_finished":
            try:
                self._finalize_process_run(event)
            except Exception as exc:
                self.training_monitor.append_log(f"任务收尾持久化失败：{exc}")
                self._set_status(f"任务收尾持久化失败：{exc}")
        self.on_job_event(event)

    def _finalize_process_run(self, event: Mapping[str, Any]) -> None:
        finisher = self._member("handle_process_finished")
        if not callable(finisher):
            return
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {}
        job_id = str(event.get("job_id") or self._active_job_id or "")
        if not job_id:
            return
        values: dict[str, Any] = {
            "success": bool(payload.get("success")),
            "exit_code": int(payload.get("exit_code", -1)),
            "cancelled": bool(payload.get("cancelled")),
            "completed_epochs": self.training_monitor._completed_epochs,
            "requested_epochs": self.training_monitor._requested_epochs,
        }
        try:
            signature = inspect.signature(finisher)
        except (TypeError, ValueError):
            finisher(job_id, success=values["success"], exit_code=values["exit_code"])
            return
        parameters = signature.parameters
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs = {
            key: value
            for key, value in values.items()
            if accepts_keywords or key in parameters
        }
        finisher(job_id, **kwargs)

    @Slot(dict)
    def _on_controller_event(self, event: dict[str, Any]) -> None:
        if not self._forwarding_controller_event:
            self.on_job_event(event)

    def _on_controller_payload(self, event_type: str, payload: object) -> None:
        if isinstance(payload, Mapping) and {
            "type",
            "payload",
        }.issubset(payload):
            self._on_controller_event(dict(payload))
            return
        data = dict(payload) if isinstance(payload, Mapping) else {"message": str(payload)}
        self._on_controller_event(
            {
                "protocol_version": UI_PROTOCOL_VERSION,
                "job_id": self._active_job_id,
                "seq": 0,
                "type": event_type,
                "payload": data,
            }
        )

    # ---- Controller helpers ---------------------------------------------

    def _member(self, *names: str) -> Any:
        for name in names:
            if self.controller is not None and hasattr(self.controller, name):
                return getattr(self.controller, name)
        return None

    def _read(self, *names: str, default: Any = None) -> Any:
        member = self._member(*names)
        if member is None:
            return default
        return member() if callable(member) else member

    @staticmethod
    def _invoke_by_arity(callable_object: Any, variants: Sequence[tuple[Any, ...]]) -> Any:
        """Choose a compatible argument tuple without masking controller errors."""

        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            return callable_object(*variants[0])
        for arguments in variants:
            try:
                signature.bind(*arguments)
            except TypeError:
                continue
            return callable_object(*arguments)
        return callable_object(*variants[0])

    # ---- Project and records --------------------------------------------

    def refresh_project(self, *, select_image_id: object | None = None) -> None:
        project = self._read("current_project", "project", default=None)
        if project is not None:
            root = _value(project, "root", _value(project, "path", project))
            self.project_label.setText(str(root))
            self.setWindowTitle(
                f"AI 数据集标注与训练 维护版 {__version__} — {Path(str(root)).name}"
            )
        self.refresh_categories()
        self.refresh_images(select_image_id=select_image_id)
        self.refresh_runs()

    def refresh_categories(self) -> None:
        records = _as_records(self._read("list_classes", "list_categories", default=[]))
        self._category_records = records
        current_id = (
            self.class_list.currentItem().data(Qt.ItemDataRole.UserRole)
            if hasattr(self, "class_list") and self.class_list.currentItem()
            else None
        )
        self.class_list.clear()
        for index, record in enumerate(records):
            class_id = str(_value(record, "id", index))
            canonical_name = str(_value(record, "name", class_id))
            explicit_alias = str(_value(record, "display_name", "") or "").strip()
            effective_name = str(
                _value(record, "effective_display_name", "")
                or explicit_alias
                or canonical_name
            )
            label = (
                f"{canonical_name} → {effective_name}"
                if explicit_alias and effective_name != canonical_name
                else canonical_name
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, canonical_name)
            item.setData(Qt.ItemDataRole.UserRole + 2, explicit_alias)
            item.setToolTip(
                f"训练规范名：{canonical_name}\n显示名称：{effective_name}"
            )
            raw_color = _value(record, "color")
            color = QColor(str(raw_color)) if raw_color else class_color(index)
            item.setForeground(color)
            self.class_list.addItem(item)
            if class_id == str(current_id):
                self.class_list.setCurrentItem(item)
        self.canvas.set_categories(records)
        if self.class_list.count() and self.class_list.currentRow() < 0:
            self.class_list.setCurrentRow(0)

    def refresh_images(
        self,
        *,
        select_image_id: object | None = None,
        navigate: bool = True,
    ) -> None:
        preserved_selection = set(self.selected_image_ids())
        current_id = (
            _value(self._current_image, "id")
            if self._current_image is not None
            else None
        )
        if select_image_id is None:
            select_image_id = current_id
        records = _as_records(self._read("list_images", "images", default=[]))
        self._image_records = records
        selected_item: QListWidgetItem | None = None
        selected_record: object | None = None
        blocker = QSignalBlocker(self.image_list)
        try:
            self.image_list.clear()
            for index, record in enumerate(records):
                image_id = _value(record, "id", index)
                name = str(
                    _value(
                        record,
                        "original_name",
                        Path(
                            str(_value(record, "relative_path", f"图片 {index + 1}"))
                        ).name,
                    )
                )
                status = _enum_text(_value(record, "review_status"), "unreviewed")
                ai_status = _enum_text(_value(record, "ai_status"), "none")
                prefix = {"verified": "✓", "draft": "AI", "unreviewed": "○"}.get(
                    status, "○"
                )
                if ai_status == "failed":
                    prefix = "!"
                selected_for_training = bool(
                    _value(record, "training_selected", True)
                )
                training_prefix = "T" if selected_for_training else "–"
                stable_index = index + 1
                item = QListWidgetItem(
                    f"[{stable_index}] {training_prefix} {prefix}  {name}"
                )
                item.setData(Qt.ItemDataRole.UserRole, record)
                item.setData(Qt.ItemDataRole.UserRole + 1, str(image_id))
                item.setData(Qt.ItemDataRole.UserRole + 2, stable_index)
                item.setToolTip(
                    f"项目全局编号：{stable_index}\n{name}\n"
                    f"状态：{status} / AI：{ai_status}\n"
                    f"参与训练：{'是' if selected_for_training else '否'}"
                )
                self.image_list.addItem(item)
                if str(image_id) in preserved_selection:
                    item.setSelected(True)
                if select_image_id is not None and str(image_id) == str(select_image_id):
                    selected_item = item
            self.apply_image_filter()
            if selected_item is None or selected_item.isHidden():
                selected_item = self._first_visible_image_item()
            if selected_item is not None:
                self.image_list.setCurrentItem(selected_item)
                if preserved_selection:
                    for index in range(self.image_list.count()):
                        item = self.image_list.item(index)
                        item.setSelected(
                            item.data(Qt.ItemDataRole.UserRole + 1)
                            in preserved_selection
                        )
                selected_record = selected_item.data(Qt.ItemDataRole.UserRole)
        finally:
            del blocker
        self._update_seed_count()
        if not records:
            self._current_image = None
            self._annotation_session_baseline = None
            self._loading_image = True
            try:
                self.canvas.set_image(None)
            finally:
                self._loading_image = False
            self.image_status_label.set_full_text("请选择图片")
            return
        if navigate and selected_record is not None:
            self.navigate_to_image(selected_record, synchronize_selection=False)
        elif current_id is not None:
            refreshed_current = self._record_by_id(current_id)
            if refreshed_current is not None:
                self._current_image = refreshed_current
                self._update_current_image_status(refreshed_current)

    def _first_visible_image_item(self) -> QListWidgetItem | None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is not None and not item.isHidden():
                return item
        return None

    def selected_image_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole + 1))
            for item in self.image_list.selectedItems()
        )

    def visible_image_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole + 1))
            for row in range(self.image_list.count())
            if (item := self.image_list.item(row)) is not None
            and not item.isHidden()
        )

    def apply_image_range_selection(self) -> bool:
        """Replace list selection from strict stable, project-global indices."""

        try:
            selected_indices = parse_image_index_expression(
                self.image_range_edit.text(),
                self.image_list.count(),
            )
        except TrainingSelectionError as exc:
            # Parse completely before touching any selection or training state.
            self._show_error("图片编号范围无效", exc)
            return False
        wanted = set(selected_indices)
        blocker = QSignalBlocker(self.image_list)
        try:
            for row in range(self.image_list.count()):
                item = self.image_list.item(row)
                if item is not None:
                    item.setSelected(
                        int(item.data(Qt.ItemDataRole.UserRole + 2)) in wanted
                    )
        finally:
            del blocker
        self._set_status(
            f"已按项目全局编号选择 {len(selected_indices)} 张图片；"
            "筛选不会改变这些编号"
        )
        return True

    def clear_image_selection(self) -> None:
        self.image_list.clearSelection()
        self.image_range_edit.clear()
        if self.select_to_here_button.isChecked():
            self.select_to_here_button.setChecked(False)
        else:
            self._selection_anchor_index = None
            self._selection_anchor_existing_indices.clear()
        self._set_status("已清除图片多选；训练候选状态未改变")

    def _toggle_select_to_here(self, checked: bool) -> None:
        if not checked:
            self._selection_anchor_index = None
            self._selection_anchor_existing_indices.clear()
            self.select_to_here_button.setText("选到这里")
            return
        current = self.image_list.currentItem()
        if current is None:
            blocker = QSignalBlocker(self.select_to_here_button)
            try:
                self.select_to_here_button.setChecked(False)
            finally:
                del blocker
            self._show_error("无法连续选择", "请先点击一张图片作为起点。")
            return
        self._selection_anchor_index = int(
            current.data(Qt.ItemDataRole.UserRole + 2)
        )
        self._selection_anchor_existing_indices = {
            int(item.data(Qt.ItemDataRole.UserRole + 2))
            for item in self.image_list.selectedItems()
        }
        self.select_to_here_button.setText("请选择终点…")
        self._set_status(
            f"连续选择起点为第 {self._selection_anchor_index} 张；请点击终点"
        )

    def _queue_select_to_here(self, current: QListWidgetItem) -> None:
        """Finish a pending range after Qt's ordinary click selection settles."""

        if (
            self._selection_anchor_index is None
            or not self.select_to_here_button.isChecked()
        ):
            return
        current_index = int(current.data(Qt.ItemDataRole.UserRole + 2))
        QTimer.singleShot(
            0,
            lambda endpoint=current_index: self._complete_select_to_here(endpoint),
        )

    def _complete_select_to_here(self, current_index: int) -> None:
        anchor = self._selection_anchor_index
        if anchor is None or not self.select_to_here_button.isChecked():
            return
        try:
            wanted = set(
                select_from_anchor(anchor, current_index, self.image_list.count())
            )
        except TrainingSelectionError as exc:
            self.select_to_here_button.setChecked(False)
            self._show_error("无法连续选择", exc)
            return
        wanted.update(self._selection_anchor_existing_indices)
        blocker = QSignalBlocker(self.image_list)
        try:
            for row in range(self.image_list.count()):
                item = self.image_list.item(row)
                if (
                    item is not None
                    and int(item.data(Qt.ItemDataRole.UserRole + 2)) in wanted
                ):
                    item.setSelected(True)
        finally:
            del blocker
        range_start = min(anchor, current_index)
        range_end = max(anchor, current_index)
        range_count = range_end - range_start + 1
        self.select_to_here_button.setChecked(False)
        self._set_status(
            f"已选择 {range_start}～{range_end}，共{range_count}张图片"
        )

    def select_all_visible_images(self) -> None:
        blocker = QSignalBlocker(self.image_list)
        try:
            for row in range(self.image_list.count()):
                item = self.image_list.item(row)
                if item is not None:
                    item.setSelected(not item.isHidden())
        finally:
            del blocker
        self._set_status(
            f"已选择当前筛选中的 {len(self.selected_image_ids())} 张图片"
        )

    def set_selected_images_training_state(self, selected: bool) -> bool:
        image_ids = self.selected_image_ids()
        if not image_ids:
            self._show_error("无法修改训练样本", "请先在左侧选择至少一张图片。")
            return False
        setter = self._member("set_training_selected")
        if not callable(setter):
            self._show_error("无法修改训练样本", "控制器未实现训练样本选择功能。")
            return False
        try:
            setter(image_ids, selected)
        except Exception as exc:
            self._show_error("修改训练样本失败", exc)
            return False
        current_id = (
            _value(self._current_image, "id")
            if self._current_image is not None
            else None
        )
        self.refresh_images(select_image_id=current_id, navigate=False)
        self._set_status(
            f"已将 {len(image_ids)} 张图片"
            f"{'加入' if selected else '移出'}训练候选；"
            "实际训练仍只使用人工已确认图片"
        )
        return True

    def select_only_selected_for_training(self) -> bool:
        image_ids = self.selected_image_ids()
        if not image_ids:
            self._show_error("无法设置训练样本", "请先在左侧选择至少一张图片。")
            return False
        if self.isVisible():
            response = QMessageBox.question(
                self,
                "仅训练所选图片",
                f"将把这 {len(image_ids)} 张设为训练候选，并把项目中其他图片"
                "移出训练候选。\n\n实际训练仍会自动跳过未人工确认图片。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return False
        setter = self._member("select_only_for_training")
        if not callable(setter):
            self._show_error("无法设置训练样本", "控制器未实现精确训练样本选择。")
            return False
        try:
            setter(image_ids)
        except Exception as exc:
            self._show_error("设置训练样本失败", exc)
            return False
        current_id = (
            _value(self._current_image, "id")
            if self._current_image is not None
            else None
        )
        self.refresh_images(select_image_id=current_id, navigate=False)
        self._set_status(
            f"训练候选已限定为所选 {len(image_ids)} 张；"
            "开始训练时只取其中人工已确认的图片"
        )
        return True

    def delete_selected_images(self) -> bool:
        image_ids = self.selected_image_ids()
        if not image_ids:
            self._show_error("无法删除图片", "请先在左侧选择至少一张图片。")
            return False
        previewer = self._member("preview_delete_images")
        deleter = self._member("delete_images")
        if not callable(previewer) or not callable(deleter):
            self._show_error("无法删除图片", "控制器未实现可恢复的批量删除功能。")
            return False
        current_id = (
            str(_value(self._current_image, "id"))
            if self._current_image is not None
            else None
        )
        if (
            current_id in image_ids
            and self._annotations_dirty
            and not self.save_current_annotations(silent=True)
        ):
            return False
        try:
            preview = previewer(image_ids)
        except Exception as exc:
            self._show_error("删除前检查失败", exc)
            return False
        image_count = int(_value(preview, "image_count", len(image_ids)))
        box_count = int(_value(preview, "box_count", 0))
        if self.isVisible():
            response = QMessageBox.question(
                self,
                "确认删除所选图片",
                f"将从项目中删除 {image_count} 张图片及其 {box_count} 个标注框。"
                "\n\n软件会先备份数据库和图片，可恢复。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return False
        try:
            report = deleter(image_ids)
        except Exception as exc:
            self._show_error("批量删除图片失败", exc)
            return False
        if current_id in image_ids:
            self._current_image = None
            self._annotations_dirty = False
        self.refresh_project()
        backup = _value(report, "backup", {})
        backup_path = _value(backup, "path", "")
        archive_path = _value(report, "archive_path", "")
        message = f"已从项目删除 {image_count} 张图片。"
        if backup_path:
            message += f" 数据库备份：{backup_path}。"
        if archive_path:
            message += f" 图片备份：{archive_path}。"
        self._set_status(message)
        if self.isVisible():
            QMessageBox.information(self, "批量删除完成", message)
        return True

    def apply_image_filter(self) -> None:
        wanted = str(self.image_filter_combo.currentData() or "all")
        query = self.image_search_edit.text().strip().casefold()
        visible = 0
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            record = item.data(Qt.ItemDataRole.UserRole)
            status = _enum_text(_value(record, "review_status"), "unreviewed")
            name = str(_value(record, "original_name", item.text())).casefold()
            hidden = (wanted != "all" and status != wanted) or bool(
                query and query not in name
            )
            item.setHidden(hidden)
            if not hidden:
                visible += 1
        self.image_count_label.setText(f"显示 {visible} / {self.image_list.count()} 张图片")

    def _on_image_item_changed(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        if current is None or self._navigation_in_progress:
            return
        if not self.navigate_to_image(
            current.data(Qt.ItemDataRole.UserRole),
            synchronize_selection=False,
        ) and previous is not None:
            blocker = QSignalBlocker(self.image_list)
            try:
                self.image_list.setCurrentItem(previous)
            finally:
                del blocker

    def open_image(self, record: object) -> bool:
        """Compatibility alias for the unified navigation entry point."""

        return self.navigate_to_image(record)

    def navigate_to_image(
        self,
        image: object,
        *,
        verify_current: bool = False,
        synchronize_selection: bool = True,
    ) -> bool:
        if self._navigation_in_progress:
            return False
        record = self._resolve_image_record(image)
        if record is None:
            return False
        target_id = _value(record, "id")
        current_id = (
            _value(self._current_image, "id")
            if self._current_image is not None
            else None
        )
        if (
            current_id is not None
            and str(current_id) == str(target_id)
            and self.canvas.has_image
        ):
            self._current_image = record
            self._update_current_image_status(record)
            if synchronize_selection:
                self._sync_image_list_selection(target_id)
            return True

        self._navigation_in_progress = True
        self._autosave_timer.stop()
        try:
            if verify_current and not self._verify_current_annotations():
                return False
            if (
                self._current_image is not None
                and not self._loading_image
                and self._annotations_dirty
                and not self.save_current_annotations(silent=True)
            ):
                return False
            image_path = self._resolve_image_path(record)
            boxes = _as_records(self._boxes_for_image(target_id))
            self._loading_image = True
            try:
                loaded = (
                    self.canvas.load_image(image_path, boxes)
                    if image_path
                    else False
                )
            finally:
                self._loading_image = False
            if not loaded:
                self._set_status(
                    "无法打开图片："
                    f"{image_path or _value(record, 'original_name', target_id)}"
                )
                if current_id is not None:
                    self._sync_image_list_selection(current_id)
                return False
            self._current_image = record
            self._annotations_dirty = False
            self._capture_annotation_session_baseline(record)
            self._update_current_image_status(record)
            self.refresh_box_list()
            if synchronize_selection:
                self._sync_image_list_selection(target_id)
            return True
        finally:
            self._navigation_in_progress = False

    def _resolve_image_record(self, image: object) -> object | None:
        if isinstance(image, Mapping) or hasattr(image, "id"):
            return image
        return self._record_by_id(image)

    def _record_by_id(self, image_id: object) -> object | None:
        return next(
            (
                record
                for record in self._image_records
                if str(_value(record, "id")) == str(image_id)
            ),
            None,
        )

    def _sync_image_list_selection(self, image_id: object) -> None:
        blocker = QSignalBlocker(self.image_list)
        try:
            for index in range(self.image_list.count()):
                item = self.image_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole + 1) == str(image_id):
                    self.image_list.setCurrentItem(item)
                    return
        finally:
            del blocker

    def _update_current_image_status(self, record: object) -> None:
        image_path = self._resolve_image_path(record)
        status = _enum_text(_value(record, "review_status"), "unreviewed")
        origin = _enum_text(_value(record, "origin"), "none")
        name = _value(record, "original_name", Path(str(image_path)).name)
        self.image_status_label.set_full_text(f"{name} · {status} · {origin}")

    def _resolve_image_path(self, record: object) -> str | None:
        for field in ("absolute_path", "path", "image_path"):
            value = _value(record, field)
            if value:
                return str(value)
        resolver = self._member("image_path", "resolve_image_path")
        if callable(resolver):
            resolved = self._invoke_by_arity(
                resolver,
                [(_value(record, "id"),), (record,)],
            )
            if resolved:
                return str(resolved)
        relative = _value(record, "relative_path")
        if not relative:
            return None
        project = self._read("current_project", "project", default=None)
        root = _value(project, "root", _value(project, "path", project))
        return str(Path(str(root)) / str(relative)) if root else str(relative)

    def _boxes_for_image(self, image_id: object) -> object:
        method = self._member("get_boxes", "boxes_for_image", "list_boxes")
        if callable(method):
            return method(image_id)
        store = getattr(self.controller, "store", None)
        if store is not None:
            method = getattr(store, "boxes_for_image", None)
            if callable(method):
                return method(image_id)
        return []

    def _capture_annotation_session_baseline(self, record: object) -> None:
        """Remember exactly what was loaded for Ctrl+Alt+Z in this session."""

        image_id = str(_value(record, "id", "")).strip()
        if not image_id:
            self._annotation_session_baseline = None
            return
        revision: int | None = None
        raw_revision = _value(record, "revision")
        try:
            revision = int(raw_revision) if raw_revision is not None else None
        except (TypeError, ValueError):
            revision = None
        self._annotation_session_baseline = {
            "image_id": image_id,
            "boxes": [dict(box) for box in self.canvas.annotations()],
            "review_status": _enum_text(
                _value(record, "review_status"), "unreviewed"
            ),
            "origin": _enum_text(_value(record, "origin"), "none"),
            "ai_status": _enum_text(_value(record, "ai_status"), "none"),
            "expected_revision": revision,
            "record": record,
        }

    def _session_baseline_for_current_image(self) -> dict[str, Any] | None:
        baseline = self._annotation_session_baseline
        if baseline is None or self._current_image is None:
            return None
        current_id = str(_value(self._current_image, "id", "")).strip()
        return baseline if current_id and baseline.get("image_id") == current_id else None

    def _update_session_baseline_revision(self, image_id: object) -> None:
        """Advance the local optimistic revision after this window saves."""

        baseline = self._annotation_session_baseline
        image_id_text = str(image_id)
        if baseline is None or baseline.get("image_id") != image_id_text:
            return
        reader = self._member("get_image", "image_by_id")
        if not callable(reader):
            return
        try:
            persisted = reader(image_id_text)
            revision = _value(persisted, "revision")
            baseline["expected_revision"] = int(revision)
        except (TypeError, ValueError):
            return
        except Exception:
            # Saving itself has already succeeded.  Keeping the old revision
            # causes the later restore to fail safely rather than overwrite a
            # state we cannot verify.
            return

    def _on_annotations_changed(self, boxes: list[dict[str, Any]]) -> None:
        del boxes
        self.refresh_box_list()
        if self._restoring_annotation_session_baseline:
            return
        if not self._loading_image and self._current_image is not None:
            self._annotations_dirty = True
            image_id = str(_value(self._current_image, "id", ""))
            if image_id:
                self._modified_image_ids.add(image_id)
            self.image_status_label.setText("当前图片有人工修改，保存后仍需按 D 确认")
            self._autosave_timer.start()

    def refresh_box_list(self) -> None:
        selected = self.canvas.selected_box()
        selected_id = selected["id"] if selected else None
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for index, box in enumerate(self.canvas.annotations(), start=1):
            class_name = self.canvas.class_style(str(box["class_id"]))[0]
            confidence = (
                f" · {float(box['confidence']):.2f}" if box.get("confidence") is not None else ""
            )
            item = QListWidgetItem(
                f"{index}. {class_name or '未分类'}"
                f" [{box['xmin']:.0f},{box['ymin']:.0f} → {box['xmax']:.0f},{box['ymax']:.0f}]"
                f"{confidence}"
            )
            item.setData(Qt.ItemDataRole.UserRole, box["id"])
            self.box_list.addItem(item)
            if box["id"] == selected_id:
                self.box_list.setCurrentItem(item)
        self.box_list.blockSignals(False)

    def _on_box_list_changed(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        del previous
        self.canvas.select_box(current.data(Qt.ItemDataRole.UserRole) if current else None)

    def _on_canvas_selection_changed(self, box: object) -> None:
        annotation_id = box.get("id") if isinstance(box, Mapping) else None
        self.box_list.blockSignals(True)
        try:
            for index in range(self.box_list.count()):
                item = self.box_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == annotation_id:
                    self.box_list.setCurrentItem(item)
                    break
            else:
                self.box_list.setCurrentItem(None)
        finally:
            self.box_list.blockSignals(False)

    def save_current_annotations(self, *, silent: bool = False) -> bool:
        if self._current_image is None:
            return False
        if not self._annotations_dirty:
            if not silent:
                self._set_status("当前图片没有需要保存的修改")
            return True
        image_id = _value(self._current_image, "id")
        boxes = self.canvas.annotations()
        method = self._member("save_boxes", "replace_boxes")
        store = getattr(self.controller, "store", None)
        if not callable(method) and store is not None:
            method = getattr(store, "replace_boxes", None)
        if not callable(method):
            return False
        try:
            method(image_id, boxes)
        except Exception as exc:  # controller boundary
            self._show_error("保存标注失败", exc)
            return False
        self._annotations_dirty = False
        self._update_session_baseline_revision(image_id)
        self._update_seed_count()
        self.image_status_label.setText("标注已自动保存，待按 D 人工确认")
        if not silent:
            self._set_status(f"已保存 {len(boxes)} 个标注框")
        return True

    def verify_and_next(self) -> bool:
        if self._current_image is None:
            return False
        current_row = self.image_list.currentRow()
        image_id = _value(self._current_image, "id")
        next_image_id = self._next_visible_image_id(current_row)
        if not self._verify_current_annotations():
            return False
        self.refresh_images(select_image_id=image_id, navigate=False)
        if next_image_id is None:
            self._set_status("当前图片已人工确认；已到最后一张")
            return True
        next_record = self._record_by_id(next_image_id)
        if next_record is None:
            self._set_status("当前图片已人工确认；下一张图片已不在项目中")
            return True
        if not self.navigate_to_image(next_record):
            return False
        self._set_status("已人工确认，已进入下一张")
        return True

    def previous_image(self) -> bool:
        """Move to the previous visible image without changing review status."""

        if self._current_image is None:
            return False
        previous_image_id = self._previous_visible_image_id(
            self.image_list.currentRow()
        )
        if previous_image_id is None:
            self._set_status("已经是当前筛选结果的第一张")
            return False
        record = self._record_by_id(previous_image_id)
        if record is None:
            return False
        if not self.navigate_to_image(record):
            return False
        self._set_status("已进入上一张")
        return True

    def _verify_current_annotations(self) -> bool:
        if self._current_image is None:
            return False
        self._autosave_timer.stop()
        image_id = _value(self._current_image, "id")
        boxes = self.canvas.annotations()
        status = _enum_text(_value(self._current_image, "review_status"), "unreviewed")
        image_rect = self.canvas.image_rect
        if boxes and image_rect.width() > 0 and image_rect.height() > 0:
            try:
                quality_report = scan_annotation_quality(
                    boxes,
                    image_width=image_rect.width(),
                    image_height=image_rect.height(),
                    overlap_threshold=0.80,
                )
            except (TypeError, ValueError) as exc:
                self._show_error("标注质量检查失败", exc)
                return False
            self._last_annotation_quality_report = quality_report
            if (
                quality_report.has_issues
                and self.isVisible()
                and not confirm_annotation_quality_warnings(self, quality_report)
            ):
                self._set_status("已返回修改；当前图片尚未人工确认")
                return False
        if not boxes and (
            status != "verified" or str(image_id) in self._modified_image_ids
        ):
            decision = self._member("confirm_empty_sample", "confirm_negative_sample")
            if callable(decision):
                if not bool(decision(image_id)):
                    return False
            elif not self.isVisible() or not confirm_empty_annotation(self):
                return False
        method = self._member("verify_and_next", "save_and_verify", "save_and_confirm")
        store = getattr(self.controller, "store", None)
        direct_store = False
        if not callable(method) and store is not None:
            method = getattr(store, "save_and_confirm", None)
            direct_store = callable(method)
        try:
            if callable(method):
                accepts_confirmation = direct_store
                with suppress(TypeError, ValueError):
                    accepts_confirmation = accepts_confirmation or (
                        "confirm_empty" in inspect.signature(method).parameters
                    )
                if accepts_confirmation:
                    method(image_id, boxes, confirm_empty=not boxes)
                else:
                    method(image_id, boxes)
            else:
                if not self.save_current_annotations(silent=True):
                    return False
                marker = self._member("mark_verified", "confirm_image", "verify_image")
                if not callable(marker) and store is not None:
                    marker = getattr(store, "confirm_image", None)
                if not callable(marker):
                    return False
                try:
                    accepts_confirmation = (
                        "confirm_empty" in inspect.signature(marker).parameters
                    )
                except (TypeError, ValueError):
                    accepts_confirmation = False
                if accepts_confirmation:
                    marker(image_id, confirm_empty=not boxes)
                else:
                    marker(image_id)
        except Exception as exc:  # controller boundary
            self._show_error("确认图片失败", exc)
            return False
        self._annotations_dirty = False
        self._modified_image_ids.discard(str(image_id))
        return True

    def _next_visible_image_id(self, after_row: int) -> object | None:
        for row in range(max(-1, after_row) + 1, self.image_list.count()):
            item = self.image_list.item(row)
            if item and not item.isHidden():
                record = item.data(Qt.ItemDataRole.UserRole)
                return _value(record, "id")
        return None

    def _previous_visible_image_id(self, before_row: int) -> object | None:
        for row in range(min(before_row, self.image_list.count()) - 1, -1, -1):
            item = self.image_list.item(row)
            if item and not item.isHidden():
                record = item.data(Qt.ItemDataRole.UserRole)
                return _value(record, "id")
        return None

    def add_class(self) -> None:
        name = self.new_class_edit.text().strip()
        if not name:
            return
        method = self._member("add_class", "add_category")
        store = getattr(self.controller, "store", None)
        if not callable(method) and store is not None:
            method = getattr(store, "add_class", None)
        if not callable(method):
            return
        try:
            self._invoke_by_arity(method, [(name,), (name, None)])
        except Exception as exc:  # controller boundary
            self._show_error("添加类别失败", exc)
            return
        self.new_class_edit.clear()
        self.refresh_categories()

    def _on_class_changed(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        del previous
        self.display_alias_edit.setEnabled(current is not None)
        self.set_display_alias_button.setEnabled(current is not None)
        self.rename_category_button.setEnabled(current is not None)
        self.clear_display_alias_button.setEnabled(current is not None)
        self.delete_empty_category_button.setEnabled(current is not None)
        if current:
            self.canvas.set_current_class(str(current.data(Qt.ItemDataRole.UserRole)))
            alias = str(current.data(Qt.ItemDataRole.UserRole + 2) or "")
            canonical = str(current.data(Qt.ItemDataRole.UserRole + 1) or "")
            self.display_alias_edit.setText(alias)
            self.display_alias_edit.setPlaceholderText(
                f"输入新名称（当前规范名：{canonical}）"
            )
        else:
            self.display_alias_edit.clear()
            self.display_alias_edit.setPlaceholderText("输入新的显示名称或规范类别名")

    def set_category_display_alias(
        self,
        _checked: bool = False,
        *,
        clear: bool = False,
    ) -> bool:
        """Update only the selected category's presentation alias."""

        del _checked
        item = self.class_list.currentItem()
        if item is None:
            self._show_error("无法设置显示别名", "请先选择一个类别。")
            return False
        category_id = str(item.data(Qt.ItemDataRole.UserRole))
        canonical = str(item.data(Qt.ItemDataRole.UserRole + 1) or category_id)
        alias = None if clear else self.display_alias_edit.text().strip() or None
        method = self._member(
            "update_category_display_name",
            "set_category_display_name",
        )
        try:
            if callable(method):
                method(category_id, alias)
            else:
                store = getattr(self.controller, "store", None)
                updater = getattr(store, "update_category", None)
                if not callable(updater):
                    raise RuntimeError("控制器未实现类别显示别名功能。")
                updater(category_id, display_name=alias)
        except Exception as exc:  # controller boundary
            self._show_error("设置显示别名失败", exc)
            return False
        self.refresh_categories()
        self.refresh_box_list()
        shown = alias or canonical
        self._set_status(f"类别 {canonical} 的显示名称已设为 {shown}；训练类别名未改变")
        return True

    def rename_category_canonical(
        self,
        _checked: bool = False,
        *,
        confirmed: bool | None = None,
    ) -> bool:
        """Fully rename the selected stable category after explicit consent."""

        del _checked
        item = self.class_list.currentItem()
        if item is None:
            self._show_error("无法完整重命名类别", "请先选择一个类别。")
            return False
        category_id = str(item.data(Qt.ItemDataRole.UserRole))
        old_name = str(item.data(Qt.ItemDataRole.UserRole + 1) or category_id)
        new_name = self.display_alias_edit.text().strip()
        if not new_name:
            self._show_error("无法完整重命名类别", "请输入新的规范类别名称。")
            return False
        if confirmed is None:
            answer = QMessageBox.question(
                self,
                "完整重命名类别",
                "这会先备份标注数据库，然后把规范类别名完整重命名：\n\n"
                f"{old_name}  →  {new_name}\n\n"
                "后续训练、导出和新生成的部署包都会使用新名称；"
                "历史模型和已经生成的部署包不会被改写。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            confirmed = answer == QMessageBox.StandardButton.Yes
        if not confirmed:
            return False

        method = self._member("rename_category_canonical", "rename_category")
        try:
            if callable(method):
                result = method(category_id, new_name)
            else:
                store = getattr(self.controller, "store", None)
                updater = getattr(store, "rename_category_canonical", None)
                if not callable(updater):
                    raise RuntimeError("控制器未实现完整类别重命名功能。")
                result = updater(category_id, new_name)
        except Exception as exc:  # controller boundary
            self._show_error("完整重命名类别失败", exc)
            return False

        self.refresh_categories()
        self.refresh_box_list()
        self._last_training_preflight = None
        self.training_snapshot_summary.setPlainText(
            "类别已变更，请重新点击“一键训练”执行训练前检查。"
        )
        backup = _value(_value(result, "backup", {}), "path", "")
        suffix = f"；备份：{backup}" if backup else ""
        self._set_status(
            f"类别已完整重命名：{old_name} → {new_name}{suffix}"
        )
        return True

    def apply_class_to_selected(self) -> None:
        item = self.class_list.currentItem()
        if item:
            self.canvas.set_selected_class(str(item.data(Qt.ItemDataRole.UserRole)))

    def delete_selected_empty_category(
        self,
        _checked: bool = False,
        *,
        confirmed: bool | None = None,
    ) -> bool:
        """Delete the selected category only when it has no boxes anywhere."""

        del _checked
        item = self.class_list.currentItem()
        if item is None:
            self._show_error("无法删除类别", "请先选择一个类别。")
            return False
        category_id = str(item.data(Qt.ItemDataRole.UserRole))
        category_name = str(
            item.data(Qt.ItemDataRole.UserRole + 1) or item.text() or category_id
        )
        if confirmed is None:
            answer = QMessageBox.question(
                self,
                "删除空类别",
                f"确定删除类别“{category_name}”吗？\n\n"
                "软件只允许删除整个项目中没有任何标注框的类别。"
                "删除前会自动备份标注数据库；图片、其他类别、标注框、模型和部署文件不会改变。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            confirmed = answer == QMessageBox.StandardButton.Yes
        if not confirmed:
            return False
        method = self._member("delete_empty_category")
        if not callable(method):
            self._show_error("无法删除类别", "当前控制器未实现安全删除空类别功能。")
            return False
        try:
            result = method(category_id)
        except Exception as exc:  # controller boundary
            self._show_error("删除空类别失败", exc)
            return False
        self.refresh_categories()
        self.refresh_box_list()
        self._last_training_preflight = None
        self.training_snapshot_summary.setPlainText(
            "空类别已删除，请重新点击“一键训练”执行训练前检查。"
        )
        backup = _value(_value(result, "backup", {}), "path", "")
        suffix = f"；备份：{backup}" if backup else ""
        self._set_status(f"已删除空类别：{category_name}{suffix}")
        return True

    # ---- Shortcuts and canvas tools -------------------------------------

    def _register_shortcut(
        self,
        sequence: str | QKeySequence,
        callback: Any,
    ) -> QShortcut:
        shortcut = QShortcut(
            sequence if isinstance(sequence, QKeySequence) else QKeySequence(sequence),
            self,
        )
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(callback)
        return shortcut

    def shortcut_allowed(self) -> bool:
        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not self:
            return False
        widget = QApplication.focusWidget()
        protected = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)
        while widget is not None:
            if isinstance(widget, protected) or bool(widget.property("protectLetterShortcuts")):
                return False
            widget = widget.parentWidget()
        return True

    def _v_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.select_tool_action.trigger()

    def _w_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.draw_tool_action.trigger()

    def _d_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.verify_next_action.trigger()

    def _a_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.previous_image_action.trigger()

    def _fit_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.fit_action.trigger()

    def _delete_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.delete_box_action.trigger()

    def _undo_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.undo_action.trigger()

    def _redo_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.redo_action.trigger()

    def _undo_all_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.undo_all_action.trigger()

    def _delete_all_annotations_shortcut(self) -> None:
        if self.shortcut_allowed():
            self.delete_all_annotations_action.trigger()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if hasattr(self, "canvas") and event.type() in {
            QEvent.Type.KeyPress,
            QEvent.Type.KeyRelease,
        }:
            key = getattr(event, "key", lambda: None)()
            repeated = bool(getattr(event, "isAutoRepeat", lambda: False)())
            if key == Qt.Key.Key_Space and not repeated:
                if event.type() == QEvent.Type.KeyRelease:
                    self.canvas.end_temporary_pan()
                    event.accept()
                    return True
                if self.shortcut_allowed():
                    self.canvas.begin_temporary_pan()
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def _on_canvas_tool_changed(self, tool: str) -> None:
        self.select_tool_action.setChecked(tool == AnnotationCanvas.TOOL_SELECT)
        self.draw_tool_action.setChecked(tool == AnnotationCanvas.TOOL_DRAW)
        self.pan_tool_action.setChecked(tool == AnnotationCanvas.TOOL_PAN)

    def _on_annotation_display_action(self, action: QAction) -> None:
        mode = str(action.data() or AnnotationDisplayMode.FULL.value)
        try:
            self.canvas.set_annotation_display_mode(mode)
        except ValueError as exc:
            self._show_error("无法切换标注显示", exc)

    def _on_annotation_display_mode_changed(self, mode: str) -> None:
        normalized = str(mode)
        actions = {
            AnnotationDisplayMode.FULL.value: self.annotation_display_full_action,
            AnnotationDisplayMode.BOX_ONLY.value: self.annotation_display_boxes_action,
            AnnotationDisplayMode.HIDDEN.value: self.annotation_display_hidden_action,
        }
        action = actions.get(normalized, self.annotation_display_full_action)
        action.setChecked(True)
        labels = {
            AnnotationDisplayMode.FULL.value: "标注显示：完整",
            AnnotationDisplayMode.BOX_ONLY.value: "标注显示：仅框",
            AnnotationDisplayMode.HIDDEN.value: "标注显示：全部隐藏（数据未删除）",
        }
        self.annotation_display_button.setText(labels.get(normalized, "标注显示"))
        self._set_status(labels.get(normalized, "标注显示已更新"))

    def undo_all_current_image(self) -> None:
        """Restore this opened image's canvas and persisted session baseline."""

        if not self.canvas.can_undo_all:
            return
        baseline = self._session_baseline_for_current_image()
        restorer = self._member(
            "restore_annotation_session_baseline",
            "restore_image_annotation_state",
        )
        if baseline is None or not callable(restorer):
            # Keep compatibility with lightweight third-party controllers.  The
            # production controller always supplies the state-preserving API.
            if self.canvas.undo_all():
                self._set_status("已撤销当前图片本次打开后的全部人工修改；可使用重做恢复。")
            return

        self._autosave_timer.stop()
        try:
            restored = restorer(
                baseline["image_id"],
                baseline["boxes"],
                review_status=baseline["review_status"],
                origin=baseline["origin"],
                ai_status=baseline["ai_status"],
                expected_revision=baseline["expected_revision"],
            )
        except Exception as exc:  # controller boundary / optimistic conflict
            self._show_error("无法全部撤销", exc)
            return

        self._restoring_annotation_session_baseline = True
        try:
            restored_canvas = self.canvas.undo_all()
        finally:
            self._restoring_annotation_session_baseline = False
        if not restored_canvas:
            # ``can_undo_all`` above makes this defensive branch unreachable in
            # normal use, but it avoids claiming a successful UI restore if a
            # custom canvas changes the history during the controller call.
            self._show_error("无法全部撤销", "当前图片的撤销历史已改变，请重新打开图片。")
            return

        self._annotations_dirty = False
        self._modified_image_ids.discard(str(baseline["image_id"]))
        self._current_image = restored if restored is not None else baseline["record"]
        with suppress(TypeError, ValueError):
            baseline["expected_revision"] = int(
                _value(self._current_image, "revision")
            )
        self._update_current_image_status(self._current_image)
        self._update_seed_count()
        self._set_status("已恢复当前图片本次打开时的标注与确认状态；可使用重做恢复。")

    # ---- Training and inference ----------------------------------------

    def _update_seed_count(self) -> None:
        count = self._read("seed_verified_count", "verified_count", default=None)
        if count is None:
            count = sum(
                _enum_text(_value(record, "review_status"), "unreviewed") == "verified"
                for record in self._image_records
            )
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 0
        self.seed_count_label.setText(f"已选入训练且人工确认：{count} / 100")
        self.seed_progress.setValue(min(100, count))
        busy = (
            self._ui_busy
            or bool(self._active_job_id)
            or self.process_bridge.is_running
        )
        self.train_button.setEnabled(count >= 100 and not busy)
        self.retrain_button.setEnabled(count > 0 and not busy)
        self.resume_training_button.setEnabled(
            self.resume_combo.currentData() is not None and not busy
        )

    def _on_model_changed(self) -> None:
        self._refresh_training_history_for_model()
        self._refresh_resume_history_for_model()

    def _on_training_history_selection_changed(self, _index: int) -> None:
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            return
        run_id = combo.currentData()
        if run_id:
            self.show_training_run_history(str(run_id))

    def show_training_run_history(self, run_id: str) -> bool:
        """Replay persisted metric/log events without changing active job state."""

        if self._ui_busy or self.process_bridge.is_running:
            return False
        loader = self._member(
            "load_training_run_history",
            "training_run_history",
        )
        if not callable(loader):
            return False
        try:
            history = loader(str(run_id))
        except Exception as exc:
            self._set_status(f"无法读取历史训练 {run_id}：{exc}")
            return False
        if not isinstance(history, Mapping):
            return False

        model = str(history.get("model_key") or "")
        status = str(history.get("status") or "")
        title = f"历史训练 · {model or run_id} · {status or '未知状态'}"
        self.training_monitor.reset(title)
        raw_events = history.get("events", ())
        if isinstance(raw_events, Sequence) and not isinstance(
            raw_events,
            str | bytes | bytearray,
        ):
            for event in raw_events:
                if isinstance(event, Mapping):
                    self.training_monitor.handle_event(event)
        console_log = history.get("console_log")
        if console_log:
            self.training_monitor.append_log(str(console_log))
        warnings = history.get("warnings", ())
        if isinstance(warnings, Sequence) and not isinstance(
            warnings,
            str | bytes | bytearray,
        ):
            for warning in warnings:
                self.training_monitor.append_log(f"历史记录警告：{warning}")
        preview = history.get("preview_path")
        if preview:
            self.training_monitor.show_preview(str(preview))
        # Replayed terminal events may replace the label; keep the selected
        # historical run unmistakable and do not make the UI look busy.
        terminal = self.training_monitor._training_end_text
        self.training_monitor.state_label.setText(
            f"{title} · {terminal}" if terminal else title
        )
        self._set_status(f"已载入训练历史：{run_id}")
        return True

    def _update_settings_summary(self) -> None:
        settings = self._training_settings
        early_stopping = (
            f"早停 patience={settings['patience']}"
            if settings.get("early_stopping_enabled", int(settings["patience"]) > 0)
            else "早停关闭"
        )
        self.settings_summary_label.setText(
            f"{settings['imgsz']}px · {settings['epochs']} epochs · "
            f"batch={settings['batch']} · device={settings['device']} · "
            f"{early_stopping}"
        )

    def open_training_settings(self) -> None:
        presets = self._read("training_presets", "list_training_presets", default={})
        if isinstance(presets, Mapping):
            self._training_presets = {str(name): dict(value) for name, value in presets.items()}
        devices = _as_records(self._read("available_training_devices", default=()))
        dialog = TrainingSettingsDialog(
            self._training_settings,
            presets=self._training_presets,
            devices=devices,
            parent=self,
        )
        dialog.presetSaveRequested.connect(self._save_training_preset)
        dialog.presetDeleteRequested.connect(self._delete_training_preset)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._training_settings = dialog.values()
            self._update_settings_summary()

    def _save_training_preset(self, name: str, settings: dict[str, Any]) -> None:
        self._training_presets[name] = settings
        method = self._member("save_training_preset")
        if callable(method):
            method(name, settings)

    def _delete_training_preset(self, name: str) -> None:
        self._training_presets.pop(name, None)
        method = self._member("delete_training_preset")
        if callable(method):
            method(name)

    def open_environment_dialog(self) -> None:
        candidates = self._read("discover_ml_environments", "list_ml_environments", default=[])
        discoverer = self._member("discover_ml_environments", "list_ml_environments")
        validator = self._member("validate_ml_environment", "validate_python")
        dialog = MLEnvironmentDialog(
            _as_records(candidates),
            selected=self._ml_environment,
            validator=validator if callable(validator) else None,
            discoverer=discoverer if callable(discoverer) else None,
            creator=self._request_create_ml_environment,
            parent=self,
        )
        self._environment_dialog = dialog
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._ml_environment = dialog.selected_path()
                name = Path(self._ml_environment).parent.name or self._ml_environment
                self.environment_button.setText(f"ML 环境：{name}")
        finally:
            self._environment_dialog = None

    def _request_create_ml_environment(self, payload: Mapping[str, Any]) -> object:
        method = self._member("create_ml_environment", "create_yolo_environment")
        if not callable(method):
            raise RuntimeError("控制器未实现 create_ml_environment。")
        result = self._invoke_by_arity(method, [(payload,), ("yolo",), ()])
        immediate = isinstance(result, Mapping) and result.get("success") is not None
        if not immediate:
            self._set_busy("environment", True)
            self._consume_job_result(result, "environment")
        return result

    def training_payload(self, *, retrain: bool = False) -> dict[str, Any]:
        model_data = dict(self.model_combo.currentData() or {})
        payload = merged_training_settings(self._training_settings)
        payload.update(model_data)
        payload["retrain"] = bool(retrain)
        payload["ml_environment"] = self._ml_environment or None
        payload["historical_run_id"] = self.history_combo.currentData()
        return payload

    def start_training(self, *, retrain: bool = False) -> bool:
        if self._ui_busy or self.process_bridge.is_running or self._active_job_id:
            self._set_status("已有任务正在运行，不能重复启动训练。")
            return False
        payload = self.training_payload(retrain=retrain)
        errors = validate_training_settings(payload)
        if (
            payload.get("start_from") in {"best", "last"}
            and not payload.get("historical_run_id")
        ):
            errors.append("选择 best.pt 或 last.pt 作为起点时，必须选择一次历史成功训练。")
        if errors:
            self._show_error("训练参数无效", "\n".join(errors))
            return False
        warnings = training_setting_warnings(payload)
        if warnings and self.isVisible():
            answer = QMessageBox.warning(
                self,
                "训练资源提醒",
                "\n".join(warnings) + "\n\n仍要继续训练吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        preflight = self._member("training_preflight")
        if callable(preflight):
            try:
                result = self._invoke_by_arity(preflight, [(payload,), ()])
            except Exception as exc:  # controller boundary
                self._show_error("训练前检查失败", exc)
                return False
            if isinstance(result, Mapping):
                current_selection_count = len(self.selected_image_ids())
                result = {
                    **dict(result),
                    "current_selection_count": current_selection_count,
                }
                payload["current_selection_count"] = current_selection_count
            if not self._accept_preflight(result):
                return False
            member_fingerprint = _value(
                result,
                "training_member_fingerprint",
                "",
            )
            if member_fingerprint:
                payload["expected_training_member_fingerprint"] = str(
                    member_fingerprint
                )
            selection_fingerprint = _value(
                result,
                "selection_fingerprint",
                "",
            )
            if selection_fingerprint:
                payload["expected_training_selection_fingerprint"] = str(
                    selection_fingerprint
                )
        method = self._member("start_training", "train")
        if not callable(method):
            self._show_error("无法启动训练", "控制器未实现 start_training。")
            return False
        self.training_monitor.reset(
            "正在创建不可变训练快照…",
            requested_epochs=int(payload["epochs"]),
        )
        self._set_busy("train", True)
        try:
            result = self._invoke_by_arity(
                method,
                [
                    (payload["model_key"], payload),
                    (payload,),
                    (payload["model_key"],),
                ],
            )
            self._consume_job_result(result, "train")
        except Exception as exc:  # controller boundary
            self._set_busy("", False)
            self._show_error("启动训练失败", exc)
            return False
        return True

    def resume_selected_training(self) -> bool:
        if self._ui_busy or self.process_bridge.is_running or self._active_job_id:
            self._set_status("已有任务正在运行，不能重复启动恢复训练。")
            return False
        run_id = self.resume_combo.currentData()
        if not run_id:
            self._show_error("无法恢复训练", "请选择一次包含 last.pt 的失败或已取消训练。")
            return False
        method = self._member("resume_training", "resume")
        if not callable(method):
            self._show_error("无法恢复训练", "控制器未实现 resume_training。")
            return False
        payload = {
            "run_id": str(run_id),
            "ml_environment": self._ml_environment or None,
        }
        self.training_monitor.reset(
            "正在校验原训练快照和 last.pt…",
            requested_epochs=int(self._training_settings.get("epochs", 100)),
        )
        self._set_busy("train", True)
        try:
            result = self._invoke_by_arity(method, [(payload,), (str(run_id), payload)])
            self._consume_job_result(result, "train")
        except Exception as exc:
            self._set_busy("", False)
            self._show_error("恢复训练失败", exc)
            return False
        return True

    def _accept_preflight(self, result: object) -> bool:
        if result is None or result is True:
            return True
        self._last_training_preflight = result
        self._set_training_snapshot_summary(result, phase="训练前检查")
        allowed = _value(result, "allowed", _value(result, "ok", True))
        errors = _value(result, "errors", ())
        if not allowed:
            message = "\n".join(map(str, errors)) if errors else str(_value(result, "message", result))
            empty_categories = _value(result, "empty_categories", ())
            if empty_categories:
                names = [
                    str(_value(category, "name", "")).strip()
                    for category in empty_categories
                ]
                names = [name for name in names if name]
                if names:
                    message += (
                        "\n\n处理方法：如果这些类别是误建且在整个项目中确实没有任何框，"
                        "请返回“标注”页，在类别列表中选择 "
                        + "、".join(names)
                        + "，点击“删除空类别”，然后重新点击“一键训练”。"
                        "如果类别在未确认或未加入训练的图片中仍有框，软件会拒绝删除，"
                        "应先整理该类别的训练样本。"
                    )
            self._show_error("训练条件不满足", message)
            return False
        if self.isVisible():
            counts = _value(result, "counts", {})
            if not isinstance(counts, Mapping):
                counts = {}
            has_unconfirmed = any(
                int(counts.get(name, 0) or 0) > 0
                for name in ("unlabeled", "ai_unconfirmed")
            )
            dialog = TrainingPreflightDialog(
                result,
                has_unconfirmed=has_unconfirmed,
                parent=self,
            )
            return dialog.exec() == QDialog.DialogCode.Accepted
        return True

    def _set_training_snapshot_summary(
        self,
        summary: Mapping[str, Any] | object,
        *,
        phase: str,
    ) -> None:
        text = training_preflight_text(summary, phase=phase)
        self.training_snapshot_summary.setPlainText(text)
        self.training_monitor.append_log(text)

    def start_autolabel(self) -> bool:
        selection = self.ai_model_combo.currentData()
        if not selection:
            self._show_error("无法自动标注", "请先选择一次成功训练的模型。")
            return False
        if isinstance(selection, Mapping):
            run_id = selection.get("run_id")
            checkpoint = selection.get("checkpoint")
            checkpoint_kind = selection.get("checkpoint_kind")
        else:
            run_id = selection
            checkpoint = None
            checkpoint_kind = "best"
        payload = {
            "run_id": run_id,
            "checkpoint": checkpoint,
            "checkpoint_kind": checkpoint_kind,
            "confidence": self.ai_confidence_spin.value(),
            "iou": self.ai_iou_spin.value(),
            "deduplicate": self.ai_dedup_check.isChecked(),
            "dedup_iou": self.ai_dedup_iou_spin.value(),
            "resume": self.ai_resume_check.isChecked(),
            "only_unverified": True,
            "ml_environment": self._ml_environment or None,
        }
        method = self._member("start_autolabel", "start_auto_label", "auto_label")
        if not callable(method):
            self._show_error("无法自动标注", "控制器未实现 start_autolabel。")
            return False
        self.ai_log.clear()
        self.ai_progress.setValue(0)
        self.ai_status_label.setText("正在排队 AI 自动标注任务…")
        self._set_busy("autolabel", True)
        try:
            result = self._invoke_by_arity(method, [(payload,), (run_id, payload), (run_id,)])
            self._consume_job_result(result, "autolabel")
        except Exception as exc:  # controller boundary
            self._set_busy("", False)
            self._show_error("启动 AI 自动标注失败", exc)
            return False
        return True

    def cleanup_historical_ai_duplicates(self) -> bool:
        """Preview, back up and clean selected historical pure-AI drafts."""

        scope = str(self.historical_dedup_scope_combo.currentData() or "selected")
        image_ids = (
            self.visible_image_ids()
            if scope == "visible"
            else self.selected_image_ids()
        )
        if not image_ids:
            label = "当前筛选结果" if scope == "visible" else "当前多选"
            self._show_error("无法清理历史 AI 重框", f"{label}中没有图片。")
            return False
        if not self.save_current_annotations(silent=True):
            return False
        previewer = self._member(
            "preview_ai_deduplication",
            "preview_historical_ai_deduplication",
        )
        applier = self._member(
            "deduplicate_ai_drafts",
            "apply_historical_ai_deduplication",
        )
        if not callable(previewer) or not callable(applier):
            self._show_error("无法清理历史 AI 重框", "控制器未实现历史 AI 草稿去重。")
            return False
        threshold = self.historical_dedup_iou_spin.value()
        try:
            preview = previewer(image_ids, iou_threshold=threshold)
        except Exception as exc:  # controller boundary
            self._show_error("AI 重框预检查失败", exc)
            return False
        requested = int(_value(preview, "requested_image_count", len(image_ids)) or 0)
        affected = int(_value(preview, "affected_image_count", 0) or 0)
        removed = int(_value(preview, "removed_box_count", 0) or 0)
        before = int(_value(preview, "before_box_count", 0) or 0)
        after = int(_value(preview, "after_box_count", before - removed) or 0)
        protected = int(_value(preview, "protected_box_count", 0) or 0)
        if removed <= 0:
            message = (
                f"已检查 {requested} 张图片，没有发现符合条件的 AI 重框。"
                f"人工确认、人工修改及非纯 AI 草稿框受保护（{protected} 个框）。"
            )
            self._set_status(message)
            if self.isVisible():
                QMessageBox.information(self, "无需清理", message)
            return True
        if self.isVisible():
            response = QMessageBox.question(
                self,
                "确认清理历史 AI 重框",
                f"阈值 IoU ≥ {threshold:.2f}\n"
                f"检查 {requested} 张，影响 {affected} 张；将从 {before} 个框中删除 "
                f"{removed} 个，保留 {after} 个。\n"
                f"另有 {protected} 个人工/混合/已确认框受保护，不会处理。\n\n"
                "执行前会自动备份标注数据库，且只删除同类别纯 AI 草稿中的低置信度重框。"
                "是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return False
        current_id = (
            _value(self._current_image, "id")
            if self._current_image is not None
            else None
        )
        try:
            report = applier(image_ids, iou_threshold=threshold)
        except Exception as exc:  # controller boundary
            self._show_error("清理历史 AI 重框失败", exc)
            return False
        applied_removed = int(_value(report, "removed_box_count", removed) or 0)
        applied_affected = int(_value(report, "affected_image_count", affected) or 0)
        backup = _value(report, "backup", None)
        backup_path = str(_value(backup, "path", "") or "")
        self._current_image = None
        self._annotations_dirty = False
        self.refresh_project(select_image_id=current_id)
        message = f"已从 {applied_affected} 张图片删除 {applied_removed} 个纯 AI 重框。"
        if backup_path:
            message += f" 备份：{backup_path}。可通过“文件 → 恢复标注备份”撤销。"
        self._set_status(message)
        if self.isVisible():
            QMessageBox.information(self, "历史 AI 重框清理完成", message)
        return True

    def _consume_job_result(self, result: object, kind: str) -> None:
        self._active_job_kind = kind
        if result is None:
            return
        if isinstance(result, Mapping):
            snapshot_summary = result.get("snapshot_summary")
            if isinstance(snapshot_summary, Mapping):
                self._set_training_snapshot_summary(
                    snapshot_summary,
                    phase="不可变训练快照已创建",
                )
            self._active_job_id = str(result.get("job_id", self._active_job_id))
            program = result.get("program") or result.get("executable")
            arguments = result.get("arguments", result.get("args", ()))
            if program:
                self.process_bridge.start(
                    str(program),
                    list(arguments or ()),
                    working_directory=result.get("working_directory", result.get("cwd")),
                    environment=result.get("environment"),
                    stdin_payload=result.get("stdin_payload"),
                    job_id=self._active_job_id,
                )
            return
        self._active_job_id = str(
            getattr(result, "job_id", result if isinstance(result, str | int) else "")
        )
        snapshot_summary = getattr(result, "snapshot_summary", None)
        if isinstance(snapshot_summary, Mapping):
            self._set_training_snapshot_summary(
                snapshot_summary,
                phase="不可变训练快照已创建",
            )
        program = getattr(result, "program", getattr(result, "executable", None))
        if program:
            self.process_bridge.start(
                str(program),
                list(getattr(result, "arguments", getattr(result, "args", ()))),
                working_directory=getattr(result, "working_directory", None),
                environment=getattr(result, "environment", None),
                stdin_payload=getattr(result, "stdin_payload", None),
                job_id=self._active_job_id,
            )

    def cancel_active_job(self) -> None:
        method = self._member("cancel_job", "cancel_active_job")
        try:
            if callable(method):
                self._invoke_by_arity(method, [(self._active_job_id,), ()])
            if self.process_bridge.is_running:
                self.process_bridge.cancel()
        except Exception as exc:  # controller boundary
            self._show_error("取消任务失败", exc)
            return
        self._set_status("已请求取消任务")

    def refresh_runs(self) -> None:
        runs = _as_records(self._read("list_runs", "training_runs", default=[]))
        successful: list[object] = []
        resumable: list[object] = []
        for run in runs:
            kind = _enum_text(_value(run, "kind"), "train")
            status = _enum_text(_value(run, "status"), "")
            checkpoints = self._run_checkpoints(run)
            if (
                kind in {"train", "training"}
                and status in {"success", "succeeded", "completed"}
                and checkpoints
            ):
                successful.append(run)
            if (
                kind in {"train", "training"}
                and status in {"failed", "cancelled", "canceled"}
                and checkpoints.get("last")
            ):
                resumable.append(run)

        self._successful_runs = successful
        self._resumable_runs = resumable
        self._refresh_training_history_for_model()
        self._refresh_resume_history_for_model()

        for combo in (self.ai_model_combo, self.deploy_checkpoint_combo):
            previous = combo.currentData()
            combo.clear()
            if not successful:
                combo.addItem("尚无成功训练", None)
                continue
            for run in successful:
                run_id = _value(run, "id", _value(run, "run_id"))
                model = _enum_text(_value(run, "model_key", _value(run, "model", "YOLO")))
                created = str(_value(run, "created_at", ""))[:19]
                for checkpoint_kind, checkpoint in self._run_checkpoints(run).items():
                    choice = {
                        "run_id": run_id,
                        "model_key": model,
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint": checkpoint,
                    }
                    combo.addItem(
                        f"{model} · {created or run_id} · {checkpoint_kind}.pt",
                        choice,
                    )
                    combo.setItemData(combo.count() - 1, run, Qt.ItemDataRole.UserRole + 1)
            index = combo.findData(previous)
            combo.setCurrentIndex(index if index >= 0 else 0)
        busy = (
            self._ui_busy
            or bool(self._active_job_id)
            or self.process_bridge.is_running
        )
        self.autolabel_button.setEnabled(bool(successful) and not busy)
        self.deploy_wizard_button.setEnabled(bool(successful) and not busy)
        self._on_deploy_checkpoint_changed()

    def _refresh_training_history_for_model(self) -> None:
        if not hasattr(self, "history_combo"):
            return
        previous = self.history_combo.currentData()
        selected_model = self.model_combo.currentText()
        matching = [
            run
            for run in self._successful_runs
            if _enum_text(_value(run, "model_key", _value(run, "model", "")))
            == selected_model
        ]
        self.history_combo.clear()
        if not matching:
            self.history_combo.addItem(f"尚无 {selected_model} 成功训练", None)
            return
        for run in matching:
            run_id = _value(run, "id", _value(run, "run_id"))
            created = str(_value(run, "created_at", ""))[:19]
            self.history_combo.addItem(f"{selected_model} · {created or run_id}", run_id)
            self.history_combo.setItemData(
                self.history_combo.count() - 1,
                run,
                Qt.ItemDataRole.UserRole + 1,
            )
        index = self.history_combo.findData(previous)
        self.history_combo.setCurrentIndex(index if index >= 0 else 0)

    def _refresh_resume_history_for_model(self) -> None:
        if not hasattr(self, "resume_combo"):
            return
        previous = self.resume_combo.currentData()
        selected_model = self.model_combo.currentText()
        matching = [
            run
            for run in self._resumable_runs
            if _enum_text(_value(run, "model_key", _value(run, "model", "")))
            == selected_model
        ]
        self.resume_combo.clear()
        if not matching:
            self.resume_combo.addItem(f"尚无 {selected_model} 可恢复训练", None)
        else:
            for run in matching:
                run_id = _value(run, "id", _value(run, "run_id"))
                created = str(_value(run, "created_at", ""))[:19]
                self.resume_combo.addItem(
                    f"{selected_model} · {created or run_id} · last.pt",
                    run_id,
                )
                self.resume_combo.setItemData(
                    self.resume_combo.count() - 1,
                    run,
                    Qt.ItemDataRole.UserRole + 1,
                )
            index = self.resume_combo.findData(previous)
            self.resume_combo.setCurrentIndex(index if index >= 0 else 0)
        busy = self._ui_busy or bool(self._active_job_id) or self.process_bridge.is_running
        self.resume_training_button.setEnabled(bool(matching) and not busy)

    def _on_deploy_checkpoint_changed(self) -> None:
        choice = self.deploy_checkpoint_combo.currentData()
        model = (
            str(choice.get("model_key"))
            if isinstance(choice, Mapping) and choice.get("model_key")
            else "—"
        )
        self.deploy_model_label.setText(f"当前部署模型：{model}")

    @staticmethod
    def _run_checkpoints(run: object) -> dict[str, str]:
        artifacts = _value(run, "artifacts", {})
        if not isinstance(artifacts, Mapping):
            artifacts = {}
        best = (
            artifacts.get("best")
            or artifacts.get("best.pt")
            or _value(run, "best_pt")
            or _value(run, "best_path")
            or _value(run, "checkpoint_path")
            or _value(run, "checkpoint")
        )
        last = (
            artifacts.get("last")
            or artifacts.get("last.pt")
            or _value(run, "last_pt")
            or _value(run, "last_path")
        )
        result: dict[str, str] = {}
        if best:
            result["best"] = str(best)
        if last:
            result["last"] = str(last)
        return result

    def _set_busy(self, kind: str, busy: bool) -> None:
        self._ui_busy = busy
        self._active_job_kind = kind if busy else ""
        if not busy:
            self._active_job_id = ""
        self.cancel_job_button.setEnabled(busy)
        self.model_combo.setEnabled(not busy)
        self.advanced_settings_button.setEnabled(not busy)
        self.environment_button.setEnabled(not busy)
        self.resume_combo.setEnabled(not busy)
        self.resume_training_button.setEnabled(
            not busy and self.resume_combo.currentData() is not None
        )
        self.autolabel_button.setEnabled(not busy and self.ai_model_combo.currentData() is not None)
        self.deploy_wizard_button.setEnabled(
            not busy and self.deploy_checkpoint_combo.currentData() is not None
        )
        self.docker_target_combo.setEnabled(not busy)
        self.docker_detect_button.setEnabled(not busy)
        self.docker_start_button.setEnabled(not busy)
        self.docker_import_button.setEnabled(not busy)
        self.docker_pull_button.setEnabled(not busy)
        self.docker_cancel_button.setEnabled(busy and kind == "docker_environment")
        self.docker_cancel_button.setVisible(busy and kind == "docker_environment")
        self._update_seed_count()

    @Slot(dict)
    def on_job_event(self, event: dict[str, Any]) -> None:
        """Public entry point for worker/controller JSONL messages."""

        if not isinstance(event, Mapping):
            return
        envelope = dict(event)
        event_type = str(envelope.get("type", "event"))
        payload = envelope.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {"message": payload}
            envelope["payload"] = payload
        job_id = envelope.get("job_id")
        if job_id:
            self._active_job_id = str(job_id)
        kind = str(
            payload.get("kind")
            or payload.get("job_kind")
            or envelope.get("kind")
            or self._active_job_kind
        )
        if kind in {"train", "training", ""}:
            self.training_monitor.handle_event(envelope)
        if kind in {"autolabel", "predict", "prediction", "infer"}:
            self._handle_ai_event(event_type, payload)
        if kind in {"maix", "deploy", "conversion", "convert"}:
            self._handle_deploy_event(event_type, payload)
        if kind in {"docker", "docker_environment", "conversion_environment"}:
            self._handle_docker_environment_event(event_type, payload)
        if kind in {"environment", "ml_environment", "env"} and self._environment_dialog:
            self._environment_dialog.handle_creation_event(envelope)

        if event_type in {"finished", "completed", "process_finished", "cancelled", "error"}:
            recoverable_error = event_type == "error" and (
                bool(payload.get("recoverable"))
                or str(payload.get("scope", "")).casefold() in {"image", "prediction"}
            )
            worker_still_running = bool(self.process_bridge.is_running)
            bridge_job_id = str(getattr(self.process_bridge, "job_id", "") or "")
            bridge_owns_event = worker_still_running and bool(job_id) and (
                not bridge_job_id or bridge_job_id == str(job_id)
            )
            terminal = not recoverable_error and (
                event_type == "process_finished" or not bridge_owns_event
            )
            if terminal:
                self._set_busy("", False)
                self.refresh_runs()
                if kind in {"autolabel", "predict", "prediction", "infer"}:
                    self.refresh_images()
        elif event_type in {"started", "status", "progress", "epoch", "batch"}:
            self.cancel_job_button.setEnabled(True)

    def _handle_ai_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if event_type in {"log", "warning", "error"}:
            self.ai_log.appendPlainText(str(payload.get("message", payload)))
        current = payload.get("current", payload.get("completed", payload.get("image_index")))
        total = payload.get("total", payload.get("image_total"))
        if total is not None:
            self.ai_progress.setMaximum(max(1, int(total)))
        if current is not None:
            self.ai_progress.setValue(max(0, int(current)))
        elif "progress" in payload:
            progress = float(payload["progress"])
            self.ai_progress.setMaximum(100)
            self.ai_progress.setValue(round(progress * 100 if progress <= 1 else progress))
        message = payload.get("message") or payload.get("status") or payload.get("stage")
        if message:
            self.ai_status_label.setText(str(message))
        if event_type in {"finished", "completed"}:
            self.ai_status_label.setText("AI 草稿已导入，请逐张人工复核并按 D 确认。")

    def _handle_deploy_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if event_type in {"log", "warning", "error"}:
            self.deploy_log.appendPlainText(str(payload.get("message", payload)))
        if "progress" in payload:
            progress = float(payload["progress"])
            self.deploy_progress.setValue(round(progress * 100 if progress <= 1 else progress))
        message = payload.get("message") or payload.get("stage") or payload.get("status")
        if message:
            self.deploy_status_label.setText(str(message))
        if event_type == "artifact":
            self._remember_deployment_artifact(payload)
        warning_code = str(payload.get("code", payload.get("reason", "")))
        if event_type in {"confirmation_required", "package_size_warning"} or warning_code in {
            "package_size_warning",
            "deployment_package_oversize",
        }:
            self._pending_deploy_confirmation = dict(payload)
            detail = _deployment_size_warning_detail(payload)
            self.deploy_status_label.setStyleSheet(f"color: {COLORS['danger']};")
            self.deploy_status_label.setText(
                "部署包超过 30,000,000 字节，等待用户确认。"
            )
            self.deploy_log.appendPlainText(detail)
            if self.isVisible():
                answer = QMessageBox.warning(
                    self,
                    "部署包体积警告",
                    detail,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                self.respond_to_deployment_size_warning(
                    answer == QMessageBox.StandardButton.Yes
                )
        if event_type in {"finished", "completed"}:
            self._last_completed_deploy_run_id = self._active_job_id
            artifacts = _as_records(payload.get("deployment_artifacts"))
            for item in artifacts:
                self._remember_deployment_artifact(item)
            app_path = str(payload.get("app_package_path") or "").strip()
            editable_path = str(payload.get("editable_project_path") or "").strip()
            model_path = str(payload.get("model_package_path") or "").strip()
            for raw_path in (app_path, editable_path, model_path):
                if raw_path:
                    self._remember_deployment_path(Path(raw_path))
            validation = str(payload.get("device_validation", "required")).casefold()
            if validation in {"verified", "passed"}:
                self.deploy_status_label.setStyleSheet(
                    f"color: {COLORS['success']};"
                )
                suffix = "已完成真机验证"
            else:
                self.deploy_status_label.setStyleSheet(
                    f"color: {COLORS['warning']};"
                )
                suffix = "尚未真机验证，状态为待真机验证"
            generated: list[str] = []
            if app_path:
                generated.append(f".maixapp：{app_path}")
            if editable_path:
                generated.append(f"可编辑工程：{editable_path}")
            if model_path:
                generated.append(f"模型包：{model_path}")
            detail = "；".join(generated) if generated else "产物路径见部署日志"
            self.deploy_status_label.setText(
                f"部署文件生成完成（尚未安装到设备）：{detail}；{suffix}。"
            )
            self.open_deploy_output_button.setEnabled(
                self._last_deploy_output_dir is not None
            )
            self.open_maixapp_button.setEnabled(self._last_maixapp_path is not None)
            self.cleanup_backups_button.setEnabled(True)

    def _handle_docker_environment_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        message = str(payload.get("message") or payload.get("status") or "")
        if message:
            self.deploy_log.appendPlainText(message)
            self.docker_status_label.setText(message)
        if event_type == "progress":
            percent = max(0.0, min(100.0, float(payload.get("percent", 0.0))))
            bytes_read = int(payload.get("bytes_read", 0))
            total_bytes = int(payload.get("total_bytes", 0))
            elapsed = float(payload.get("elapsed_seconds", 0.0))
            speed = float(payload.get("bytes_per_second", 0.0))
            self.docker_import_progress.setVisible(True)
            self.docker_import_progress.setValue(round(percent))
            heartbeat = bool(payload.get("heartbeat"))
            detail = (
                f"已读取 {self._format_bytes(bytes_read)} / "
                f"{self._format_bytes(total_bytes)}（{percent:.1f}%） · "
                f"已用 {elapsed:.1f} 秒 · {self._format_bytes(speed)}/s"
            )
            if heartbeat:
                detail += " · Docker 仍在运行"
            if bool(payload.get("diagnostic")):
                stalled = float(payload.get("stalled_seconds", 0.0))
                diagnostic = str(payload.get("diagnostic_message") or "")
                detail += f" · 已停滞 {stalled:.1f} 秒，诊断已记录"
                self.deploy_log.appendPlainText(
                    "Docker 导入停滞诊断："
                    + (diagnostic or "长时间没有新的归档字节进度。")
                    + f" bytes={bytes_read}/{total_bytes}, elapsed={elapsed:.1f}s"
                )
            self.docker_import_detail_label.setVisible(True)
            self.docker_import_detail_label.setText(detail)
            self.docker_status_label.setStyleSheet(f"color: {COLORS['warning']};")
            self.docker_status_label.setText("正在导入 Docker 转换镜像…")
        elif event_type == "artifact" and payload.get("kind") == "docker_image":
            identity = str(payload.get("image_id") or "")
            digests = _as_records(payload.get("repo_digests"))
            if digests:
                identity = str(digests[0])
            self.deploy_log.appendPlainText(
                f"镜像校验：{payload.get('name') or '未知'} · "
                f"{payload.get('status') or '未知'}"
                + (f" · {identity}" if identity else "")
            )
        elif event_type == "error":
            self._docker_terminal_event_seen = True
            self.docker_status_label.setStyleSheet(f"color: {COLORS['danger']};")
            if not message:
                self.docker_status_label.setText("Docker 镜像操作失败。")
        elif event_type in {"cancelled", "canceled"}:
            self._docker_terminal_event_seen = True
            self.docker_status_label.setStyleSheet(f"color: {COLORS['warning']};")
            self.docker_status_label.setText(message or "Docker 镜像操作已取消。")
        elif event_type in {"finished", "completed"}:
            self._docker_terminal_event_seen = True
            self.docker_import_progress.setVisible(True)
            self.docker_import_progress.setValue(100)
            self.docker_import_detail_label.setVisible(True)
            self.docker_import_detail_label.setText(
                "镜像数据已导入，并已校验镜像标签、ID 与摘要。"
            )
            action = self._docker_environment_action or "Docker 镜像操作"
            self.docker_status_label.setStyleSheet(f"color: {COLORS['success']};")
            self.docker_status_label.setText(f"{action}完成，正在重新检测转换环境…")
            self.inspect_docker_environment()
        elif event_type == "process_finished":
            if not self._docker_terminal_event_seen:
                success = bool(payload.get("success"))
                action = self._docker_environment_action or "Docker 镜像操作"
                if success:
                    self.docker_status_label.setStyleSheet(
                        f"color: {COLORS['success']};"
                    )
                    self.docker_status_label.setText(
                        f"{action}完成，正在重新检测转换环境…"
                    )
                    self.inspect_docker_environment()
                else:
                    exit_code = payload.get("exit_code")
                    suffix = (
                        f"（退出码 {exit_code}）" if exit_code is not None else ""
                    )
                    self.docker_status_label.setStyleSheet(
                        f"color: {COLORS['danger']};"
                    )
                    self.docker_status_label.setText(
                        f"{action}失败{suffix}，请查看部署日志。"
                    )
            self._docker_environment_action = ""
            self._docker_terminal_event_seen = False

    @staticmethod
    def _format_bytes(value: float | int) -> str:
        amount = max(0.0, float(value))
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024.0 or unit == "TiB":
                decimals = 0 if unit == "B" else 1
                return f"{amount:.{decimals}f} {unit}"
            amount /= 1024.0
        return f"{amount:.1f} TiB"

    def respond_to_deployment_size_warning(self, accepted: bool) -> None:
        """Resume or stop packaging after the non-blocking 30 MB warning."""

        payload = self._pending_deploy_confirmation or {}
        response = {
            "job_id": self._active_job_id,
            "accepted": bool(accepted),
            "reason": "package_size_warning",
            "payload": payload,
        }
        method = self._member(
            "respond_to_job_confirmation",
            "confirm_deployment_size",
            "confirm_job",
        )
        if callable(method):
            self._invoke_by_arity(
                method,
                [
                    (self._active_job_id, bool(accepted)),
                    (bool(accepted),),
                    (response,),
                ],
            )
        elif self.process_bridge.is_running:
            self.process_bridge.write_message(
                {
                    "protocol_version": UI_PROTOCOL_VERSION,
                    "job_id": self._active_job_id,
                    "type": "confirmation",
                    "accepted": bool(accepted),
                    "reason": "package_size_warning",
                }
            )
        self._pending_deploy_confirmation = None
        self.deploy_status_label.setStyleSheet(
            f"color: {COLORS['warning' if accepted else 'danger']};"
        )
        self.deploy_status_label.setText(
            "已确认超限，继续生成部署包。" if accepted else "用户取消了超限部署包生成。"
        )

    # ---- Maix deployment ------------------------------------------------

    def _persist_docker_target(self) -> None:
        target = str(self.docker_target_combo.currentData() or "maixcam2")
        method = self._member("set_last_maix_target")
        if callable(method):
            try:
                method(target)
            except Exception as exc:
                self.deploy_log.appendPlainText(f"保存目标设备选择失败：{exc}")

    def start_docker_desktop(self) -> bool:
        method = self._member("start_docker_desktop")
        if not callable(method):
            self._show_error(
                "无法启动 Docker Desktop",
                "控制器未实现 Docker Desktop 恢复接口。",
            )
            return False
        if self.isVisible():
            answer = QMessageBox.question(
                self,
                "启动 Docker Desktop",
                "软件将启动本机已安装的 Docker Desktop，并每 2 秒检测一次 "
                "Linux Engine，最多等待 120 秒。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        target = str(self.docker_target_combo.currentData() or "maixcam2")
        try:
            status = method({"target": target, "confirmed": True})
        except Exception as exc:
            self._show_error("启动 Docker Desktop 失败", exc)
            return False
        self._docker_recovery_started_at = time.monotonic()
        self.docker_start_button.setEnabled(False)
        self._show_docker_recovery_status(status)
        state = str(_value(status, "state", ""))
        if state == "ready":
            self.inspect_docker_environment()
        elif bool(_value(status, "should_poll", state == "starting")):
            self._docker_recovery_timer.start()
        return True

    def _poll_docker_recovery(self) -> None:
        method = self._member("docker_desktop_recovery_status")
        if not callable(method):
            self._docker_recovery_timer.stop()
            return
        target = str(self.docker_target_combo.currentData() or "maixcam2")
        elapsed = max(0.0, time.monotonic() - self._docker_recovery_started_at)
        try:
            status = method(
                target,
                launch_requested=True,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            self._docker_recovery_timer.stop()
            self.docker_status_label.setStyleSheet(f"color: {COLORS['danger']};")
            self.docker_status_label.setText(f"等待 Docker Desktop 失败：{exc}")
            return
        self._show_docker_recovery_status(status)
        state = str(_value(status, "state", ""))
        if state == "ready":
            self._docker_recovery_timer.stop()
            self.inspect_docker_environment()
        elif not bool(_value(status, "should_poll", False)):
            self._docker_recovery_timer.stop()
            self.docker_start_button.setEnabled(bool(_value(status, "can_start", False)))

    def _show_docker_recovery_status(self, status: object) -> None:
        state = str(_value(status, "state", ""))
        message = str(_value(status, "message", "") or "")
        elapsed = float(_value(status, "elapsed_seconds", 0.0) or 0.0)
        timeout = float(_value(status, "timeout_seconds", 120.0) or 120.0)
        if state == "starting":
            message = f"{message} 已等待 {elapsed:.0f}/{timeout:.0f} 秒。"
        color = (
            COLORS["success"]
            if state == "ready"
            else COLORS["warning"]
            if state in {"starting", "stopped"}
            else COLORS["danger"]
        )
        self.docker_status_label.setStyleSheet(f"color: {color};")
        self.docker_status_label.setText(message or "Docker Desktop 状态未知。")

    def inspect_docker_environment(self) -> bool:
        """Read Docker conversion prerequisites without changing the host."""

        method = self._member(
            "inspect_conversion_environment",
            "inspect_docker_environment",
        )
        if not callable(method):
            self._show_error(
                "无法检测转换环境",
                "控制器未实现 Docker 转换环境检测接口。",
            )
            return False
        target = str(self.docker_target_combo.currentData() or "")
        self.docker_status_label.setStyleSheet("")
        self.docker_status_label.setText("正在检测 Docker、WSL2、转换镜像和目录挂载…")
        QApplication.processEvents()
        try:
            report = self._invoke_by_arity(method, [(target,), ()])
        except Exception as exc:
            self.docker_status_label.setStyleSheet(f"color: {COLORS['danger']};")
            self.docker_status_label.setText(f"转换环境检测失败：{exc}")
            return False
        ready = self._show_docker_environment_report(report)
        self._set_status("Docker 转换环境可用" if ready else "Docker 转换环境尚未就绪")
        return ready

    def _show_docker_environment_report(self, report: object) -> bool:
        converter = getattr(report, "to_dict", None)
        if callable(converter):
            values = converter()
        elif isinstance(report, Mapping):
            values = dict(report)
        else:
            values = {
                name: getattr(report, name, None)
                for name in (
                    "ready",
                    "executable",
                    "client_version",
                    "server_version",
                    "daemon_ready",
                    "wsl2_ready",
                    "mount_ready",
                    "images",
                    "errors",
                    "warnings",
                )
            }
        ready = bool(values.get("ready"))
        lines = [
            f"Docker CLI：{values.get('executable') or '未找到'}",
            (
                "Docker daemon："
                + (
                    f"可用（Client {values.get('client_version') or '未知'} / "
                    f"Server {values.get('server_version') or '未知'}）"
                    if values.get("daemon_ready")
                    else "不可用"
                )
            ),
            f"WSL2：{self._readiness_text(values.get('wsl2_ready'))}",
            f"目录挂载：{self._readiness_text(values.get('mount_ready'))}",
        ]
        for image in _as_records(values.get("images")):
            available = _value(image, "available")
            image_id = str(_value(image, "image_id", "") or "")
            digests = _as_records(_value(image, "repo_digests", ()))
            identity = image_id[:19] if image_id else ""
            if digests:
                identity = str(digests[0])
            detail = f"（{identity}）" if identity else ""
            if available is not True and _value(image, "error"):
                detail = f"（{_value(image, 'error')}）"
            status = (
                "已加载"
                if available is True
                else "缺失"
                if available is False
                else "未检查"
            )
            lines.append(
                f"镜像 {_value(image, 'name', '未知')}："
                f"{status}{detail}"
            )
        warnings = [str(item) for item in _as_records(values.get("warnings"))]
        errors = [str(item) for item in _as_records(values.get("errors"))]
        if warnings:
            lines.append("警告：" + "；".join(warnings))
        if errors:
            lines.append("错误：" + "；".join(errors))
        self.docker_status_label.setText("\n".join(lines))
        color = COLORS["success"] if ready else COLORS["danger"] if errors else COLORS["warning"]
        self.docker_status_label.setStyleSheet(f"color: {color};")
        self.docker_start_button.setEnabled(False)
        if not values.get("daemon_ready"):
            recovery = self._member("docker_desktop_recovery_status")
            if callable(recovery):
                try:
                    target = str(
                        self.docker_target_combo.currentData() or "maixcam2"
                    )
                    status_report = recovery(target)
                    self.docker_start_button.setEnabled(
                        bool(_value(status_report, "can_start", False))
                    )
                    recovery_message = str(
                        _value(status_report, "message", "") or ""
                    )
                    if recovery_message:
                        self.docker_status_label.setText(
                            self.docker_status_label.text()
                            + f"\n恢复建议：{recovery_message}"
                        )
                except Exception as exc:
                    self.deploy_log.appendPlainText(
                        f"Docker Desktop 恢复状态检查失败：{exc}"
                    )
        return ready

    @staticmethod
    def _readiness_text(value: object) -> str:
        if value is True:
            return "可用"
        if value is False:
            return "不可用"
        return "未检查"

    def import_docker_image(self) -> bool:
        archive, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Docker 转换镜像归档",
            "",
            "Docker 镜像归档 (*.tar *.tar.gz *.tgz);;所有文件 (*)",
        )
        if not archive:
            return False
        answer = QMessageBox.question(
            self,
            "确认导入转换镜像",
            "Docker 将从以下归档导入镜像，这会占用本机磁盘空间：\n"
            f"{archive}\n\n确认继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        method = self._member("import_converter_image")
        if not callable(method):
            self._show_error("无法导入转换镜像", "控制器未实现镜像导入接口。")
            return False
        try:
            result = method(
                {
                    "path": archive,
                    "target": str(self.docker_target_combo.currentData() or ""),
                    "confirmed": True,
                }
            )
            self._start_docker_environment_job(result, "正在导入 Docker 转换镜像…")
        except Exception as exc:
            self._set_busy("", False)
            self._show_error("导入转换镜像失败", exc)
            return False
        return True

    def pull_docker_image(self) -> bool:
        target = str(self.docker_target_combo.currentData() or "")
        if target == "maixcam2":
            QMessageBox.information(
                self,
                "请导入 Pulsar2 镜像归档",
                "Pulsar2 官方转换环境以 tar 归档发布，不能直接 docker pull。\n\n"
                "官方下载页：\n"
                "https://huggingface.co/AXERA-TECH/Pulsar2/tree/main/6.0\n\n"
                "下载 ax_pulsar2_6.0.tar.gz（或 lite 版本）后，"
                "点击“导入镜像”执行 docker load。",
            )
            return False
        image = {
            "maixcam_pro": "sophgo/tpuc_dev:latest",
        }.get(target, target)
        answer = QMessageBox.question(
            self,
            "确认拉取转换镜像",
            "Docker 将从官方镜像源下载转换环境，下载量和磁盘占用可能较大：\n"
            f"{image}\n\n确认继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        method = self._member("pull_converter_image")
        if not callable(method):
            self._show_error("无法拉取转换镜像", "控制器未实现镜像拉取接口。")
            return False
        try:
            result = method({"target": target, "confirmed": True})
            self._start_docker_environment_job(result, f"正在拉取 {image}…")
        except Exception as exc:
            self._set_busy("", False)
            self._show_error("拉取转换镜像失败", exc)
            return False
        return True

    def _start_docker_environment_job(self, result: object, message: str) -> None:
        self._docker_environment_action = message.rstrip("…")
        self._docker_terminal_event_seen = False
        self.deploy_log.appendPlainText(message)
        self.docker_status_label.setStyleSheet("")
        self.docker_status_label.setText(message)
        self.docker_import_progress.setVisible(True)
        self.docker_import_progress.setValue(0)
        self.docker_import_detail_label.setVisible(True)
        self.docker_import_detail_label.setText("等待 Docker 返回首个进度事件…")
        self._set_busy("docker_environment", True)
        self._consume_job_result(result, "docker_environment")

    def _remember_deployment_artifact(self, payload: object) -> None:
        kind = str(_value(payload, "kind", "") or "").casefold()
        path_value = str(_value(payload, "path", "") or "").strip()
        if not path_value or kind not in {
            "maix_app_package",
            "maix_editable_project",
            "maix_model_package",
            "maixapp",
            "editable-project",
            "model-only",
        }:
            return
        path = Path(path_value)
        if kind in {"maix_app_package", "maixapp"} or path.suffix.casefold() == (
            ".maixapp"
        ):
            self._last_maixapp_path = path
        self._remember_deployment_path(path)

    def _remember_deployment_path(self, path: Path) -> None:
        self._last_deploy_output_dir = path.parent

    def open_deploy_output_directory(self) -> bool:
        directory = self._last_deploy_output_dir
        if directory is None or not directory.is_dir():
            self._show_error("无法打开产物目录", "尚未找到已生成的产物目录。")
            return False
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        if not opened:
            self._show_error("无法打开产物目录", f"系统未能打开：{directory}")
            return False
        return True

    def open_generated_maixapp(self) -> bool:
        path = self._last_maixapp_path
        if path is None or not path.is_file():
            self._show_error("无法打开 .maixapp", "尚未找到已生成的 .maixapp 文件。")
            return False
        if self.isVisible():
            answer = QMessageBox.question(
                self,
                "交给 MaixVision 打开",
                "这一步只会把 .maixapp 交给系统关联的 MaixVision/默认程序，"
                "不会自动证明已经安装到设备。请在 MaixVision 中连接设备并确认安装。\n\n"
                f"文件：{path}\n\n继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if not opened:
            self._show_error(
                "无法打开 .maixapp",
                "系统没有可用的文件关联。请先启动 MaixVision，再手工选择该文件。",
            )
            return False
        return True

    def cleanup_old_backups_after_deploy(self) -> bool:
        previewer = self._member(
            "preview_backup_cleanup",
            "preview_annotation_backup_cleanup",
        )
        cleaner = self._member(
            "cleanup_old_backups",
            "cleanup_old_annotation_backups",
        )
        if not callable(previewer) or not callable(cleaner):
            self._show_error("无法清理备份", "控制器未实现受保护的备份清理接口。")
            return False
        if self.isVisible():
            verified = QMessageBox.warning(
                self,
                "确认已完成真机验证",
                "只有当部署文件已经安装到 MaixCAM，并且在真机上运行验证成功后，"
                "才能继续。\n\n我确认：本次部署已经在真机验证成功。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if verified != QMessageBox.StandardButton.Yes:
                return False
        marker = self._member("mark_deployment_verified")
        if callable(marker) and self._last_completed_deploy_run_id:
            try:
                marker(self._last_completed_deploy_run_id)
            except Exception as exc:
                self._show_error("记录真机验证状态失败", exc)
                return False
        try:
            preview = previewer(keep_latest=0, include_recovery_trash=True)
        except Exception as exc:
            self._show_error("备份清理预览失败", exc)
            return False
        count = int(_value(preview, "backup_count", 0) or 0)
        total_bytes = int(_value(preview, "total_bytes", 0) or 0)
        if count == 0:
            if self.isVisible():
                QMessageBox.information(self, "无需清理", "当前项目没有可清理的备份。")
            return True
        if self.isVisible():
            confirm = QMessageBox.warning(
                self,
                "永久删除全部旧备份",
                f"将从电脑永久删除 {count} 个旧备份"
                f"（{self._format_bytes(total_bytes)}）。此操作不可撤销。\n\n"
                "不会删除当前标注数据库、图片、类别、模型或部署文件。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return False
        try:
            report = cleaner(
                keep_latest=0,
                deployment_verified=True,
                permanently_delete=True,
            )
        except Exception as exc:
            self._show_error("清理备份失败", exc)
            return False
        deleted = int(_value(report, "deleted_count", 0) or 0)
        message = f"已永久删除 {deleted} 个旧备份文件。"
        self._set_status(message)
        if self.isVisible():
            QMessageBox.information(self, "备份清理完成", message)
        return True

    def open_deploy_wizard(self) -> None:
        checkpoints: list[dict[str, Any]] = []
        for index in range(self.deploy_checkpoint_combo.count()):
            choice = self.deploy_checkpoint_combo.itemData(index)
            if isinstance(choice, Mapping) and choice.get("checkpoint"):
                checkpoints.append(
                    {
                        "name": self.deploy_checkpoint_combo.itemText(index),
                        **dict(choice),
                    }
                )
        project = self._read("current_project", "project", default=None)
        project_root = _value(project, "root", _value(project, "path", project))
        project_config = _value(project, "config")
        project_id = _value(
            project_config,
            "project_id",
            _value(project, "project_id"),
        )
        current_images = _as_records(self._read("list_images", "images", default=[]))
        verified_images = [
            image
            for image in current_images
            if _enum_text(_value(image, "review_status"), "unreviewed") == "verified"
        ]
        recommended_calibration_ids: Sequence[str] = ()
        recommender = self._member(
            "recommend_calibration_image_ids",
            "recommend_calibration_images",
        )
        if callable(recommender):
            try:
                recommended_calibration_ids = tuple(
                    str(image_id)
                    for image_id in self._invoke_by_arity(
                        recommender,
                        [(100, 42), (100,), ()],
                    )
                )
            except Exception as exc:
                self.deploy_log.appendPlainText(
                    f"校准图片智能推荐失败，已回退到固定顺序：{exc}"
                )
        selected_choice = self.deploy_checkpoint_combo.currentData()
        selected_model = (
            str(selected_choice.get("model_key"))
            if isinstance(selected_choice, Mapping) and selected_choice.get("model_key")
            else self.model_combo.currentText()
        )
        dialog = MaixDeployDialog(
            checkpoints,
            model_key=selected_model,
            calibration_images=verified_images,
            recommended_calibration_ids=recommended_calibration_ids,
            verified_image_count=len(verified_images),
            project_root=project_root,
            project_id=None if project_id is None else str(project_id),
            initial_target=str(
                self.docker_target_combo.currentData() or "maixcam2"
            ),
            parent=self,
        )
        selected = self.deploy_checkpoint_combo.currentData()
        index = dialog.checkpoint_combo.findData(selected)
        if index >= 0:
            dialog.checkpoint_combo.setCurrentIndex(index)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dialog.deployment_config()
        payload["ml_environment"] = self._ml_environment or None
        method = self._member(
            "start_maix_deploy",
            "prepare_maix_deployment",
            "export_maix",
            "deploy_maix",
        )
        if not callable(method):
            self._show_error("无法生成部署包", "控制器未实现 MaixCAM-Pro 部署接口。")
            return
        self.deploy_log.clear()
        self.deploy_progress.setValue(0)
        self.deploy_status_label.setText("正在准备模型转换…")
        self._last_deploy_output_dir = None
        self._last_maixapp_path = None
        self._last_completed_deploy_run_id = ""
        self.open_deploy_output_button.setEnabled(False)
        self.open_maixapp_button.setEnabled(False)
        self.cleanup_backups_button.setEnabled(False)
        self._set_busy("deploy", True)
        try:
            result = method(payload)
            self._consume_job_result(result, "deploy")
        except Exception as exc:  # controller boundary
            self._set_busy("", False)
            self._show_error("启动部署转换失败", exc)

    # ---- File actions ---------------------------------------------------

    def new_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择新项目目录")
        if not path:
            return
        method = self._member("new_project", "create_project")
        if not callable(method):
            self._show_error("无法新建项目", "控制器未实现 new_project。")
            return
        try:
            method(path)
        except Exception as exc:
            self._show_error("新建项目失败", exc)
            return
        self._current_image = None
        self.refresh_project()

    def open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "打开数据标注项目")
        if not path:
            return
        method = self._member("open_project")
        if not callable(method):
            self._show_error("无法打开项目", "控制器未实现 open_project。")
            return
        try:
            method(path)
        except Exception as exc:
            self._show_error("打开项目失败", exc)
            return
        self._current_image = None
        self.refresh_project()

    def import_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if not paths:
            return
        method = self._member("import_images")
        if not callable(method):
            self._show_error("无法导入图片", "控制器未实现 import_images。")
            return
        try:
            report = method(paths)
        except Exception as exc:
            self._show_error("导入图片失败", exc)
            return
        self.refresh_images()
        if isinstance(report, Mapping) and self.isVisible():
            ImportReportDialog(report, self).exec()

    def import_voc_dataset(self) -> None:
        """Import a validated MaixHub/Pascal VOC folder through the controller."""

        source = QFileDialog.getExistingDirectory(
            self,
            "选择包含 images 和 annotations 的 MaixHub/VOC 数据集目录",
        )
        if not source:
            return

        original_button_text = self.import_voc_button.text()
        stage_cursor_active = False

        def begin_stage(message: str) -> None:
            nonlocal stage_cursor_active
            self.import_voc_button.setText(message)
            self.import_voc_button.setEnabled(False)
            self.import_voc_action.setEnabled(False)
            self._set_status(message)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            stage_cursor_active = True
            QApplication.processEvents()

        def end_stage() -> None:
            nonlocal stage_cursor_active
            if stage_cursor_active:
                QApplication.restoreOverrideCursor()
                stage_cursor_active = False
            self.import_voc_button.setText(original_button_text)
            self.import_voc_button.setEnabled(True)
            self.import_voc_action.setEnabled(True)
            QApplication.processEvents()

        inspector = self._member("inspect_voc_import", "preflight_voc_import")
        importer = self._member("import_voc_dataset")
        if not callable(inspector) or not callable(importer):
            self._show_error("无法导入 VOC", "控制器未实现 MaixHub/VOC 导入功能。")
            return
        begin_stage("正在检查 VOC 数据集…")
        logger.info("VOC 导入开始预检查：source=%s", source)
        try:
            preflight = inspector(source)
        except Exception as exc:  # controller boundary
            logger.exception("VOC 数据集初始预检查失败：source=%s", source)
            end_stage()
            self._show_error("VOC 数据集检查失败", exc)
            return
        end_stage()
        if not isinstance(preflight, Mapping):
            self._show_error("VOC 数据集检查失败", "控制器返回了无效的预检查结果。")
            return
        logger.info(
            "VOC 数据集初始预检查完成：source=%s images=%s boxes=%s",
            source,
            preflight.get("image_count", 0),
            preflight.get("box_count", 0),
        )
        project = self._read("current_project", "project", default=None)
        dialog = VocImportDialog(
            source,
            preflight,
            existing_category_names=[
                str(_value(category, "name", ""))
                for category in self._category_records
                if str(_value(category, "name", "")).strip()
            ],
            current_project_root=(
                None if project is None else _value(project, "root", _value(project, "path"))
            ),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dialog.payload()
        mode = str(payload["mode"])
        if mode == "merge":
            begin_stage(
                f"正在重新预检 {int(preflight.get('image_count', 0))} 张图片…"
            )
            try:
                mapped_preflight = inspector(
                    source,
                    category_mapping=payload["category_mapping"],
                )
            except Exception as exc:
                logger.exception("VOC 合并映射预检查失败：source=%s", source)
                end_stage()
                self._show_error("VOC 合并预检查失败", exc)
                return
            end_stage()
            if not isinstance(mapped_preflight, Mapping):
                self._show_error("VOC 合并预检查失败", "控制器返回了无效的预检查结果。")
                return
            imported = int(mapped_preflight.get("new_image_count", 0))
            upgraded = int(mapped_preflight.get("upgraded_image_count", 0))
            conflicts = int(mapped_preflight.get("conflict_count", 0))
            annotated = int(mapped_preflight.get("annotated_image_count", 0))
            negatives = int(mapped_preflight.get("verified_negative_count", 0))
            unconfirmed = int(mapped_preflight.get("unconfirmed_image_count", 0))
            preserved_unconfirmed = int(
                mapped_preflight.get("preserved_unconfirmed_count", 0)
            )
            if self.isVisible():
                response = QMessageBox.question(
                    self,
                    "确认合并 VOC 标注",
                    "本次将导入 "
                    f"{imported} 张新图片，安全升级 {upgraded} 张 AI 草稿，"
                    f"保留 {conflicts} 张人工标注冲突图片。\n"
                    f"源数据：有框已确认 {annotated} 张，确认空负样本 {negatives} 张，"
                    f"无 XML 未确认 {unconfirmed} 张；其中 {preserved_unconfirmed} 张重复的"
                    "无 XML 图片将保留项目原状态。\n\n"
                    "人工确认或人工修改过的图片不会被覆盖。是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if response != QMessageBox.StandardButton.Yes:
                    logger.info("用户取消 VOC 合并：source=%s", source)
                    return

        # An empty project has no current image.  There is nothing to save in
        # that case, and it must not silently block a valid VOC import.
        if self._current_image is not None and not self.save_current_annotations(
            silent=True
        ):
            logger.warning("VOC 导入未开始：当前图片标注保存失败，source=%s", source)
            self._set_status("VOC 导入未开始：当前图片标注保存失败")
            return

        image_count = int(preflight.get("image_count", 0))
        begin_stage(f"正在导入 {image_count} 张 VOC 图片，请勿关闭软件…")
        logger.info(
            "VOC 正式导入开始：source=%s mode=%s images=%s",
            source,
            mode,
            image_count,
        )
        try:
            result = importer(
                source,
                mode=mode,
                destination=payload.get("destination"),
                project_name=payload.get("project_name"),
                category_mapping=payload.get("category_mapping"),
            )
        except Exception as exc:  # controller boundary
            logger.exception("VOC 正式导入失败：source=%s mode=%s", source, mode)
            end_stage()
            self._show_error("VOC 导入失败", exc)
            return
        end_stage()
        logger.info("VOC 正式导入完成：source=%s mode=%s", source, mode)
        self._current_image = None
        self._annotations_dirty = False
        self.refresh_project()
        result_map = result if isinstance(result, Mapping) else {}
        if mode == "merge":
            message = (
                "VOC 合并完成："
                f"新图片 {int(result_map.get('imported_image_count', 0))} 张，"
                f"升级 AI 草稿 {int(result_map.get('upgraded_image_count', 0))} 张，"
                f"保留冲突 {int(result_map.get('conflict_image_count', 0))} 张；"
                f"源数据有框已确认 {int(result_map.get('source_annotated_image_count', 0))} 张，"
                f"确认空负样本 {int(result_map.get('source_verified_negative_count', 0))} 张，"
                f"无 XML 未确认 {int(result_map.get('source_unconfirmed_image_count', 0))} 张。"
            )
        else:
            message = (
                "VOC 新项目创建完成："
                f"{int(result_map.get('image_count', 0))} 张图片，"
                f"{int(result_map.get('box_count', 0))} 个标注框；"
                f"有框已确认 {int(result_map.get('annotated_image_count', 0))} 张，"
                f"确认空负样本 {int(result_map.get('verified_negative_count', 0))} 张，"
                f"无 XML 未确认 {int(result_map.get('unconfirmed_image_count', 0))} 张。"
            )
        report_path = result_map.get("report_path")
        if report_path:
            message += f" 报告：{report_path}"
        self._set_status(message)
        if self.isVisible():
            QMessageBox.information(self, "VOC 导入完成", message)

    def clear_all_annotations(self) -> None:
        """Show a safe multi-image selection dialog and clear selected boxes."""

        if not self._image_records:
            self._show_error("无法删除标记", "当前项目没有可处理的图片。")
            return
        current_id = _value(self._current_image, "id") if self._current_image is not None else None
        dialog = BulkAnnotationClearDialog(
            self._image_records,
            current_image_id=current_id,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        image_ids = dialog.selected_image_ids()
        if not image_ids:
            return
        if not self.save_current_annotations(silent=True):
            return
        previewer = self._member(
            "preview_clear_all_annotations",
            "preview_delete_all_annotations",
        )
        clearer = self._member("clear_all_annotations", "delete_all_annotations")
        if not callable(previewer) or not callable(clearer):
            self._show_error("无法删除标记", "控制器未实现批量删除标记功能。")
            return
        try:
            preview = previewer(image_ids)
        except Exception as exc:  # controller boundary
            self._show_error("删除前检查失败", exc)
            return
        image_count = int(_value(preview, "image_count", len(image_ids)))
        box_count = int(_value(preview, "box_count", 0))
        if self.isVisible():
            response = QMessageBox.question(
                self,
                "确认删除所有标记",
                f"将删除 {image_count} 张图片中的 {box_count} 个标注框。\n\n"
                "图片不会删除，操作前会自动创建可恢复备份。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        try:
            report = clearer(image_ids)
        except Exception as exc:  # controller boundary
            self._show_error("删除所有标记失败", exc)
            return
        backup = _value(report, "backup", {})
        backup_path = _value(backup, "path", "")
        self._current_image = None
        self._annotations_dirty = False
        self.refresh_project(select_image_id=current_id)
        message = (
            f"已删除 {int(_value(report, 'box_count', box_count))} 个标注框，"
            f"影响 {int(_value(report, 'image_count', image_count))} 张图片。"
        )
        if backup_path:
            message += f" 备份：{backup_path}"
        self._set_status(message)
        if self.isVisible():
            QMessageBox.information(self, "删除所有标记完成", message)

    def restore_annotation_backup(self) -> None:
        """Restore a project-local annotation backup after an explicit confirmation."""

        project = self._read("current_project", "project", default=None)
        root = _value(project, "root", None)
        if root is None:
            self._show_error("无法恢复标注备份", "请先打开项目。")
            return
        backups_root = Path(str(root)) / "backups"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标注数据库备份",
            str(backups_root),
            "标注备份 (*.db)",
        )
        if not path:
            return
        restorer = self._member("restore_annotation_backup")
        if not callable(restorer):
            self._show_error("无法恢复标注备份", "控制器未实现标注备份恢复功能。")
            return
        if self.isVisible():
            response = QMessageBox.question(
                self,
                "确认恢复标注备份",
                "恢复会替换当前项目的标注数据库，并在恢复前再创建一份安全备份。"
                "图片、模型和训练结果不会删除。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        if not self.save_current_annotations(silent=True):
            return
        current_id = _value(self._current_image, "id") if self._current_image is not None else None
        try:
            report = restorer(path)
        except Exception as exc:  # controller boundary
            self._show_error("恢复标注备份失败", exc)
            return
        self._current_image = None
        self._annotations_dirty = False
        self.refresh_project(select_image_id=current_id)
        safety = _value(report, "safety_backup", {})
        safety_path = _value(safety, "path", "")
        message = "标注备份已恢复。"
        if safety_path:
            message += f" 恢复前安全备份：{safety_path}"
        self._set_status(message)
        if self.isVisible():
            QMessageBox.information(self, "恢复完成", message)

    def export_yolo(self) -> None:
        target = QFileDialog.getExistingDirectory(self, "选择 YOLO 导出目录")
        if not target:
            return
        method = self._member("export_yolo")
        if not callable(method):
            self._show_error("无法导出", "控制器未实现 export_yolo。")
            return
        try:
            result = method(target)
        except Exception as exc:
            self._show_error("YOLO 导出失败", exc)
            return
        self._set_status(f"YOLO 数据集已导出到：{result or target}")

    # ---- Miscellaneous --------------------------------------------------

    def _show_process_error(self, message: str) -> None:
        self._show_error("ML 子进程错误", message)

    def _show_error(self, title: str, error: object) -> None:
        message = str(error)
        self._set_status(f"{title}：{message}")
        if self.isVisible():
            QMessageBox.critical(self, title, message)

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(str(message), 8000)
        self.requestStatus.emit(str(message))

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self._closing:
            event.accept()
            return
        self.save_current_annotations(silent=True)
        busy = (
            self._ui_busy
            or self.process_bridge.is_running
            or bool(self._active_job_id)
        )
        if busy and self.isVisible():
            answer = QMessageBox.question(
                self,
                "任务仍在运行",
                "训练、推理或部署任务仍在运行。确认取消任务并退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_active_job()
        self._closing = True
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        event.accept()


__all__ = [
    "AnnotationController",
    "ElidingStatusLabel",
    "JsonlProcessBridge",
    "MODEL_OPTIONS",
    "MainWindow",
    "MetricsPlotWidget",
    "NullController",
    "TrainingMonitorWidget",
    "run_ui_demo",
]


def run_ui_demo(argv: Sequence[str] | None = None) -> int:
    """Launch the complete window with an empty controller for visual review."""

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow(NullController())
    window.show()
    return application.exec() if owns_application else 0


if __name__ == "__main__":
    raise SystemExit(run_ui_demo())
