"""Visual constants for the desktop application.

The UI intentionally keeps all colours in one module.  Apart from making the
application pleasant to use for long annotation sessions, this also prevents
annotation colours from being accidentally coupled to the widget palette.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

COLORS = {
    "window": "#171a21",
    "panel": "#20242d",
    "panel_alt": "#282d38",
    "canvas": "#111318",
    "border": "#373d4b",
    "text": "#eef1f6",
    "muted": "#9ca6b5",
    "accent": "#4c8dff",
    "accent_hover": "#6aa1ff",
    "success": "#45c486",
    "warning": "#f3b64c",
    "danger": "#ef6262",
}

_CLASS_PALETTE = (
    "#42a5f5",
    "#ef5350",
    "#66bb6a",
    "#ffa726",
    "#ab47bc",
    "#26c6da",
    "#ec407a",
    "#d4e157",
    "#7e57c2",
    "#8d6e63",
)


def class_color(index: int) -> QColor:
    """Return a stable, high-contrast colour for a category index."""

    return QColor(_CLASS_PALETTE[index % len(_CLASS_PALETTE)])


APP_STYLESHEET = f"""
QWidget {{
    color: {COLORS["text"]};
}}
QMainWindow, QDialog {{ background: {COLORS["window"]}; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QFrame#panel, QGroupBox, QTabWidget::pane {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 7px;
}}
QGroupBox {{
    margin-top: 11px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 4px;
}}
QPushButton, QToolButton {{
    background: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 5px;
    padding: 6px 10px;
}}
QPushButton:hover, QToolButton:hover {{
    border-color: {COLORS["accent_hover"]};
    background: #303747;
}}
QPushButton:pressed, QToolButton:pressed {{ background: #1d4f9d; }}
QPushButton:disabled, QToolButton:disabled {{
    color: #667080;
    background: #20242b;
}}
QPushButton[primary="true"] {{
    background: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
    color: white;
    font-weight: 600;
}}
QPushButton[danger="true"] {{ color: #ff9999; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background: #151820;
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {COLORS["accent"]};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {COLORS["accent"]}; }}
QListWidget, QTreeWidget, QTableWidget {{
    background: #151820;
    alternate-background-color: #1b1f28;
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item, QTreeWidget::item {{ padding: 5px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: #275aab;
    color: white;
}}
QHeaderView::section {{
    background: {COLORS["panel_alt"]};
    color: {COLORS["muted"]};
    border: 0;
    border-right: 1px solid {COLORS["border"]};
    padding: 5px;
}}
QProgressBar {{
    background: #151820;
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {COLORS["accent"]};
    border-radius: 3px;
}}
QTabBar::tab {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    padding: 7px 11px;
}}
QTabBar::tab:selected {{ background: {COLORS["panel_alt"]}; color: white; }}
QSplitter::handle {{ background: {COLORS["border"]}; }}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: #151820;
    border: none;
    width: 11px;
    height: 11px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #4a5262;
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}}
QStatusBar {{ color: {COLORS["muted"]}; }}
"""


def apply_theme(application: QApplication) -> None:
    """Apply the supported dark palette and stylesheet to ``application``."""

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["window"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor("#151820"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["panel"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["panel_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    application.setPalette(palette)
    application.setStyleSheet(APP_STYLESHEET)
