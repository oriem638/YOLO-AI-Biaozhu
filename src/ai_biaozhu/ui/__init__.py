"""PySide6 desktop user interface."""

from .canvas import HANDLE_NAMES, AnnotationCanvas, AnnotationRectItem
from .dialogs import (
    DEFAULT_TRAINING_SETTINGS,
    MaixDeployDialog,
    MLEnvironmentDialog,
    TrainingSettingsDialog,
    training_setting_warnings,
    validate_maix_deployment,
    validate_training_settings,
)
from .main_window import (
    MODEL_OPTIONS,
    AnnotationController,
    JsonlProcessBridge,
    MainWindow,
    MetricsPlotWidget,
    NullController,
    TrainingMonitorWidget,
    run_ui_demo,
)
from .theme import APP_STYLESHEET, COLORS, apply_theme

__all__ = [
    "APP_STYLESHEET",
    "COLORS",
    "DEFAULT_TRAINING_SETTINGS",
    "HANDLE_NAMES",
    "MODEL_OPTIONS",
    "AnnotationCanvas",
    "AnnotationController",
    "AnnotationRectItem",
    "JsonlProcessBridge",
    "MLEnvironmentDialog",
    "MainWindow",
    "MaixDeployDialog",
    "MetricsPlotWidget",
    "NullController",
    "TrainingMonitorWidget",
    "TrainingSettingsDialog",
    "apply_theme",
    "run_ui_demo",
    "training_setting_warnings",
    "validate_maix_deployment",
    "validate_training_settings",
]
