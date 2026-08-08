"""Desktop application entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ai_biaozhu import __version__
from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-biaozhu")
    parser.add_argument(
        "--project",
        type=Path,
        help="启动后打开指定项目目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="在日志中记录调试信息",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = AppPaths.discover().ensure()
    paths.apply_process_environment()
    log_path = configure_logging(paths.logs, verbose=args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("AI 标注 %s 启动；日志：%s", __version__, log_path)

    # Import Qt after writable runtime paths have been configured.
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox

    from ai_biaozhu.ui.dialogs import merged_training_settings
    from ai_biaozhu.ui.fonts import configure_application_font
    from ai_biaozhu.ui.main_window import MainWindow

    # Qt requires the policy to be set before QApplication is constructed.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QCoreApplication.setOrganizationName("AI-Biaozhu-Maintenance")
    QCoreApplication.setApplicationName("AI标注-维护版-0.2")
    QCoreApplication.setApplicationVersion(__version__)
    application = QApplication([sys.argv[0], *(argv or ())])
    application.setApplicationDisplayName(
        f"AI 数据集标注与训练 维护版 {__version__}"
    )
    font_diagnostics = configure_application_font(application, logger=logger)
    screen = application.primaryScreen()
    if screen is not None:
        logger.info(
            "Display diagnostics: screen=%s logical=%sx%s dpr=%.2f font=%s glyphs_ok=%s",
            screen.name(),
            screen.availableGeometry().width(),
            screen.availableGeometry().height(),
            screen.devicePixelRatio(),
            font_diagnostics.family,
            font_diagnostics.supports_required_glyphs,
        )

    controller = ApplicationController(paths)
    startup_error: Exception | None = None
    try:
        if args.project is not None:
            controller.open_project(args.project)
        else:
            controller.reopen_last_project()
    except Exception as exc:
        startup_error = exc
        logger.exception("打开启动项目失败")

    window = MainWindow(controller)
    stored_environment = controller.settings.get("ml_python")
    if stored_environment:
        window._ml_environment = str(stored_environment)
        window.environment_button.setText(
            f"ML 环境：{Path(str(stored_environment)).parent.name}"
        )
    stored_training = controller.settings.mapping("last_training_settings")
    if stored_training:
        window._training_settings = merged_training_settings(stored_training)
        window._update_settings_summary()

    def process_finished(payload: dict[str, object]) -> None:
        controller.handle_process_finished(
            window.process_bridge.job_id,
            success=bool(payload.get("success")),
            exit_code=int(payload.get("exit_code", -1)),
        )

    window.process_bridge.finished.connect(process_finished)
    application.aboutToQuit.connect(controller.close_project)
    window.show()
    if startup_error is not None:
        QMessageBox.warning(
            window,
            "启动项目无法打开",
            f"{startup_error}\n\n应用已继续启动，请手动选择项目。",
        )
    exit_code = application.exec()
    logger.info("AI 标注退出，代码 %s", exit_code)
    return int(exit_code)


if __name__ == "__main__":
    # This branch is useful when app.py is launched directly during diagnosis.
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
