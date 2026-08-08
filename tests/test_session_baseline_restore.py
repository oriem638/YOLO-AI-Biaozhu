from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.core.domain import AIStatus, AnnotationOrigin, BoxInput, ReviewStatus
from ai_biaozhu.core.exceptions import RevisionConflictError
from ai_biaozhu.data.project import create_project
from ai_biaozhu.settings import SettingsStore


def _box_key(box: object) -> tuple[object, ...]:
    return tuple(
        getattr(box, name)
        for name in (
            "id",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "origin",
            "confidence",
            "model_run_id",
            "prediction_id",
        )
    )


def _add_image_record(project, *, image_id: str = "image-1"):
    return project.repository.add_image_record(
        image_id=image_id,
        relative_path=f"images/{image_id}.jpg",
        original_name=f"{image_id}.jpg",
        source_path=None,
        sha256="a" * 64,
        width=80,
        height=60,
    )


def test_session_baseline_restore_preserves_metadata_and_revision_safety(
    tmp_path: Path,
) -> None:
    project = create_project(
        tmp_path / "project",
        name="session",
        categories=({"name": "钢球", "color": "#45C486"},),
    )
    try:
        category = project.repository.list_categories()[0]
        image = _add_image_record(project)
        baseline = project.save_and_confirm(
            image.id,
            [BoxInput(category.id, 4, 5, 30, 35)],
            confirm_empty=True,
        )
        baseline_boxes = project.list_boxes(image.id)

        project.save_boxes(image.id, [BoxInput(category.id, 10, 10, 40, 45)])
        changed = project.repository.get_image(image.id)
        assert changed.review_status is ReviewStatus.UNREVIEWED

        with pytest.raises(RevisionConflictError):
            project.restore_annotation_session_baseline(
                image.id,
                baseline_boxes,
                review_status=baseline.review_status.value,
                origin=baseline.origin.value,
                ai_status=baseline.ai_status.value,
                expected_revision=baseline.revision,
            )
        assert project.repository.get_image(image.id).revision == changed.revision

        restored = project.restore_annotation_session_baseline(
            image.id,
            baseline_boxes,
            review_status=baseline.review_status.value,
            origin=baseline.origin.value,
            ai_status=baseline.ai_status.value,
            expected_revision=changed.revision,
        )
        assert restored.review_status is ReviewStatus.VERIFIED
        assert restored.origin is AnnotationOrigin.MANUAL
        assert restored.ai_status is AIStatus.NONE
        assert restored.revision == changed.revision + 1
        assert [_box_key(box) for box in project.list_boxes(image.id)] == [
            _box_key(box) for box in baseline_boxes
        ]
    finally:
        project.close()


def _controller(tmp_path: Path) -> ApplicationController:
    paths = AppPaths(
        data=tmp_path / "app-data",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        models=tmp_path / "models",
        yolo_config=tmp_path / "yolo-config",
    )
    return ApplicationController(
        paths,
        settings=SettingsStore(tmp_path / "settings.json"),
        source_root=Path(__file__).resolve().parents[1],
    )


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_ctrl_alt_z_restores_persisted_loaded_state_after_autosave(
    tmp_path: Path, app
) -> None:
    from PySide6.QtCore import QRectF

    from ai_biaozhu.ui.main_window import MainWindow

    source = tmp_path / "source.jpg"
    Image.new("RGB", (80, 60), (20, 30, 40)).save(source)
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "project", "session-ui")
    category = project.repository.list_categories()[0]
    controller.import_images([source])
    image = project.list_images()[0]
    baseline_record = project.save_and_confirm(
        image.id,
        [BoxInput(category.id, 4, 5, 30, 35)],
        confirm_empty=True,
    )

    window = MainWindow(controller)
    try:
        assert window._current_image is not None
        baseline_canvas = window.canvas.annotations()
        assert window.canvas.add_box(QRectF(42, 8, 20, 20), category.id)
        assert window.save_current_annotations(silent=True)
        changed = project.repository.get_image(image.id)
        assert changed.review_status is ReviewStatus.UNREVIEWED

        window.undo_all_current_image()

        restored = project.repository.get_image(image.id)
        assert restored.review_status is baseline_record.review_status
        assert restored.origin is baseline_record.origin
        assert restored.ai_status is baseline_record.ai_status
        assert restored.revision == changed.revision + 1
        assert window.canvas.annotations() == baseline_canvas
        assert not window._annotations_dirty
        assert not window._autosave_timer.isActive()
    finally:
        window.close()
        app.processEvents()
        controller.close_project()
