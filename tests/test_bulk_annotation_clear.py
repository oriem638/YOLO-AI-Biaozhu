from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ai_biaozhu.core import AIPrediction, BoxInput, ReviewStatus
from ai_biaozhu.data import create_project


def _add_image(project, image_id: str, number: int):
    return project.repository.add_image_record(
        image_id=image_id,
        relative_path=f"images/{image_id}.jpg",
        original_name=f"{image_id}.jpg",
        source_path=None,
        sha256=f"{number:064x}",
        width=100,
        height=80,
    )


def _build_populated_project(tmp_path: Path):
    project = create_project(
        tmp_path / "project",
        name="bulk-clear",
        categories=["小刚球"],
    )
    category = project.repository.list_categories()[0]
    first = _add_image(project, "first", 1)
    second = _add_image(project, "second", 2)
    untouched = _add_image(project, "untouched", 3)
    prediction_run = project.repository.create_run(
        "predict", "YOLO26n", run_id="predict-1"
    )
    project.repository.import_ai_predictions(
        prediction_run.id,
        first.id,
        [
            AIPrediction(
                image_id=first.id,
                class_id=category.id,
                x1=5,
                y1=6,
                x2=20,
                y2=22,
                confidence=0.85,
                prediction_id="ai-1",
            )
        ],
    )
    first_state = project.repository.get_image(first.id)
    project.repository.add_box(
        first.id,
        BoxInput(category.id, 30, 10, 50, 35),
        expected_revision=first_state.revision,
    )
    first_state = project.repository.get_image(first.id)
    project.verify_image(first.id, expected_revision=first_state.revision)
    project.save_and_confirm(
        second.id,
        [BoxInput(category.id, 10, 10, 30, 30)],
    )
    project.save_and_confirm(
        untouched.id,
        [BoxInput(category.id, 15, 15, 35, 35)],
    )
    return project, category, prediction_run


def test_bulk_clear_previews_backs_up_and_restores_selected_images(
    tmp_path: Path,
) -> None:
    project, category, prediction_run = _build_populated_project(tmp_path)
    try:
        first_before = project.repository.get_image("first")
        second_before = project.repository.get_image("second")
        untouched_before = project.repository.get_image("untouched")
        untouched_boxes = project.list_boxes("untouched")

        preview = project.preview_clear_all_annotations(
            ["first", "second", "first"]
        )
        assert preview.image_ids == ("first", "second")
        assert preview.image_count == 2
        assert preview.box_count == 3
        assert preview.manual_box_count == 2
        assert preview.ai_box_count == 1
        assert preview.mixed_box_count == 0
        assert preview.verified_image_count == 2
        assert preview.ai_import_count == 1

        report = project.clear_all_annotations(preview.image_ids)
        assert report.image_count == 2
        assert report.box_count == 3
        assert report.backup.valid
        assert report.backup.path.parent == project.root / "backups"
        assert report.backup.path.is_file()
        assert report.backup.image_count == 3
        assert report.backup.box_count == 4

        for image_id, old_revision in (
            ("first", first_before.revision),
            ("second", second_before.revision),
        ):
            image = project.repository.get_image(image_id)
            assert project.list_boxes(image_id) == ()
            assert image.review_status is ReviewStatus.UNREVIEWED
            assert image.origin.value == "none"
            assert image.ai_status.value == "none"
            assert image.revision == old_revision + 1
        assert project.repository.get_image("untouched") == untouched_before
        assert project.list_boxes("untouched") == untouched_boxes
        assert project.repository.list_categories()[0] == category
        assert project.repository.get_run(prediction_run.id) == prediction_run
        assert project.list_ai_imported_image_ids(prediction_run.id) == ()

        backups = project.list_annotation_backups()
        assert backups == (report.backup,)
        restored = project.restore_annotation_backup(report.backup)
        assert restored.restored_backup.path == report.backup.path
        assert restored.safety_backup.path.is_file()
        assert len(project.list_boxes("first")) == 2
        assert len(project.list_boxes("second")) == 1
        assert project.repository.get_image("first") == first_before
        assert project.repository.get_image("second") == second_before
        assert project.list_boxes("untouched") == untouched_boxes
        assert project.list_ai_imported_image_ids(prediction_run.id) == ("first",)
    finally:
        project.close()


def test_bulk_clear_rolls_back_every_selected_image_on_midway_failure(
    tmp_path: Path,
) -> None:
    project, _category, _prediction_run = _build_populated_project(tmp_path)
    try:
        first_before = project.repository.get_image("first")
        second_before = project.repository.get_image("second")
        first_boxes = project.list_boxes("first")
        second_boxes = project.list_boxes("second")
        project.repository.connection.executescript(
            """
            CREATE TRIGGER force_second_clear_failure
            BEFORE UPDATE OF review_status ON images
            WHEN NEW.id = 'second' AND NEW.review_status = 'unreviewed'
            BEGIN
                SELECT RAISE(ABORT, 'forced bulk clear failure');
            END;
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="forced bulk clear failure"):
            project.clear_all_annotations(["first", "second"])

        assert project.repository.get_image("first") == first_before
        assert project.repository.get_image("second") == second_before
        assert project.list_boxes("first") == first_boxes
        assert project.list_boxes("second") == second_boxes
        backups = project.list_annotation_backups()
        assert len(backups) == 1
        assert backups[0].valid
        assert backups[0].box_count == 4
    finally:
        project.close()


def test_bulk_clear_never_mutates_when_required_backup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _category, _prediction_run = _build_populated_project(tmp_path)
    try:
        image_before = project.repository.get_image("first")
        boxes_before = project.list_boxes("first")

        def fail_backup(_destination):
            raise OSError("disk full")

        monkeypatch.setattr(project.repository, "backup_database", fail_backup)
        with pytest.raises(OSError, match="disk full"):
            project.clear_all_annotations(["first"])

        assert project.repository.get_image("first") == image_before
        assert project.list_boxes("first") == boxes_before
        assert project.list_annotation_backups() == ()
    finally:
        project.close()


def test_clear_resets_a_confirmed_empty_image_instead_of_creating_a_negative(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path / "project", name="negative")
    try:
        image = _add_image(project, "negative", 1)
        verified = project.verify_image(image.id, confirm_empty=True)
        report = project.clear_all_annotations(image.id)

        cleared = project.repository.get_image(image.id)
        assert report.image_count == 1
        assert report.box_count == 0
        assert report.preview.verified_image_count == 1
        assert cleared.review_status is ReviewStatus.UNREVIEWED
        assert cleared.origin.value == "none"
        assert cleared.revision == verified.revision + 1
    finally:
        project.close()


def test_restore_rejects_paths_outside_the_project_backup_directory(
    tmp_path: Path,
) -> None:
    project, _category, _prediction_run = _build_populated_project(tmp_path)
    try:
        outside = tmp_path / "outside.db"
        project.repository.backup_database(outside)
        with pytest.raises(Exception, match="backups"):
            project.restore_annotation_backup(outside)
    finally:
        project.close()
