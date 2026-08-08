from __future__ import annotations

from pathlib import Path

import pytest

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.core import BoxInput
from ai_biaozhu.core.exceptions import DataIntegrityError
from ai_biaozhu.data.project import create_project
from ai_biaozhu.settings import SettingsStore


def _image(project, *, image_id: str, sha256: str):
    return project.repository.add_image_record(
        image_id=image_id,
        relative_path=f"images/{image_id}.jpg",
        original_name=f"{image_id}.jpg",
        source_path=None,
        sha256=sha256,
        width=640,
        height=480,
    )


def _controller(tmp_path: Path) -> ApplicationController:
    paths = AppPaths(
        data=tmp_path / "app-data",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        models=tmp_path / "models",
        yolo_config=tmp_path / "ultralytics",
    )
    return ApplicationController(
        paths,
        settings=SettingsStore(tmp_path / "settings.json"),
        source_root=Path(__file__).resolve().parents[1],
    )


def test_delete_empty_category_backs_up_and_unblocks_training_preflight(
    tmp_path: Path,
) -> None:
    with create_project(
        tmp_path / "project",
        name="empty-category",
        categories=["BALL", "钢球"],
    ) as project:
        ball, accidental = project.repository.list_categories()
        image = _image(project, image_id="verified-ball", sha256="1" * 64)
        project.save_and_confirm(
            image.id,
            [BoxInput(ball.id, 100, 100, 140, 140)],
        )

        before = project.training_preflight(minimum=1)
        assert not before.ok
        assert before.class_instance_counts == {ball.id: 1, accidental.id: 0}
        assert any("钢球" in error for error in before.errors)

        deleted, backup = project.delete_empty_category(accidental.id)

        assert deleted.id == accidental.id
        assert deleted.name == "钢球"
        assert backup.valid
        assert backup.path.is_file()
        assert backup.reason == "before-empty-category-delete"
        assert [category.name for category in project.repository.list_categories()] == [
            "BALL"
        ]
        boxes = project.repository.list_boxes(image.id)
        assert len(boxes) == 1
        assert boxes[0].class_id == ball.id
        assert project.repository.get_image(image.id).training_selected
        assert project.training_preflight(minimum=1).ok


def test_delete_empty_category_uses_global_box_count_not_training_subset(
    tmp_path: Path,
) -> None:
    with create_project(
        tmp_path / "project",
        name="global-usage-guard",
        categories=["BALL", "钢球"],
    ) as project:
        _ball, steel = project.repository.list_categories()
        draft = _image(project, image_id="unreviewed-steel", sha256="2" * 64)
        project.repository.add_box(
            draft.id,
            BoxInput(steel.id, 10, 20, 30, 40),
        )
        project.repository.set_training_selected((draft.id,), False)
        backups_before = project.list_annotation_backups()

        # This class has zero trainable instances, but it is not globally empty.
        preflight = project.training_preflight(minimum=1)
        assert preflight.class_instance_counts[steel.id] == 0
        with pytest.raises(DataIntegrityError, match=r"钢球.*1 个标注框"):
            project.delete_empty_category(steel.id)

        assert project.repository.get_category(steel.id).name == "钢球"
        assert project.repository.category_box_count(steel.id) == 1
        assert project.repository.list_boxes(draft.id)[0].class_id == steel.id
        assert project.list_annotation_backups() == backups_before


def test_controller_delete_empty_category_reports_backup_and_preserves_other_data(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    project = controller.new_project(tmp_path / "controller-project", "controller")
    try:
        ball = project.repository.update_category(
            project.repository.list_categories()[0].id,
            name="BALL",
        )
        accidental = project.repository.add_category("钢球")
        image = _image(project, image_id="controller-ball", sha256="3" * 64)
        project.save_and_confirm(
            image.id,
            [BoxInput(ball.id, 50, 60, 80, 90)],
        )

        result = controller.delete_empty_category(accidental.id)

        assert result["category"]["id"] == accidental.id
        assert result["category"]["name"] == "钢球"
        assert result["backup"]["valid"] is True
        assert Path(result["backup"]["path"]).is_file()
        assert [category.name for category in controller.list_classes()] == ["BALL"]
        assert project.repository.list_boxes(image.id)[0].class_id == ball.id
        assert controller.training_preflight()["class_box_counts"] == {"BALL": 1}
    finally:
        controller.close_project()


def test_delete_empty_category_refuses_to_remove_the_last_category(
    tmp_path: Path,
) -> None:
    with create_project(
        tmp_path / "project",
        name="last-category-guard",
        categories=["BALL"],
    ) as project:
        category = project.repository.list_categories()[0]
        backups_before = project.list_annotation_backups()

        with pytest.raises(DataIntegrityError, match="至少保留一个类别"):
            project.delete_empty_category(category.id)

        assert project.repository.get_category(category.id).name == "BALL"
        assert project.list_annotation_backups() == backups_before
