from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from ai_biaozhu.core import (
    AIPrediction,
    BoxInput,
    DataIntegrityError,
    EmptyAnnotationConfirmationRequired,
    ModelKey,
    ReviewStatus,
    RevisionConflictError,
    SplitConfig,
    TrainingConfig,
)
from ai_biaozhu.data import create_project, open_project


def _make_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    Image.new("RGB", size, (30, 90, 150)).save(path)


def test_training_config_accepts_exact_model_matrix() -> None:
    expected = {
        "YOLOv5n": "yolov5n.pt",
        "YOLOv5s": "yolov5s.pt",
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLO11n": "yolo11n.pt",
        "YOLO11s": "yolo11s.pt",
        "YOLO26n": "yolo26n.pt",
        "YOLO26s": "yolo26s.pt",
    }
    assert {item.value for item in ModelKey} == set(expected)
    for key, weight in expected.items():
        assert TrainingConfig(model_key=key).weight_name == weight
    assert TrainingConfig().model_key is ModelKey.YOLO26N
    with pytest.raises(ValueError):
        TrainingConfig(model_key="YOLOv10n")
    with pytest.raises(ValueError):
        TrainingConfig(imgsz=650)
    with pytest.raises(ValueError):
        TrainingConfig(batch=0)
    with pytest.raises(ValueError):
        TrainingConfig(imgsz=128)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=10, patience=11)
    with pytest.raises(ValueError):
        TrainingConfig(workers=33)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=1.5)  # type: ignore[arg-type]
    assert SplitConfig(seed=1, val_ratio=0.3).train_ratio == pytest.approx(0.7)
    with pytest.raises(ValueError):
        SplitConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.1)


def test_model_runs_can_be_filtered_and_are_newest_first(tmp_path: Path) -> None:
    with create_project(tmp_path / "project", name="runs") as project:
        train = project.repository.create_run("train", "YOLO26n", run_id="a-train")
        predict = project.repository.create_run(
            "predict", "YOLO11s", run_id="b-predict"
        )
        project.repository.update_run(train.id, status="completed", progress=1)
        assert [item.id for item in project.list_runs()] == [predict.id, train.id]
        assert [item.id for item in project.list_runs(kind="train")] == [train.id]
        assert [item.id for item in project.list_runs(status="completed")] == [train.id]
        deploy = project.repository.create_run("deploy", "YOLO11s", run_id="c-deploy")
        package = project.repository.create_deployment_package(
            deploy.id,
            target="maixcam2",
            checkpoint_role="best",
            npu_mode="NPU1+NPU2",
            warnings=["校准图片较少"],
            package_id="package-1",
        )
        updated = project.repository.update_deployment_package(
            package.id,
            status="completed",
            model_package_path="deployments/model.zip",
            zip_bytes=1024,
            payload_bytes=4096,
        )
        assert updated.status == "completed"
        assert updated.warnings == ("校准图片较少",)
        assert [
            value.id
            for value in project.list_deployment_packages(
                target="maixcam2", status="completed"
            )
        ] == [package.id]
        project.repository.delete_deployment_package(package.id)
        assert project.list_deployment_packages() == ()


def test_schema_v1_is_migrated_to_current_version(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    project = create_project(root, name="legacy", categories=["target"])
    category = project.repository.list_categories()[0]
    image = project.repository.add_image_record(
        image_id="legacy-image",
        relative_path="images/legacy.png",
        original_name="legacy.png",
        source_path=None,
        sha256="a" * 64,
        width=20,
        height=20,
    )
    predict = project.repository.create_run(
        "predict", "YOLO26n", run_id="legacy-predict"
    )
    project.repository.import_ai_predictions(
        predict.id,
        image.id,
        [
            AIPrediction(
                image_id=image.id,
                class_id=category.id,
                x1=1,
                y1=1,
                x2=10,
                y2=10,
            )
        ],
    )
    project.close()
    config_path = root / "project.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = 1
    config.pop("deployments_dir")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (root / "deployments").rmdir()
    connection = sqlite3.connect(root / "annotations.db")
    connection.execute("DROP TABLE deployment_packages")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    migrated = open_project(root)
    assert migrated.config.schema_version == 5
    assert migrated.repository.get_image(image.id).training_selected is True
    assert migrated.deployments_dir.is_dir()
    assert (
        sqlite3.connect(root / "annotations.db")
        .execute("PRAGMA user_version")
        .fetchone()[0]
        == 5
    )
    assert migrated.repository.get_run(predict.id).id == predict.id
    assert len(migrated.list_boxes(image.id)) == 1
    assert migrated.repository.list_ai_imported_image_ids(predict.id) == (image.id,)
    upgrade_backups = migrated.list_annotation_backups()
    assert any("before-schema-v1-to-v5" in item.reason for item in upgrade_backups)
    migrated.repository.create_run("deploy", "YOLO26n")
    migrated.close()


def test_schema_v4_alias_migration_is_backed_up_and_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "schema-v4"
    project = create_project(root, name="schema-v4", categories=["BALL"])
    category = project.repository.list_categories()[0]
    image = project.repository.add_image_record(
        image_id="schema-v4-image",
        relative_path="images/schema-v4.png",
        original_name="schema-v4.png",
        source_path=None,
        sha256="4" * 64,
        width=20,
        height=20,
    )
    project.repository.replace_boxes(
        image.id,
        [
            BoxInput(
                class_id=category.id,
                x1=1,
                y1=2,
                x2=10,
                y2=12,
            )
        ],
    )
    project.close()

    config_path = root / "project.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")
    database_path = root / "annotations.db"
    connection = sqlite3.connect(database_path)
    connection.execute("DROP TABLE category_name_aliases")
    connection.execute("PRAGMA user_version = 4")
    connection.commit()
    connection.close()

    migrated = open_project(root)
    assert migrated.config.schema_version == 5
    assert migrated.repository.connection.execute(
        "PRAGMA user_version"
    ).fetchone()[0] == 5
    assert migrated.repository.connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name = 'category_name_aliases'"
    ).fetchone() is not None
    assert migrated.repository.get_category(category.id).name == "BALL"
    assert len(migrated.list_boxes(image.id)) == 1

    backups = [
        item
        for item in migrated.list_annotation_backups()
        if item.reason == "before-schema-v4-to-v5"
    ]
    assert len(backups) == 1
    backup_connection = sqlite3.connect(backups[0].path)
    try:
        assert backup_connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert backup_connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'category_name_aliases'"
        ).fetchone() is None
        assert backup_connection.execute("SELECT COUNT(*) FROM boxes").fetchone()[0] == 1
    finally:
        backup_connection.close()

    migrated.repository.rename_category_canonical(category.id, "ball")
    migrated.close()

    reopened = open_project(root)
    try:
        assert reopened.repository.resolve_category_name("BALL").name == "ball"
        assert len(
            [
                item
                for item in reopened.list_annotation_backups()
                if item.reason == "before-schema-v4-to-v5"
            ]
        ) == 1
    finally:
        reopened.close()


def test_category_display_alias_is_independent_and_migrated(tmp_path: Path) -> None:
    with create_project(
        tmp_path / "alias-project", name="alias", categories=["小刚球", "缺陷"]
    ) as project:
        ball, defect = project.repository.list_categories()
        updated = project.repository.update_category(ball.id, display_name="BALL")
        assert updated.name == "小刚球"
        assert updated.display_name == "BALL"
        assert updated.effective_display_name == "BALL"
        assert defect.effective_display_name == "缺陷"
        with pytest.raises(Exception, match="显示名称已存在"):
            project.repository.update_category(defect.id, display_name="ball")
        cleared = project.repository.update_category(ball.id, display_name=None)
        assert cleared.display_name is None
        assert cleared.effective_display_name == "小刚球"


def test_full_category_rename_preserves_id_boxes_and_historical_import_alias(
    tmp_path: Path,
) -> None:
    with create_project(
        tmp_path / "rename-project",
        name="rename",
        categories=["BALL", "defect"],
    ) as project:
        ball, _defect = project.repository.list_categories()
        image = project.repository.add_image_record(
            image_id="rename-image",
            relative_path="images/rename.jpg",
            original_name="rename.jpg",
            source_path=None,
            sha256="1" * 64,
            width=100,
            height=100,
        )
        project.save_and_confirm(
            image.id,
            [BoxInput(ball.id, 10, 10, 30, 30)],
        )
        project.repository.update_category(ball.id, display_name="Ball shown")

        lower, first_backup = project.rename_category_canonical(ball.id, "ball")
        assert lower.id == ball.id
        assert lower.name == "ball"
        assert lower.display_name is None
        assert first_backup.valid and first_backup.path.is_file()
        assert project.list_boxes(image.id)[0].class_id == ball.id
        assert project.repository.resolve_category_name("BALL").id == ball.id

        chinese, second_backup = project.rename_category_canonical(ball.id, "钢球")
        assert chinese.id == ball.id
        assert chinese.name == "钢球"
        assert second_backup.valid and second_backup.path.is_file()
        assert project.repository.resolve_category_name("ball").id == ball.id
        assert project.repository.resolve_category_name("钢球").id == ball.id
        assert project.repository.list_category_name_aliases(ball.id)


def test_full_category_rename_rejects_cross_category_label_conflicts(
    tmp_path: Path,
) -> None:
    with create_project(
        tmp_path / "rename-conflict",
        name="rename-conflict",
        categories=["BALL", "defect"],
    ) as project:
        ball, defect = project.repository.list_categories()
        project.repository.update_category(defect.id, display_name="target")
        with pytest.raises(Exception, match="已存在"):
            project.repository.rename_category_canonical(ball.id, "TARGET")
        assert project.repository.get_category(ball.id).name == "BALL"


def test_ai_import_deduplicates_only_same_class_ai_predictions(tmp_path: Path) -> None:
    with create_project(
        tmp_path / "dedup-project",
        name="dedup",
        categories=["小刚球"],
    ) as project:
        category = project.repository.list_categories()[0]
        image = project.repository.add_image_record(
            image_id="image-dedup",
            relative_path="images/dedup.jpg",
            original_name="dedup.jpg",
            source_path=None,
            sha256="b" * 64,
            width=100,
            height=100,
        )
        project.repository.replace_boxes(
            image.id,
            [BoxInput(category.id, 2, 2, 8, 8, origin="manual")],
        )
        run = project.repository.create_run(
            "predict",
            "YOLO26n",
            parameters={"deduplicate": True, "dedup_iou": 0.8},
        )
        result = project.repository.import_ai_predictions(
            run.id,
            image.id,
            [
                AIPrediction(
                    image.id, category.id, 10, 10, 50, 50, confidence=0.40
                ),
                AIPrediction(
                    image.id, category.id, 11, 11, 51, 51, confidence=0.90
                ),
                AIPrediction(
                    image.id, category.id, 65, 65, 90, 90, confidence=0.70
                ),
            ],
        )
        boxes = project.repository.list_boxes(image.id)
        assert result.imported_count == 2
        assert sum(box.origin.value == "manual" for box in boxes) == 1
        assert sorted(
            box.confidence
            for box in boxes
            if box.origin.value == "ai"
        ) == [0.7, 0.9]


@pytest.mark.parametrize("threshold", [0.69, 0.96])
def test_ai_import_rejects_dedup_threshold_outside_maintenance_range(
    tmp_path: Path,
    threshold: float,
) -> None:
    with create_project(
        tmp_path / f"dedup-threshold-{threshold}",
        name="dedup-threshold",
        categories=["小刚球"],
    ) as project:
        image = project.repository.add_image_record(
            image_id="image",
            relative_path="images/image.jpg",
            original_name="image.jpg",
            source_path=None,
            sha256="e" * 64,
            width=100,
            height=100,
        )
        run = project.repository.create_run(
            "predict",
            "YOLO26n",
            parameters={"deduplicate": True, "dedup_iou": threshold},
        )
        with pytest.raises(DataIntegrityError, match="0.70～0.95"):
            project.repository.import_ai_predictions(run.id, image.id, [])


def test_historical_ai_dedup_preview_and_apply_protects_human_work(
    tmp_path: Path,
) -> None:
    with create_project(
        tmp_path / "historical-dedup", name="dedup", categories=["小刚球"]
    ) as project:
        category = project.repository.list_categories()[0]
        ai_image = project.repository.add_image_record(
            image_id="ai-image",
            relative_path="images/ai.jpg",
            original_name="ai.jpg",
            source_path=None,
            sha256="c" * 64,
            width=100,
            height=100,
        )
        human_image = project.repository.add_image_record(
            image_id="human-image",
            relative_path="images/human.jpg",
            original_name="human.jpg",
            source_path=None,
            sha256="d" * 64,
            width=100,
            height=100,
        )
        run = project.repository.create_run(
            "predict", "YOLO26n", parameters={"deduplicate": False}
        )
        duplicates = [
            AIPrediction(
                ai_image.id, category.id, 10, 10, 50, 50, confidence=0.35
            ),
            AIPrediction(
                ai_image.id, category.id, 11, 11, 51, 51, confidence=0.91
            ),
        ]
        project.repository.import_ai_predictions(run.id, ai_image.id, duplicates)
        project.repository.import_ai_predictions(
            run.id,
            human_image.id,
            [
                AIPrediction(
                    human_image.id,
                    category.id,
                    10,
                    10,
                    50,
                    50,
                    confidence=0.30,
                ),
                AIPrediction(
                    human_image.id,
                    category.id,
                    11,
                    11,
                    51,
                    51,
                    confidence=0.90,
                ),
            ],
        )
        first_human_box = project.repository.list_boxes(human_image.id)[0]
        project.repository.update_box(first_human_box.id, x1=9)

        preview = project.repository.preview_ai_deduplication(iou_threshold=0.8)
        assert preview.affected_image_ids == (ai_image.id,)
        assert preview.removed_box_count == 1
        assert preview.removals[0].removed_confidence == pytest.approx(0.35)
        before_human = project.repository.list_boxes(human_image.id)

        applied = project.repository.deduplicate_ai_drafts(iou_threshold=0.8)
        assert applied == preview
        remaining = project.repository.list_boxes(ai_image.id)
        assert len(remaining) == 1
        assert remaining[0].confidence == pytest.approx(0.91)
        assert project.repository.list_boxes(human_image.id) == before_human
        with pytest.raises(ValueError, match="between 0.70 and 0.95"):
            project.repository.preview_ai_deduplication(iou_threshold=0.60)


def test_project_ai_dedup_creates_restorable_backup(tmp_path: Path) -> None:
    with create_project(
        tmp_path / "dedup-backup", name="dedup", categories=["ball"]
    ) as project:
        category = project.repository.list_categories()[0]
        image = project.repository.add_image_record(
            image_id="ai",
            relative_path="images/ai.jpg",
            original_name="ai.jpg",
            source_path=None,
            sha256="e" * 64,
            width=100,
            height=100,
        )
        run = project.repository.create_run("predict", "YOLO26n")
        project.repository.import_ai_predictions(
            run.id,
            image.id,
            [
                AIPrediction(image.id, category.id, 10, 10, 50, 50, confidence=0.2),
                AIPrediction(image.id, category.id, 11, 11, 51, 51, confidence=0.9),
            ],
        )
        report = project.deduplicate_ai_drafts(iou_threshold=0.8)
        assert report.preview.removed_box_count == 1
        assert report.backup is not None and report.backup.path.is_file()
        assert len(project.repository.list_boxes(image.id)) == 1
        project.restore_annotation_backup(report.backup)
        assert len(project.repository.list_boxes(image.id)) == 2


def test_backup_cleanup_requires_device_confirmation_and_is_recoverable(
    tmp_path: Path,
) -> None:
    with create_project(tmp_path / "cleanup", name="cleanup") as project:
        for _index in range(5):
            project._create_annotation_backup("cleanup-test")
        preview = project.preview_backup_cleanup(keep_latest=2)
        assert preview.backup_count == 3
        with pytest.raises(PermissionError, match="设备上验证成功"):
            project.cleanup_old_backups(keep_latest=2)
        report = project.cleanup_old_backups(
            keep_latest=2,
            deployment_verified=True,
        )
        assert len(project.list_annotation_backups()) == 2
        assert report.recovery_directory.is_dir()
        assert sum(path.suffix == ".db" for path in report.moved_paths) == 3

        permanent_preview = project.preview_backup_cleanup(
            keep_latest=0,
            include_recovery_trash=True,
        )
        assert permanent_preview.backup_count == 5
        permanent = project.cleanup_old_backups(
            keep_latest=0,
            deployment_verified=True,
            permanently_delete=True,
        )
        assert permanent.permanently_deleted
        assert len(permanent.deleted_paths) >= 10
        assert project.list_annotation_backups() == ()
        assert not (project.backups_dir / ".trash").exists()


def test_training_snapshot_uses_only_verified_selected_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(4):
        _make_image(source / f"{index}.png", (100 + index, 80))
    with create_project(
        tmp_path / "selection-project",
        name="selection",
        categories=["小刚球"],
    ) as project:
        imported = project.import_images([source]).imported
        category = project.repository.list_categories()[0]
        for image in imported:
            project.save_and_confirm(
                image.id,
                [BoxInput(category.id, 10, 10, 30, 30)],
            )
        selected = (imported[0].id, imported[2].id)
        project.select_only_for_training(selected)
        preflight = project.training_preflight(minimum=1)
        assert preflight.verified_count == 2
        assert {
            image.id
            for image in project.list_images(training_selected=True)
        } == set(selected)
        snapshot = project.create_snapshot(
            "selected-snapshot",
            minimum=1,
            split=SplitConfig(seed=1, train_ratio=0.5, val_ratio=0.5),
        )
        assert snapshot.train_count + snapshot.val_count + snapshot.test_count == 2


def test_bulk_image_delete_creates_database_and_image_backups(tmp_path: Path) -> None:
    source = tmp_path / "delete-source"
    source.mkdir()
    _make_image(source / "keep.png", (100, 80))
    _make_image(source / "delete.png", (101, 80))
    with create_project(
        tmp_path / "delete-project",
        name="delete",
        categories=["小刚球"],
    ) as project:
        imported = project.import_images([source]).imported
        target = imported[1]
        live_path = project.image_path(target)
        report = project.delete_images([target.id])
        assert report.image_count == 1
        assert report.backup.path.is_file()
        assert report.manifest_path.is_file()
        assert (report.archive_path / target.relative_path).is_file()
        assert not live_path.exists()
        assert {image.id for image in project.list_images()} == {imported[0].id}


def test_project_create_open_and_project_id_guard(tmp_path: Path) -> None:
    root = tmp_path / "中文 project"
    project = create_project(root, name="测试项目", categories=["零件", "划痕"])
    project_id = project.config.project_id
    assert (root / "project.json").is_file()
    assert (root / "annotations.db").is_file()
    assert project.deployments_dir == root / "deployments"
    assert project.deployments_dir.is_dir()
    assert [item.name for item in project.repository.list_categories()] == [
        "零件",
        "划痕",
    ]
    project.close()

    reopened = open_project(root)
    assert reopened.config.project_id == project_id
    reopened.close()

    # Version-1 projects created before deployments support are upgraded safely.
    config_path = root / "project.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("deployments_dir")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (root / "deployments").rmdir()
    migrated = open_project(root)
    assert migrated.deployments_dir.is_dir()
    migrated.close()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["project_id"] = "wrong-project"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Exception, match="项目 ID"):
        open_project(root)


def test_image_import_normalizes_exif_deduplicates_and_reports_damage(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "输入 图片"
    sources.mkdir()
    base = sources / "普通.png"
    duplicate = sources / "普通副本.png"
    _make_image(base, (8, 6))
    shutil.copyfile(base, duplicate)

    oriented = sources / "旋转.jpg"
    image = Image.new("RGB", (4, 2), (200, 40, 10))
    exif = Image.Exif()
    exif[274] = 6
    image.save(oriented, exif=exif)
    damaged = sources / "损坏.jpg"
    damaged.write_bytes(b"not-an-image")

    with create_project(tmp_path / "project", name="import") as project:
        report = project.import_images([base, duplicate, oriented, damaged])
        assert report.requested == 4
        assert report.imported_count == 2
        assert report.duplicate_count == 1
        assert report.failed_count == 1
        assert report.report_path is not None and report.report_path.is_file()
        report_json = json.loads(report.report_path.read_text(encoding="utf-8"))
        assert report_json["requested"] == 4
        assert len(report_json["duplicates"]) == 1
        assert len(report_json["failures"]) == 1

        rotated_record = next(
            value for value in report.imported if value.original_name == oriented.name
        )
        assert (rotated_record.width, rotated_record.height) == (2, 4)
        with Image.open(project.image_path(rotated_record)) as normalized:
            assert normalized.size == (2, 4)
            assert normalized.getexif().get(274) is None


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
def test_all_supported_image_formats_are_copied(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"supported{suffix}"
    Image.new("RGBA" if suffix in {".png", ".webp"} else "RGB", (13, 7)).save(source)
    with create_project(tmp_path / "project", name="formats") as project:
        report = project.import_images([source])
        assert report.imported_count == 1
        stored = project.image_path(report.imported[0])
        assert stored.parent == project.images_dir
        with Image.open(stored) as image:
            assert image.size == (13, 7)


def test_box_crud_revision_conflict_and_atomic_d_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    _make_image(source)
    with create_project(
        tmp_path / "project", name="boxes", categories=["target"]
    ) as project:
        category = project.repository.list_categories()[0]
        image = project.import_images([source]).imported[0]

        with pytest.raises(EmptyAnnotationConfirmationRequired):
            project.verify_image(image.id)
        assert project.repository.get_image(image.id).revision == 0

        box = project.repository.add_box(
            image.id,
            BoxInput(category.id, 10, 12, 50, 60),
            expected_revision=0,
        )
        image = project.repository.get_image(image.id)
        assert image.revision == 1
        assert image.review_status is ReviewStatus.UNREVIEWED
        with pytest.raises(RevisionConflictError):
            project.repository.update_box(box.id, x1=11, expected_revision=0)

        image = project.verify_image(image.id, expected_revision=1)
        assert image.review_status is ReviewStatus.VERIFIED
        assert image.revision == 2
        project.repository.update_box(box.id, x1=15, expected_revision=2)
        image = project.repository.get_image(image.id)
        assert image.review_status is ReviewStatus.UNREVIEWED

        before = project.list_boxes(image.id)
        with pytest.raises(ValueError, match="边界"):
            project.save_and_confirm(
                image.id,
                [
                    BoxInput(category.id, 20, 20, 70, 70),
                    BoxInput(category.id, 0, 0, 101, 40),
                ],
                expected_revision=image.revision,
            )
        assert project.list_boxes(image.id) == before
        assert project.repository.get_image(image.id).revision == image.revision

        verified = project.save_and_confirm(
            image.id,
            [BoxInput(category.id, 20, 20, 70, 70)],
            expected_revision=image.revision,
        )
        assert verified.review_status is ReviewStatus.VERIFIED
        assert len(project.list_boxes(image.id)) == 1


def test_explicit_empty_negative_and_ai_draft_rules(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _make_image(first)
    _make_image(second)
    Image.new("RGB", (100, 80), (220, 40, 30)).save(second)
    with create_project(
        tmp_path / "project", name="states", categories=["target"]
    ) as project:
        category = project.repository.list_categories()[0]
        records = project.import_images([first, second]).imported
        negative = project.verify_image(records[0].id, confirm_empty=True)
        assert negative.review_status is ReviewStatus.VERIFIED

        run = project.repository.create_run("predict", "YOLO26n")
        result = project.repository.import_ai_predictions(
            run.id,
            records[1].id,
            [
                AIPrediction(
                    image_id=records[1].id,
                    class_id=category.id,
                    x1=1,
                    y1=2,
                    x2=20,
                    y2=30,
                    confidence=0.8,
                    prediction_id="p1",
                )
            ],
        )
        assert result.imported_count == 1
        draft = project.repository.get_image(records[1].id)
        assert draft.review_status is ReviewStatus.DRAFT

        # Same run/image import is a true no-op, including the revision.
        repeated = project.repository.import_ai_predictions(run.id, records[1].id, [])
        assert repeated.imported_count == 0
        assert project.repository.list_ai_imported_image_ids(run.id) == (records[1].id,)
        assert project.repository.get_image(records[1].id).revision == draft.revision

        manual = project.repository.add_box(
            records[1].id,
            BoxInput(category.id, 40, 10, 60, 30),
            expected_revision=draft.revision,
        )
        replacement_run = project.repository.create_run("predict", "YOLO11n")
        project.repository.import_ai_predictions(
            replacement_run.id,
            records[1].id,
            [
                AIPrediction(
                    image_id=records[1].id,
                    class_id=category.id,
                    x1=5,
                    y1=5,
                    x2=25,
                    y2=35,
                    prediction_id="replacement",
                )
            ],
        )
        replacement_boxes = project.list_boxes(records[1].id)
        assert len(replacement_boxes) == 2
        assert {box.id for box in replacement_boxes if box.x1 == 40} == {manual.id}
        assert any(box.x1 == 5 for box in replacement_boxes)

        preflight = project.preflight(minimum=1)
        assert preflight.verified_count == 1
        assert preflight.negative_image_count == 1
        assert preflight.instance_count == 0
        assert not preflight.ok

        reviewed = project.verify_image(records[1].id)
        assert reviewed.review_status is ReviewStatus.VERIFIED
        preflight = project.preflight(minimum=2)
        assert preflight.ok
        assert preflight.verified_count == 2
        assert preflight.positive_image_count == 1
        assert preflight.negative_image_count == 1


def test_training_threshold_missing_class_and_drafts_excluded(tmp_path: Path) -> None:
    with create_project(
        tmp_path / "project", name="threshold", categories=["present", "missing"]
    ) as project:
        categories = project.repository.list_categories()
        for index in range(101):
            image = project.repository.add_image_record(
                image_id=f"image-{index}",
                relative_path=f"images/image-{index}.png",
                original_name=f"{index}.png",
                source_path=None,
                sha256=f"{index + 1:064x}",
                width=100,
                height=100,
            )
            if index < 99:
                project.repository.confirm_image(image.id, confirm_empty=True)
            elif index == 99:
                project.repository.save_and_confirm(
                    image.id,
                    [BoxInput(categories[0].id, 1, 1, 10, 10)],
                )
            else:
                # The 101st image remains an unreviewed/AI-style draft.
                project.repository.add_box(
                    image.id, BoxInput(categories[0].id, 1, 1, 10, 10)
                )

        result = project.preflight()
        assert result.verified_count == 100
        assert result.positive_image_count == 1
        assert result.negative_image_count == 99
        assert any("missing" in error for error in result.errors)
        project.repository.update_category(categories[1].id, enabled=False)
        assert project.preflight().ok
