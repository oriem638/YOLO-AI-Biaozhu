from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_biaozhu.core import (
    AIPrediction,
    AnnotationOrigin,
    BoxInput,
    ModelKey,
    ReviewStatus,
    RunKind,
)
from ai_biaozhu.core.exceptions import ProjectFormatError
from ai_biaozhu.data import (
    create_project,
    create_project_from_voc,
    open_project,
    read_voc_dataset,
)
from ai_biaozhu.data.voc import (
    VocMergeDisposition,
    merge_voc_into_project,
    preflight_voc_merge,
)


def _write_annotation(
    path: Path,
    filename: str,
    *,
    width: int = 40,
    height: int = 30,
    objects: tuple[tuple[str, int, int, int, int], ...] = (),
) -> None:
    object_xml = "".join(
        (
            "<object>"
            f"<name>{name}</name>"
            "<bndbox>"
            f"<xmin>{x1}</xmin><ymin>{y1}</ymin>"
            f"<xmax>{x2}</xmax><ymax>{y2}</ymax>"
            "</bndbox>"
            "</object>"
        )
        for name, x1, y1, x2, y2 in objects
    )
    path.write_text(
        (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<annotation>"
            f"<filename>{filename}</filename>"
            f"<size><width>{width}</width><height>{height}</height>"
            "<depth>3</depth></size>"
            f"{object_xml}"
            "</annotation>"
        ),
        encoding="utf-8",
    )


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "MaixHub 导出"
    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    Image.new("RGB", (40, 30), (20, 30, 40)).save(images / "一.jpg")
    Image.new("RGB", (40, 30), (80, 90, 100)).save(images / "二.jpg")
    _write_annotation(
        annotations / "一.xml",
        "一.jpg",
        objects=(("小刚球", 1, 2, 10, 12), ("大钢球", 15, 3, 25, 18)),
    )
    _write_annotation(annotations / "二.xml", "二.jpg")
    (root / "train.txt").write_text("二.jpg\n一.jpg\n", encoding="utf-8")
    return root


def _make_single_image_dataset(
    tmp_path: Path,
    *,
    directory_name: str,
    filename: str = "钢球.jpg",
    color: tuple[int, int, int] = (20, 30, 40),
    objects: tuple[tuple[str, int, int, int, int], ...] = (
        ("小刚球", 2, 3, 12, 14),
    ),
) -> Path:
    root = tmp_path / directory_name
    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    Image.new("RGB", (40, 30), color).save(images / filename)
    _write_annotation(
        annotations / f"{Path(filename).stem}.xml",
        filename,
        objects=objects,
    )
    return root


def test_read_voc_validates_and_uses_train_order(tmp_path: Path) -> None:
    dataset = read_voc_dataset(_make_dataset(tmp_path))
    assert [item.filename for item in dataset.images] == ["二.jpg", "一.jpg"]
    assert dataset.category_names == ("小刚球", "大钢球")
    assert dataset.box_count == 2


def test_create_project_from_voc_preserves_boxes_and_verifies_images(
    tmp_path: Path,
) -> None:
    source = _make_dataset(tmp_path)
    destination = tmp_path / "native-project"
    result = create_project_from_voc(
        source,
        destination,
        name="钢球",
        category_renames={"小刚球": "小钢球"},
    )
    assert result.image_count == 2
    assert result.verified_count == 2
    assert result.box_count == 2
    assert result.category_names == ("小钢球", "大钢球")
    assert result.import_report_path.is_file()

    with open_project(destination) as project:
        images = project.list_images()
        assert [item.original_name for item in images] == ["二.jpg", "一.jpg"]
        assert all(item.review_status is ReviewStatus.VERIFIED for item in images)
        categories = {
            item.id: item.name for item in project.repository.list_categories()
        }
        first_boxes = project.list_boxes(images[0].id)
        second_boxes = sorted(project.list_boxes(images[1].id), key=lambda box: box.x1)
        assert first_boxes == ()
        assert [categories[box.class_id] for box in second_boxes] == [
            "小钢球",
            "大钢球",
        ]
        assert (second_boxes[0].x1, second_boxes[0].y1) == (1, 2)
        assert (second_boxes[0].x2, second_boxes[0].y2) == (10, 12)


def test_invalid_voc_does_not_create_destination(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path)
    _write_annotation(
        source / "annotations" / "一.xml",
        "一.jpg",
        objects=(("钢球", 1, 2, 41, 12),),
    )
    destination = tmp_path / "should-not-exist"
    with pytest.raises(ProjectFormatError, match="超出图片范围"):
        create_project_from_voc(source, destination)
    assert not destination.exists()


def test_train_list_must_cover_exact_dataset(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path)
    (source / "train.txt").write_text("一.jpg\n", encoding="utf-8")
    with pytest.raises(ProjectFormatError, match="漏掉"):
        read_voc_dataset(source)


def test_merge_voc_imports_new_image_and_preserves_category_spelling(
    tmp_path: Path,
) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="new-voc",
        objects=(("小刚球", 2, 3, 12, 14),),
    )
    with create_project(tmp_path / "project", name="合并项目") as project:
        plan = preflight_voc_merge(project, source)
        assert plan.new_image_count == 1
        assert plan.upgraded_image_count == 0
        assert plan.conflict_count == 0
        assert plan.created_category_names == ("小刚球",)

        report = merge_voc_into_project(project, plan)
        assert report.imported_image_count == 1
        assert report.upgraded_image_count == 0
        assert report.conflict_image_count == 0
        assert report.applied_box_count == 1
        assert report.category_names == ("小刚球",)
        assert report.report_path.is_file()
        assert report.image_import_report_path is not None
        assert report.image_import_report_path.is_file()

        image = project.list_images()[0]
        assert image.review_status is ReviewStatus.VERIFIED
        assert image.origin is AnnotationOrigin.MANUAL
        category = project.repository.list_categories()[0]
        assert category.name == "小刚球"
        box = project.list_boxes(image.id)[0]
        assert box.class_id == category.id
        assert box.origin is AnnotationOrigin.MANUAL


def test_merge_voc_marks_new_empty_xml_as_verified_negative(tmp_path: Path) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="negative-voc",
        objects=(),
    )
    with create_project(tmp_path / "project", name="负样本") as project:
        report = merge_voc_into_project(project, source)
        assert report.imported_image_count == 1
        assert report.applied_box_count == 0
        image = project.list_images()[0]
        assert image.review_status is ReviewStatus.VERIFIED
        assert image.origin is AnnotationOrigin.MANUAL
        assert project.list_boxes(image.id) == ()


def test_merge_voc_maps_source_category_to_existing_project_category(
    tmp_path: Path,
) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="mapped-voc",
        objects=(("小刚球", 2, 3, 12, 14),),
    )
    with create_project(
        tmp_path / "project",
        name="类别映射",
        categories=("钢球",),
    ) as project:
        existing = project.repository.list_categories()[0]
        plan = preflight_voc_merge(
            project,
            source,
            category_mapping={"小刚球": "钢球"},
        )
        assert plan.created_category_names == ()
        assert plan.categories[0].existing_category_id == existing.id

        report = merge_voc_into_project(project, plan)
        assert report.category_names == ("钢球",)
        assert project.list_boxes(project.list_images()[0].id)[0].class_id == existing.id
        assert [item.name for item in project.repository.list_categories()] == ["钢球"]


def test_merge_voc_reuses_historical_name_after_full_category_rename(
    tmp_path: Path,
) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="historical-name-voc",
        objects=(("BALL", 2, 3, 12, 14),),
    )
    with create_project(
        tmp_path / "project",
        name="历史名称兼容",
        categories=("BALL",),
    ) as project:
        category = project.repository.list_categories()[0]
        renamed, _backup = project.rename_category_canonical(category.id, "钢球")

        plan = preflight_voc_merge(project, source)
        assert plan.created_category_names == ()
        assert plan.category_mapping == (("BALL", "钢球"),)
        assert plan.categories[0].existing_category_id == renamed.id

        report = merge_voc_into_project(project, plan)
        assert report.category_names == ("钢球",)
        assert project.list_boxes(project.list_images()[0].id)[0].class_id == renamed.id
        assert [item.name for item in project.repository.list_categories()] == ["钢球"]


def test_merge_voc_safely_upgrades_unreviewed_duplicate(tmp_path: Path) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="upgrade-voc",
        objects=(("小刚球", 4, 5, 20, 22),),
    )
    with create_project(tmp_path / "project", name="升级项目") as project:
        imported = project.import_images([source / "images" / "钢球.jpg"])
        image = imported.imported[0]
        assert image.review_status is ReviewStatus.UNREVIEWED

        plan = preflight_voc_merge(project, source)
        assert plan.new_image_count == 0
        assert plan.upgraded_image_count == 1
        assert plan.items[0].disposition is VocMergeDisposition.UPGRADE_EXISTING

        report = merge_voc_into_project(project, plan)
        assert report.upgraded_image_count == 1
        assert report.imported_image_count == 0
        upgraded = project.repository.get_image(image.id)
        assert upgraded.review_status is ReviewStatus.VERIFIED
        assert upgraded.origin is AnnotationOrigin.MANUAL
        box = project.list_boxes(image.id)[0]
        assert (box.x1, box.y1, box.x2, box.y2) == (4, 5, 20, 22)
        assert box.origin is AnnotationOrigin.MANUAL


def test_merge_voc_safely_upgrades_pure_ai_draft(tmp_path: Path) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="ai-upgrade-voc",
        objects=(("小刚球", 6, 7, 21, 23),),
    )
    with create_project(
        tmp_path / "project",
        name="AI 草稿",
        categories=("小刚球",),
    ) as project:
        image = project.import_images(
            [source / "images" / "钢球.jpg"]
        ).imported[0]
        category = project.repository.list_categories()[0]
        run = project.repository.create_run(RunKind.PREDICT, ModelKey.YOLO26N)
        imported_ai = project.repository.import_ai_predictions(
            run.id,
            image.id,
            (
                AIPrediction(
                    image_id=image.id,
                    class_id=category.id,
                    x1=1,
                    y1=2,
                    x2=10,
                    y2=12,
                    confidence=0.8,
                ),
            ),
        )
        assert imported_ai.imported_count == 1
        draft = project.repository.get_image(image.id)
        assert draft.review_status is ReviewStatus.DRAFT
        assert draft.origin is AnnotationOrigin.AI

        plan = preflight_voc_merge(project, source)
        assert plan.upgraded_image_count == 1
        report = merge_voc_into_project(project, plan)
        assert report.upgraded_image_count == 1
        box = project.list_boxes(image.id)[0]
        assert (box.x1, box.y1, box.x2, box.y2) == (6, 7, 21, 23)
        assert box.origin is AnnotationOrigin.MANUAL
        assert box.confidence is None
        assert project.repository.get_image(image.id).review_status is ReviewStatus.VERIFIED


def test_merge_empty_xml_converts_pure_ai_draft_to_manual_negative(
    tmp_path: Path,
) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="ai-negative-voc",
        objects=(),
    )
    with create_project(
        tmp_path / "project",
        name="AI 负样本升级",
        categories=("小刚球",),
    ) as project:
        image = project.import_images(
            [source / "images" / "钢球.jpg"]
        ).imported[0]
        category = project.repository.list_categories()[0]
        run = project.repository.create_run(RunKind.PREDICT, ModelKey.YOLO26N)
        project.repository.import_ai_predictions(
            run.id,
            image.id,
            (
                AIPrediction(
                    image_id=image.id,
                    class_id=category.id,
                    x1=1,
                    y1=2,
                    x2=10,
                    y2=12,
                    confidence=0.8,
                ),
            ),
        )

        report = merge_voc_into_project(project, source)
        assert report.upgraded_image_count == 1
        assert report.applied_box_count == 0
        upgraded = project.repository.get_image(image.id)
        assert upgraded.review_status is ReviewStatus.VERIFIED
        assert upgraded.origin is AnnotationOrigin.MANUAL
        assert project.list_boxes(image.id) == ()


def test_merge_voc_preserves_human_duplicate_as_conflict(tmp_path: Path) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="conflict-voc",
        objects=(("小刚球", 15, 10, 30, 25),),
    )
    with create_project(
        tmp_path / "project",
        name="人工冲突",
        categories=("小刚球",),
    ) as project:
        image = project.import_images(
            [source / "images" / "钢球.jpg"]
        ).imported[0]
        category = project.repository.list_categories()[0]
        project.save_boxes(
            image.id,
            (BoxInput(category.id, 1, 2, 11, 13),),
        )
        before = project.list_boxes(image.id)

        plan = preflight_voc_merge(project, source)
        assert plan.conflict_count == 1
        assert plan.created_category_names == ()
        assert (
            plan.items[0].disposition
            is VocMergeDisposition.PRESERVE_CONFLICT
        )

        report = merge_voc_into_project(project, plan)
        assert report.conflict_image_count == 1
        assert report.applied_box_count == 0
        assert project.list_boxes(image.id) == before
        current = project.repository.get_image(image.id)
        assert current.review_status is ReviewStatus.UNREVIEWED
        assert current.origin is AnnotationOrigin.MANUAL


def test_merge_voc_invalid_input_leaves_project_unchanged(tmp_path: Path) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="invalid-merge-voc",
    )
    _write_annotation(
        source / "annotations" / "钢球.xml",
        "钢球.jpg",
        objects=(("新类别", 1, 2, 41, 12),),
    )
    with create_project(
        tmp_path / "project",
        name="原项目",
        categories=("原类别",),
    ) as project:
        before_categories = project.repository.list_categories()
        with pytest.raises(ProjectFormatError, match="超出图片范围"):
            merge_voc_into_project(project, source)
        assert project.list_images() == ()
        assert project.repository.list_categories() == before_categories
        assert tuple(project.images_dir.iterdir()) == ()


def test_merge_voc_runtime_failure_rolls_back_database_and_copied_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_single_image_dataset(
        tmp_path,
        directory_name="rollback-voc",
        objects=(("新类别", 2, 3, 12, 14),),
    )
    with create_project(tmp_path / "project", name="回滚项目") as project:
        from ai_biaozhu.data import voc as voc_module

        def fail_report(_path: Path, _value: object) -> None:
            raise OSError("模拟报告写入失败")

        monkeypatch.setattr(voc_module, "write_json", fail_report)
        with pytest.raises(OSError, match="模拟报告写入失败"):
            merge_voc_into_project(project, source)
        assert project.list_images() == ()
        assert project.repository.list_categories() == ()
        assert tuple(project.images_dir.iterdir()) == ()


def test_read_voc_250_images_and_1122_boxes_invariant(tmp_path: Path) -> None:
    root = tmp_path / "250-1122"
    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    for index in range(250):
        filename = f"{index:03d}.png"
        Image.new(
            "RGB",
            (8, 8),
            (index % 251, (index * 3) % 251, (index * 7) % 251),
        ).save(images / filename)
        count = 5 if index < 122 else 4
        _write_annotation(
            annotations / f"{index:03d}.xml",
            filename,
            width=8,
            height=8,
            objects=tuple(("小刚球", 1, 1, 4, 4) for _ in range(count)),
        )

    dataset = read_voc_dataset(root)
    assert len(dataset.images) == 250
    assert dataset.box_count == 1122
    assert dataset.category_names == ("小刚球",)
