from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_biaozhu.core import AnnotationOrigin, BoxInput, ReviewStatus
from ai_biaozhu.core.exceptions import ProjectFormatError
from ai_biaozhu.data import create_project, create_project_from_voc, open_project
from ai_biaozhu.data.voc import (
    VocAnnotationState,
    VocMergeDisposition,
    merge_voc_into_project,
    preflight_voc_merge,
    read_voc_dataset,
)


def _write_xml(
    path: Path,
    filename: str,
    *,
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
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<annotation>"
            f"<filename>{filename}</filename>"
            "<size><width>40</width><height>30</height><depth>3</depth></size>"
            f"{object_xml}"
            "</annotation>"
        ),
        encoding="utf-8",
    )


def _mixed_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "maixhub-mixed"
    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    Image.new("RGB", (40, 30), (10, 20, 30)).save(images / "有框.jpg")
    Image.new("RGB", (40, 30), (40, 50, 60)).save(images / "负样本.jpg")
    Image.new("RGB", (40, 30), (70, 80, 90)).save(images / "待标注.jpg")
    _write_xml(
        annotations / "有框.xml",
        "有框.jpg",
        objects=(("钢球", 2, 3, 16, 18),),
    )
    _write_xml(annotations / "负样本.xml", "负样本.jpg")
    (root / "train.txt").write_text(
        "待标注.jpg\n负样本.jpg\n有框.jpg\n",
        encoding="utf-8",
    )
    return root


def test_read_mixed_voc_distinguishes_three_annotation_states(tmp_path: Path) -> None:
    dataset = read_voc_dataset(_mixed_dataset(tmp_path))

    assert [image.filename for image in dataset.images] == [
        "待标注.jpg",
        "负样本.jpg",
        "有框.jpg",
    ]
    assert [image.annotation_state for image in dataset.images] == [
        VocAnnotationState.UNCONFIRMED,
        VocAnnotationState.VERIFIED_NEGATIVE,
        VocAnnotationState.ANNOTATED,
    ]
    assert dataset.annotated_image_count == 1
    assert dataset.verified_negative_count == 1
    assert dataset.confirmed_image_count == 2
    assert dataset.unconfirmed_image_count == 1
    assert dataset.box_count == 1
    assert dataset.images[0].annotation_path is None
    assert dataset.images[1].annotation_path is not None


def test_new_project_confirms_only_images_with_xml(tmp_path: Path) -> None:
    source = _mixed_dataset(tmp_path)
    destination = tmp_path / "native"

    result = create_project_from_voc(source, destination)

    assert result.image_count == 3
    assert result.verified_count == 2
    assert result.annotated_image_count == 1
    assert result.verified_negative_count == 1
    assert result.unconfirmed_image_count == 1
    with open_project(destination) as project:
        images = {image.original_name: image for image in project.list_images()}
        assert images["有框.jpg"].review_status is ReviewStatus.VERIFIED
        assert images["负样本.jpg"].review_status is ReviewStatus.VERIFIED
        assert images["待标注.jpg"].review_status is ReviewStatus.UNREVIEWED
        assert images["待标注.jpg"].origin is AnnotationOrigin.NONE
        assert len(project.list_boxes(images["有框.jpg"].id)) == 1
        assert project.list_boxes(images["负样本.jpg"].id) == ()
        assert project.list_boxes(images["待标注.jpg"].id) == ()


def test_merge_mixed_voc_keeps_unconfirmed_new_image_unreviewed(tmp_path: Path) -> None:
    source = _mixed_dataset(tmp_path)
    with create_project(tmp_path / "project", name="mixed") as project:
        report = merge_voc_into_project(project, source)

        assert report.imported_image_count == 3
        assert report.source_annotated_image_count == 1
        assert report.source_verified_negative_count == 1
        assert report.source_unconfirmed_image_count == 1
        assert [item.annotation_state for item in report.items] == [
            VocAnnotationState.UNCONFIRMED,
            VocAnnotationState.VERIFIED_NEGATIVE,
            VocAnnotationState.ANNOTATED,
        ]
        images = {image.original_name: image for image in project.list_images()}
        assert images["待标注.jpg"].review_status is ReviewStatus.UNREVIEWED
        assert images["负样本.jpg"].review_status is ReviewStatus.VERIFIED
        assert images["有框.jpg"].review_status is ReviewStatus.VERIFIED


def test_unconfirmed_duplicate_preserves_existing_annotations(tmp_path: Path) -> None:
    source = tmp_path / "unconfirmed-duplicate"
    (source / "annotations").mkdir(parents=True)
    (source / "images").mkdir()
    image_path = source / "images" / "same.jpg"
    Image.new("RGB", (40, 30), (3, 4, 5)).save(image_path)

    with create_project(
        tmp_path / "existing-project",
        name="existing",
        categories=["钢球"],
    ) as project:
        record = project.import_images([image_path]).imported[0]
        category = project.repository.list_categories()[0]
        project.save_and_confirm(
            record.id,
            [BoxInput(category.id, 2, 3, 16, 18)],
        )
        before = project.repository.get_image(record.id)
        before_boxes = project.list_boxes(record.id)

        plan = preflight_voc_merge(project, source)
        assert plan.items[0].annotation_state is VocAnnotationState.UNCONFIRMED
        assert (
            plan.items[0].disposition
            is VocMergeDisposition.PRESERVE_UNCONFIRMED
        )
        assert plan.preserved_unconfirmed_count == 1

        report = merge_voc_into_project(project, plan)
        after = project.repository.get_image(record.id)
        assert report.preserved_unconfirmed_image_count == 1
        assert report.items[0].annotation_state is VocAnnotationState.UNCONFIRMED
        assert after.revision == before.revision
        assert after.review_status is ReviewStatus.VERIFIED
        assert project.list_boxes(record.id) == before_boxes


def test_train_list_must_include_images_without_xml(tmp_path: Path) -> None:
    source = _mixed_dataset(tmp_path)
    (source / "train.txt").write_text(
        "负样本.jpg\n有框.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectFormatError, match="数据集图片"):
        read_voc_dataset(source)


def test_empty_images_directory_is_not_a_valid_unconfirmed_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty"
    (source / "annotations").mkdir(parents=True)
    (source / "images").mkdir()

    with pytest.raises(ProjectFormatError, match="没有支持的图片"):
        read_voc_dataset(source)
