from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from ai_biaozhu.core import (
    BoxInput,
    DataIntegrityError,
    SnapshotExistsError,
    SplitConfig,
    YoloFormatError,
)
from ai_biaozhu.data import (
    create_project,
    parse_yolo_detection,
    read_yolo_export,
    yolo_boxes_to_pixels,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (80, 40), color).save(path)


def _populated_project(tmp_path: Path, count: int = 5):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = []
    for index in range(count):
        path = source_dir / f"图像 {index}.png"
        _image(path, (index * 20, 80, 120))
        sources.append(path)
    project = create_project(tmp_path / "project", name="yolo", categories=["目标"])
    category = project.repository.list_categories()[0]
    records = project.import_images(sources).imported
    for index, record in enumerate(records):
        boxes = [BoxInput(category.id, 8, 4, 40, 20)] if index == 0 else []
        project.save_and_confirm(record.id, boxes, confirm_empty=True)
    return project


def test_yolo_label_strict_parse_and_pixel_roundtrip() -> None:
    values = parse_yolo_detection("0 0.3 0.4 0.2 0.4\n", class_count=1)
    pixels = yolo_boxes_to_pixels(
        values, image_width=100, image_height=50, class_ids=["class-id"]
    )
    assert pixels[0].class_id == "class-id"
    assert pixels[0].x1 == pytest.approx(20)
    assert pixels[0].y1 == pytest.approx(10)
    assert pixels[0].x2 == pytest.approx(40)
    assert pixels[0].y2 == pytest.approx(30)
    with pytest.raises(YoloFormatError):
        parse_yolo_detection("1 0.5 0.5 0.2 0.2", class_count=1)
    with pytest.raises(YoloFormatError):
        parse_yolo_detection("0 0.01 0.5 0.2 0.2")
    with pytest.raises(YoloFormatError):
        parse_yolo_detection("0 0.5 0.5 0.2")


def test_immutable_snapshot_split_export_and_readback(tmp_path: Path) -> None:
    with _populated_project(tmp_path) as project:
        snapshot = project.snapshot(
            "run-001", minimum=5, split=SplitConfig(seed=42, val_ratio=0.2)
        )
        assert snapshot.train_count == 4
        assert snapshot.val_count == 1
        assert snapshot.test_count == 0
        assert (snapshot.root / "images" / "test").is_dir()
        assert (snapshot.root / "labels" / "test").is_dir()
        assert snapshot.data_yaml.is_file()
        assert snapshot.manifest_path.is_file()
        assert (snapshot.root / ".immutable").is_file()
        snapshot_yaml = yaml.safe_load(snapshot.data_yaml.read_text(encoding="utf-8"))
        assert snapshot_yaml["path"] == str(snapshot.root.resolve())

        readback = read_yolo_export(snapshot.root)
        assert len(readback.images) == 5
        assert readback.class_names == ("目标",)
        assert sum(len(item.boxes) for item in readback.images) == 1
        assert readback.dataset_sha256 == snapshot.dataset_sha256

        with pytest.raises(SnapshotExistsError):
            project.snapshot("run-001", minimum=5)

        exported = project.export_yolo(tmp_path / "YOLO 导出")
        assert exported.image_count == 5
        assert exported.box_count == 1
        export_yaml = yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))
        assert export_yaml["path"] == "."
        export_readback = read_yolo_export(exported.root)
        assert len(export_readback.images) == 5
        assert export_readback.dataset_sha256 == exported.dataset_sha256


def test_snapshot_supports_deterministic_70_20_10_split(tmp_path: Path) -> None:
    with _populated_project(tmp_path, count=10) as project:
        config = SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=7)
        first = project.snapshot("split-a", minimum=10, split=config)
        second = project.snapshot("split-b", minimum=10, split=config)
        assert (first.train_count, first.val_count, first.test_count) == (7, 2, 1)
        first_assignments = {
            item.image_id: item.split for item in read_yolo_export(first.root).images
        }
        second_assignments = {
            item.image_id: item.split for item in read_yolo_export(second.root).images
        }
        assert first_assignments == second_assignments


def test_snapshot_multilabel_split_keeps_rare_classes_in_training(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = []
    for index in range(20):
        path = source_dir / f"{index:02d}.png"
        _image(path, (index * 10, 60, 120))
        sources.append(path)
    with create_project(
        tmp_path / "project",
        name="stratified",
        categories=["common", "rare"],
    ) as project:
        common, rare = project.repository.list_categories()
        records = project.import_images(sources).imported
        for index, record in enumerate(records):
            boxes = []
            if index < 12:
                boxes.append(BoxInput(common.id, 4, 4, 30, 20))
            if index < 2:
                boxes.append(BoxInput(rare.id, 35, 4, 60, 20))
            project.save_and_confirm(record.id, boxes, confirm_empty=True)

        snapshot = project.snapshot(
            "stratified",
            minimum=20,
            split=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1,
                seed=11,
            ),
        )
        assert (snapshot.train_count, snapshot.val_count, snapshot.test_count) == (14, 4, 2)
        readback = read_yolo_export(snapshot.root)
        train_classes = {
            box.class_index
            for image in readback.images
            if image.split == "train"
            for box in image.boxes
        }
        assert train_classes == {0, 1}


def test_snapshot_rejects_ratio_that_cannot_cover_all_classes_in_train(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = []
    for index in range(10):
        path = source_dir / f"{index:02d}.png"
        _image(path, (index * 20, 30, 100))
        sources.append(path)
    with create_project(
        tmp_path / "project",
        name="coverage",
        categories=["left", "right"],
    ) as project:
        left, right = project.repository.list_categories()
        records = project.import_images(sources).imported
        for index, record in enumerate(records):
            class_id = left.id if index == 0 else right.id if index == 1 else None
            boxes = [BoxInput(class_id, 4, 4, 30, 20)] if class_id else []
            project.save_and_confirm(record.id, boxes, confirm_empty=True)
        with pytest.raises(DataIntegrityError, match="训练集缺少"):
            project.snapshot(
                "impossible",
                minimum=10,
                split=SplitConfig(
                    train_ratio=0.1,
                    val_ratio=0.8,
                    test_ratio=0.1,
                    seed=1,
                ),
            )


def test_snapshot_is_independent_and_hash_tamper_is_detected(tmp_path: Path) -> None:
    with _populated_project(tmp_path, count=2) as project:
        snapshot = project.snapshot("run", minimum=2)
        first = read_yolo_export(snapshot.root).images[0]
        original = first.image_path.read_bytes()
        # Changing project annotations does not mutate the already written snapshot.
        record = project.repository.get_image(first.image_id)
        project.save_and_confirm(
            record.id,
            [],
            confirm_empty=True,
            expected_revision=record.revision,
        )
        assert first.image_path.read_bytes() == original

        first.label_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        with pytest.raises(YoloFormatError, match="哈希"):
            read_yolo_export(snapshot.root)
