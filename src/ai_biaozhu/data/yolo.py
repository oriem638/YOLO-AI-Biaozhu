"""Immutable YOLO Detection snapshots, exports, and strict label readback."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

from ai_biaozhu.core.domain import (
    BoundingBox,
    BoxInput,
    ExportResult,
    ImageRecord,
    SnapshotResult,
    SplitConfig,
    YoloBox,
)
from ai_biaozhu.core.exceptions import (
    DataIntegrityError,
    SnapshotExistsError,
    YoloFormatError,
)

from .utils import canonical_json_bytes, sha256_file, utc_now, write_json

if TYPE_CHECKING:
    from .project import AnnotationProject


@dataclass(frozen=True, slots=True)
class YoloReadbackImage:
    image_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    split: str
    boxes: tuple[YoloBox, ...]


@dataclass(frozen=True, slots=True)
class YoloReadback:
    root: Path
    class_names: tuple[str, ...]
    images: tuple[YoloReadbackImage, ...]
    dataset_sha256: str


def create_training_snapshot(
    project: AnnotationProject,
    run_id: str,
    *,
    split: SplitConfig | None = None,
    minimum: int = 100,
) -> SnapshotResult:
    """Freeze verified annotations into a create-once 80/20 YOLO dataset."""

    split = split or SplitConfig()
    _validate_component(run_id, "run_id")
    destination = project.runs_dir / run_id / "snapshot"
    if destination.exists():
        raise SnapshotExistsError(f"训练快照已存在，不能覆盖：{destination}")

    with project.repository.transaction():
        preflight = project.repository.training_preflight(minimum=minimum)
        if not preflight.ok:
            raise DataIntegrityError("；".join(preflight.errors))
        categories = project.repository.list_categories(enabled_only=True)
        images = project.repository.list_images(
            review_status="verified",
            training_selected=True,
        )
        boxes_by_image = {
            image.id: tuple(
                box
                for box in project.repository.list_boxes(image.id)
                if box.class_id in {category.id for category in categories}
            )
            for image in images
        }

    split_assignments = _split_assignments(
        images,
        split,
        boxes_by_image=boxes_by_image,
    )
    split_distribution = _split_distribution(
        split_assignments,
        boxes_by_image=boxes_by_image,
        category_ids=tuple(category.id for category in categories),
    )
    missing_from_train = [
        category.name
        for category in categories
        if split_distribution["train"]["class_image_counts"][category.id] == 0
    ]
    if missing_from_train:
        raise DataIntegrityError(
            "训练集缺少以下启用类别，无法开始训练："
            + ", ".join(missing_from_train)
        )
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".snapshot-{uuid4().hex}.tmp"
    if staging.exists():
        raise SnapshotExistsError(f"临时快照路径已存在：{staging}")
    try:
        manifest, data_yaml = _write_dataset(
            project,
            staging,
            published_root=destination,
            categories=categories,
            images=images,
            boxes_by_image=boxes_by_image,
            split_for_image=lambda image: split_assignments[image.id],
            declared_splits=("train", "val", "test"),
            dataset_kind="training-snapshot",
            extra_manifest={
                "run_id": run_id,
                "split": split.to_dict(),
                "split_distribution": split_distribution,
                "preflight": {
                    "minimum": preflight.minimum,
                    "verified_count": preflight.verified_count,
                    "positive_image_count": preflight.positive_image_count,
                    "negative_image_count": preflight.negative_image_count,
                    "instance_count": preflight.instance_count,
                    "class_instance_counts": dict(preflight.class_instance_counts),
                    "warnings": list(preflight.warnings),
                },
            },
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return SnapshotResult(
        root=destination,
        data_yaml=destination / data_yaml.name,
        manifest_path=destination / manifest.name,
        train_count=sum(value == "train" for value in split_assignments.values()),
        val_count=sum(value == "val" for value in split_assignments.values()),
        test_count=sum(value == "test" for value in split_assignments.values()),
        class_count=len(categories),
        dataset_sha256=json.loads(
            (destination / manifest.name).read_text(encoding="utf-8")
        )["dataset_sha256"],
    )


def export_yolo_detection(
    project: AnnotationProject,
    destination: Path | str,
) -> ExportResult:
    """Export all and only human-verified images as a YOLO Detection dataset."""

    destination = Path(destination).resolve()
    if destination.exists():
        raise SnapshotExistsError(f"导出目标已存在，不能覆盖：{destination}")
    categories = project.repository.list_categories(enabled_only=True)
    if not categories:
        raise DataIntegrityError("至少需要启用一个类别才能导出")
    images = project.repository.list_images(review_status="verified")
    if not images:
        raise DataIntegrityError("没有人工确认图片可供导出")
    category_ids = {category.id for category in categories}
    boxes_by_image = {
        image.id: tuple(
            box
            for box in project.repository.list_boxes(image.id)
            if box.class_id in category_ids
        )
        for image in images
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}-{uuid4().hex}.tmp"
    try:
        manifest, data_yaml = _write_dataset(
            project,
            staging,
            categories=categories,
            images=images,
            boxes_by_image=boxes_by_image,
            split_for_image=lambda _image: "all",
            declared_splits=("all",),
            dataset_kind="yolo-detection-export",
            extra_manifest={},
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest_value = json.loads(
        (destination / manifest.name).read_text(encoding="utf-8")
    )
    return ExportResult(
        root=destination,
        data_yaml=destination / data_yaml.name,
        manifest_path=destination / manifest.name,
        image_count=len(images),
        box_count=sum(len(values) for values in boxes_by_image.values()),
        dataset_sha256=str(manifest_value["dataset_sha256"]),
    )


def parse_yolo_detection(
    text: str,
    *,
    class_count: int | None = None,
    source: str = "<memory>",
) -> tuple[YoloBox, ...]:
    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise YoloFormatError(
                f"{source}:{line_number} 应有 5 列，实际为 {len(parts)} 列"
            )
        try:
            class_index = int(parts[0])
            coordinates = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise YoloFormatError(f"{source}:{line_number} 包含无效数值") from exc
        if class_count is not None and not 0 <= class_index < class_count:
            raise YoloFormatError(
                f"{source}:{line_number} 类别索引 {class_index} 超出范围"
            )
        try:
            boxes.append(YoloBox(class_index, *coordinates))
        except ValueError as exc:
            raise YoloFormatError(f"{source}:{line_number} {exc}") from exc
    return tuple(boxes)


def read_yolo_label(
    path: Path | str, *, class_count: int | None = None
) -> tuple[YoloBox, ...]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise YoloFormatError(f"{path} 不是 UTF-8 文本") from exc
    return parse_yolo_detection(text, class_count=class_count, source=str(path))


def yolo_boxes_to_pixels(
    boxes: Iterable[YoloBox],
    *,
    image_width: int,
    image_height: int,
    class_ids: Sequence[str],
) -> tuple[BoxInput, ...]:
    values: list[BoxInput] = []
    for box in boxes:
        if box.class_index >= len(class_ids):
            raise YoloFormatError(f"类别索引 {box.class_index} 超出类别表范围")
        values.append(
            box.to_pixels(image_width, image_height, class_ids[box.class_index])
        )
    return tuple(values)


def read_yolo_export(root: Path | str, *, verify_hashes: bool = True) -> YoloReadback:
    """Read back an export/snapshot created by this application.

    The manifest makes filename mapping unambiguous and lets tests or workers
    detect a modified supposedly immutable snapshot.
    """

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise YoloFormatError(f"无法读取导出清单：{exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "ai-biaozhu-yolo":
        raise YoloFormatError("manifest.json 不是受支持的 YOLO 导出清单")
    classes = manifest.get("classes")
    images = manifest.get("images")
    if not isinstance(classes, list) or not isinstance(images, list):
        raise YoloFormatError("manifest.json 缺少 classes 或 images")
    class_names = tuple(str(item["name"]) for item in classes)
    records: list[YoloReadbackImage] = []
    digest_entries: list[dict[str, Any]] = []
    for item in images:
        if not isinstance(item, dict):
            raise YoloFormatError("manifest.json 的图片记录格式无效")
        image_path = _safe_manifest_member(root, str(item["image"]))
        label_path = _safe_manifest_member(root, str(item["label"]))
        if not image_path.is_file() or not label_path.is_file():
            raise YoloFormatError(f"导出文件缺失：{image_path} 或 {label_path}")
        image_hash = sha256_file(image_path)
        label_hash = sha256_file(label_path)
        if verify_hashes and (
            image_hash != item.get("image_sha256")
            or label_hash != item.get("label_sha256")
        ):
            raise YoloFormatError(f"导出文件哈希校验失败：{item.get('image_id')}")
        boxes = read_yolo_label(label_path, class_count=len(class_names))
        if int(item.get("box_count", -1)) != len(boxes):
            raise YoloFormatError(f"标注框数量与清单不一致：{item.get('image_id')}")
        records.append(
            YoloReadbackImage(
                image_id=str(item["image_id"]),
                image_path=image_path,
                label_path=label_path,
                width=int(item["width"]),
                height=int(item["height"]),
                split=str(item["split"]),
                boxes=boxes,
            )
        )
        digest_entries.append(
            {
                "image_id": str(item["image_id"]),
                "split": str(item["split"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "box_count": len(boxes),
            }
        )
    digest_payload = {
        "classes": [
            {
                "index": int(item["index"]),
                "id": str(item["id"]),
                "name": str(item["name"]),
            }
            for item in classes
        ],
        "images": digest_entries,
    }
    actual_dataset_hash = hashlib.sha256(
        canonical_json_bytes(digest_payload)
    ).hexdigest()
    expected_dataset_hash = str(manifest.get("dataset_sha256", ""))
    if verify_hashes and actual_dataset_hash != expected_dataset_hash:
        raise YoloFormatError("数据集总体哈希校验失败")
    return YoloReadback(
        root=root,
        class_names=class_names,
        images=tuple(records),
        dataset_sha256=actual_dataset_hash,
    )


verify_yolo_export = read_yolo_export


def _write_dataset(
    project: AnnotationProject,
    root: Path,
    *,
    published_root: Path | None = None,
    categories: Sequence,
    images: Sequence[ImageRecord],
    boxes_by_image: Mapping[str, Sequence[BoundingBox]],
    split_for_image,
    declared_splits: Sequence[str],
    dataset_kind: str,
    extra_manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=False)
    class_index = {category.id: index for index, category in enumerate(categories)}
    class_manifest = [
        {"index": index, "id": category.id, "name": category.name}
        for index, category in enumerate(categories)
    ]
    manifest_images: list[dict[str, Any]] = []
    digest_images: list[dict[str, Any]] = []
    for split_name in declared_splits:
        (root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for image in images:
        split_name = split_for_image(image)
        suffix = Path(image.relative_path).suffix.lower() or ".jpg"
        image_relative = Path("images") / split_name / f"{image.id}{suffix}"
        label_relative = Path("labels") / split_name / f"{image.id}.txt"
        destination_image = root / image_relative
        destination_label = root / label_relative
        shutil.copy2(project.image_path(image), destination_image)
        boxes = boxes_by_image[image.id]
        destination_label.write_text(
            _serialize_boxes(
                boxes,
                image_width=image.width,
                image_height=image.height,
                class_index=class_index,
            ),
            encoding="utf-8",
            newline="\n",
        )
        image_hash = sha256_file(destination_image)
        label_hash = sha256_file(destination_label)
        entry = {
            "image_id": image.id,
            "original_name": image.original_name,
            "split": split_name,
            "width": image.width,
            "height": image.height,
            "image": image_relative.as_posix(),
            "label": label_relative.as_posix(),
            "image_sha256": image_hash,
            "label_sha256": label_hash,
            "box_count": len(boxes),
        }
        manifest_images.append(entry)
        digest_images.append(
            {
                "image_id": image.id,
                "split": split_name,
                "width": image.width,
                "height": image.height,
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "box_count": len(boxes),
            }
        )

    digest_payload = {"classes": class_manifest, "images": digest_images}
    dataset_hash = hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
    manifest_path = root / "manifest.json"
    write_json(
        manifest_path,
        {
            "format": "ai-biaozhu-yolo",
            "version": 1,
            "kind": dataset_kind,
            "created_at": utc_now(),
            "project_id": project.config.project_id,
            "dataset_sha256": dataset_hash,
            "classes": class_manifest,
            "images": manifest_images,
            **dict(extra_manifest),
        },
    )
    data_yaml = root / "data.yaml"
    yaml_value: dict[str, Any] = {
        # Ultralytics resolves a relative ``path`` against its global
        # ``datasets_dir``, not against data.yaml.  Training snapshots therefore
        # need the final (post-staging-rename) absolute root.  Portable manual
        # exports intentionally retain ".".
        "path": str(published_root.resolve()) if published_root is not None else ".",
        "names": {index: category.name for index, category in enumerate(categories)},
    }
    if tuple(declared_splits) == ("all",):
        yaml_value.update(train="images/all", val="images/all")
    else:
        yaml_value.update(
            train="images/train",
            val="images/val",
            test="images/test",
        )
    data_yaml.write_text(
        yaml.safe_dump(
            yaml_value,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (root / ".immutable").write_text(
        f"dataset_sha256={dataset_hash}\n", encoding="ascii", newline="\n"
    )
    return manifest_path, data_yaml


def _serialize_boxes(
    boxes: Sequence[BoundingBox],
    *,
    image_width: int,
    image_height: int,
    class_index: Mapping[str, int],
) -> str:
    lines: list[str] = []
    for box in boxes:
        if box.class_id not in class_index:
            continue
        center_x = (box.x1 + box.x2) / 2.0 / image_width
        center_y = (box.y1 + box.y2) / 2.0 / image_height
        width = (box.x2 - box.x1) / image_width
        height = (box.y2 - box.y1) / image_height
        values = (center_x, center_y, width, height)
        lines.append(
            f"{class_index[box.class_id]} "
            + " ".join(_format_coordinate(value) for value in values)
        )
    return "".join(f"{line}\n" for line in lines)


def _split_assignments(
    images: Sequence[ImageRecord],
    split: SplitConfig,
    *,
    boxes_by_image: Mapping[str, Sequence[BoundingBox]] | None = None,
) -> dict[str, str]:
    """Create a deterministic, approximately multilabel-stratified split.

    A plain shuffle can put every example of a rare class into validation or
    test.  This allocator first reserves the smallest deterministic set of
    images needed to cover every class in training, then greedily fills the
    exact split capacities according to per-label deficits.  Empty/negative
    images are treated as their own label so they are distributed as well.
    """

    if not images:
        return {}
    count = len(images)
    val_count, test_count = _validation_and_test_counts(count, split)
    target_counts = {
        "train": count - val_count - test_count,
        "val": val_count,
        "test": test_count,
    }
    labels_by_image: dict[str, frozenset[str]] = {}
    for image in images:
        labels = {
            box.class_id
            for box in (boxes_by_image or {}).get(image.id, ())
        }
        labels_by_image[image.id] = frozenset(labels or {"__negative__"})
    label_frequencies: dict[str, int] = {}
    for labels in labels_by_image.values():
        for label in labels:
            label_frequencies[label] = label_frequencies.get(label, 0) + 1

    rank = {
        image.id: hashlib.sha256(f"{split.seed}:{image.id}".encode()).digest()
        for image in images
    }
    assignments: dict[str, str] = {}
    remaining = dict(target_counts)
    current_label_counts = {
        split_name: {label: 0 for label in label_frequencies}
        for split_name in target_counts
    }

    # Negative samples do not need to be part of the mandatory coverage set.
    uncovered = set(label_frequencies).difference({"__negative__"})
    while uncovered:
        candidates = [
            image
            for image in images
            if image.id not in assignments
            and labels_by_image[image.id].intersection(uncovered)
        ]
        if not candidates or remaining["train"] <= 0:
            break
        chosen = min(
            candidates,
            key=lambda image: (
                -len(labels_by_image[image.id].intersection(uncovered)),
                sum(
                    label_frequencies[label]
                    for label in labels_by_image[image.id].intersection(uncovered)
                ),
                rank[image.id],
            ),
        )
        _assign_split(
            chosen.id,
            "train",
            labels_by_image=labels_by_image,
            assignments=assignments,
            remaining=remaining,
            current_label_counts=current_label_counts,
        )
        uncovered.difference_update(labels_by_image[chosen.id])

    ordered = sorted(
        (image for image in images if image.id not in assignments),
        key=lambda image: (
            min(label_frequencies[label] for label in labels_by_image[image.id]),
            -len(labels_by_image[image.id]),
            rank[image.id],
        ),
    )
    desired = {
        split_name: {
            label: frequency * target_counts[split_name] / count
            for label, frequency in label_frequencies.items()
        }
        for split_name in target_counts
    }
    split_order = ("train", "val", "test")
    for image in ordered:
        available = [
            split_name for split_name in split_order if remaining[split_name] > 0
        ]
        if not available:
            raise DataIntegrityError("数据划分容量计算错误")
        image_labels = labels_by_image[image.id]

        def score(
            split_name: str,
            labels: frozenset[str] = image_labels,
        ) -> tuple[float, float, int]:
            label_deficit = sum(
                max(
                    0.0,
                    desired[split_name][label]
                    - current_label_counts[split_name][label],
                )
                / max(1.0, desired[split_name][label])
                for label in labels
            )
            capacity = remaining[split_name] / max(1, target_counts[split_name])
            # The final term makes ties stable and keeps the public split order.
            return label_deficit, capacity, -split_order.index(split_name)

        selected = max(available, key=score)
        _assign_split(
            image.id,
            selected,
            labels_by_image=labels_by_image,
            assignments=assignments,
            remaining=remaining,
            current_label_counts=current_label_counts,
        )
    return assignments


def _validation_and_test_counts(
    count: int,
    split: SplitConfig,
) -> tuple[int, int]:
    val_count = round(count * split.val_ratio)
    test_count = round(count * split.test_ratio)
    if count >= 2:
        val_count = max(1, val_count)
    if split.test_ratio > 0 and count >= 3:
        test_count = max(1, test_count)
    while val_count + test_count >= count:
        if test_count > 0 and (
            test_count / max(split.test_ratio, 1e-12) >= val_count / split.val_ratio
        ):
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    return val_count, test_count


def _assign_split(
    image_id: str,
    split_name: str,
    *,
    labels_by_image: Mapping[str, frozenset[str]],
    assignments: dict[str, str],
    remaining: dict[str, int],
    current_label_counts: dict[str, dict[str, int]],
) -> None:
    assignments[image_id] = split_name
    remaining[split_name] -= 1
    for label in labels_by_image[image_id]:
        current_label_counts[split_name][label] += 1


def _split_distribution(
    assignments: Mapping[str, str],
    *,
    boxes_by_image: Mapping[str, Sequence[BoundingBox]],
    category_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        name: {
            "image_count": 0,
            "negative_image_count": 0,
            "class_image_counts": {category_id: 0 for category_id in category_ids},
            "class_instance_counts": {category_id: 0 for category_id in category_ids},
        }
        for name in ("train", "val", "test")
    }
    for image_id, split_name in assignments.items():
        entry = result[split_name]
        entry["image_count"] += 1
        boxes = boxes_by_image[image_id]
        present = {box.class_id for box in boxes}
        if not boxes:
            entry["negative_image_count"] += 1
        for category_id in present:
            entry["class_image_counts"][category_id] += 1
        for box in boxes:
            entry["class_instance_counts"][box.class_id] += 1
    return result


def _format_coordinate(value: float) -> str:
    if -1e-12 < value < 0:
        value = 0.0
    if 1 < value < 1 + 1e-12:
        value = 1.0
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _validate_component(value: str, name: str) -> None:
    if not value or value in {".", ".."} or any(char in value for char in r'\/:*?"<>|'):
        raise ValueError(f"{name} 不是安全的 Windows 路径组件")


def _safe_manifest_member(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative.replace("\\", "/")).parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise YoloFormatError(f"清单路径越界：{relative}") from exc
    return path
