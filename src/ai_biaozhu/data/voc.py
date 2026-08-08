"""Validated Pascal VOC ingestion for externally labelled datasets."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from ai_biaozhu.core.domain import (
    AnnotationOrigin,
    BoundingBox,
    BoxInput,
    ImageRecord,
    ReviewStatus,
)
from ai_biaozhu.core.exceptions import ProjectExistsError, ProjectFormatError

from .image_import import SUPPORTED_IMAGE_SUFFIXES
from .project import AnnotationProject, create_project
from .utils import sha256_file, utc_now, write_json

_CATEGORY_COLORS = (
    "#22C55E",
    "#3B82F6",
    "#F59E0B",
    "#EF4444",
    "#8B5CF6",
    "#06B6D4",
    "#EC4899",
    "#84CC16",
)


@dataclass(frozen=True, slots=True)
class VocBox:
    category_name: str
    x1: float
    y1: float
    x2: float
    y2: float


class VocAnnotationState(StrEnum):
    """How confidently one source image is labelled by the VOC export."""

    ANNOTATED = "annotated"
    VERIFIED_NEGATIVE = "verified_negative"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True)
class VocImage:
    filename: str
    image_path: Path
    annotation_path: Path | None
    width: int
    height: int
    boxes: tuple[VocBox, ...]

    @property
    def annotation_state(self) -> VocAnnotationState:
        if self.annotation_path is None:
            return VocAnnotationState.UNCONFIRMED
        if self.boxes:
            return VocAnnotationState.ANNOTATED
        return VocAnnotationState.VERIFIED_NEGATIVE

    @property
    def has_annotation(self) -> bool:
        """Whether an XML explicitly confirms this image, even as a negative."""

        return self.annotation_path is not None


@dataclass(frozen=True, slots=True)
class VocDataset:
    root: Path
    images: tuple[VocImage, ...]
    category_names: tuple[str, ...]

    @property
    def box_count(self) -> int:
        return sum(len(image.boxes) for image in self.images)

    @property
    def annotated_image_count(self) -> int:
        return sum(image.annotation_state is VocAnnotationState.ANNOTATED for image in self.images)

    @property
    def verified_negative_count(self) -> int:
        return sum(
            image.annotation_state is VocAnnotationState.VERIFIED_NEGATIVE
            for image in self.images
        )

    @property
    def confirmed_image_count(self) -> int:
        return self.annotated_image_count + self.verified_negative_count

    @property
    def unconfirmed_image_count(self) -> int:
        return sum(
            image.annotation_state is VocAnnotationState.UNCONFIRMED
            for image in self.images
        )


@dataclass(frozen=True, slots=True)
class VocProjectImport:
    destination: Path
    image_count: int
    verified_count: int
    box_count: int
    category_names: tuple[str, ...]
    import_report_path: Path
    annotated_image_count: int = 0
    verified_negative_count: int = 0
    unconfirmed_image_count: int = 0


class VocMergeDisposition(StrEnum):
    """The non-destructive decision for one VOC image."""

    IMPORT_NEW = "import_new"
    UPGRADE_EXISTING = "upgrade_existing"
    PRESERVE_CONFLICT = "preserve_conflict"
    PRESERVE_UNCONFIRMED = "preserve_unconfirmed"


@dataclass(frozen=True, slots=True)
class VocCategoryPlan:
    source_name: str
    target_name: str
    existing_category_id: str | None

    @property
    def creates_category(self) -> bool:
        return self.existing_category_id is None


@dataclass(frozen=True, slots=True)
class VocMergeItemPlan:
    filename: str
    sha256: str
    box_count: int
    disposition: VocMergeDisposition
    existing_image_id: str | None = None
    expected_revision: int | None = None
    reason: str | None = None
    annotation_state: VocAnnotationState = VocAnnotationState.ANNOTATED


@dataclass(frozen=True, slots=True)
class VocMergePlan:
    """A read-only, revision-aware preview that can be presented by the UI."""

    project_id: str
    dataset: VocDataset
    category_mapping: tuple[tuple[str, str], ...]
    categories: tuple[VocCategoryPlan, ...]
    items: tuple[VocMergeItemPlan, ...]

    @property
    def new_image_count(self) -> int:
        return sum(
            item.disposition is VocMergeDisposition.IMPORT_NEW for item in self.items
        )

    @property
    def upgraded_image_count(self) -> int:
        return sum(
            item.disposition is VocMergeDisposition.UPGRADE_EXISTING
            for item in self.items
        )

    @property
    def conflict_count(self) -> int:
        return sum(
            item.disposition is VocMergeDisposition.PRESERVE_CONFLICT
            for item in self.items
        )

    @property
    def preserved_unconfirmed_count(self) -> int:
        return sum(
            item.disposition is VocMergeDisposition.PRESERVE_UNCONFIRMED
            for item in self.items
        )

    @property
    def created_category_names(self) -> tuple[str, ...]:
        applied_filenames = {
            item.filename
            for item in self.items
            if item.disposition
            not in {
                VocMergeDisposition.PRESERVE_CONFLICT,
                VocMergeDisposition.PRESERVE_UNCONFIRMED,
            }
        }
        required_source_names = {
            box.category_name
            for image in self.dataset.images
            if image.filename in applied_filenames
            for box in image.boxes
        }
        result: list[str] = []
        for item in self.categories:
            if (
                item.source_name in required_source_names
                and item.creates_category
                and item.target_name not in result
            ):
                result.append(item.target_name)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class VocMergeItemResult:
    filename: str
    sha256: str
    disposition: VocMergeDisposition
    image_id: str | None
    box_count: int
    reason: str | None = None
    annotation_state: VocAnnotationState = VocAnnotationState.ANNOTATED


@dataclass(frozen=True, slots=True)
class VocMergeReport:
    project_id: str
    source: Path
    source_image_count: int
    source_box_count: int
    imported_image_count: int
    upgraded_image_count: int
    conflict_image_count: int
    applied_box_count: int
    category_names: tuple[str, ...]
    created_category_names: tuple[str, ...]
    items: tuple[VocMergeItemResult, ...]
    report_path: Path
    image_import_report_path: Path | None
    source_annotated_image_count: int = 0
    source_verified_negative_count: int = 0
    source_unconfirmed_image_count: int = 0
    preserved_unconfirmed_image_count: int = 0


def read_voc_dataset(root: Path | str) -> VocDataset:
    """Read and fully validate a MaixHub/Pascal VOC detection dataset."""

    root = Path(root).resolve()
    annotations_dir = root / "annotations"
    images_dir = root / "images"
    if not root.is_dir():
        raise ProjectFormatError(f"数据集目录不存在：{root}")
    if not annotations_dir.is_dir():
        raise ProjectFormatError(f"找不到标注目录：{annotations_dir}")
    if not images_dir.is_dir():
        raise ProjectFormatError(f"找不到图片目录：{images_dir}")

    parsed_by_filename: dict[str, VocImage] = {}
    category_names: list[str] = []
    seen_categories: set[str] = set()
    for annotation_path in sorted(annotations_dir.rglob("*.xml")):
        item = _read_annotation(annotation_path, images_dir)
        if item.filename in parsed_by_filename:
            raise ProjectFormatError(f"多个 XML 引用了同一图片：{item.filename}")
        parsed_by_filename[item.filename] = item
        for box in item.boxes:
            if box.category_name not in seen_categories:
                seen_categories.add(box.category_name)
                category_names.append(box.category_name)

    actual_images = {
        path.relative_to(images_dir).as_posix()
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    if not actual_images:
        raise ProjectFormatError(f"图片目录中没有支持的图片：{images_dir}")
    annotated_images = set(parsed_by_filename)
    missing_images = sorted(annotated_images - actual_images)
    if missing_images:
        raise ProjectFormatError(
            f"{len(missing_images)} 份 XML 找不到图片，例如：{missing_images[0]}"
        )

    # MaixHub exports can intentionally mix labelled images with images that
    # have not been reviewed yet.  A missing XML is therefore distinct from an
    # existing, object-free XML: the former remains unconfirmed while the
    # latter is an explicitly reviewed negative sample.
    for filename in sorted(actual_images - annotated_images):
        parsed_by_filename[filename] = _read_unconfirmed_image(filename, images_dir)

    dataset_images = set(parsed_by_filename)

    train_path = root / "train.txt"
    if train_path.is_file():
        ordered_names = _read_train_list(train_path)
        listed = set(ordered_names)
        if len(listed) != len(ordered_names):
            raise ProjectFormatError("train.txt 中存在重复图片")
        missing_from_list = sorted(dataset_images - listed)
        unknown_in_list = sorted(listed - dataset_images)
        if missing_from_list:
            raise ProjectFormatError(
                f"train.txt 漏掉了 {len(missing_from_list)} 张数据集图片，例如："
                f"{missing_from_list[0]}"
            )
        if unknown_in_list:
            raise ProjectFormatError(
                f"train.txt 引用了不存在的图片，例如：{unknown_in_list[0]}"
            )
    else:
        ordered_names = sorted(parsed_by_filename)

    return VocDataset(
        root=root,
        images=tuple(parsed_by_filename[name] for name in ordered_names),
        category_names=tuple(category_names),
    )


def create_project_from_voc(
    source: Path | str,
    destination: Path | str,
    *,
    name: str | None = None,
    category_renames: Mapping[str, str] | None = None,
) -> VocProjectImport:
    """Atomically create a native project from a validated VOC dataset."""

    dataset = read_voc_dataset(source)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ProjectExistsError(f"目标项目目录已存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    renames = {
        str(source_name): str(target_name).strip()
        for source_name, target_name in (category_renames or {}).items()
    }
    unknown_renames = sorted(set(renames) - set(dataset.category_names))
    if unknown_renames:
        raise ValueError(f"类别重命名找不到源类别：{unknown_renames[0]}")
    if any(not value for value in renames.values()):
        raise ValueError("类别重命名后的名称不能为空")

    final_category_names: list[str] = []
    for source_name in dataset.category_names:
        target_name = renames.get(source_name, source_name)
        if target_name not in final_category_names:
            final_category_names.append(target_name)

    temporary = destination.parent / (
        f".{destination.name}.voc-import-{uuid4().hex}.tmp"
    )
    project = None
    report_relative: Path | None = None
    try:
        project = create_project(
            temporary,
            name=(name or destination.name).strip(),
            categories=[
                {
                    "name": category_name,
                    "color": _CATEGORY_COLORS[index % len(_CATEGORY_COLORS)],
                }
                for index, category_name in enumerate(final_category_names)
            ],
        )
        categories = {
            category.name: category for category in project.repository.list_categories()
        }
        report = project.import_images(
            [item.image_path for item in dataset.images],
            recursive=False,
        )
        if report.failed_count or report.duplicate_count:
            detail = (
                report.failures[0].reason
                if report.failures
                else "数据集中存在内容重复的图片"
            )
            raise ProjectFormatError(f"导入图片失败：{detail}")
        if report.imported_count != len(dataset.images):
            raise ProjectFormatError(
                f"图片导入数量不一致：预期 {len(dataset.images)}，"
                f"实际 {report.imported_count}"
            )

        records_by_source = {
            Path(record.source_path).resolve(): record
            for record in report.imported
            if record.source_path
        }
        for item in dataset.images:
            record = records_by_source.get(item.image_path.resolve())
            if record is None:
                raise ProjectFormatError(f"找不到已导入图片记录：{item.filename}")
            if (record.width, record.height) != (item.width, item.height):
                raise ProjectFormatError(
                    f"图片尺寸与 XML 不一致：{item.filename}；"
                    f"XML={item.width}x{item.height}，"
                    f"图片={record.width}x{record.height}"
                )
            if not item.has_annotation:
                # No XML means nobody has reviewed the image yet.  Keep the
                # image import defaults (unreviewed / origin none) intact.
                continue
            boxes = [
                BoxInput(
                    categories[renames.get(box.category_name, box.category_name)].id,
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                )
                for box in item.boxes
            ]
            project.save_and_confirm(record.id, boxes, confirm_empty=True)

        verified_count = len(
            project.list_images(review_status=ReviewStatus.VERIFIED)
        )
        if verified_count != dataset.confirmed_image_count:
            raise ProjectFormatError(
                f"确认状态数量不一致：预期 {dataset.confirmed_image_count}，"
                f"实际 {verified_count}"
            )
        if report.report_path is None:
            raise ProjectFormatError("图片导入报告未生成")
        report_relative = report.report_path.relative_to(temporary)
        project.close()
        project = None
        if destination.exists():
            raise ProjectExistsError(f"目标项目目录已存在：{destination}")
        temporary.rename(destination)
    except Exception:
        if project is not None:
            project.close()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    if report_relative is None:
        raise AssertionError("导入报告路径未设置")
    return VocProjectImport(
        destination=destination,
        image_count=len(dataset.images),
        verified_count=dataset.confirmed_image_count,
        box_count=dataset.box_count,
        category_names=tuple(final_category_names),
        import_report_path=destination / report_relative,
        annotated_image_count=dataset.annotated_image_count,
        verified_negative_count=dataset.verified_negative_count,
        unconfirmed_image_count=dataset.unconfirmed_image_count,
    )


def preflight_voc_merge(
    project: AnnotationProject,
    source: Path | str,
    *,
    category_mapping: Mapping[str, str] | None = None,
) -> VocMergePlan:
    """Validate and plan a safe VOC merge without changing the project.

    Categories with the same exact name are reused automatically. A mapping
    value is a target category *name*, not an ID, which lets the UI map several
    VOC classes to one existing/new project class.
    """

    dataset = read_voc_dataset(source)
    requested_mapping = _normalize_category_mapping(dataset, category_mapping)
    # Canonical names may change over a project's lifetime.  Historical names
    # are import-only aliases, so a VOC dataset exported before the rename is
    # merged into the same stable category instead of creating a duplicate.
    mapping = tuple(
        (
            source_name,
            resolved.name if resolved is not None else target_name,
        )
        for source_name, target_name in requested_mapping
        for resolved in (project.repository.resolve_category_name(target_name),)
    )
    existing_categories = {
        category.name: category for category in project.repository.list_categories()
    }
    categories = tuple(
        VocCategoryPlan(
            source_name=source_name,
            target_name=target_name,
            existing_category_id=(
                existing_categories[target_name].id
                if target_name in existing_categories
                else None
            ),
        )
        for source_name, target_name in mapping
    )

    seen_hashes: dict[str, str] = {}
    items: list[VocMergeItemPlan] = []
    for image in dataset.images:
        image_hash = sha256_file(image.image_path)
        earlier = seen_hashes.get(image_hash)
        if earlier is not None:
            raise ProjectFormatError(
                f"VOC 数据集中存在内容重复图片：{earlier} 与 {image.filename}"
            )
        seen_hashes[image_hash] = image.filename

        existing = project.repository.find_image_by_sha256(image_hash)
        if existing is None:
            items.append(
                VocMergeItemPlan(
                    filename=image.filename,
                    sha256=image_hash,
                    box_count=len(image.boxes),
                    disposition=VocMergeDisposition.IMPORT_NEW,
                    annotation_state=image.annotation_state,
                )
            )
            continue
        if not image.has_annotation:
            items.append(
                VocMergeItemPlan(
                    filename=image.filename,
                    sha256=image_hash,
                    box_count=0,
                    disposition=VocMergeDisposition.PRESERVE_UNCONFIRMED,
                    existing_image_id=existing.id,
                    expected_revision=existing.revision,
                    reason="源图片没有 XML，保留项目中的现有状态和标注",
                    annotation_state=image.annotation_state,
                )
            )
            continue
        if (existing.width, existing.height) != (image.width, image.height):
            raise ProjectFormatError(
                f"项目重复图片尺寸与 VOC XML 不一致：{image.filename}，"
                f"项目={existing.width}x{existing.height}，"
                f"XML={image.width}x{image.height}"
            )

        boxes = project.list_boxes(existing.id)
        disposition, reason = _duplicate_disposition(existing, boxes)
        items.append(
            VocMergeItemPlan(
                filename=image.filename,
                sha256=image_hash,
                box_count=len(image.boxes),
                disposition=disposition,
                existing_image_id=existing.id,
                expected_revision=existing.revision,
                reason=reason,
                annotation_state=image.annotation_state,
            )
        )

    return VocMergePlan(
        project_id=project.config.project_id,
        dataset=dataset,
        category_mapping=mapping,
        categories=categories,
        items=tuple(items),
    )


def merge_voc_into_project(
    project: AnnotationProject,
    source_or_plan: Path | str | VocMergePlan,
    *,
    category_mapping: Mapping[str, str] | None = None,
) -> VocMergeReport:
    """Atomically merge VOC labels using the safe-upgrade policy.

    A :class:`VocMergePlan` can be shown by the UI before confirmation. It is
    re-preflighted at execution time, so source changes, category changes and
    image revision changes fail before any write. New project image files are
    removed if the surrounding database transaction rolls back.
    """

    if isinstance(source_or_plan, VocMergePlan):
        if category_mapping is not None:
            raise ValueError("传入 VocMergePlan 时不能再次指定 category_mapping")
        plan = source_or_plan
    else:
        plan = preflight_voc_merge(
            project,
            source_or_plan,
            category_mapping=category_mapping,
        )
    if plan.project_id != project.config.project_id:
        raise ProjectFormatError("VOC 合并计划不属于当前项目")

    refreshed = preflight_voc_merge(
        project,
        plan.dataset.root,
        category_mapping=dict(plan.category_mapping),
    )
    if refreshed != plan:
        raise ProjectFormatError("VOC 数据集或当前项目已变化，请重新预检查后再导入")

    items_by_filename = {item.filename: item for item in plan.items}
    images_by_filename = {item.filename: item for item in plan.dataset.images}
    baseline_image_files = {
        path.resolve()
        for path in project.images_dir.iterdir()
        if path.is_file()
    }
    report_path: Path | None = None
    temporary_report_path: Path | None = None
    image_import_report_path: Path | None = None
    imported_records_by_filename = {}
    results: list[VocMergeItemResult] = []

    try:
        with project.repository.transaction():
            categories_by_name = {
                category.name: category
                for category in project.repository.list_categories()
            }
            for name in plan.created_category_names:
                if name not in categories_by_name:
                    created = project.add_category(
                        name,
                        color=_CATEGORY_COLORS[
                            len(categories_by_name) % len(_CATEGORY_COLORS)
                        ],
                    )
                    categories_by_name[name] = created

            new_images = [
                images_by_filename[item.filename]
                for item in plan.items
                if item.disposition is VocMergeDisposition.IMPORT_NEW
            ]
            if new_images:
                image_report = project.import_images(
                    [item.image_path for item in new_images],
                    recursive=False,
                )
                image_import_report_path = image_report.report_path
                if image_report.failed_count or image_report.duplicate_count:
                    detail = (
                        image_report.failures[0].reason
                        if image_report.failures
                        else "预检查后检测到重复图片"
                    )
                    raise ProjectFormatError(f"VOC 图片合并失败：{detail}")
                if image_report.imported_count != len(new_images):
                    raise ProjectFormatError(
                        "VOC 图片导入数量不一致："
                        f"预期 {len(new_images)}，实际 {image_report.imported_count}"
                    )
                records_by_source = {
                    Path(record.source_path).resolve(): record
                    for record in image_report.imported
                    if record.source_path
                }
                for image in new_images:
                    record = records_by_source.get(image.image_path.resolve())
                    if record is None:
                        raise ProjectFormatError(
                            f"找不到新导入图片记录：{image.filename}"
                        )
                    imported_records_by_filename[image.filename] = record

            target_name_by_source = dict(plan.category_mapping)
            for image in plan.dataset.images:
                item_plan = items_by_filename[image.filename]
                if (
                    item_plan.disposition
                    in {
                        VocMergeDisposition.PRESERVE_CONFLICT,
                        VocMergeDisposition.PRESERVE_UNCONFIRMED,
                    }
                ):
                    results.append(
                        VocMergeItemResult(
                            filename=image.filename,
                            sha256=item_plan.sha256,
                            disposition=item_plan.disposition,
                            image_id=item_plan.existing_image_id,
                            box_count=0,
                            reason=item_plan.reason,
                            annotation_state=image.annotation_state,
                        )
                    )
                    continue

                if item_plan.disposition is VocMergeDisposition.IMPORT_NEW:
                    record = imported_records_by_filename[image.filename]
                    expected_revision = record.revision
                else:
                    if item_plan.existing_image_id is None:
                        raise AssertionError("升级项目缺少 existing_image_id")
                    record = project.repository.get_image(
                        item_plan.existing_image_id
                    )
                    expected_revision = item_plan.expected_revision

                if not image.has_annotation:
                    # A newly imported image without XML remains unreviewed.
                    # Existing copies took the preservation branch above.
                    results.append(
                        VocMergeItemResult(
                            filename=image.filename,
                            sha256=item_plan.sha256,
                            disposition=item_plan.disposition,
                            image_id=record.id,
                            box_count=0,
                            reason="源图片没有 XML，作为未确认图片导入",
                            annotation_state=image.annotation_state,
                        )
                    )
                    continue

                boxes = [
                    BoxInput(
                        categories_by_name[
                            target_name_by_source[box.category_name]
                        ].id,
                        box.x1,
                        box.y1,
                        box.x2,
                        box.y2,
                        origin=AnnotationOrigin.MANUAL,
                    )
                    for box in image.boxes
                ]
                if boxes:
                    project.save_and_confirm(
                        record.id,
                        boxes,
                        expected_revision=expected_revision,
                    )
                else:
                    # An empty XML is a human-reviewed negative sample. Saving
                    # the empty canvas first intentionally changes a former
                    # pure-AI draft's origin to manual before confirmation.
                    project.save_boxes(
                        record.id,
                        (),
                        expected_revision=expected_revision,
                    )
                    saved = project.repository.get_image(record.id)
                    project.verify_image(
                        record.id,
                        confirm_empty=True,
                        expected_revision=saved.revision,
                    )
                results.append(
                    VocMergeItemResult(
                        filename=image.filename,
                        sha256=item_plan.sha256,
                        disposition=item_plan.disposition,
                        image_id=record.id,
                        box_count=len(boxes),
                        annotation_state=image.annotation_state,
                    )
                )

            reports_dir = project.exports_dir / "voc-merge-reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            stem = (
                utc_now().replace(":", "").replace("-", "").replace(".", "")
                + f"-{uuid4().hex[:8]}"
            )
            report_path = reports_dir / f"voc-merge-{stem}.json"
            temporary_report_path = reports_dir / f".voc-merge-{stem}.tmp"
            report_payload = _voc_merge_report_payload(
                plan,
                results,
                image_import_report_path=image_import_report_path,
            )
            write_json(temporary_report_path, report_payload)
            temporary_report_path.replace(report_path)
            temporary_report_path = None
    except Exception:
        # A failed COMMIT can leave SQLite in a transaction. Rolling back is
        # harmless after an ordinary transactional exception and keeps file/DB
        # cleanup aligned.
        project.repository.connection.rollback()
        _remove_project_files_created_after(
            project.images_dir, baseline_image_files
        )
        if temporary_report_path is not None:
            temporary_report_path.unlink(missing_ok=True)
        if report_path is not None:
            report_path.unlink(missing_ok=True)
        if image_import_report_path is not None:
            image_import_report_path.unlink(missing_ok=True)
        raise

    if report_path is None:
        raise AssertionError("VOC 合并报告路径未设置")
    imported_count = sum(
        item.disposition is VocMergeDisposition.IMPORT_NEW for item in results
    )
    upgraded_count = sum(
        item.disposition is VocMergeDisposition.UPGRADE_EXISTING for item in results
    )
    conflict_count = sum(
        item.disposition is VocMergeDisposition.PRESERVE_CONFLICT
        for item in results
    )
    preserved_unconfirmed_count = sum(
        item.disposition is VocMergeDisposition.PRESERVE_UNCONFIRMED
        for item in results
    )
    applied_box_count = sum(item.box_count for item in results)
    return VocMergeReport(
        project_id=plan.project_id,
        source=plan.dataset.root,
        source_image_count=len(plan.dataset.images),
        source_box_count=plan.dataset.box_count,
        imported_image_count=imported_count,
        upgraded_image_count=upgraded_count,
        conflict_image_count=conflict_count,
        applied_box_count=applied_box_count,
        category_names=tuple(
            dict.fromkeys(target for _source, target in plan.category_mapping)
        ),
        created_category_names=plan.created_category_names,
        items=tuple(results),
        report_path=report_path,
        image_import_report_path=image_import_report_path,
        source_annotated_image_count=plan.dataset.annotated_image_count,
        source_verified_negative_count=plan.dataset.verified_negative_count,
        source_unconfirmed_image_count=plan.dataset.unconfirmed_image_count,
        preserved_unconfirmed_image_count=preserved_unconfirmed_count,
    )


def _normalize_category_mapping(
    dataset: VocDataset,
    values: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    provided = {
        str(source_name): str(target_name).strip()
        for source_name, target_name in (values or {}).items()
    }
    unknown = sorted(set(provided) - set(dataset.category_names))
    if unknown:
        raise ValueError(f"类别映射找不到 VOC 源类别：{unknown[0]}")
    for target_name in provided.values():
        if not target_name:
            raise ValueError("类别映射后的名称不能为空")
        if any(char in target_name for char in "\r\n,"):
            raise ValueError("类别映射后的名称不能包含换行或逗号")
        if len(target_name) > 128:
            raise ValueError("类别映射后的名称不能超过 128 个字符")
    return tuple(
        (source_name, provided.get(source_name, source_name))
        for source_name in dataset.category_names
    )


def _duplicate_disposition(
    existing: ImageRecord,
    boxes: tuple[BoundingBox, ...],
) -> tuple[VocMergeDisposition, str]:
    if existing.review_status is ReviewStatus.VERIFIED:
        return VocMergeDisposition.PRESERVE_CONFLICT, "项目图片已经人工确认"
    if existing.origin in {AnnotationOrigin.MANUAL, AnnotationOrigin.MIXED}:
        return VocMergeDisposition.PRESERVE_CONFLICT, "项目图片包含人工修改"
    if any(box.origin is not AnnotationOrigin.AI for box in boxes):
        return VocMergeDisposition.PRESERVE_CONFLICT, "项目图片包含非 AI 标注框"
    return (
        VocMergeDisposition.UPGRADE_EXISTING,
        "未确认图片或纯 AI 草稿，可由 VOC 人工标注安全升级",
    )


def _voc_merge_report_payload(
    plan: VocMergePlan,
    results: list[VocMergeItemResult],
    *,
    image_import_report_path: Path | None,
) -> dict[str, object]:
    imported_count = sum(
        item.disposition is VocMergeDisposition.IMPORT_NEW for item in results
    )
    upgraded_count = sum(
        item.disposition is VocMergeDisposition.UPGRADE_EXISTING for item in results
    )
    conflict_count = sum(
        item.disposition is VocMergeDisposition.PRESERVE_CONFLICT
        for item in results
    )
    preserved_unconfirmed_count = sum(
        item.disposition is VocMergeDisposition.PRESERVE_UNCONFIRMED
        for item in results
    )
    return {
        "format": "ai-biaozhu-voc-merge-report",
        "created_at": utc_now(),
        "project_id": plan.project_id,
        "source": str(plan.dataset.root),
        "source_images": len(plan.dataset.images),
        "source_boxes": plan.dataset.box_count,
        "source_annotated_images": plan.dataset.annotated_image_count,
        "source_verified_negatives": plan.dataset.verified_negative_count,
        "source_unconfirmed_images": plan.dataset.unconfirmed_image_count,
        "imported_images": imported_count,
        "upgraded_images": upgraded_count,
        "preserved_conflicts": conflict_count,
        "preserved_unconfirmed": preserved_unconfirmed_count,
        "applied_boxes": sum(item.box_count for item in results),
        "categories": [
            {
                "source_name": item.source_name,
                "target_name": item.target_name,
                "existing_category_id": item.existing_category_id,
                "created": item.target_name in plan.created_category_names,
            }
            for item in plan.categories
        ],
        "image_import_report": (
            None
            if image_import_report_path is None
            else str(image_import_report_path)
        ),
        "items": [
            {
                "filename": item.filename,
                "sha256": item.sha256,
                "action": item.disposition.value,
                "image_id": item.image_id,
                "box_count": item.box_count,
                "reason": item.reason,
                "annotation_state": item.annotation_state.value,
            }
            for item in results
        ],
    }


def _remove_project_files_created_after(
    images_dir: Path,
    baseline: set[Path],
) -> None:
    for path in images_dir.iterdir():
        if path.is_file() and path.resolve() not in baseline:
            path.unlink(missing_ok=True)


def _read_annotation(annotation_path: Path, images_dir: Path) -> VocImage:
    try:
        xml_text = annotation_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProjectFormatError(f"无法读取 XML：{annotation_path}：{exc}") from exc
    upper = xml_text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise ProjectFormatError(f"XML 不允许 DTD 或实体声明：{annotation_path}")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ProjectFormatError(f"XML 格式错误：{annotation_path}：{exc}") from exc
    if root.tag != "annotation":
        raise ProjectFormatError(f"XML 根节点不是 annotation：{annotation_path}")

    filename = _safe_relative_name(_required_node_text(root, "filename", annotation_path))
    image_path = (images_dir / Path(*PurePosixPath(filename).parts)).resolve()
    try:
        image_path.relative_to(images_dir.resolve())
    except ValueError as exc:
        raise ProjectFormatError(f"XML 图片路径越界：{filename}") from exc
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ProjectFormatError(f"XML 引用了不支持的图片格式：{filename}")
    if not image_path.is_file():
        raise ProjectFormatError(f"XML 引用的图片不存在：{image_path}")

    size = root.find("size")
    if size is None:
        raise ProjectFormatError(f"XML 缺少 size：{annotation_path}")
    width = _positive_int_node(size, "width", annotation_path)
    height = _positive_int_node(size, "height", annotation_path)
    _verify_image_size(image_path, width, height)

    boxes: list[VocBox] = []
    for index, object_node in enumerate(root.findall("object"), start=1):
        category_name = _required_node_text(
            object_node, "name", annotation_path
        ).strip()
        if any(char in category_name for char in "\r\n,"):
            raise ProjectFormatError(
                f"类别名称含换行或逗号：{annotation_path} 第 {index} 个 object"
            )
        if len(category_name) > 128:
            raise ProjectFormatError(f"类别名称超过 128 个字符：{annotation_path}")
        bounds = object_node.find("bndbox")
        if bounds is None:
            raise ProjectFormatError(
                f"XML 缺少 bndbox：{annotation_path} 第 {index} 个 object"
            )
        x1 = _finite_float_node(bounds, "xmin", annotation_path)
        y1 = _finite_float_node(bounds, "ymin", annotation_path)
        x2 = _finite_float_node(bounds, "xmax", annotation_path)
        y2 = _finite_float_node(bounds, "ymax", annotation_path)
        if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
            raise ProjectFormatError(
                f"标注框坐标无效：{annotation_path} 第 {index} 个 object"
            )
        if x2 > width or y2 > height:
            raise ProjectFormatError(
                f"标注框超出图片范围：{annotation_path} 第 {index} 个 object"
            )
        boxes.append(VocBox(category_name, x1, y1, x2, y2))

    return VocImage(
        filename=filename,
        image_path=image_path,
        annotation_path=annotation_path,
        width=width,
        height=height,
        boxes=tuple(boxes),
    )


def _read_unconfirmed_image(filename: str, images_dir: Path) -> VocImage:
    """Create a validated source record for an image that has no XML."""

    safe_name = _safe_relative_name(filename)
    image_path = (images_dir / Path(*PurePosixPath(safe_name).parts)).resolve()
    try:
        image_path.relative_to(images_dir.resolve())
    except ValueError as exc:
        raise ProjectFormatError(f"未标注图片路径越界：{filename}") from exc
    width, height = _read_image_size(image_path)
    return VocImage(
        filename=safe_name,
        image_path=image_path,
        annotation_path=None,
        width=width,
        height=height,
        boxes=(),
    )


def _read_train_list(path: Path) -> list[str]:
    try:
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProjectFormatError(f"无法读取 train.txt：{exc}") from exc
    names = [_safe_relative_name(line.strip()) for line in raw_lines if line.strip()]
    if not names:
        raise ProjectFormatError("train.txt 为空")
    return names


def _safe_relative_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    if normalized.startswith("images/"):
        normalized = normalized[len("images/") :]
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or str(path) == "."
        or ":" in path.parts[0]
    ):
        raise ProjectFormatError(f"不安全的图片相对路径：{value}")
    return path.as_posix()


def _required_node_text(parent: ET.Element, name: str, path: Path) -> str:
    node = parent.find(name)
    value = "" if node is None or node.text is None else node.text.strip()
    if not value:
        raise ProjectFormatError(f"XML 缺少 {name}：{path}")
    return value


def _positive_int_node(parent: ET.Element, name: str, path: Path) -> int:
    value = _finite_float_node(parent, name, path)
    if not value.is_integer() or value <= 0:
        raise ProjectFormatError(f"XML 的 {name} 必须是正整数：{path}")
    return int(value)


def _finite_float_node(parent: ET.Element, name: str, path: Path) -> float:
    text = _required_node_text(parent, name, path)
    try:
        value = float(text)
    except ValueError as exc:
        raise ProjectFormatError(f"XML 的 {name} 不是数字：{path}") from exc
    if not isfinite(value):
        raise ProjectFormatError(f"XML 的 {name} 必须是有限数值：{path}")
    return value


def _verify_image_size(image_path: Path, width: int, height: int) -> None:
    actual = _read_image_size(image_path)
    if actual != (width, height):
        raise ProjectFormatError(
            f"图片尺寸与 XML 不一致：{image_path.name}；"
            f"XML={width}x{height}，图片={actual[0]}x{actual[1]}"
        )


def _read_image_size(image_path: Path) -> tuple[int, int]:
    normalized = None
    try:
        with Image.open(image_path) as opened:
            opened.load()
            normalized = ImageOps.exif_transpose(opened)
            actual = normalized.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ProjectFormatError(f"图片损坏或无法解码：{image_path}：{exc}") from exc
    finally:
        if normalized is not None:
            normalized.close()
    return int(actual[0]), int(actual[1])
