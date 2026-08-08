"""Transactional SQLite repository for annotations and ML run metadata."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_biaozhu.core.annotation_quality import (
    MAX_DEDUPLICATION_IOU,
    MIN_DEDUPLICATION_IOU,
    deduplicate_ai_draft_boxes,
)
from ai_biaozhu.core.domain import (
    AIPrediction,
    AIPredictionImportResult,
    AIStatus,
    AnnotationOrigin,
    BoundingBox,
    BoxInput,
    Category,
    DeploymentPackage,
    ImageRecord,
    ModelKey,
    ReviewStatus,
    RunKind,
    RunRecord,
    RunStatus,
    TrainingPreflight,
    validate_box_bounds,
)
from ai_biaozhu.core.exceptions import (
    DataIntegrityError,
    EmptyAnnotationConfirmationRequired,
    RecordNotFoundError,
    RevisionConflictError,
)

from .schema import connect_database
from .utils import utc_now

_UNSET = object()


@dataclass(frozen=True, slots=True)
class BulkAnnotationClearPreview:
    """Immutable summary of the exact rows a bulk clear will affect."""

    image_ids: tuple[str, ...]
    box_count: int
    manual_box_count: int
    ai_box_count: int
    mixed_box_count: int
    verified_image_count: int
    draft_image_count: int
    unreviewed_image_count: int
    ai_import_count: int
    image_revisions: tuple[tuple[str, int], ...]

    @property
    def image_count(self) -> int:
        return len(self.image_ids)


@dataclass(frozen=True, slots=True)
class AIDedupRemovalRecord:
    image_id: str
    removed_box_id: str
    kept_box_id: str
    class_id: str
    iou: float
    removed_confidence: float | None
    kept_confidence: float | None


@dataclass(frozen=True, slots=True)
class AIDedupPreview:
    """Exact, immutable preview for conservative historical AI cleanup."""

    requested_image_ids: tuple[str, ...]
    affected_image_ids: tuple[str, ...]
    removals: tuple[AIDedupRemovalRecord, ...]
    before_box_count: int
    after_box_count: int
    protected_box_count: int
    iou_threshold: float

    @property
    def requested_image_count(self) -> int:
        return len(self.requested_image_ids)

    @property
    def affected_image_count(self) -> int:
        return len(self.affected_image_ids)

    @property
    def removed_box_count(self) -> int:
        return len(self.removals)


class AnnotationRepository:
    """The sole mutable source of project annotations.

    Public write methods create an immediate transaction. An outer
    :meth:`transaction` can group multiple calls; nested calls use savepoints.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.connection = connect_database(self.database_path)
        self._savepoint_counter = 0

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> AnnotationRepository:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            self._savepoint_counter += 1
            name = f"repo_sp_{self._savepoint_counter}"
            self.connection.execute(f"SAVEPOINT {name}")
            try:
                yield self.connection
            except Exception:
                self.connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
                self.connection.execute(f"RELEASE SAVEPOINT {name}")
                raise
            else:
                self.connection.execute(f"RELEASE SAVEPOINT {name}")
            return

        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    # ------------------------------------------------------------------
    # Categories

    def add_category(
        self,
        name: str,
        *,
        display_name: str | None = None,
        color: str = "#22C55E",
        enabled: bool = True,
        category_id: str | None = None,
        position: int | None = None,
    ) -> Category:
        name = _validate_category_name(name)
        display_name = _validate_display_name(display_name)
        color = _validate_color(color)
        category_id = category_id or uuid4().hex
        now = utc_now()
        with self.transaction() as connection:
            _assert_category_label_unique(connection, name)
            if display_name is not None:
                _assert_category_label_unique(connection, display_name)
            if position is None:
                row = connection.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM categories"
                ).fetchone()
                position = int(row[0])
            if position < 0:
                raise ValueError("类别顺序不能小于 0")
            try:
                connection.execute(
                    """
                    INSERT INTO categories
                        (id, name, display_name, color, position, enabled,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        category_id,
                        name,
                        display_name,
                        color,
                        position,
                        int(enabled),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DataIntegrityError(f"类别名称或 ID 已存在：{name}") from exc
        return self.get_category(category_id)

    create_category = add_category

    def get_category(self, category_id: str) -> Category:
        row = self.connection.execute(
            "SELECT * FROM categories WHERE id = ?", (category_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"类别不存在：{category_id}")
        return _category_from_row(row)

    def list_categories(self, *, enabled_only: bool = False) -> tuple[Category, ...]:
        where = "WHERE enabled = 1" if enabled_only else ""
        rows = self.connection.execute(
            f"SELECT * FROM categories {where} ORDER BY position, id"
        ).fetchall()
        return tuple(_category_from_row(row) for row in rows)

    def update_category(
        self,
        category_id: str,
        *,
        name: str | None = None,
        display_name: str | None | object = _UNSET,
        color: str | None = None,
        enabled: bool | None = None,
        position: int | None = None,
    ) -> Category:
        current = self.get_category(category_id)
        new_name = current.name if name is None else _validate_category_name(name)
        new_display_name = (
            current.display_name
            if display_name is _UNSET
            else _validate_display_name(display_name)
        )
        new_color = current.color if color is None else _validate_color(color)
        new_enabled = current.enabled if enabled is None else bool(enabled)
        new_position = current.position if position is None else int(position)
        if new_position < 0:
            raise ValueError("类别顺序不能小于 0")
        with self.transaction() as connection:
            _assert_category_label_unique(
                connection,
                new_name,
                exclude_category_id=category_id,
            )
            if new_display_name is not None:
                _assert_category_label_unique(
                    connection,
                    new_display_name,
                    exclude_category_id=category_id,
                )
            try:
                connection.execute(
                    """
                    UPDATE categories
                    SET name = ?, display_name = ?, color = ?, enabled = ?,
                        position = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        new_name,
                        new_display_name,
                        new_color,
                        int(new_enabled),
                        new_position,
                        utc_now(),
                        category_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DataIntegrityError(f"类别名称已存在：{new_name}") from exc
        return self.get_category(category_id)

    def rename_category_canonical(
        self,
        category_id: str,
        name: str,
    ) -> Category:
        """Rename a dataset class without changing its stable ID or boxes.

        The previous canonical name is retained as an import-only alias.  A
        canonical rename deliberately clears the presentation alias so every
        newly generated training/export/deployment artifact visibly uses the
        new canonical name.
        """

        current = self.get_category(category_id)
        new_name = _validate_category_name(name)
        if current.name == new_name and current.display_name is None:
            return current
        with self.transaction(immediate=True) as connection:
            _assert_category_label_unique(
                connection,
                new_name,
                exclude_category_id=category_id,
            )
            existing_alias = connection.execute(
                """
                SELECT category_id FROM category_name_aliases
                WHERE alias = ? COLLATE NOCASE
                """,
                (current.name,),
            ).fetchone()
            if existing_alias is not None and str(existing_alias[0]) != category_id:
                raise DataIntegrityError(
                    f"类别旧名称已属于其他类别：{current.name}"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO category_name_aliases
                    (alias, category_id, created_at)
                VALUES (?, ?, ?)
                """,
                (current.name, category_id, utc_now()),
            )
            try:
                connection.execute(
                    """
                    UPDATE categories
                    SET name = ?, display_name = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_name, utc_now(), category_id),
                )
            except sqlite3.IntegrityError as exc:
                raise DataIntegrityError(f"类别名称已存在：{new_name}") from exc
        return self.get_category(category_id)

    def list_category_name_aliases(
        self,
        category_id: str | None = None,
    ) -> tuple[tuple[str, str], ...]:
        values: tuple[object, ...] = ()
        where = ""
        if category_id is not None:
            where = "WHERE category_id = ?"
            values = (str(category_id),)
        rows = self.connection.execute(
            f"""
            SELECT alias, category_id FROM category_name_aliases
            {where} ORDER BY alias COLLATE NOCASE
            """,
            values,
        ).fetchall()
        return tuple((str(row["alias"]), str(row["category_id"])) for row in rows)

    def resolve_category_name(self, name: str) -> Category | None:
        """Resolve a canonical or historical import name case-insensitively."""

        candidate = str(name).strip()
        if not candidate:
            return None
        row = self.connection.execute(
            "SELECT * FROM categories WHERE name = ? COLLATE NOCASE",
            (candidate,),
        ).fetchone()
        if row is not None:
            return _category_from_row(row)
        row = self.connection.execute(
            """
            SELECT categories.*
            FROM category_name_aliases
            JOIN categories ON categories.id = category_name_aliases.category_id
            WHERE category_name_aliases.alias = ? COLLATE NOCASE
            """,
            (candidate,),
        ).fetchone()
        return None if row is None else _category_from_row(row)

    def delete_category(self, category_id: str) -> None:
        with self.transaction() as connection:
            try:
                cursor = connection.execute(
                    "DELETE FROM categories WHERE id = ?", (category_id,)
                )
            except sqlite3.IntegrityError as exc:
                raise DataIntegrityError("该类别仍被标注框使用，不能删除") from exc
            if cursor.rowcount != 1:
                raise RecordNotFoundError(f"类别不存在：{category_id}")

    def category_box_count(self, category_id: str) -> int:
        """Return the number of boxes linked to one category in the project."""

        self.get_category(category_id)
        row = self.connection.execute(
            "SELECT COUNT(*) FROM boxes WHERE class_id = ?",
            (category_id,),
        ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Images

    def add_image_record(
        self,
        *,
        image_id: str,
        relative_path: str,
        original_name: str,
        source_path: str | None,
        sha256: str,
        width: int,
        height: int,
    ) -> ImageRecord:
        if width <= 0 or height <= 0:
            raise ValueError("图像尺寸必须大于 0")
        if len(sha256) != 64:
            raise ValueError("sha256 必须是 64 位十六进制字符串")
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise ValueError("sha256 必须是十六进制字符串") from exc
        now = utc_now()
        with self.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO images (
                        id, relative_path, original_name, source_path, sha256,
                        width, height, review_status, origin, ai_status,
                        revision, imported_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'unreviewed', 'none', 'none', 0, ?, ?)
                    """,
                    (
                        image_id,
                        relative_path,
                        original_name,
                        source_path,
                        sha256.lower(),
                        width,
                        height,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DataIntegrityError("图片 ID、路径或内容哈希已存在") from exc
        return self.get_image(image_id)

    def get_image(self, image_id: str) -> ImageRecord:
        row = self.connection.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"图片不存在：{image_id}")
        return _image_from_row(row)

    def find_image_by_sha256(self, sha256: str) -> ImageRecord | None:
        row = self.connection.execute(
            "SELECT * FROM images WHERE sha256 = ?", (sha256.lower(),)
        ).fetchone()
        return None if row is None else _image_from_row(row)

    def list_images(
        self,
        *,
        review_status: ReviewStatus | str | None = None,
        ai_status: AIStatus | str | None = None,
        training_selected: bool | None = None,
    ) -> tuple[ImageRecord, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if review_status is not None:
            clauses.append("review_status = ?")
            parameters.append(ReviewStatus(review_status).value)
        if ai_status is not None:
            clauses.append("ai_status = ?")
            parameters.append(AIStatus(ai_status).value)
        if training_selected is not None:
            clauses.append("training_selected = ?")
            parameters.append(1 if training_selected else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM images {where} ORDER BY imported_at, id",
            tuple(parameters),
        ).fetchall()
        return tuple(_image_from_row(row) for row in rows)

    def delete_image(self, image_id: str) -> ImageRecord:
        current = self.get_image(image_id)
        with self.transaction() as connection:
            connection.execute("DELETE FROM images WHERE id = ?", (image_id,))
        return current

    def delete_images(self, image_ids: Iterable[str]) -> tuple[ImageRecord, ...]:
        """Atomically delete selected image rows and their cascaded annotations."""

        normalized = _normalize_image_ids(image_ids)
        records = tuple(self.get_image(image_id) for image_id in normalized)
        with self.transaction() as connection:
            connection.executemany(
                "DELETE FROM images WHERE id = ?",
                ((image_id,) for image_id in normalized),
            )
        return records

    def set_training_selected(
        self,
        image_ids: Iterable[str],
        selected: bool,
    ) -> tuple[ImageRecord, ...]:
        """Persist whether selected images may enter future training snapshots."""

        normalized = _normalize_image_ids(image_ids)
        # Resolve every ID before mutating so a bad selection cannot partly apply.
        tuple(self.get_image(image_id) for image_id in normalized)
        with self.transaction() as connection:
            connection.executemany(
                """
                UPDATE images
                SET training_selected = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    (1 if selected else 0, utc_now(), image_id)
                    for image_id in normalized
                ),
            )
        return tuple(self.get_image(image_id) for image_id in normalized)

    def select_only_for_training(
        self,
        image_ids: Iterable[str],
    ) -> tuple[ImageRecord, ...]:
        """Make the supplied IDs the exact project-wide training selection."""

        normalized = _normalize_image_ids(image_ids)
        tuple(self.get_image(image_id) for image_id in normalized)
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE images
                SET training_selected = 0, updated_at = ?
                WHERE training_selected != 0
                """,
                (now,),
            )
            connection.executemany(
                """
                UPDATE images
                SET training_selected = 1, updated_at = ?
                WHERE id = ?
                """,
                ((now, image_id) for image_id in normalized),
            )
        return tuple(self.get_image(image_id) for image_id in normalized)

    def preview_clear_all_annotations(
        self, image_ids: Iterable[str]
    ) -> BulkAnnotationClearPreview:
        """Validate and count a prospective multi-image annotation clear.

        IDs are de-duplicated while preserving caller order. Every requested
        image must exist; callers therefore never receive a misleading partial
        preview.
        """

        normalized = _normalize_image_ids(image_ids)
        images = tuple(self.get_image(image_id) for image_id in normalized)
        origin_counts = {
            AnnotationOrigin.MANUAL: 0,
            AnnotationOrigin.AI: 0,
            AnnotationOrigin.MIXED: 0,
        }
        box_count = 0
        ai_import_count = 0
        for image in images:
            rows = self.connection.execute(
                """
                SELECT origin, COUNT(*) AS count
                FROM boxes
                WHERE image_id = ?
                GROUP BY origin
                """,
                (image.id,),
            ).fetchall()
            for row in rows:
                count = int(row["count"])
                box_count += count
                origin_counts[AnnotationOrigin(str(row["origin"]))] += count
            ai_import_count += int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM ai_imports WHERE image_id = ?",
                    (image.id,),
                ).fetchone()[0]
            )
        return BulkAnnotationClearPreview(
            image_ids=normalized,
            box_count=box_count,
            manual_box_count=origin_counts[AnnotationOrigin.MANUAL],
            ai_box_count=origin_counts[AnnotationOrigin.AI],
            mixed_box_count=origin_counts[AnnotationOrigin.MIXED],
            verified_image_count=sum(
                image.review_status is ReviewStatus.VERIFIED for image in images
            ),
            draft_image_count=sum(
                image.review_status is ReviewStatus.DRAFT for image in images
            ),
            unreviewed_image_count=sum(
                image.review_status is ReviewStatus.UNREVIEWED for image in images
            ),
            ai_import_count=ai_import_count,
            image_revisions=tuple((image.id, image.revision) for image in images),
        )

    def clear_all_annotations(
        self,
        image_ids: Iterable[str],
        *,
        expected_revisions: Mapping[str, int] | None = None,
    ) -> BulkAnnotationClearPreview:
        """Atomically remove every box from selected images.

        Images are deliberately reset to ``unreviewed``/``none`` rather than
        becoming verified negatives. AI import markers are also removed so a
        user can run auto-labeling again, while model and job history remain.
        The returned preview describes the state immediately before mutation.
        """

        normalized = _normalize_image_ids(image_ids)
        expected = (
            {}
            if expected_revisions is None
            else {str(key): int(value) for key, value in expected_revisions.items()}
        )
        now = utc_now()
        with self.transaction() as connection:
            images = tuple(
                self._checked_image(connection, image_id, expected.get(image_id))
                for image_id in normalized
            )
            before = self.preview_clear_all_annotations(normalized)
            for image in images:
                connection.execute(
                    "DELETE FROM boxes WHERE image_id = ?",
                    (image.id,),
                )
                connection.execute(
                    "DELETE FROM ai_imports WHERE image_id = ?",
                    (image.id,),
                )
                connection.execute(
                    """
                    UPDATE images
                    SET review_status = 'unreviewed', origin = 'none',
                        ai_status = 'none', revision = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (image.revision + 1, now, image.id),
                )
        return before

    # UI/controller-friendly aliases.
    preview_delete_all_annotations = preview_clear_all_annotations
    delete_all_annotations = clear_all_annotations

    def preview_ai_deduplication(
        self,
        image_ids: Iterable[str] | None = None,
        *,
        iou_threshold: float = 0.80,
    ) -> AIDedupPreview:
        """Preview duplicate removal without mutating any project data."""

        normalized = (
            tuple(image.id for image in self.list_images())
            if image_ids is None
            else _normalize_image_ids(image_ids)
        )
        images = tuple(self.get_image(image_id) for image_id in normalized)
        removals: list[AIDedupRemovalRecord] = []
        affected: list[str] = []
        before_count = 0
        after_count = 0
        protected_count = 0
        threshold = deduplicate_ai_draft_boxes(
            (), iou_threshold=float(iou_threshold)
        ).iou_threshold
        for image in images:
            boxes = self.list_boxes(image.id)
            result = deduplicate_ai_draft_boxes(
                boxes,
                iou_threshold=threshold,
                image_review_status=image.review_status,
                image_origin=image.origin,
            )
            before_count += result.stats.before_count
            after_count += result.stats.after_count
            protected_count += result.stats.protected_count
            if result.changed:
                affected.append(image.id)
            removals.extend(
                AIDedupRemovalRecord(
                    image_id=image.id,
                    removed_box_id=removal.removed_id,
                    kept_box_id=removal.kept_id,
                    class_id=removal.class_id,
                    iou=removal.iou,
                    removed_confidence=removal.removed_confidence,
                    kept_confidence=removal.kept_confidence,
                )
                for removal in result.removals
            )
            threshold = result.iou_threshold
        return AIDedupPreview(
            requested_image_ids=normalized,
            affected_image_ids=tuple(affected),
            removals=tuple(removals),
            before_box_count=before_count,
            after_box_count=after_count,
            protected_box_count=protected_count,
            iou_threshold=threshold,
        )

    def deduplicate_ai_drafts(
        self,
        image_ids: Iterable[str] | None = None,
        *,
        iou_threshold: float = 0.80,
    ) -> AIDedupPreview:
        """Atomically apply an exact preview of protected AI-draft cleanup."""

        with self.transaction() as connection:
            preview = self.preview_ai_deduplication(
                image_ids,
                iou_threshold=iou_threshold,
            )
            now = utc_now()
            for removal in preview.removals:
                cursor = connection.execute(
                    """
                    DELETE FROM boxes
                    WHERE id = ? AND image_id = ? AND origin = 'ai'
                    """,
                    (removal.removed_box_id, removal.image_id),
                )
                if cursor.rowcount != 1:
                    raise DataIntegrityError(
                        f"AI 去重期间标注框状态发生变化：{removal.removed_box_id}"
                    )
            for image_id in preview.affected_image_ids:
                connection.execute(
                    """
                    UPDATE images
                    SET revision = revision + 1, updated_at = ?
                    WHERE id = ? AND review_status = 'draft' AND origin = 'ai'
                    """,
                    (now, image_id),
                )
                connection.execute(
                    """
                    UPDATE ai_imports
                    SET box_count = (
                        SELECT COUNT(*) FROM boxes
                        WHERE boxes.image_id = ai_imports.image_id
                          AND boxes.model_run_id = ai_imports.run_id
                    )
                    WHERE image_id = ?
                    """,
                    (image_id,),
                )
        return preview

    def backup_database(self, destination: Path | str) -> Path:
        """Write a consistent SQLite snapshot using the online backup API."""

        destination = Path(destination).resolve()
        if destination == self.database_path.resolve():
            raise ValueError("备份目标不能是当前标注数据库")
        if destination.exists():
            raise FileExistsError(f"备份目标已经存在：{destination}")
        if self.connection.in_transaction:
            raise RuntimeError("数据库事务进行中，暂时不能创建备份")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target: sqlite3.Connection | None = None
        try:
            target = sqlite3.connect(destination, timeout=30.0, isolation_level=None)
            self.connection.backup(target)
            integrity = str(target.execute("PRAGMA quick_check").fetchone()[0])
            if integrity.casefold() != "ok":
                raise DataIntegrityError(f"新建备份完整性校验失败：{integrity}")
        except Exception:
            if target is not None:
                target.close()
                target = None
            destination.unlink(missing_ok=True)
            raise
        finally:
            if target is not None:
                target.close()
        return destination

    def validate_database_backup(
        self,
        source: Path | str,
        *,
        expected_project_id: str | None = None,
    ) -> tuple[str, int, int]:
        """Validate a candidate backup without changing the live database."""

        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"标注数据库备份不存在：{source}")
        candidate = sqlite3.connect(
            f"{source.as_uri()}?mode=ro",
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
        try:
            integrity = str(candidate.execute("PRAGMA quick_check").fetchone()[0])
            if integrity.casefold() != "ok":
                raise DataIntegrityError(f"备份完整性校验失败：{integrity}")
            row = candidate.execute(
                "SELECT value FROM project_meta WHERE key = 'project_id'"
            ).fetchone()
            if row is None:
                raise DataIntegrityError("备份缺少项目 ID")
            project_id = str(row[0])
            if expected_project_id is not None and project_id != expected_project_id:
                raise DataIntegrityError("备份属于另一个项目，禁止恢复")
            image_count = int(
                candidate.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            )
            box_count = int(
                candidate.execute("SELECT COUNT(*) FROM boxes").fetchone()[0]
            )
        except sqlite3.DatabaseError as exc:
            raise DataIntegrityError(f"无效的标注数据库备份：{exc}") from exc
        finally:
            candidate.close()
        return project_id, image_count, box_count

    def restore_database(
        self,
        source: Path | str,
        *,
        expected_project_id: str | None = None,
    ) -> None:
        """Atomically restore a validated snapshot into the live connection."""

        source = Path(source).resolve()
        self.validate_database_backup(
            source, expected_project_id=expected_project_id
        )
        if self.connection.in_transaction:
            raise RuntimeError("数据库事务进行中，暂时不能恢复备份")
        candidate = sqlite3.connect(
            f"{source.as_uri()}?mode=ro",
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
        try:
            candidate.backup(self.connection)
        finally:
            candidate.close()
        integrity = str(
            self.connection.execute("PRAGMA quick_check").fetchone()[0]
        )
        if integrity.casefold() != "ok":
            raise DataIntegrityError(f"恢复后的数据库完整性校验失败：{integrity}")
        violation = self.connection.execute("PRAGMA foreign_key_check").fetchone()
        if violation is not None:
            raise DataIntegrityError(
                f"恢复后的数据库外键校验失败：{tuple(violation)}"
            )

    def set_ai_status(
        self,
        image_id: str,
        status: AIStatus | str,
        *,
        expected_revision: int | None = None,
        bump_revision: bool = True,
    ) -> ImageRecord:
        status = AIStatus(status)
        with self.transaction() as connection:
            image = self._checked_image(connection, image_id, expected_revision)
            connection.execute(
                """
                UPDATE images
                SET ai_status = ?, revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    image.revision + (1 if bump_revision else 0),
                    utc_now(),
                    image_id,
                ),
            )
        return self.get_image(image_id)

    # ------------------------------------------------------------------
    # Boxes and D confirmation

    def list_boxes(self, image_id: str) -> tuple[BoundingBox, ...]:
        self.get_image(image_id)
        rows = self.connection.execute(
            "SELECT * FROM boxes WHERE image_id = ? ORDER BY created_at, id",
            (image_id,),
        ).fetchall()
        return tuple(_box_from_row(row) for row in rows)

    def get_box(self, box_id: str) -> BoundingBox:
        row = self.connection.execute(
            "SELECT * FROM boxes WHERE id = ?", (box_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"标注框不存在：{box_id}")
        return _box_from_row(row)

    def add_box(
        self,
        image_id: str,
        box: BoxInput | Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> BoundingBox:
        value = BoxInput.from_value(box)
        if value.origin == AnnotationOrigin.NONE:
            value = _manual_box(value)
        with self.transaction() as connection:
            image = self._checked_image(connection, image_id, expected_revision)
            self._validate_box(connection, image, value)
            box_id = value.id or uuid4().hex
            self._insert_box(connection, image_id, box_id, value)
            self._mark_annotation_change(connection, image, human_edit=True)
        return self.get_box(box_id)

    def update_box(
        self,
        box_id: str,
        *,
        class_id: str | None = None,
        x1: float | None = None,
        y1: float | None = None,
        x2: float | None = None,
        y2: float | None = None,
        expected_revision: int | None = None,
    ) -> BoundingBox:
        current = self.get_box(box_id)
        updated = BoxInput(
            id=current.id,
            class_id=class_id or current.class_id,
            x1=current.x1 if x1 is None else x1,
            y1=current.y1 if y1 is None else y1,
            x2=current.x2 if x2 is None else x2,
            y2=current.y2 if y2 is None else y2,
            origin=AnnotationOrigin.MANUAL,
        )
        with self.transaction() as connection:
            image = self._checked_image(connection, current.image_id, expected_revision)
            self._validate_box(connection, image, updated)
            now = utc_now()
            connection.execute(
                """
                UPDATE boxes
                SET class_id = ?, x1 = ?, y1 = ?, x2 = ?, y2 = ?,
                    origin = 'manual', confidence = NULL, model_run_id = NULL,
                    prediction_id = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated.class_id,
                    updated.x1,
                    updated.y1,
                    updated.x2,
                    updated.y2,
                    now,
                    box_id,
                ),
            )
            self._mark_annotation_change(connection, image, human_edit=True)
        return self.get_box(box_id)

    def delete_box(
        self, box_id: str, *, expected_revision: int | None = None
    ) -> ImageRecord:
        current = self.get_box(box_id)
        with self.transaction() as connection:
            image = self._checked_image(connection, current.image_id, expected_revision)
            connection.execute("DELETE FROM boxes WHERE id = ?", (box_id,))
            self._mark_annotation_change(connection, image, human_edit=True)
        return self.get_image(current.image_id)

    def replace_boxes(
        self,
        image_id: str,
        boxes: Iterable[BoxInput | BoundingBox | Mapping[str, Any]],
        *,
        expected_revision: int | None = None,
    ) -> tuple[BoundingBox, ...]:
        values = tuple(BoxInput.from_value(box) for box in boxes)
        with self.transaction() as connection:
            image = self._checked_image(connection, image_id, expected_revision)
            self._replace_boxes(connection, image, values)
            self._mark_annotation_change(connection, image, human_edit=True)
        return self.list_boxes(image_id)

    def restore_annotation_session_baseline(
        self,
        image_id: str,
        boxes: Iterable[BoxInput | BoundingBox | Mapping[str, Any]],
        *,
        review_status: ReviewStatus | str,
        origin: AnnotationOrigin | str,
        ai_status: AIStatus | str,
        expected_revision: int | None = None,
    ) -> ImageRecord:
        """Restore one opened image to its captured annotation session baseline.

        Unlike :meth:`replace_boxes`, this is deliberately not a new human
        edit.  It restores the caller-provided image review/origin/AI state
        along with every box field, so an ``undo all`` after autosave does not
        demote an originally verified image.  ``expected_revision`` protects
        against overwriting a concurrent update after the session began.
        """

        values = tuple(BoxInput.from_value(box) for box in boxes)
        restored_review_status = ReviewStatus(review_status)
        restored_origin = AnnotationOrigin(origin)
        restored_ai_status = AIStatus(ai_status)
        with self.transaction() as connection:
            image = self._checked_image(connection, image_id, expected_revision)
            self._replace_boxes(connection, image, values)
            connection.execute(
                """
                UPDATE images
                SET review_status = ?, origin = ?, ai_status = ?,
                    revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    restored_review_status.value,
                    restored_origin.value,
                    restored_ai_status.value,
                    image.revision + 1,
                    utc_now(),
                    image.id,
                ),
            )
        return self.get_image(image_id)

    def confirm_image(
        self,
        image_id: str,
        *,
        confirm_empty: bool = False,
        expected_revision: int | None = None,
    ) -> ImageRecord:
        with self.transaction() as connection:
            image = self._checked_image(connection, image_id, expected_revision)
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM boxes WHERE image_id = ?", (image_id,)
                ).fetchone()[0]
            )
            if (
                count == 0
                and image.review_status != ReviewStatus.VERIFIED
                and not confirm_empty
            ):
                raise EmptyAnnotationConfirmationRequired(image_id)
            origin = self._derive_origin(
                connection,
                image_id,
                empty_fallback=(
                    image.origin
                    if image.origin in (AnnotationOrigin.AI, AnnotationOrigin.MIXED)
                    else AnnotationOrigin.MANUAL
                ),
            )
            connection.execute(
                """
                UPDATE images
                SET review_status = 'verified', origin = ?,
                    revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (origin.value, image.revision + 1, utc_now(), image_id),
            )
        return self.get_image(image_id)

    def save_and_confirm(
        self,
        image_id: str,
        boxes: Iterable[BoxInput | BoundingBox | Mapping[str, Any]],
        *,
        confirm_empty: bool = False,
        expected_revision: int | None = None,
    ) -> ImageRecord:
        """Atomically persist the current canvas and perform the D confirmation."""

        values = tuple(BoxInput.from_value(box) for box in boxes)
        with self.transaction() as connection:
            image = self._checked_image(connection, image_id, expected_revision)
            if (
                not values
                and image.review_status != ReviewStatus.VERIFIED
                and not confirm_empty
            ):
                raise EmptyAnnotationConfirmationRequired(image_id)
            self._replace_boxes(connection, image, values)
            origin = self._derive_origin(
                connection,
                image_id,
                empty_fallback=(
                    image.origin
                    if image.origin in (AnnotationOrigin.AI, AnnotationOrigin.MIXED)
                    else AnnotationOrigin.MANUAL
                ),
            )
            connection.execute(
                """
                UPDATE images
                SET review_status = 'verified', origin = ?,
                    revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (origin.value, image.revision + 1, utc_now(), image_id),
            )
        return self.get_image(image_id)

    save_boxes = replace_boxes
    verify_image = confirm_image

    def import_ai_predictions(
        self,
        run_id: str,
        image_id: str,
        predictions: Sequence[AIPrediction | Mapping[str, Any]],
        *,
        expected_revision: int | None = None,
    ) -> AIPredictionImportResult:
        """Idempotently import one image's AI suggestions without touching manual boxes."""

        run = self.get_run(run_id)
        if run.kind != RunKind.PREDICT:
            raise DataIntegrityError("AI 预测只能关联 predict 类型运行")
        values = tuple(AIPrediction.from_value(item) for item in predictions)
        if bool(run.parameters.get("deduplicate", False)):
            try:
                dedup_iou = float(run.parameters.get("dedup_iou", 0.8))
            except (TypeError, ValueError, OverflowError) as exc:
                raise DataIntegrityError("AI 去重 IoU 不是有效数字") from exc
            if not MIN_DEDUPLICATION_IOU <= dedup_iou <= MAX_DEDUPLICATION_IOU:
                raise DataIntegrityError("AI 去重 IoU 必须位于 0.70～0.95")
            values = _deduplicate_ai_predictions(values, dedup_iou)
        prediction_revisions = {
            value.expected_revision
            for value in values
            if value.expected_revision is not None
        }
        if len(prediction_revisions) > 1:
            raise DataIntegrityError("同一图片的 AI 预测包含不同 expected_revision")
        if prediction_revisions:
            prediction_revision = next(iter(prediction_revisions))
            if (
                expected_revision is not None
                and prediction_revision != expected_revision
            ):
                raise DataIntegrityError("AI 预测 expected_revision 与任务参数不一致")
            expected_revision = prediction_revision
        imported_count = 0
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"图片不存在：{image_id}")
            image = _image_from_row(row)
            already_imported = connection.execute(
                "SELECT box_count FROM ai_imports WHERE run_id = ? AND image_id = ?",
                (run_id, image_id),
            ).fetchone()
            if already_imported is not None:
                return AIPredictionImportResult(
                    run_id=run_id,
                    image_id=image_id,
                    imported_count=0,
                    skipped_verified=image.review_status == ReviewStatus.VERIFIED,
                    revision_conflict=False,
                )
            if image.review_status == ReviewStatus.VERIFIED:
                return AIPredictionImportResult(
                    run_id=run_id,
                    image_id=image_id,
                    imported_count=0,
                    skipped_verified=True,
                    revision_conflict=False,
                )
            if expected_revision is not None and image.revision != expected_revision:
                return AIPredictionImportResult(
                    run_id=run_id,
                    image_id=image_id,
                    imported_count=0,
                    skipped_verified=False,
                    revision_conflict=True,
                )

            categories = self.list_categories(enabled_only=True)
            category_ids = {category.id for category in categories}
            # A newer AI run replaces only untouched AI suggestions. Manual
            # boxes (including AI boxes a human edited) are never deleted.
            connection.execute(
                "DELETE FROM boxes WHERE image_id = ? AND origin = 'ai'",
                (image_id,),
            )
            for index, prediction in enumerate(values):
                if prediction.image_id != image_id:
                    raise DataIntegrityError("AI 预测的 image_id 与当前图片不一致")
                class_id = prediction.class_id
                if class_id is None and prediction.class_index is not None:
                    if not 0 <= prediction.class_index < len(categories):
                        raise DataIntegrityError("AI 预测类别索引超出项目类别范围")
                    class_id = categories[prediction.class_index].id
                if class_id is None or class_id not in category_ids:
                    raise DataIntegrityError("AI 预测引用了不存在的类别")
                box = BoxInput(
                    class_id=class_id,
                    x1=prediction.x1,
                    y1=prediction.y1,
                    x2=prediction.x2,
                    y2=prediction.y2,
                    origin=AnnotationOrigin.AI,
                    confidence=prediction.confidence,
                    model_run_id=run_id,
                    prediction_id=prediction.prediction_id or str(index),
                )
                self._validate_box(connection, image, box)
                try:
                    self._insert_box(connection, image_id, uuid4().hex, box)
                except sqlite3.IntegrityError as exc:
                    if "idx_boxes_prediction_idempotency" not in str(exc):
                        # SQLite normally reports column names, so verify whether
                        # this is simply an already imported prediction.
                        exists = connection.execute(
                            """
                            SELECT 1 FROM boxes
                            WHERE model_run_id = ? AND image_id = ? AND prediction_id = ?
                            """,
                            (run_id, image_id, box.prediction_id),
                        ).fetchone()
                        if exists is None:
                            raise
                    continue
                imported_count += 1

            origin = self._derive_origin(
                connection, image_id, empty_fallback=AnnotationOrigin.AI
            )
            connection.execute(
                """
                UPDATE images
                SET review_status = 'draft', origin = ?, ai_status = 'ready',
                    revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (origin.value, image.revision + 1, utc_now(), image_id),
            )
            connection.execute(
                """
                INSERT INTO ai_imports(run_id, image_id, box_count, imported_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, image_id, imported_count, utc_now()),
            )
        return AIPredictionImportResult(
            run_id=run_id,
            image_id=image_id,
            imported_count=imported_count,
            skipped_verified=False,
            revision_conflict=False,
        )

    def list_ai_imported_image_ids(self, run_id: str) -> tuple[str, ...]:
        rows = self.connection.execute(
            """
            SELECT image_id
            FROM ai_imports
            WHERE run_id = ?
            ORDER BY imported_at, image_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(str(row["image_id"]) for row in rows)

    # ------------------------------------------------------------------
    # Training selection and run records

    def training_preflight(self, *, minimum: int = 100) -> TrainingPreflight:
        if minimum < 1:
            raise ValueError("minimum 必须大于 0")
        categories = self.list_categories(enabled_only=True)
        verified_count = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM images
                WHERE review_status = 'verified' AND training_selected = 1
                """
            ).fetchone()[0]
        )
        counts = {category.id: 0 for category in categories}
        rows = self.connection.execute(
            """
            SELECT b.class_id, COUNT(*) AS count
            FROM boxes b
            JOIN images i ON i.id = b.image_id
            JOIN categories c ON c.id = b.class_id
            WHERE i.review_status = 'verified'
              AND i.training_selected = 1
              AND c.enabled = 1
            GROUP BY b.class_id
            """
        ).fetchall()
        for row in rows:
            counts[str(row["class_id"])] = int(row["count"])
        instance_count = sum(counts.values())
        positive_image_count = int(
            self.connection.execute(
                """
                SELECT COUNT(DISTINCT i.id)
                FROM images i
                JOIN boxes b ON b.image_id = i.id
                JOIN categories c ON c.id = b.class_id
                WHERE i.review_status = 'verified'
                  AND i.training_selected = 1
                  AND c.enabled = 1
                """
            ).fetchone()[0]
        )
        errors: list[str] = []
        warnings: list[str] = []
        if verified_count < minimum:
            errors.append(
                f"至少需要 {minimum} 张人工确认图片，当前为 {verified_count} 张"
            )
        if not categories:
            errors.append("至少需要启用一个类别")
        if instance_count == 0:
            errors.append("训练数据中没有任何正样本")
        missing = [category.name for category in categories if counts[category.id] == 0]
        if missing:
            errors.append(f"以下启用类别没有实例：{', '.join(missing)}")

        nonzero = [count for count in counts.values() if count > 0]
        if len(nonzero) >= 2 and max(nonzero) / min(nonzero) >= 10:
            warnings.append("类别实例数量相差超过 10 倍，可能存在明显类别失衡")
        return TrainingPreflight(
            minimum=minimum,
            verified_count=verified_count,
            positive_image_count=positive_image_count,
            negative_image_count=verified_count - positive_image_count,
            instance_count=instance_count,
            class_instance_counts=counts,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    preflight = training_preflight

    def create_run(
        self,
        kind: RunKind | str,
        model_key: ModelKey | str,
        *,
        parameters: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        run_id = run_id or uuid4().hex
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO model_runs (
                    id, kind, model_key, status, parameters_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 'created', ?, ?, ?)
                """,
                (
                    run_id,
                    RunKind(kind).value,
                    ModelKey(model_key).value,
                    _json(parameters or {}),
                    now,
                    now,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        row = self.connection.execute(
            "SELECT * FROM model_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"模型运行不存在：{run_id}")
        return _run_from_row(row)

    def list_runs(
        self,
        *,
        kind: RunKind | str | None = None,
        status: RunStatus | str | None = None,
    ) -> tuple[RunRecord, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(RunKind(kind).value)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(RunStatus(status).value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT * FROM model_runs
            {where}
            ORDER BY created_at DESC, id DESC
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | str | None = None,
        progress: float | None = None,
        snapshot_path: str | None = None,
        metrics_jsonl_path: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        checkpoint_path: str | None = None,
        error: str | None = None,
    ) -> RunRecord:
        current = self.get_run(run_id)
        new_status = current.status if status is None else RunStatus(status)
        new_progress = current.progress if progress is None else float(progress)
        if not 0.0 <= new_progress <= 1.0:
            raise ValueError("progress 必须位于 0 到 1")
        completed_at = (
            utc_now()
            if new_status
            in {
                RunStatus.COMPLETED,
                RunStatus.COMPLETED_WITH_ERRORS,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
            else current.completed_at
        )
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE model_runs
                SET status = ?, progress = ?, snapshot_path = ?,
                    metrics_jsonl_path = ?, metrics_json = ?, artifacts_json = ?,
                    checkpoint_path = ?, error = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    new_status.value,
                    new_progress,
                    current.snapshot_path if snapshot_path is None else snapshot_path,
                    (
                        current.metrics_jsonl_path
                        if metrics_jsonl_path is None
                        else metrics_jsonl_path
                    ),
                    _json(current.metrics if metrics is None else metrics),
                    _json(current.artifacts if artifacts is None else artifacts),
                    (
                        current.checkpoint_path
                        if checkpoint_path is None
                        else checkpoint_path
                    ),
                    current.error if error is None else error,
                    utc_now(),
                    completed_at,
                    run_id,
                ),
            )
        return self.get_run(run_id)

    # ------------------------------------------------------------------
    # Deployment packages

    def create_deployment_package(
        self,
        run_id: str,
        *,
        target: str,
        checkpoint_role: str,
        npu_mode: str,
        status: str = "created",
        model_package_path: str | None = None,
        app_package_path: str | None = None,
        report_path: str | None = None,
        zip_bytes: int | None = None,
        payload_bytes: int | None = None,
        warnings: Sequence[str] = (),
        package_id: str | None = None,
    ) -> DeploymentPackage:
        run = self.get_run(run_id)
        if run.kind != RunKind.DEPLOY:
            raise DataIntegrityError("部署包只能关联 deploy 类型运行")
        target = _required_text(target, "target")
        checkpoint_role = _required_text(checkpoint_role, "checkpoint_role")
        npu_mode = _required_text(npu_mode, "npu_mode")
        status = _required_text(status, "status")
        zip_bytes = _optional_nonnegative_integer(zip_bytes, "zip_bytes")
        payload_bytes = _optional_nonnegative_integer(payload_bytes, "payload_bytes")
        package_id = package_id or uuid4().hex
        warning_values = tuple(str(value) for value in warnings)
        now = utc_now()
        with self.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO deployment_packages (
                        id, run_id, target, checkpoint_role, npu_mode, status,
                        model_package_path, app_package_path, report_path,
                        zip_bytes, payload_bytes, warnings_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        package_id,
                        run_id,
                        target,
                        checkpoint_role,
                        npu_mode,
                        status,
                        model_package_path,
                        app_package_path,
                        report_path,
                        zip_bytes,
                        payload_bytes,
                        json.dumps(warning_values, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DataIntegrityError("部署包 ID 重复或引用的运行不存在") from exc
        return self.get_deployment_package(package_id)

    def get_deployment_package(self, package_id: str) -> DeploymentPackage:
        row = self.connection.execute(
            "SELECT * FROM deployment_packages WHERE id = ?", (package_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"部署包不存在：{package_id}")
        return _deployment_package_from_row(row)

    def list_deployment_packages(
        self,
        *,
        run_id: str | None = None,
        target: str | None = None,
        status: str | None = None,
    ) -> tuple[DeploymentPackage, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        for column, value in (
            ("run_id", run_id),
            ("target", target),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(str(value))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT * FROM deployment_packages
            {where}
            ORDER BY created_at DESC, id DESC
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(_deployment_package_from_row(row) for row in rows)

    def update_deployment_package(
        self,
        package_id: str,
        *,
        status: str | None = None,
        model_package_path: str | None = None,
        app_package_path: str | None = None,
        report_path: str | None = None,
        zip_bytes: int | None = None,
        payload_bytes: int | None = None,
        warnings: Sequence[str] | None = None,
    ) -> DeploymentPackage:
        current = self.get_deployment_package(package_id)
        new_status = (
            current.status if status is None else _required_text(status, "status")
        )
        new_zip_bytes = (
            current.zip_bytes
            if zip_bytes is None
            else _optional_nonnegative_integer(zip_bytes, "zip_bytes")
        )
        new_payload_bytes = (
            current.payload_bytes
            if payload_bytes is None
            else _optional_nonnegative_integer(payload_bytes, "payload_bytes")
        )
        new_warnings = (
            current.warnings
            if warnings is None
            else tuple(str(value) for value in warnings)
        )
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE deployment_packages
                SET status = ?, model_package_path = ?, app_package_path = ?,
                    report_path = ?, zip_bytes = ?, payload_bytes = ?,
                    warnings_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_status,
                    (
                        current.model_package_path
                        if model_package_path is None
                        else model_package_path
                    ),
                    (
                        current.app_package_path
                        if app_package_path is None
                        else app_package_path
                    ),
                    current.report_path if report_path is None else report_path,
                    new_zip_bytes,
                    new_payload_bytes,
                    json.dumps(new_warnings, ensure_ascii=False),
                    utc_now(),
                    package_id,
                ),
            )
        return self.get_deployment_package(package_id)

    def delete_deployment_package(self, package_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM deployment_packages WHERE id = ?", (package_id,)
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError(f"部署包不存在：{package_id}")

    # ------------------------------------------------------------------
    # Internals

    def _checked_image(
        self,
        connection: sqlite3.Connection,
        image_id: str,
        expected_revision: int | None,
    ) -> ImageRecord:
        row = connection.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"图片不存在：{image_id}")
        image = _image_from_row(row)
        if expected_revision is not None and image.revision != expected_revision:
            raise RevisionConflictError(image_id, expected_revision, image.revision)
        return image

    def _validate_box(
        self,
        connection: sqlite3.Connection,
        image: ImageRecord,
        box: BoxInput,
    ) -> None:
        validate_box_bounds(box, image.width, image.height)
        exists = connection.execute(
            "SELECT 1 FROM categories WHERE id = ?", (box.class_id,)
        ).fetchone()
        if exists is None:
            raise DataIntegrityError(f"标注框引用了不存在的类别：{box.class_id}")
        if box.origin == AnnotationOrigin.NONE:
            raise DataIntegrityError("标注框 origin 不能为 none")

    def _insert_box(
        self,
        connection: sqlite3.Connection,
        image_id: str,
        box_id: str,
        box: BoxInput,
    ) -> None:
        now = utc_now()
        connection.execute(
            """
            INSERT INTO boxes (
                id, image_id, class_id, x1, y1, x2, y2, origin,
                confidence, model_run_id, prediction_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                box_id,
                image_id,
                box.class_id,
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                box.origin.value,
                box.confidence,
                box.model_run_id,
                box.prediction_id,
                now,
                now,
            ),
        )

    def _replace_boxes(
        self,
        connection: sqlite3.Connection,
        image: ImageRecord,
        boxes: Sequence[BoxInput],
    ) -> None:
        ids: set[str] = set()
        for box in boxes:
            self._validate_box(connection, image, box)
            if box.id is not None and box.id in ids:
                raise DataIntegrityError(f"重复的标注框 ID：{box.id}")
            if box.id is not None:
                ids.add(box.id)
        connection.execute("DELETE FROM boxes WHERE image_id = ?", (image.id,))
        for box in boxes:
            value = box if box.origin != AnnotationOrigin.NONE else _manual_box(box)
            self._insert_box(connection, image.id, value.id or uuid4().hex, value)

    def _mark_annotation_change(
        self,
        connection: sqlite3.Connection,
        image: ImageRecord,
        *,
        human_edit: bool,
    ) -> None:
        status = (
            ReviewStatus.DRAFT
            if image.review_status == ReviewStatus.DRAFT
            else ReviewStatus.UNREVIEWED
        )
        origin = self._derive_origin(
            connection,
            image.id,
            empty_fallback=(AnnotationOrigin.MANUAL if human_edit else image.origin),
        )
        connection.execute(
            """
            UPDATE images
            SET review_status = ?, origin = ?, revision = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                origin.value,
                image.revision + 1,
                utc_now(),
                image.id,
            ),
        )

    @staticmethod
    def _derive_origin(
        connection: sqlite3.Connection,
        image_id: str,
        *,
        empty_fallback: AnnotationOrigin,
    ) -> AnnotationOrigin:
        values = {
            AnnotationOrigin(str(row[0]))
            for row in connection.execute(
                "SELECT DISTINCT origin FROM boxes WHERE image_id = ?", (image_id,)
            ).fetchall()
        }
        if not values:
            return empty_fallback
        if values == {AnnotationOrigin.MANUAL}:
            return AnnotationOrigin.MANUAL
        if values == {AnnotationOrigin.AI}:
            return AnnotationOrigin.AI
        return AnnotationOrigin.MIXED


def _category_from_row(row: sqlite3.Row) -> Category:
    return Category(
        id=str(row["id"]),
        name=str(row["name"]),
        color=str(row["color"]),
        position=int(row["position"]),
        enabled=bool(row["enabled"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        display_name=(
            None
            if "display_name" not in set(row.keys()) or row["display_name"] is None
            else str(row["display_name"])
        ),
    )


def _image_from_row(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        id=str(row["id"]),
        relative_path=str(row["relative_path"]),
        original_name=str(row["original_name"]),
        source_path=None if row["source_path"] is None else str(row["source_path"]),
        sha256=str(row["sha256"]),
        width=int(row["width"]),
        height=int(row["height"]),
        review_status=ReviewStatus(row["review_status"]),
        origin=AnnotationOrigin(row["origin"]),
        ai_status=AIStatus(row["ai_status"]),
        revision=int(row["revision"]),
        imported_at=str(row["imported_at"]),
        updated_at=str(row["updated_at"]),
        training_selected=bool(row["training_selected"]),
    )


def _box_from_row(row: sqlite3.Row) -> BoundingBox:
    return BoundingBox(
        id=str(row["id"]),
        image_id=str(row["image_id"]),
        class_id=str(row["class_id"]),
        x1=float(row["x1"]),
        y1=float(row["y1"]),
        x2=float(row["x2"]),
        y2=float(row["y2"]),
        origin=AnnotationOrigin(row["origin"]),
        confidence=None if row["confidence"] is None else float(row["confidence"]),
        model_run_id=(
            None if row["model_run_id"] is None else str(row["model_run_id"])
        ),
        prediction_id=(
            None if row["prediction_id"] is None else str(row["prediction_id"])
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=str(row["id"]),
        kind=RunKind(row["kind"]),
        model_key=ModelKey(row["model_key"]),
        status=RunStatus(row["status"]),
        parameters=_load_json(row["parameters_json"]),
        snapshot_path=(
            None if row["snapshot_path"] is None else str(row["snapshot_path"])
        ),
        metrics_jsonl_path=(
            None
            if row["metrics_jsonl_path"] is None
            else str(row["metrics_jsonl_path"])
        ),
        metrics=_load_json(row["metrics_json"]),
        artifacts=_load_json(row["artifacts_json"]),
        checkpoint_path=(
            None if row["checkpoint_path"] is None else str(row["checkpoint_path"])
        ),
        progress=float(row["progress"]),
        error=None if row["error"] is None else str(row["error"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=(
            None if row["completed_at"] is None else str(row["completed_at"])
        ),
    )


def _deployment_package_from_row(row: sqlite3.Row) -> DeploymentPackage:
    try:
        raw_warnings = json.loads(row["warnings_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("部署包 warnings_json 无效") from exc
    if not isinstance(raw_warnings, list) or not all(
        isinstance(value, str) for value in raw_warnings
    ):
        raise DataIntegrityError("部署包 warnings_json 必须是字符串数组")
    return DeploymentPackage(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        target=str(row["target"]),
        checkpoint_role=str(row["checkpoint_role"]),
        npu_mode=str(row["npu_mode"]),
        status=str(row["status"]),
        model_package_path=(
            None
            if row["model_package_path"] is None
            else str(row["model_package_path"])
        ),
        app_package_path=(
            None if row["app_package_path"] is None else str(row["app_package_path"])
        ),
        report_path=None if row["report_path"] is None else str(row["report_path"]),
        zip_bytes=None if row["zip_bytes"] is None else int(row["zip_bytes"]),
        payload_bytes=(
            None if row["payload_bytes"] is None else int(row["payload_bytes"])
        ),
        warnings=tuple(raw_warnings),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _normalize_image_ids(image_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(image_ids, str):
        image_ids = (image_ids,)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in image_ids:
        image_id = str(value).strip()
        if not image_id:
            raise ValueError("图片 ID 不能为空")
        if image_id not in seen:
            normalized.append(image_id)
            seen.add(image_id)
    if not normalized:
        raise ValueError("至少选择一张图片")
    return tuple(normalized)


def _deduplicate_ai_predictions(
    predictions: Sequence[AIPrediction],
    iou_threshold: float,
) -> tuple[AIPrediction, ...]:
    """Apply a conservative same-class NMS pass to incoming AI predictions."""

    ranked = sorted(
        enumerate(predictions),
        key=lambda item: (
            -(
                float(item[1].confidence)
                if item[1].confidence is not None
                else -1.0
            ),
            item[0],
        ),
    )
    kept: list[tuple[int, AIPrediction]] = []
    for original_index, candidate in ranked:
        candidate_class = _prediction_class_key(candidate)
        duplicate = any(
            candidate_class == _prediction_class_key(existing)
            and _prediction_iou(candidate, existing) >= iou_threshold
            for _index, existing in kept
        )
        if not duplicate:
            kept.append((original_index, candidate))
    kept.sort(key=lambda item: item[0])
    return tuple(value for _index, value in kept)


def _prediction_class_key(prediction: AIPrediction) -> tuple[str, str | int | None]:
    if prediction.class_id is not None:
        return ("id", prediction.class_id)
    return ("index", prediction.class_index)


def _prediction_iou(left: AIPrediction, right: AIPrediction) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _validate_category_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("类别名称不能为空")
    if any(char in name for char in ",，、") or any(
        unicodedata.category(char) == "Cc" for char in name
    ):
        raise ValueError("类别名称不能包含控制字符、换行或类别分隔符")
    if len(name) > 128:
        raise ValueError("类别名称不能超过 128 个字符")
    return name


def _validate_display_name(name: object | None) -> str | None:
    if name is None:
        return None
    value = str(name).strip()
    if not value:
        return None
    if any(char in value for char in ",，、") or any(
        unicodedata.category(char) == "Cc" for char in value
    ):
        raise ValueError("显示名称不能包含控制字符、换行或类别分隔符")
    if len(value) > 128:
        raise ValueError("显示名称不能超过 128 个字符")
    return value


def _assert_category_label_unique(
    connection: sqlite3.Connection,
    candidate: str,
    *,
    exclude_category_id: str | None = None,
) -> None:
    normalized = candidate.casefold()
    rows = connection.execute("SELECT id, name, display_name FROM categories").fetchall()
    for row in rows:
        if exclude_category_id is not None and str(row["id"]) == exclude_category_id:
            continue
        for raw in (row["name"], row["display_name"]):
            if raw is not None and str(raw).strip().casefold() == normalized:
                raise DataIntegrityError(f"类别名称或显示名称已存在：{candidate}")
    aliases = connection.execute(
        "SELECT alias, category_id FROM category_name_aliases"
    ).fetchall()
    for row in aliases:
        if exclude_category_id is not None and str(row["category_id"]) == exclude_category_id:
            continue
        if str(row["alias"]).strip().casefold() == normalized:
            raise DataIntegrityError(f"类别历史名称已存在：{candidate}")


def _validate_color(color: str) -> str:
    value = str(color).strip().upper()
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError("颜色必须使用 #RRGGBB 格式")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError("颜色必须使用 #RRGGBB 格式") from exc
    return value


def _required_text(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} 不能为空")
    return value


def _optional_nonnegative_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


def _manual_box(box: BoxInput) -> BoxInput:
    return BoxInput(
        id=box.id,
        class_id=box.class_id,
        x1=box.x1,
        y1=box.y1,
        x2=box.x2,
        y2=box.y2,
        origin=AnnotationOrigin.MANUAL,
    )


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(value: str) -> Mapping[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise DataIntegrityError("数据库 JSON 字段必须是对象")
    return loaded


# A short alias for consumers that do not need to spell out the storage type.
Repository = AnnotationRepository
