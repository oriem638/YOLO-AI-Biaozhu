"""Creation and opening of self-contained annotation projects."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from ai_biaozhu.core.domain import (
    PROJECT_SCHEMA_VERSION,
    BoundingBox,
    BoxInput,
    Category,
    ImageRecord,
    ImportReport,
    ProjectConfig,
    TrainingPreflight,
)
from ai_biaozhu.core.exceptions import (
    DataIntegrityError,
    ProjectExistsError,
    ProjectFormatError,
)

from .image_import import ImageImporter
from .repository import (
    AIDedupPreview,
    AnnotationRepository,
    BulkAnnotationClearPreview,
)
from .utils import utc_now, write_json

PROJECT_FILE = "project.json"
BACKUPS_DIRECTORY = "backups"


@dataclass(frozen=True, slots=True)
class AnnotationDatabaseBackup:
    path: Path
    created_at: str
    reason: str
    size_bytes: int
    project_id: str | None
    image_count: int | None
    box_count: int | None
    valid: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BulkAnnotationClearReport:
    preview: BulkAnnotationClearPreview
    backup: AnnotationDatabaseBackup
    cleared_at: str

    @property
    def image_count(self) -> int:
        return self.preview.image_count

    @property
    def box_count(self) -> int:
        return self.preview.box_count

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self.preview.image_ids


@dataclass(frozen=True, slots=True)
class AnnotationDatabaseRestoreReport:
    restored_backup: AnnotationDatabaseBackup
    safety_backup: AnnotationDatabaseBackup
    restored_at: str


@dataclass(frozen=True, slots=True)
class AIDeduplicationReport:
    preview: AIDedupPreview
    backup: AnnotationDatabaseBackup | None
    completed_at: str


@dataclass(frozen=True, slots=True)
class BackupCleanupPreview:
    backups: tuple[AnnotationDatabaseBackup, ...]
    total_bytes: int
    keep_latest: int

    @property
    def backup_count(self) -> int:
        return len(self.backups)


@dataclass(frozen=True, slots=True)
class BackupCleanupReport:
    preview: BackupCleanupPreview
    recovery_directory: Path
    moved_paths: tuple[Path, ...]
    completed_at: str
    deleted_paths: tuple[Path, ...] = ()
    permanently_deleted: bool = False


@dataclass(frozen=True, slots=True)
class BulkImageDeleteReport:
    image_ids: tuple[str, ...]
    box_count: int
    backup: AnnotationDatabaseBackup
    archive_path: Path
    manifest_path: Path
    deleted_at: str
    warnings: tuple[str, ...] = ()

    @property
    def image_count(self) -> int:
        return len(self.image_ids)


class AnnotationProject:
    def __init__(
        self,
        root: Path,
        config: ProjectConfig,
        repository: AnnotationRepository,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.repository = repository
        self.database_path = _member(self.root, config.database)
        self.images_dir = _member(self.root, config.images_dir)
        self.runs_dir = _member(self.root, config.runs_dir)
        self.exports_dir = _member(self.root, config.exports_dir)
        self.deployments_dir = _member(self.root, config.deployments_dir)
        self.thumbnails_dir = _member(self.root, config.thumbnails_dir)
        self.backups_dir = self.root / BACKUPS_DIRECTORY

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        name: str,
        categories: Sequence[str | Mapping[str, Any]] = (),
    ) -> AnnotationProject:
        root = Path(root).resolve()
        if root.exists():
            if not root.is_dir():
                raise ProjectExistsError(f"目标路径不是目录：{root}")
            if any(root.iterdir()):
                raise ProjectExistsError(f"目标目录不是空目录：{root}")
        else:
            root.mkdir(parents=True)

        now = utc_now()
        config = ProjectConfig(
            project_id=uuid4().hex,
            name=name.strip(),
            created_at=now,
            updated_at=now,
        )
        for directory in (
            config.images_dir,
            config.runs_dir,
            config.exports_dir,
            config.deployments_dir,
            config.thumbnails_dir,
        ):
            _member(root, directory).mkdir(parents=True, exist_ok=False)

        config_path = root / PROJECT_FILE
        temporary_config = root / f".{PROJECT_FILE}.{uuid4().hex}.tmp"
        write_json(temporary_config, config.to_dict())
        temporary_config.replace(config_path)

        repository = AnnotationRepository(_member(root, config.database))
        try:
            repository.connection.execute(
                """
                INSERT OR REPLACE INTO project_meta(key, value)
                VALUES ('project_id', ?)
                """,
                (config.project_id,),
            )
            for item in categories:
                if isinstance(item, str):
                    repository.add_category(item)
                else:
                    repository.add_category(
                        str(item["name"]),
                        display_name=(
                            None
                            if item.get("display_name") is None
                            else str(item.get("display_name"))
                        ),
                        color=str(item.get("color", "#22C55E")),
                        enabled=bool(item.get("enabled", True)),
                        category_id=(
                            None if item.get("id") is None else str(item.get("id"))
                        ),
                        position=(
                            None
                            if item.get("position") is None
                            else int(item["position"])
                        ),
                    )
        except Exception:
            repository.close()
            raise
        return cls(root, config, repository)

    @classmethod
    def open(cls, root: Path | str) -> AnnotationProject:
        root = Path(root).resolve()
        config_path = root / PROJECT_FILE
        if not config_path.is_file():
            raise ProjectFormatError(f"找不到项目文件：{config_path}")
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("project.json 根节点必须是对象")
            legacy_without_deployments = "deployments_dir" not in raw
            stored_schema_version = int(raw.get("schema_version", 1))
            config = ProjectConfig.from_dict(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProjectFormatError(f"无法读取 project.json：{exc}") from exc

        deployments_dir = _member(root, config.deployments_dir)
        if legacy_without_deployments and not deployments_dir.exists():
            deployments_dir.mkdir(parents=True)
        required_directories = (
            _member(root, config.images_dir),
            _member(root, config.runs_dir),
            _member(root, config.exports_dir),
            deployments_dir,
            _member(root, config.thumbnails_dir),
        )
        missing = [str(path) for path in required_directories if not path.is_dir()]
        if missing:
            raise ProjectFormatError(f"项目目录不完整：{', '.join(missing)}")
        database_path = _member(root, config.database)
        if not database_path.is_file():
            raise ProjectFormatError(f"项目数据库不存在：{database_path}")

        if stored_schema_version < PROJECT_SCHEMA_VERSION:
            _backup_database_before_schema_upgrade(
                root,
                database_path,
                project_id=config.project_id,
                source_version=stored_schema_version,
                target_version=PROJECT_SCHEMA_VERSION,
            )
        repository = AnnotationRepository(database_path)
        row = repository.connection.execute(
            "SELECT value FROM project_meta WHERE key = 'project_id'"
        ).fetchone()
        if row is None or str(row[0]) != config.project_id:
            repository.close()
            raise ProjectFormatError("project.json 与 annotations.db 的项目 ID 不一致")
        if stored_schema_version < config.schema_version:
            upgraded = config.to_dict()
            upgraded["updated_at"] = utc_now()
            temporary = root / f".{PROJECT_FILE}.{uuid4().hex}.tmp"
            write_json(temporary, upgraded)
            temporary.replace(config_path)
            config = ProjectConfig.from_dict(upgraded)
        return cls(root, config, repository)

    def close(self) -> None:
        self.repository.close()

    def __enter__(self) -> AnnotationProject:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def image_path(self, image: ImageRecord | str) -> Path:
        record = self.repository.get_image(image) if isinstance(image, str) else image
        path = _member(self.root, record.relative_path)
        if not path.is_file():
            raise ProjectFormatError(f"项目图片文件不存在：{path}")
        return path

    def import_images(
        self,
        paths: Iterable[Path | str],
        *,
        recursive: bool = True,
    ) -> ImportReport:
        importer = ImageImporter(
            self.repository,
            project_root=self.root,
            images_dir=self.images_dir,
            reports_dir=self.exports_dir / "import-reports",
        )
        return importer.import_paths(paths, recursive=recursive)

    def list_images(self, **filters: Any) -> tuple[ImageRecord, ...]:
        return self.repository.list_images(**filters)

    def list_boxes(self, image_id: str) -> tuple[BoundingBox, ...]:
        return self.repository.list_boxes(image_id)

    def list_runs(self, **filters: Any):
        return self.repository.list_runs(**filters)

    def list_deployment_packages(self, **filters: Any):
        return self.repository.list_deployment_packages(**filters)

    def create_deployment_package(self, run_id: str, **values: Any):
        return self.repository.create_deployment_package(run_id, **values)

    def list_ai_imported_image_ids(self, run_id: str) -> tuple[str, ...]:
        return self.repository.list_ai_imported_image_ids(run_id)

    def save_boxes(
        self,
        image_id: str,
        boxes: Iterable[BoxInput | Mapping[str, Any]],
        *,
        expected_revision: int | None = None,
    ) -> tuple[BoundingBox, ...]:
        return self.repository.replace_boxes(
            image_id, boxes, expected_revision=expected_revision
        )

    def restore_annotation_session_baseline(
        self,
        image_id: str,
        boxes: Iterable[BoxInput | Mapping[str, Any]],
        *,
        review_status: str,
        origin: str,
        ai_status: str,
        expected_revision: int | None = None,
    ) -> ImageRecord:
        """Restore one image to the immutable baseline captured on opening."""

        return self.repository.restore_annotation_session_baseline(
            image_id,
            boxes,
            review_status=review_status,
            origin=origin,
            ai_status=ai_status,
            expected_revision=expected_revision,
        )

    def preview_clear_all_annotations(
        self, image_ids: Iterable[str]
    ) -> BulkAnnotationClearPreview:
        return self.repository.preview_clear_all_annotations(image_ids)

    def clear_all_annotations(
        self, image_ids: Iterable[str]
    ) -> BulkAnnotationClearReport:
        """Back up, then atomically clear boxes from the selected images."""

        preview = self.preview_clear_all_annotations(image_ids)
        backup = self._create_annotation_backup("clear-all-annotations")
        expected_revisions = dict(preview.image_revisions)
        before = self.repository.clear_all_annotations(
            preview.image_ids,
            expected_revisions=expected_revisions,
        )
        return BulkAnnotationClearReport(
            preview=before,
            backup=backup,
            cleared_at=utc_now(),
        )

    def preview_delete_images(self, image_ids: Iterable[str]) -> dict[str, Any]:
        normalized = tuple(dict.fromkeys(str(value) for value in image_ids))
        if not normalized:
            raise ValueError("至少选择一张图片")
        records = tuple(self.repository.get_image(image_id) for image_id in normalized)
        return {
            "image_ids": normalized,
            "image_count": len(records),
            "box_count": sum(
                len(self.repository.list_boxes(record.id)) for record in records
            ),
            "verified_count": sum(
                record.review_status.value == "verified" for record in records
            ),
        }

    def delete_images(self, image_ids: Iterable[str]) -> BulkImageDeleteReport:
        """Recoverably remove selected samples from the project.

        A consistent database backup and a copy of every selected image are
        created before the live rows/files are removed.
        """

        preview = self.preview_delete_images(image_ids)
        normalized = tuple(str(value) for value in preview["image_ids"])
        records = tuple(self.repository.get_image(image_id) for image_id in normalized)
        backup = self._create_annotation_backup("delete-images")
        created_at = utc_now()
        stamp = (
            created_at.replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("Z", "Z")
        )
        archive = self.backups_dir / (
            f"deleted-images-{stamp}-{uuid4().hex[:8]}"
        )
        archive.mkdir(parents=True, exist_ok=False)
        copied: list[dict[str, Any]] = []
        for record in records:
            source = self.image_path(record)
            relative = PurePosixPath(record.relative_path.replace("\\", "/"))
            target = archive.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(
                {
                    "id": record.id,
                    "relative_path": record.relative_path,
                    "original_name": record.original_name,
                    "sha256": record.sha256,
                }
            )
        manifest = archive / "manifest.json"
        write_json(
            manifest,
            {
                "format": "ai-biaozhu-deleted-images-backup",
                "created_at": created_at,
                "database_backup": str(backup.path),
                "images": copied,
            },
        )
        self.repository.delete_images(normalized)
        warnings: list[str] = []
        for record in records:
            source = _member(self.root, record.relative_path)
            try:
                source.unlink(missing_ok=True)
            except OSError as exc:
                warnings.append(f"原图片文件未能移除：{source}（{exc}）")
        return BulkImageDeleteReport(
            image_ids=normalized,
            box_count=int(preview["box_count"]),
            backup=backup,
            archive_path=archive,
            manifest_path=manifest,
            deleted_at=created_at,
            warnings=tuple(warnings),
        )

    def set_training_selected(
        self,
        image_ids: Iterable[str],
        selected: bool,
    ) -> tuple[ImageRecord, ...]:
        return self.repository.set_training_selected(image_ids, selected)

    def select_only_for_training(
        self,
        image_ids: Iterable[str],
    ) -> tuple[ImageRecord, ...]:
        return self.repository.select_only_for_training(image_ids)

    def preview_ai_deduplication(
        self,
        image_ids: Iterable[str] | None = None,
        *,
        iou_threshold: float = 0.80,
    ) -> AIDedupPreview:
        return self.repository.preview_ai_deduplication(
            image_ids,
            iou_threshold=iou_threshold,
        )

    def deduplicate_ai_drafts(
        self,
        image_ids: Iterable[str] | None = None,
        *,
        iou_threshold: float = 0.80,
    ) -> AIDeduplicationReport:
        preview = self.preview_ai_deduplication(
            image_ids,
            iou_threshold=iou_threshold,
        )
        if not preview.removals:
            return AIDeduplicationReport(
                preview=preview,
                backup=None,
                completed_at=utc_now(),
            )
        backup = self._create_annotation_backup("before-ai-dedup")
        applied = self.repository.deduplicate_ai_drafts(
            preview.requested_image_ids,
            iou_threshold=preview.iou_threshold,
        )
        return AIDeduplicationReport(
            preview=applied,
            backup=backup,
            completed_at=utc_now(),
        )

    # UI/controller-friendly aliases.
    preview_delete_all_annotations = preview_clear_all_annotations
    delete_all_annotations = clear_all_annotations

    def list_annotation_backups(self) -> tuple[AnnotationDatabaseBackup, ...]:
        if not self.backups_dir.is_dir():
            return ()
        backups: list[AnnotationDatabaseBackup] = []
        root = self.backups_dir.resolve()
        for path in self.backups_dir.glob("*.db"):
            resolved = path.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            backups.append(self._describe_annotation_backup(resolved))
        return tuple(
            sorted(
                backups,
                key=lambda value: (value.created_at, value.path.name),
                reverse=True,
            )
        )

    def preview_backup_cleanup(
        self,
        *,
        keep_latest: int = 3,
        include_recovery_trash: bool = False,
    ) -> BackupCleanupPreview:
        if isinstance(keep_latest, bool) or keep_latest < 0:
            raise ValueError("keep_latest 必须是非负整数")
        backups = list(self.list_annotation_backups())
        if include_recovery_trash:
            trash_root = (self.backups_dir / ".trash").resolve()
            backups.extend(
                self._describe_annotation_backup(path)
                for path in self._safe_recovery_backup_paths(trash_root)
            )
            backups.sort(
                key=lambda value: (value.created_at, value.path.name),
                reverse=True,
            )
        candidates = tuple(backups[keep_latest:])
        return BackupCleanupPreview(
            backups=candidates,
            total_bytes=sum(item.size_bytes for item in candidates),
            keep_latest=keep_latest,
        )

    def cleanup_old_backups(
        self,
        *,
        keep_latest: int = 3,
        deployment_verified: bool = False,
        permanently_delete: bool = False,
    ) -> BackupCleanupReport:
        """Clean old backups after explicit device verification.

        The safe default moves candidates into project-local recovery trash.
        ``permanently_delete`` is reserved for the separately confirmed UI
        action required by the maintenance specification; it also includes
        backups from previous recovery-trash cleanups so disk space is really
        released instead of merely moving the accumulation.
        """

        if not deployment_verified:
            raise PermissionError("必须先明确确认部署已在设备上验证成功")
        preview = self.preview_backup_cleanup(
            keep_latest=keep_latest,
            include_recovery_trash=permanently_delete,
        )
        stamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
        recovery = (self.backups_dir / ".trash" / stamp).resolve()
        root = self.backups_dir.resolve()
        recovery.relative_to(root)
        moved: list[Path] = []
        active_backups = tuple(
            backup
            for backup in preview.backups
            if backup.path.resolve().parent == root
        )
        if active_backups:
            recovery.mkdir(parents=True, exist_ok=False)
        for backup in preview.backups:
            source = self._resolve_annotation_backup_path(backup.path)
            auxiliary_sources = tuple(
                source.with_name(source.name + suffix)
                for suffix in ("-wal", "-shm", "-journal")
            )
            if source.parent == root:
                target = recovery / source.name
                source.replace(target)
            else:
                target = source
            moved.append(target)
            metadata = source.with_suffix(".json")
            if metadata.is_file():
                if source.parent == root:
                    metadata_target = recovery / metadata.name
                    metadata.replace(metadata_target)
                else:
                    metadata_target = metadata
                moved.append(metadata_target)
            for auxiliary in auxiliary_sources:
                if not auxiliary.is_file() or auxiliary.is_symlink():
                    continue
                if source.parent == root:
                    auxiliary_target = recovery / auxiliary.name
                    auxiliary.replace(auxiliary_target)
                else:
                    auxiliary_target = auxiliary
                moved.append(auxiliary_target)
        deleted: list[Path] = []
        if permanently_delete:
            trash_root = (self.backups_dir / ".trash").resolve()
            for path in moved:
                resolved = path.resolve()
                resolved.relative_to(trash_root)
                if resolved.is_symlink() or not resolved.is_file():
                    raise ProjectFormatError(f"拒绝永久删除不安全的备份路径：{resolved}")
                resolved.unlink()
                deleted.append(resolved)
            if trash_root.is_dir():
                directories = sorted(
                    (path for path in trash_root.rglob("*") if path.is_dir()),
                    key=lambda path: len(path.parts),
                    reverse=True,
                )
                for directory in directories:
                    if not directory.is_symlink():
                        with suppress(OSError):
                            directory.rmdir()
                with suppress(OSError):
                    trash_root.rmdir()
        return BackupCleanupReport(
            preview=preview,
            recovery_directory=recovery,
            moved_paths=tuple(moved),
            completed_at=utc_now(),
            deleted_paths=tuple(deleted),
            permanently_deleted=permanently_delete,
        )

    def _safe_recovery_backup_paths(self, trash_root: Path) -> tuple[Path, ...]:
        if not trash_root.is_dir() or trash_root.is_symlink():
            return ()
        backups: list[Path] = []
        for path in trash_root.glob("*/*.db"):
            resolved = path.resolve()
            try:
                resolved.relative_to(trash_root)
            except ValueError:
                continue
            if (
                path.is_symlink()
                or not resolved.is_file()
                or resolved.parent.parent != trash_root
            ):
                continue
            backups.append(resolved)
        return tuple(backups)

    def restore_annotation_backup(
        self, backup: AnnotationDatabaseBackup | Path | str
    ) -> AnnotationDatabaseRestoreReport:
        source = self._resolve_annotation_backup_path(
            backup.path if isinstance(backup, AnnotationDatabaseBackup) else backup
        )
        restored = self._describe_annotation_backup(source)
        if not restored.valid:
            raise ProjectFormatError(
                restored.error or f"标注数据库备份无效：{source}"
            )
        safety = self._create_annotation_backup("before-restore")
        try:
            self.repository.restore_database(
                source,
                expected_project_id=self.config.project_id,
            )
        except Exception as restore_error:
            try:
                self.repository.restore_database(
                    safety.path,
                    expected_project_id=self.config.project_id,
                )
            except Exception as recovery_error:
                raise ProjectFormatError(
                    "恢复标注备份失败，而且自动恢复安全备份也失败："
                    f"{restore_error}; {recovery_error}"
                ) from restore_error
            raise
        return AnnotationDatabaseRestoreReport(
            restored_backup=restored,
            safety_backup=safety,
            restored_at=utc_now(),
        )

    def _create_annotation_backup(
        self, reason: str
    ) -> AnnotationDatabaseBackup:
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        created_at = utc_now()
        stamp = (
            created_at.replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("Z", "Z")
        )
        safe_reason = "".join(
            char if char.isalnum() or char in {"-", "_"} else "-"
            for char in str(reason).strip().casefold()
        ).strip("-") or "manual"
        filename = f"annotations-{stamp}-{safe_reason}-{uuid4().hex[:8]}.db"
        final_path = self.backups_dir / filename
        partial_path = self.backups_dir / f".{filename}.partial"
        temporary_metadata: Path | None = None
        try:
            self.repository.backup_database(partial_path)
            partial_path.replace(final_path)
            backup = self._describe_annotation_backup(
                final_path,
                created_at=created_at,
                reason=safe_reason,
            )
            if not backup.valid:
                raise ProjectFormatError(
                    backup.error or "刚创建的标注数据库备份无效"
                )
            metadata_path = final_path.with_suffix(".json")
            temporary_metadata = metadata_path.with_name(
                f".{metadata_path.name}.{uuid4().hex}.tmp"
            )
            write_json(
                temporary_metadata,
                {
                    "format": "ai-biaozhu-annotation-backup",
                    "database": final_path.name,
                    "project_id": backup.project_id,
                    "created_at": backup.created_at,
                    "reason": backup.reason,
                    "size_bytes": backup.size_bytes,
                    "image_count": backup.image_count,
                    "box_count": backup.box_count,
                },
            )
            temporary_metadata.replace(metadata_path)
            return backup
        except Exception:
            partial_path.unlink(missing_ok=True)
            if temporary_metadata is not None:
                temporary_metadata.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            final_path.with_suffix(".json").unlink(missing_ok=True)
            raise

    def rename_category_canonical(
        self,
        category_id: str,
        name: str,
    ) -> tuple[Category, AnnotationDatabaseBackup]:
        """Back up the database, then rename one stable category atomically."""

        backup = self._create_annotation_backup("before-category-rename")
        category = self.repository.rename_category_canonical(category_id, name)
        return category, backup

    def delete_empty_category(
        self,
        category_id: str,
    ) -> tuple[Category, AnnotationDatabaseBackup]:
        """Delete an unused category after creating a restorable DB backup."""

        category = self.repository.get_category(category_id)
        box_count = self.repository.category_box_count(category_id)
        if box_count:
            raise DataIntegrityError(
                f"类别“{category.name}”仍有 {box_count} 个标注框，不能删除。"
            )
        if len(self.repository.list_categories()) <= 1:
            raise DataIntegrityError("项目必须至少保留一个类别，不能删除最后一个类别。")
        backup = self._create_annotation_backup("before-empty-category-delete")
        self.repository.delete_category(category_id)
        return category, backup

    def _describe_annotation_backup(
        self,
        path: Path,
        *,
        created_at: str | None = None,
        reason: str | None = None,
    ) -> AnnotationDatabaseBackup:
        metadata: dict[str, Any] = {}
        metadata_path = path.with_suffix(".json")
        if metadata_path.is_file():
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("database") == path.name:
                    metadata = raw
            except (OSError, json.JSONDecodeError):
                metadata = {}
        created_at = str(
            created_at
            or metadata.get("created_at")
            or _file_timestamp(path)
        )
        reason = str(reason or metadata.get("reason") or "unknown")
        try:
            project_id, image_count, box_count = (
                self.repository.validate_database_backup(
                    path,
                    expected_project_id=self.config.project_id,
                )
            )
        except Exception as exc:
            return AnnotationDatabaseBackup(
                path=path,
                created_at=created_at,
                reason=reason,
                size_bytes=path.stat().st_size if path.is_file() else 0,
                project_id=None,
                image_count=None,
                box_count=None,
                valid=False,
                error=str(exc),
            )
        return AnnotationDatabaseBackup(
            path=path,
            created_at=created_at,
            reason=reason,
            size_bytes=path.stat().st_size,
            project_id=project_id,
            image_count=image_count,
            box_count=box_count,
        )

    def _resolve_annotation_backup_path(self, path: Path | str) -> Path:
        resolved = Path(path).resolve()
        root = self.backups_dir.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ProjectFormatError("只能恢复当前项目 backups 目录中的备份") from exc
        in_recovery_trash = (
            len(relative.parts) >= 3
            and relative.parts[0] == ".trash"
            and resolved.parent.parent == root / ".trash"
        )
        if (
            resolved.suffix.casefold() != ".db"
            or (resolved.parent != root and not in_recovery_trash)
        ):
            raise ProjectFormatError("标注数据库备份路径无效")
        if not resolved.is_file():
            raise ProjectFormatError(f"标注数据库备份不存在：{resolved}")
        return resolved

    def verify_image(
        self,
        image_id: str,
        *,
        confirm_empty: bool = False,
        expected_revision: int | None = None,
    ) -> ImageRecord:
        return self.repository.confirm_image(
            image_id,
            confirm_empty=confirm_empty,
            expected_revision=expected_revision,
        )

    def add_category(
        self, name: str, *, color: str = "#22C55E", enabled: bool = True
    ) -> Category:
        return self.repository.add_category(name, color=color, enabled=enabled)

    def save_and_confirm(
        self,
        image_id: str,
        boxes: Iterable[BoxInput | Mapping[str, Any]],
        *,
        confirm_empty: bool = False,
        expected_revision: int | None = None,
    ) -> ImageRecord:
        return self.repository.save_and_confirm(
            image_id,
            boxes,
            confirm_empty=confirm_empty,
            expected_revision=expected_revision,
        )

    def training_preflight(self, *, minimum: int = 100) -> TrainingPreflight:
        return self.repository.training_preflight(minimum=minimum)

    preflight = training_preflight

    def create_snapshot(self, run_id: str, **options: Any):
        from .yolo import create_training_snapshot

        return create_training_snapshot(self, run_id, **options)

    snapshot = create_snapshot

    def export_yolo(self, destination: Path | str):
        from .yolo import export_yolo_detection

        return export_yolo_detection(self, destination)


Project = AnnotationProject


def create_project(
    root: Path | str,
    *,
    name: str,
    categories: Sequence[str | Mapping[str, Any]] = (),
) -> AnnotationProject:
    return AnnotationProject.create(root, name=name, categories=categories)


def open_project(root: Path | str) -> AnnotationProject:
    return AnnotationProject.open(root)


def _backup_database_before_schema_upgrade(
    root: Path,
    database_path: Path,
    *,
    project_id: str,
    source_version: int,
    target_version: int,
) -> Path:
    """Freeze a consistent pre-migration database before opening it writable."""

    backups = (root / BACKUPS_DIRECTORY).resolve()
    backups.mkdir(parents=True, exist_ok=True)
    created_at = utc_now()
    stamp = created_at.replace("-", "").replace(":", "").replace(".", "")
    filename = (
        f"annotations-{stamp}-before-schema-v{source_version}-to-v{target_version}-"
        f"{uuid4().hex[:8]}.db"
    )
    destination = backups / filename
    partial = backups / f".{filename}.partial"
    source: sqlite3.Connection | None = None
    target: sqlite3.Connection | None = None
    try:
        source = sqlite3.connect(
            f"{database_path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
        row = source.execute(
            "SELECT value FROM project_meta WHERE key = 'project_id'"
        ).fetchone()
        if row is None or str(row[0]) != project_id:
            raise ProjectFormatError("升级前备份发现数据库项目 ID 不一致")
        target = sqlite3.connect(partial, timeout=30.0, isolation_level=None)
        source.backup(target)
        integrity = str(target.execute("PRAGMA quick_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise ProjectFormatError(f"升级前数据库备份校验失败：{integrity}")
        target.close()
        target = None
        source.close()
        source = None
        partial.replace(destination)
        write_json(
            destination.with_suffix(".json"),
            {
                "format": "ai-biaozhu-annotation-backup",
                "database": destination.name,
                "project_id": project_id,
                "created_at": created_at,
                "reason": f"before-schema-v{source_version}-to-v{target_version}",
                "size_bytes": destination.stat().st_size,
                "source_schema_version": source_version,
                "target_schema_version": target_version,
            },
        )
        return destination
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        destination.with_suffix(".json").unlink(missing_ok=True)
        raise
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()


def _member(root: Path, relative: str) -> Path:
    candidate = root.joinpath(
        *PurePosixPath(relative.replace("\\", "/")).parts
    ).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ProjectFormatError(f"项目成员路径越界：{relative}") from exc
    return candidate


def _file_timestamp(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
