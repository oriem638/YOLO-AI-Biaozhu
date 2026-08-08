"""Application controller joining the Qt UI to the project and worker layers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import yaml
from PIL import Image as PillowImage

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.core.annotation_quality import (
    MAX_DEDUPLICATION_IOU,
    MIN_DEDUPLICATION_IOU,
)
from ai_biaozhu.core.domain import (
    ModelKey,
    RunKind,
    RunRecord,
    RunStatus,
    SplitConfig,
    TrainingConfig,
)
from ai_biaozhu.core.exceptions import RecordNotFoundError
from ai_biaozhu.core.training_preflight import (
    TrainingSample,
    build_training_preflight,
)
from ai_biaozhu.data.project import AnnotationProject
from ai_biaozhu.data.utils import sha256_file, utc_now, write_json
from ai_biaozhu.data.voc import (
    create_project_from_voc,
    merge_voc_into_project,
    preflight_voc_merge,
    read_voc_dataset,
)
from ai_biaozhu.data.yolo import read_yolo_export
from ai_biaozhu.deploy.environment import (
    assess_docker_desktop_recovery,
    find_docker_desktop_executable,
    inspect_docker_environment,
)
from ai_biaozhu.deploy.maix import MAIXCAM2_IMAGE, MAIXCAM_PRO_IMAGE
from ai_biaozhu.errors import (
    JobAlreadyRunningError,
    ProjectNotOpenError,
    ValidationError,
)
from ai_biaozhu.ml.environment import (
    EnvironmentCandidate,
    EnvironmentReport,
    discover_environments,
    inspect_environment,
)
from ai_biaozhu.ml.importer import AIResultImporter
from ai_biaozhu.ml.model_registry import ModelBackend, get_model
from ai_biaozhu.ml.protocol import ProtocolError, ProtocolEvent, read_jsonl_events
from ai_biaozhu.ml.training_results import resolve_training_end
from ai_biaozhu.settings import SettingsStore

EnvironmentInspector = Callable[[EnvironmentCandidate | str | Path], EnvironmentReport]


class ApplicationController:
    """Stateful, UI-agnostic orchestration boundary.

    SQLite is always mutated in this process.  The worker receives immutable JSON
    manifests and can only report results through the versioned JSONL protocol.
    """

    def __init__(
        self,
        paths: AppPaths,
        *,
        settings: SettingsStore | None = None,
        environment_inspector: EnvironmentInspector = inspect_environment,
        source_root: str | Path | None = None,
    ) -> None:
        self.paths = paths.ensure()
        self.settings = settings or SettingsStore(self.paths.data / "settings.json")
        self._environment_inspector = environment_inspector
        self.source_root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.current_project: AnnotationProject | None = None
        self._event_lock = RLock()
        self._last_event_seq: dict[str, int] = {}
        self._active_job_id: str | None = None
        self._active_job_kind: str | None = None
        self._prediction_importer: AIResultImporter | None = None
        self._job_failures: dict[str, int] = {}
        self._external_job_cancel_files: dict[str, Path] = {}
        self._environment_cache: dict[str, EnvironmentReport] = {}
        self._last_reconciled_run_ids: tuple[str, ...] = ()

    @property
    def project(self) -> AnnotationProject | None:
        return self.current_project

    @property
    def store(self) -> Any | None:
        project = self.current_project
        return project.repository if project is not None else None

    @property
    def seed_verified_count(self) -> int:
        project = self.current_project
        if project is None:
            return 0
        return sum(
            image.review_status.value == "verified" and image.training_selected
            for image in project.list_images()
        )

    verified_count = seed_verified_count

    # ------------------------------------------------------------------
    # Projects and annotations

    def new_project(self, root: str | Path, name: str | None = None) -> AnnotationProject:
        destination = Path(root).resolve()
        project_name = (name or destination.name or "未命名项目").strip()
        project = AnnotationProject.create(
            destination,
            name=project_name,
            categories=({"name": "目标", "color": "#45C486"},),
        )
        self._replace_project(project)
        return project

    create_project = new_project

    def open_project(self, root: str | Path) -> AnnotationProject:
        project = AnnotationProject.open(root)
        self._replace_project(project)
        return project

    def close_project(self) -> None:
        if self.current_project is not None:
            self.current_project.close()
        self.current_project = None
        self._prediction_importer = None

    def reopen_last_project(self) -> AnnotationProject | None:
        raw = self.settings.get("last_project")
        if not raw:
            return None
        path = Path(str(raw))
        if not (path / "project.json").is_file():
            self.settings.remove("last_project")
            return None
        try:
            return self.open_project(path)
        except Exception:
            # A moved or damaged project must not make the application unstartable.
            return None

    def _replace_project(self, project: AnnotationProject) -> None:
        previous = self.current_project
        reconciled = self.reconcile_interrupted_runs(project)
        self._last_reconciled_run_ids = tuple(run.id for run in reconciled)
        self.current_project = project
        self._prediction_importer = AIResultImporter(project.repository)
        self.settings.set("last_project", str(project.root))
        if previous is not None and previous is not project:
            previous.close()

    @property
    def last_reconciled_run_ids(self) -> tuple[str, ...]:
        """Runs safely closed while the current project was opened."""

        return self._last_reconciled_run_ids

    def reconcile_interrupted_runs(
        self,
        project: AnnotationProject | None = None,
    ) -> tuple[RunRecord, ...]:
        """Fail non-terminal runs that no longer have a controller-owned worker.

        A worker started by this controller is explicitly protected.  On a fresh
        application start there is no protected job, so runs left in an active
        phase by an earlier crash become resumable failures instead of remaining
        permanently stuck in ``training``.
        """

        target = project or self._require_project()
        active_statuses = {
            RunStatus.CREATED,
            RunStatus.PREFLIGHT,
            RunStatus.SNAPSHOTTING,
            RunStatus.TRAINING,
            RunStatus.EVALUATING,
            RunStatus.INFERENCING,
            RunStatus.IMPORTING,
        }
        protected: set[str] = set()
        if self._active_job_id:
            current = self.current_project
            if current is target or (
                current is not None
                and current.root.resolve() == target.root.resolve()
            ):
                protected.add(self._active_job_id)

        reconciled: list[RunRecord] = []
        for run in target.repository.list_runs():
            if run.id in protected or run.status not in active_statuses:
                continue
            artifacts = dict(run.artifacts)
            checkpoint_path = run.checkpoint_path
            recovered_roles: list[str] = []
            if run.kind is RunKind.TRAIN:
                discovered = self._discover_training_checkpoints(target, run)
                for role, path in discovered.items():
                    if not artifacts.get(role):
                        artifacts[role] = str(path)
                        recovered_roles.append(role)
                if discovered.get("last") is not None:
                    checkpoint_path = str(discovered["last"])
                elif discovered.get("best") is not None and not checkpoint_path:
                    checkpoint_path = str(discovered["best"])

            message = (
                "应用上次退出时任务子进程未完成；重新打开项目时已安全标记为失败。"
            )
            if run.kind is RunKind.TRAIN:
                message += (
                    " 已保留 last.pt，可从原不可变快照恢复训练。"
                    if artifacts.get("last")
                    else " 未发现 last.pt，不能执行断点恢复。"
                )
            if recovered_roles:
                message += f" 已恢复 checkpoint 记录：{', '.join(recovered_roles)}。"
            if run.error:
                message = f"{run.error}\n{message}"
            reconciled_run = target.repository.update_run(
                run.id,
                status=RunStatus.FAILED,
                artifacts=artifacts,
                checkpoint_path=checkpoint_path,
                error=message,
            )
            if run.kind is RunKind.DEPLOY:
                packages = target.repository.list_deployment_packages(run_id=run.id)
                for package in packages:
                    if package.status not in {
                        "ready",
                        "needs_device_validation",
                        "failed",
                        "cancelled",
                    }:
                        target.repository.update_deployment_package(
                            package.id,
                            status="failed",
                        )
            reconciled.append(reconciled_run)
        return tuple(reconciled)

    def _require_project(self) -> AnnotationProject:
        if self.current_project is None:
            raise ProjectNotOpenError("请先新建或打开标注项目。")
        return self.current_project

    def list_images(self) -> tuple[Any, ...]:
        return self.current_project.list_images() if self.current_project is not None else ()

    images = list_images

    def list_classes(self) -> tuple[Any, ...]:
        if self.current_project is None:
            return ()
        return self.current_project.repository.list_categories()

    list_categories = list_classes

    def list_runs(self) -> tuple[RunRecord, ...]:
        return self.current_project.list_runs() if self.current_project is not None else ()

    training_runs = list_runs

    def load_training_run_history(self, run_id: str) -> dict[str, Any]:
        """Load persisted metrics and logs for display after an application restart."""

        project = self._require_project()
        run = project.repository.get_run(str(run_id))
        if run.kind is not RunKind.TRAIN:
            raise ValidationError("只能查看训练运行的历史指标。")
        run_dir = (project.runs_dir / run.id).resolve()
        if not _is_within(run_dir, project.runs_dir):
            raise ValidationError("训练运行目录越过当前项目 runs 目录。")

        events: dict[tuple[str, int, str], ProtocolEvent] = {}
        warnings: list[str] = []

        def read_events(path: Path, *, metric_only: bool = False) -> None:
            if not path.is_file():
                return
            try:
                persisted = read_jsonl_events(path, ignore_incomplete_tail=True)
            except (OSError, ProtocolError) as exc:
                warnings.append(f"无法完整读取 {path.name}：{exc}")
                return
            for event in persisted:
                if event.job_id != run.id:
                    warnings.append(f"{path.name} 中已忽略其他任务的事件。")
                    continue
                if metric_only and event.type != "metrics":
                    continue
                events[(event.job_id, event.seq, event.type)] = event

        metrics_path: Path | None = None
        if run.metrics_jsonl_path:
            candidate = Path(run.metrics_jsonl_path).resolve()
            if _is_within(candidate, run_dir):
                metrics_path = candidate
                read_events(candidate, metric_only=True)
            else:
                warnings.append("运行记录中的 metrics.jsonl 路径不安全，已忽略。")

        event_path = run_dir / "events.jsonl"
        read_events(event_path)
        if not any(event.type == "metrics" for event in events.values()) and run.metrics:
            fallback = ProtocolEvent.create(
                job_id=run.id,
                seq=0,
                event_type="metrics",
                payload=dict(run.metrics),
                timestamp=run.updated_at,
            )
            events[(fallback.job_id, fallback.seq, fallback.type)] = fallback

        ordered = sorted(
            events.values(),
            key=lambda event: (event.timestamp, event.seq, event.type),
        )
        console_log = ""
        console_path = run_dir / "console.log"
        if console_path.is_file():
            try:
                lines = console_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
                console_log = "\n".join(lines[-5000:])
            except OSError as exc:
                warnings.append(f"无法读取 console.log：{exc}")

        preview_path: str | None = None
        for role in (
            "training_visual",
            "results_plot",
            "results",
            "confusion_matrix",
            "pr_curve",
            "preview",
        ):
            value = run.artifacts.get(role)
            if not value:
                continue
            candidate = Path(str(value)).resolve()
            if _is_within(candidate, run_dir) and candidate.is_file():
                preview_path = str(candidate)
                break

        return {
            "run_id": run.id,
            "model_key": run.model_key.value,
            "status": run.status.value,
            "events": [event.to_dict() for event in ordered],
            "console_log": console_log,
            "preview_path": preview_path,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "warnings": tuple(warnings),
        }

    training_run_history = load_training_run_history

    def recommend_calibration_image_ids(
        self,
        count: int = 100,
        seed: int = 42,
    ) -> tuple[str, ...]:
        """Choose a deterministic, class-balanced subset of verified images."""

        project = self._require_project()
        if isinstance(count, bool) or not 1 <= int(count) <= 200:
            raise ValidationError("校准图片推荐数量必须位于 1–200。")
        if isinstance(seed, bool):
            raise ValidationError("校准图片随机种子必须是整数。")
        requested = int(count)
        seed = int(seed)
        images = list(project.repository.list_images(review_status="verified"))
        if not images:
            return ()
        category_ids = {
            category.id
            for category in project.repository.list_categories(enabled_only=True)
        }
        labels_by_image: dict[str, frozenset[str]] = {}
        frequencies: dict[str, int] = {}
        for image in images:
            labels = {
                box.class_id
                for box in project.repository.list_boxes(image.id)
                if box.class_id in category_ids
            }
            frozen = frozenset(labels or {"__negative__"})
            labels_by_image[image.id] = frozen
            for label in frozen:
                frequencies[label] = frequencies.get(label, 0) + 1

        rank = {
            image.id: hashlib.sha256(
                f"{project.config.project_id}:{seed}:{image.id}:{image.sha256}".encode()
            ).digest()
            for image in images
        }
        selected: list[str] = []
        selected_set: set[str] = set()
        selected_counts = {label: 0 for label in frequencies}
        limit = min(requested, len(images))
        while len(selected) < limit:
            candidates = [image for image in images if image.id not in selected_set]

            def score(image: Any) -> tuple[float, float, float, bytes]:
                labels = labels_by_image[image.id]
                uncovered = sum(
                    1.0
                    for label in labels
                    if label != "__negative__" and selected_counts[label] == 0
                )
                balance = sum(
                    (0.5 if label == "__negative__" else 1.0)
                    / (selected_counts[label] + 1.0)
                    for label in labels
                )
                rarity = sum(
                    (0.5 if label == "__negative__" else 1.0)
                    / frequencies[label]
                    for label in labels
                )
                return uncovered, balance, rarity, rank[image.id]

            chosen = max(candidates, key=score)
            selected.append(chosen.id)
            selected_set.add(chosen.id)
            for label in labels_by_image[chosen.id]:
                selected_counts[label] += 1
        return tuple(selected)

    recommend_calibration_images = recommend_calibration_image_ids

    def image_path(self, image: Any) -> Path:
        project = self._require_project()
        image_id = getattr(image, "id", image)
        return project.image_path(str(image_id))

    resolve_image_path = image_path

    def get_image(self, image_id: object) -> Any:
        """Return the latest persisted record for an image.

        The UI uses this after an autosave to retain the current optimistic
        revision for a later session-baseline restore.
        """

        return self._require_project().repository.get_image(str(image_id))

    image_by_id = get_image

    def get_boxes(self, image_id: object) -> tuple[Any, ...]:
        if self.current_project is None:
            return ()
        return self.current_project.list_boxes(str(image_id))

    list_boxes = get_boxes
    boxes_for_image = get_boxes

    def save_boxes(
        self,
        image_id: object,
        boxes: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, ...]:
        return self._require_project().save_boxes(str(image_id), boxes)

    replace_boxes = save_boxes

    def restore_annotation_session_baseline(
        self,
        image_id: object,
        boxes: Sequence[Mapping[str, Any]],
        *,
        review_status: object,
        origin: object,
        ai_status: object,
        expected_revision: int | None = None,
    ) -> Any:
        """Restore an opened image's saved baseline without treating it as an edit."""

        return self._require_project().restore_annotation_session_baseline(
            str(image_id),
            boxes,
            review_status=str(getattr(review_status, "value", review_status)),
            origin=str(getattr(origin, "value", origin)),
            ai_status=str(getattr(ai_status, "value", ai_status)),
            expected_revision=expected_revision,
        )

    restore_image_annotation_state = restore_annotation_session_baseline

    def verify_and_next(
        self,
        image_id: object,
        boxes: Sequence[Mapping[str, Any]],
        *,
        confirm_empty: bool | None = None,
    ) -> Any | None:
        project = self._require_project()
        image_id_text = str(image_id)
        images = list(project.list_images())
        current_index = next(
            (index for index, image in enumerate(images) if image.id == image_id_text),
            -1,
        )
        project.save_and_confirm(
            image_id_text,
            boxes,
            confirm_empty=False if confirm_empty is None else bool(confirm_empty),
        )
        if not images:
            return None
        next_index = (current_index + 1) % len(images) if current_index >= 0 else 0
        return project.repository.get_image(images[next_index].id)

    save_and_confirm = verify_and_next
    save_and_verify = verify_and_next

    def add_class(self, name: str, color: str | None = None) -> Any:
        return self._require_project().add_category(
            name,
            color=color or "#45C486",
        )

    add_category = add_class

    def update_category_display_name(
        self,
        category_id: object,
        display_name: str | None,
    ) -> Any:
        """Set a presentation-only alias without changing the canonical class."""

        normalized = None if display_name is None else str(display_name).strip() or None
        return self._require_project().repository.update_category(
            str(category_id),
            display_name=normalized,
        )

    set_category_display_name = update_category_display_name

    def rename_category_canonical(
        self,
        category_id: object,
        name: str,
    ) -> dict[str, Any]:
        """Fully rename a category after creating a restorable DB backup."""

        category, backup = self._require_project().rename_category_canonical(
            str(category_id),
            str(name),
        )
        return {
            "category": _json_safe(category),
            "backup": {
                "path": str(backup.path),
                "created_at": backup.created_at,
                "reason": backup.reason,
                "size_bytes": backup.size_bytes,
                "valid": backup.valid,
            },
        }

    rename_category = rename_category_canonical

    def delete_empty_category(self, category_id: object) -> dict[str, Any]:
        """Safely remove one zero-instance category after a database backup."""

        category, backup = self._require_project().delete_empty_category(
            str(category_id)
        )
        return {
            "category": _json_safe(category),
            "backup": {
                "path": str(backup.path),
                "created_at": backup.created_at,
                "reason": backup.reason,
                "size_bytes": backup.size_bytes,
                "valid": backup.valid,
            },
        }

    def import_images(self, paths: Sequence[str | Path]) -> dict[str, Any]:
        report = self._require_project().import_images(paths)
        return {
            "requested": report.requested,
            "imported": report.imported_count,
            "duplicates": report.duplicate_count,
            "failed": report.failed_count,
            "duplicate_paths": [str(path) for path in report.duplicate_paths],
            "failures": [
                {"path": str(item.path), "reason": item.reason}
                for item in report.failures
            ],
            "report_path": str(report.report_path) if report.report_path else None,
        }

    def inspect_voc_import(
        self,
        source: str | Path,
        *,
        category_mapping: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Validate a MaixHub/Pascal VOC folder before showing its import dialog."""

        dataset = read_voc_dataset(source)
        result: dict[str, Any] = {
            "source": str(dataset.root),
            "image_count": len(dataset.images),
            "box_count": dataset.box_count,
            "annotated_image_count": dataset.annotated_image_count,
            "verified_negative_count": dataset.verified_negative_count,
            "unconfirmed_image_count": dataset.unconfirmed_image_count,
            # Compatibility field for older UI builds.  It means an explicit
            # empty XML only; an image without XML is never a known negative.
            "negative_count": dataset.verified_negative_count,
            "category_names": list(dataset.category_names),
        }
        if self.current_project is not None:
            plan = preflight_voc_merge(
                self.current_project,
                dataset.root,
                category_mapping=category_mapping,
            )
            result["merge_plan"] = _json_safe(plan)
            result["new_image_count"] = plan.new_image_count
            result["upgraded_image_count"] = plan.upgraded_image_count
            result["conflict_count"] = plan.conflict_count
            result["preserved_unconfirmed_count"] = plan.preserved_unconfirmed_count
        return result

    preflight_voc_import = inspect_voc_import

    def import_voc_dataset(
        self,
        source: str | Path,
        *,
        mode: str,
        destination: str | Path | None = None,
        project_name: str | None = None,
        category_mapping: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create or safely merge a validated MaixHub/Pascal VOC dataset."""

        import_mode = str(mode).strip().casefold()
        mapping = {
            str(source_name): str(target_name).strip()
            for source_name, target_name in (category_mapping or {}).items()
        }
        if import_mode == "new":
            if destination is None:
                raise ValidationError("新建 VOC 项目必须提供目标目录。")
            result = create_project_from_voc(
                source,
                destination,
                name=project_name,
                category_renames=mapping,
            )
            project = AnnotationProject.open(result.destination)
            self._replace_project(project)
            return {
                "mode": "new",
                "destination": str(result.destination),
                "image_count": result.image_count,
                "verified_count": result.verified_count,
                "annotated_image_count": result.annotated_image_count,
                "verified_negative_count": result.verified_negative_count,
                "unconfirmed_image_count": result.unconfirmed_image_count,
                "box_count": result.box_count,
                "category_names": list(result.category_names),
                "report_path": str(result.import_report_path),
            }
        if import_mode != "merge":
            raise ValidationError("VOC 导入方式只能是 new 或 merge。")
        project = self._require_project()
        plan = preflight_voc_merge(project, source, category_mapping=mapping)
        report = merge_voc_into_project(project, plan)
        return {
            "mode": "merge",
            "preflight": _json_safe(plan),
            "result": _json_safe(report),
            "image_count": report.source_image_count,
            "box_count": report.source_box_count,
            "imported_image_count": report.imported_image_count,
            "upgraded_image_count": report.upgraded_image_count,
            "conflict_image_count": report.conflict_image_count,
            "source_annotated_image_count": report.source_annotated_image_count,
            "source_verified_negative_count": report.source_verified_negative_count,
            "source_unconfirmed_image_count": report.source_unconfirmed_image_count,
            "preserved_unconfirmed_image_count": report.preserved_unconfirmed_image_count,
            "applied_box_count": report.applied_box_count,
            "report_path": str(report.report_path),
        }

    def preview_clear_all_annotations(
        self, image_ids: Sequence[str | object]
    ) -> dict[str, Any]:
        """Return the exact impact of clearing annotations without mutating data."""

        preview = self._require_project().preview_clear_all_annotations(
            tuple(str(image_id) for image_id in image_ids)
        )
        result = _json_safe(preview)
        if isinstance(result, dict):
            # Dataclass serialization deliberately omits computed convenience
            # properties; make the UI contract explicit.
            result["image_count"] = preview.image_count
            result["box_count"] = preview.box_count
            result["image_ids"] = list(preview.image_ids)
        return result

    preview_delete_all_annotations = preview_clear_all_annotations

    def clear_all_annotations(
        self, image_ids: Sequence[str | object]
    ) -> dict[str, Any]:
        """Back up then transactionally clear all annotations for selected images."""

        report = self._require_project().clear_all_annotations(
            tuple(str(image_id) for image_id in image_ids)
        )
        result = _json_safe(report)
        if isinstance(result, dict):
            result["image_count"] = report.image_count
            result["box_count"] = report.box_count
            result["image_ids"] = list(report.image_ids)
        return result

    delete_all_annotations = clear_all_annotations

    def preview_ai_deduplication(
        self,
        image_ids: Sequence[str | object],
        *,
        iou_threshold: float = 0.80,
    ) -> dict[str, Any]:
        """Preview exact same-class, untouched AI-draft duplicate cleanup."""

        preview = self._require_project().preview_ai_deduplication(
            tuple(str(image_id) for image_id in image_ids),
            iou_threshold=float(iou_threshold),
        )
        result = _json_safe(preview)
        if isinstance(result, dict):
            result.update(
                {
                    "requested_image_count": preview.requested_image_count,
                    "affected_image_count": preview.affected_image_count,
                    "removed_box_count": preview.removed_box_count,
                }
            )
        return result

    preview_historical_ai_deduplication = preview_ai_deduplication

    def deduplicate_ai_drafts(
        self,
        image_ids: Sequence[str | object],
        *,
        iou_threshold: float = 0.80,
    ) -> dict[str, Any]:
        """Back up then atomically remove only preview-eligible AI duplicates."""

        report = self._require_project().deduplicate_ai_drafts(
            tuple(str(image_id) for image_id in image_ids),
            iou_threshold=float(iou_threshold),
        )
        result = _json_safe(report)
        if isinstance(result, dict):
            result.update(
                {
                    "requested_image_count": report.preview.requested_image_count,
                    "affected_image_count": report.preview.affected_image_count,
                    "removed_box_count": report.preview.removed_box_count,
                }
            )
        return result

    apply_historical_ai_deduplication = deduplicate_ai_drafts

    def preview_delete_images(
        self, image_ids: Sequence[str | object]
    ) -> dict[str, Any]:
        return self._require_project().preview_delete_images(
            tuple(str(image_id) for image_id in image_ids)
        )

    def delete_images(
        self, image_ids: Sequence[str | object]
    ) -> dict[str, Any]:
        return _json_safe(
            self._require_project().delete_images(
                tuple(str(image_id) for image_id in image_ids)
            )
        )

    def set_training_selected(
        self,
        image_ids: Sequence[str | object],
        selected: bool,
    ) -> tuple[Any, ...]:
        return self._require_project().set_training_selected(
            tuple(str(image_id) for image_id in image_ids),
            bool(selected),
        )

    def select_only_for_training(
        self,
        image_ids: Sequence[str | object],
    ) -> tuple[Any, ...]:
        return self._require_project().select_only_for_training(
            tuple(str(image_id) for image_id in image_ids)
        )

    def list_annotation_backups(self) -> list[dict[str, Any]]:
        return _json_safe(self._require_project().list_annotation_backups())

    def preview_backup_cleanup(
        self,
        *,
        keep_latest: int = 3,
        include_recovery_trash: bool = False,
    ) -> dict[str, Any]:
        """Preview recoverable cleanup of old annotation database backups."""

        preview = self._require_project().preview_backup_cleanup(
            keep_latest=keep_latest,
            include_recovery_trash=include_recovery_trash,
        )
        result = _json_safe(preview)
        result["backup_count"] = preview.backup_count
        return result

    def cleanup_old_backups(
        self,
        *,
        keep_latest: int = 3,
        deployment_verified: bool = False,
        permanently_delete: bool = False,
    ) -> dict[str, Any]:
        """Move old backups to project-local trash after explicit verification."""

        if deployment_verified is not True:
            raise PermissionError("必须先明确确认部署已在设备上验证成功")
        report = self._require_project().cleanup_old_backups(
            keep_latest=keep_latest,
            deployment_verified=True,
            permanently_delete=permanently_delete,
        )
        result = _json_safe(report)
        result.update(
            {
                "backup_count": report.preview.backup_count,
                "moved_count": len(report.moved_paths),
                "deleted_count": len(report.deleted_paths),
                "permanently_deleted": report.permanently_deleted,
                "total_bytes": report.preview.total_bytes,
                "keep_latest": report.preview.keep_latest,
            }
        )
        return result

    # Explicit aliases for UI code that names the protected resource.
    preview_annotation_backup_cleanup = preview_backup_cleanup
    cleanup_old_annotation_backups = cleanup_old_backups

    def mark_deployment_verified(self, run_id: str) -> dict[str, Any]:
        """Record an explicit user confirmation after real-device validation."""

        project = self._require_project()
        packages = project.repository.list_deployment_packages(run_id=str(run_id))
        if not packages:
            raise ValidationError(f"部署任务没有可验证的产物记录：{run_id}")
        package = packages[0]
        if package.status not in {"needs_device_validation", "device_verified"}:
            raise ValidationError(
                f"部署任务当前状态不能标记为真机已验证：{package.status}"
            )
        updated = project.repository.update_deployment_package(
            package.id,
            status="device_verified",
        )
        return _json_safe(updated)

    def restore_annotation_backup(self, backup: str | Path) -> dict[str, Any]:
        report = self._require_project().restore_annotation_backup(backup)
        return _json_safe(report)

    def export_yolo(self, destination: str | Path) -> Path:
        return self._require_project().export_yolo(destination).root

    # ------------------------------------------------------------------
    # User settings and environment selection

    @property
    def training_presets(self) -> dict[str, dict[str, Any]]:
        raw = self.settings.mapping("training_presets")
        return {
            str(name): dict(value)
            for name, value in raw.items()
            if isinstance(value, Mapping)
        }

    list_training_presets = training_presets

    def save_training_preset(self, name: str, values: Mapping[str, Any]) -> None:
        key = name.strip()
        if not key:
            raise ValueError("预设名称不能为空。")
        presets = self.training_presets
        presets[key] = _json_safe(values)
        self.settings.set("training_presets", presets)

    def delete_training_preset(self, name: str) -> None:
        presets = self.training_presets
        presets.pop(str(name), None)
        self.settings.set("training_presets", presets)

    def discover_ml_environments(self) -> list[EnvironmentCandidate]:
        manual: list[Path] = []
        selected = self.settings.get("ml_python")
        if selected:
            manual.append(Path(str(selected)))
        common = Path(r"E:\tools\anacondass\envs\yolo")
        if common.exists():
            manual.append(common)
        runner = self._conda_aware_runner()
        return discover_environments(manual_paths=manual, runner=runner)

    list_ml_environments = discover_ml_environments

    def validate_ml_environment(self, value: str | Path) -> EnvironmentReport:
        key = str(Path(value)).casefold()
        report = self._environment_inspector(value)
        self._environment_cache[key] = report
        if report.valid:
            self.settings.set("ml_python", str(report.candidate.python))
        return report

    validate_python = validate_ml_environment

    def available_training_devices(self) -> tuple[dict[str, str], ...]:
        """List CPU/CUDA choices without importing Torch into the UI process."""

        devices: list[dict[str, str]] = [
            {"value": "auto", "label": "自动选择（auto）"},
            {"value": "cpu", "label": "CPU"},
        ]
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return tuple(devices)
        if completed.returncode:
            return tuple(devices)
        for raw_line in completed.stdout.splitlines():
            index, separator, name = raw_line.partition(",")
            index = index.strip()
            name = name.strip()
            if separator and index.isdigit():
                devices.append(
                    {
                        "value": index,
                        "label": f"CUDA {index} — {name or '已检测到的 GPU'}",
                    }
                )
        return tuple(devices)

    def create_ml_environment(
        self,
        payload: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        if isinstance(payload, Mapping) and not bool(payload.get("confirmed")):
            raise ValidationError("创建 Conda 环境需要用户明确确认。")
        conda = self._find_conda_executable()
        if conda is None:
            raise ValidationError("未找到 conda.exe；请先安装 Miniconda/Anaconda 或手动选择环境。")
        script = self.source_root / "scripts" / "create_yolo_env.ps1"
        if not script.is_file():
            raise ValidationError(f"环境创建脚本不存在：{script}")
        environment_name = (
            str(payload.get("name", "yolo"))
            if isinstance(payload, Mapping)
            else str(payload or "yolo")
        )
        if environment_name.casefold() != "yolo":
            raise ValidationError("首版仅允许创建锁定的 Conda yolo 环境。")
        job_id = f"environment-{uuid4().hex}"
        return {
            "job_id": job_id,
            "program": _powershell_executable(),
            "arguments": [
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-CondaExe",
                str(conda),
                "-UpdateExisting",
            ],
            "working_directory": str(self.source_root),
            "environment": self._worker_environment(),
        }

    create_yolo_environment = create_ml_environment

    # ------------------------------------------------------------------
    # Training and inference jobs

    def training_preflight(
        self,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = self._require_project()
        minimum = 1 if self._successful_training_runs(project) else 100
        legacy = project.training_preflight(minimum=minimum)
        categories = project.repository.list_categories(enabled_only=True)
        enabled_category_ids = {category.id for category in categories}
        samples: list[TrainingSample] = []
        for index, image in enumerate(project.list_images(), start=1):
            box_count = sum(
                box.class_id in enabled_category_ids
                for box in project.repository.list_boxes(image.id)
            )
            samples.append(
                TrainingSample(
                    index=index,
                    image_id=image.id,
                    filename=image.original_name,
                    review_status=image.review_status,
                    box_count=box_count,
                    training_selected=image.training_selected,
                    revision=image.revision,
                    image_sha256=image.sha256,
                )
            )
        structured = build_training_preflight(samples)
        split_source: Mapping[str, Any] = settings or {}
        split_raw = split_source.get("split")
        split = (
            dict(split_raw)
            if isinstance(split_raw, Mapping)
            else dict(split_source)
        )
        train_ratio = float(split.get("train_ratio", 0.8))
        val_ratio = float(split.get("val_ratio", 0.2))
        test_ratio = float(split.get("test_ratio", 0.0))
        split_counts = _training_split_counts(
            structured.trainable_count,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        warnings = list(legacy.warnings)
        for name, ratio in (
            ("训练集", train_ratio),
            ("验证集", val_ratio),
            ("测试集", test_ratio),
        ):
            split_name = {"训练集": "train", "验证集": "val", "测试集": "test"}[name]
            exact = split_counts[split_name]
            if ratio > 0 and exact < 10:
                warnings.append(f"{name}只有 {exact} 张图片，指标可能不稳定。")
        value = structured.to_dict()
        value.update(
            {
                "allowed": legacy.ok,
                "ok": legacy.ok,
                "minimum": minimum,
                "verified_count": legacy.verified_count,
                "positive_image_count": legacy.positive_image_count,
                "negative_image_count": legacy.negative_image_count,
                "instance_count": legacy.instance_count,
                "class_instance_counts": dict(legacy.class_instance_counts),
                "class_box_counts": {
                    category.name: int(
                        legacy.class_instance_counts.get(category.id, 0)
                    )
                    for category in categories
                },
                "empty_categories": [
                    {"id": category.id, "name": category.name}
                    for category in categories
                    if int(legacy.class_instance_counts.get(category.id, 0)) == 0
                ],
                "split_counts": split_counts,
                "errors": list(legacy.errors),
                "warnings": warnings,
                "manifest_path": None,
            }
        )
        if isinstance(settings, Mapping) and settings.get(
            "current_selection_count"
        ) is not None:
            value["current_selection_count"] = max(
                0,
                int(settings["current_selection_count"]),
            )
        return value

    def start_training(
        self,
        model_key: str | Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(model_key, Mapping):
            values = dict(model_key)
            model_key = str(values.get("model_key", "YOLO26n"))
        else:
            values = dict(settings or {})
            values.setdefault("model_key", model_key)
        project = self._require_project()
        self._ensure_no_active_job()

        model = ModelKey(str(model_key))
        requested_patience = int(values.get("patience", 20))
        early_stopping_enabled = bool(
            values.get("early_stopping_enabled", requested_patience > 0)
        )
        values["early_stopping_enabled"] = early_stopping_enabled
        values["patience"] = requested_patience if early_stopping_enabled else 0
        values["early_stopping_monitor"] = "fitness"
        config = TrainingConfig(
            model_key=model,
            imgsz=int(values.get("imgsz", 640)),
            epochs=int(values.get("epochs", 100)),
            patience=int(values["patience"]),
            batch=_batch_value(values.get("batch", "auto")),
            device=_device_value(values.get("device", 0)),
            workers=int(values.get("workers", 0)),
            seed=int(values.get("seed", 42)),
        )
        split_raw = values.get("split")
        split_values = dict(split_raw) if isinstance(split_raw, Mapping) else values
        split = SplitConfig(
            seed=int(split_values.get("seed", config.seed)),
            train_ratio=float(split_values.get("train_ratio", 0.8)),
            val_ratio=float(split_values.get("val_ratio", 0.2)),
            test_ratio=float(split_values.get("test_ratio", 0.0)),
        )
        augmentation = _augmentation_values(values.get("augmentation"))
        _validate_augmentation(augmentation)
        remembered_values = {
            key: value
            for key, value in values.items()
            if not str(key).startswith("expected_training_")
            and key != "current_selection_count"
        }
        self.settings.set(
            "last_training_settings",
            _json_safe(remembered_values),
        )

        preflight = self.training_preflight(values)
        minimum = int(preflight["minimum"])
        if not bool(preflight["ok"]):
            raise ValidationError("；".join(map(str, preflight["errors"])))
        expected_member_fingerprint = str(
            values.get("expected_training_member_fingerprint") or ""
        )
        if (
            expected_member_fingerprint
            and expected_member_fingerprint
            != preflight["training_member_fingerprint"]
        ):
            raise ValidationError(
                "训练样本在确认后发生变化，请重新执行训练前检查。"
            )
        expected_selection_fingerprint = str(
            values.get("expected_training_selection_fingerprint") or ""
        )
        if (
            expected_selection_fingerprint
            and expected_selection_fingerprint
            != preflight["selection_fingerprint"]
        ):
            raise ValidationError(
                "训练选择范围在确认后发生变化，请重新执行训练前检查。"
            )

        checkpoint_source = self._training_checkpoint_source(
            project,
            model,
            values,
        )
        python = self._select_worker_python(
            values.get("ml_environment"),
            device=config.device,
        )
        run = project.repository.create_run(
            RunKind.TRAIN,
            model,
            parameters={
                **_json_safe(values),
                "model_key": model.value,
                "weight": get_model(model.value).weight,
                "split": split.to_dict(),
                "augmentation": augmentation,
                "checkpoint_source": (
                    str(checkpoint_source) if checkpoint_source is not None else "official"
                ),
                "minimum_verified": minimum,
            },
        )
        run_dir = project.runs_dir / run.id
        cancel_file = run_dir / "cancel.requested"
        metrics_path = run_dir / "metrics.jsonl"
        manifest_path = run_dir / "job.json"
        try:
            project.repository.update_run(run.id, status=RunStatus.SNAPSHOTTING)
            snapshot = project.create_snapshot(run.id, split=split, minimum=minimum)
            manifest = {
                "job_id": run.id,
                "model_key": model.value,
                "data_yaml": str(snapshot.data_yaml),
                "output_dir": str(run_dir / "training"),
                "run_name": "model",
                "config": config.to_dict(),
                "augmentation": augmentation,
                "checkpoint_source": (
                    str(checkpoint_source) if checkpoint_source is not None else None
                ),
                "legacy_yolov5_repo": (
                    str(self._legacy_yolov5_repository())
                    if get_model(model.value).backend is ModelBackend.LEGACY_YOLOV5
                    else None
                ),
                "python_executable": str(python),
                "cancel_file": str(cancel_file),
                "metrics_path": str(metrics_path),
                "weight_cache_dir": str(self.paths.models),
                "weight_lock_path": str(
                    Path(__file__).resolve().parent / "ml" / "weights.lock.json"
                ),
                "offline_weights": bool(values.get("offline_weights", False)),
                "snapshot_sha256": snapshot.dataset_sha256,
                "training_member_fingerprint": preflight[
                    "training_member_fingerprint"
                ],
            }
            write_json(manifest_path, manifest)
            project.repository.update_run(
                run.id,
                status=RunStatus.TRAINING,
                snapshot_path=str(snapshot.root),
                metrics_jsonl_path=str(metrics_path),
            )
        except Exception as exc:
            project.repository.update_run(
                run.id,
                status=RunStatus.FAILED,
                error=str(exc),
            )
            raise
        self._activate_job(run.id, "train")
        snapshot_summary = {
            **preflight,
            "split_counts": {
                "train": snapshot.train_count,
                "val": snapshot.val_count,
                "test": snapshot.test_count,
            },
            "manifest_path": str(snapshot.manifest_path),
            "snapshot_path": str(snapshot.root),
            "snapshot_sha256": snapshot.dataset_sha256,
        }
        launch = self._worker_launch(python, "train", manifest_path, run.id)
        launch["snapshot_summary"] = snapshot_summary
        return launch

    train = start_training

    def resume_training(
        self,
        values: Mapping[str, Any] | str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resume a failed/cancelled run from its own immutable snapshot and last.pt."""

        options = dict(values) if isinstance(values, Mapping) else dict(extra or {})
        if not isinstance(values, Mapping):
            options.setdefault("run_id", values)
        project = self._require_project()
        self._ensure_no_active_job()
        source = project.repository.get_run(str(options.get("run_id") or ""))
        if source.kind is not RunKind.TRAIN or source.status not in {
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            raise ValidationError("只能恢复已失败或已取消的训练运行。")
        last_checkpoint = self._checkpoint_from_run(project, source, "last")
        if not source.snapshot_path:
            raise ValidationError("原训练没有不可变数据快照，不能恢复。")
        snapshot_root = Path(source.snapshot_path).resolve()
        if not _is_within(snapshot_root, project.runs_dir):
            raise ValidationError("原训练快照不属于当前项目。")
        snapshot = read_yolo_export(snapshot_root)
        source_manifest_path = project.runs_dir / source.id / "job.json"
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"原训练 manifest 无法读取：{exc}") from exc
        if source_manifest.get("snapshot_sha256") != snapshot.dataset_sha256:
            raise ValidationError("原训练快照哈希不一致，禁止恢复。")

        python = self._select_worker_python(
            options.get("ml_environment"),
            device=source.parameters.get("device", 0),
        )
        resumed = project.repository.create_run(
            RunKind.TRAIN,
            source.model_key,
            parameters={
                **dict(source.parameters),
                "resume_of": source.id,
                "resume_checkpoint": str(last_checkpoint),
            },
        )
        run_dir = project.runs_dir / resumed.id
        manifest_path = run_dir / "job.json"
        metrics_path = run_dir / "metrics.jsonl"
        cancel_file = run_dir / "cancel.requested"
        manifest = dict(source_manifest)
        config = dict(manifest.get("config") or {})
        config["resume"] = str(last_checkpoint)
        manifest.update(
            {
                "job_id": resumed.id,
                "data_yaml": str(snapshot_root / "data.yaml"),
                "output_dir": str(run_dir / "training"),
                "config": config,
                "checkpoint_source": str(last_checkpoint),
                "python_executable": str(python),
                "cancel_file": str(cancel_file),
                "metrics_path": str(metrics_path),
                "snapshot_sha256": snapshot.dataset_sha256,
                "resume_from_run_id": source.id,
            }
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        project.repository.update_run(
            resumed.id,
            status=RunStatus.TRAINING,
            snapshot_path=str(snapshot_root),
            metrics_jsonl_path=str(metrics_path),
        )
        self._activate_job(resumed.id, "train")
        return self._worker_launch(python, "train", manifest_path, resumed.id)

    resume = resume_training

    def start_autolabel(
        self,
        settings: Mapping[str, Any] | str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = dict(settings) if isinstance(settings, Mapping) else dict(extra or {})
        if not isinstance(settings, Mapping):
            values.setdefault("run_id", settings)
        project = self._require_project()
        self._ensure_no_active_job()
        source_run_id = str(values.get("run_id") or "")
        if not source_run_id:
            raise ValidationError("请选择一次成功训练。")
        source_run = project.repository.get_run(source_run_id)
        if source_run.kind is not RunKind.TRAIN or source_run.status is not RunStatus.COMPLETED:
            raise ValidationError("AI 自动标注只能使用当前项目中成功完成的训练运行。")
        checkpoint_kind = str(values.get("checkpoint_kind") or "best").casefold()
        checkpoint = self._checkpoint_from_run(
            project,
            source_run,
            checkpoint_kind,
            supplied=values.get("checkpoint"),
        )
        try:
            confidence = float(values.get("confidence", 0.25))
            iou = float(values.get("iou", 0.7))
            dedup_iou = float(values.get("dedup_iou", 0.8))
            imgsz = int(values.get("imgsz", source_run.parameters.get("imgsz", 640)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("AI 标注阈值或输入尺寸不是有效数字。") from exc
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValidationError("AI 标注置信度阈值必须位于 0～1。")
        if not isfinite(iou) or not 0.0 <= iou <= 1.0:
            raise ValidationError("AI 标注 IoU 阈值必须位于 0～1。")
        if (
            bool(values.get("deduplicate", False))
            and (
                not isfinite(dedup_iou)
                or not MIN_DEDUPLICATION_IOU <= dedup_iou <= MAX_DEDUPLICATION_IOU
            )
        ):
            raise ValidationError("AI 去重 IoU 阈值必须位于 0.70～0.95。")
        deduplicate = bool(values.get("deduplicate", False))
        if imgsz < 32 or imgsz > 4096 or imgsz % 32:
            raise ValidationError("AI 标注输入尺寸必须是 32～4096 之间的 32 的倍数。")
        device = _device_value(
            values.get("device", source_run.parameters.get("device", 0))
        )
        python = self._select_worker_python(
            values.get("ml_environment"),
            device=device,
        )

        prediction_run = self._resumable_prediction_run(
            project,
            source_run=source_run,
            checkpoint_kind=checkpoint_kind,
            deduplicate=deduplicate,
            dedup_iou=dedup_iou,
            enabled=bool(values.get("resume", True)),
        )
        prediction_run_id = (
            prediction_run.id if prediction_run is not None else uuid4().hex
        )
        imported = set(project.list_ai_imported_image_ids(prediction_run_id))
        images = [
            image
            for image in project.list_images()
            if image.review_status.value != "verified" and image.id not in imported
        ]
        if not images:
            raise ValidationError("没有尚未人工确认、且未完成本次推理的图片。")
        categories = project.repository.list_categories(enabled_only=True)
        run_dir = project.runs_dir / prediction_run_id
        manifest_path = run_dir / "job.json"
        cancel_file = run_dir / "cancel.requested"
        # Queueing images, recording the run and materializing its immutable
        # manifest form one database transaction. A disk or revision failure
        # therefore cannot leave only part of the batch queued.
        with project.repository.transaction():
            if prediction_run is None:
                prediction_run = project.repository.create_run(
                    RunKind.PREDICT,
                    source_run.model_key,
                    run_id=prediction_run_id,
                    parameters={
                        **_json_safe(values),
                        "deduplicate": deduplicate,
                        "dedup_iou": dedup_iou,
                        "source_run_id": source_run.id,
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint": str(checkpoint),
                    },
                )
            images = [
                project.repository.set_ai_status(
                    image.id,
                    "queued",
                    expected_revision=image.revision,
                )
                for image in images
            ]
            manifest = {
                "job_id": prediction_run.id,
                "model_key": source_run.model_key.value,
                "checkpoint": str(checkpoint),
                "images": [
                    {
                        "image_id": image.id,
                        "path": str(project.image_path(image)),
                        "expected_revision": image.revision,
                        "width": image.width,
                        "height": image.height,
                    }
                    for image in images
                ],
                "output_dir": str(run_dir / "predictions"),
                "class_ids": [category.id for category in categories],
                "confidence": confidence,
                "iou": iou,
                # Persist the import policy in both the immutable job manifest
                # and the prediction run.  SQLite applies the actual NMS pass
                # while importing events; keeping it here makes the effective
                # policy auditable and prevents a resume from silently reusing
                # a run created with different settings.
                "deduplicate": deduplicate,
                "dedup_iou": dedup_iou,
                "imgsz": imgsz,
                "device": device,
                "legacy_yolov5_repo": (
                    str(self._legacy_yolov5_repository())
                    if get_model(source_run.model_key.value).backend
                    is ModelBackend.LEGACY_YOLOV5
                    else None
                ),
                "python_executable": str(python),
                "cancel_file": str(cancel_file),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(manifest_path, manifest)
            project.repository.update_run(
                prediction_run.id,
                status=RunStatus.INFERENCING,
                progress=0.0,
            )
        self._prediction_importer = AIResultImporter(project.repository)
        self._activate_job(prediction_run.id, "predict")
        return self._worker_launch(python, "predict", manifest_path, prediction_run.id)

    start_auto_label = start_autolabel
    auto_label = start_autolabel

    def cancel_job(self, job_id: str | None = None) -> bool:
        selected = str(job_id or self._active_job_id or "")
        if not selected:
            return False
        external_cancel = self._external_job_cancel_files.get(selected)
        if external_cancel is not None:
            external_cancel.parent.mkdir(parents=True, exist_ok=True)
            external_cancel.write_text(
                f"requested_at={utc_now()}\n",
                encoding="ascii",
                newline="\n",
            )
            return True
        project = self._require_project()
        try:
            run = project.repository.get_run(selected)
        except RecordNotFoundError:
            run = None
        root = (
            project.deployments_dir
            if run is not None and run.kind is RunKind.DEPLOY
            else project.runs_dir
        )
        run_dir = root / selected
        if not _is_within(run_dir, root):
            raise ValidationError("取消文件路径越过项目任务目录。")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "cancel.requested").write_text(
            f"requested_at={utc_now()}\n",
            encoding="ascii",
            newline="\n",
        )
        return True

    cancel_active_job = cancel_job

    # ------------------------------------------------------------------
    # Worker event persistence

    def handle_job_event(self, raw: Mapping[str, Any]) -> Any | None:
        if raw.get("_internal"):
            if raw.get("type") == "log":
                self._append_process_log(raw)
            return None
        event = ProtocolEvent.from_dict(raw)
        if event.job_id in self._external_job_cancel_files:
            # Docker environment jobs do not belong to a project/model_runs
            # row.  Their structured events are consumed by the UI only.
            return None
        project = self._require_project()
        with self._event_lock:
            last = self._last_event_seq.get(event.job_id, -1)
            if event.seq <= last:
                relation = "重复" if event.seq == last else "乱序"
                diagnostic = (
                    f"已忽略{relation} worker 事件：job_id={event.job_id!r}, "
                    f"seq={event.seq}, last_seq={last}, type={event.type!r}"
                )
                self._append_process_log(
                    {
                        "_internal": True,
                        "job_id": event.job_id,
                        "type": "log",
                        "payload": {
                            "message": diagnostic,
                            "stream": "protocol",
                            "protocol_diagnostic": True,
                            "ignored_seq": event.seq,
                            "last_seq": last,
                            "ignored_type": event.type,
                        },
                    }
                )
                return None
            self._append_event(project, event)
            self._last_event_seq[event.job_id] = event.seq

        run = project.repository.get_run(event.job_id)
        result: Any | None = None
        payload = dict(event.payload)
        if event.type == "prediction":
            importer = self._prediction_importer or AIResultImporter(project.repository)
            result = importer.import_event(event)
        elif event.type == "progress":
            project.repository.update_run(
                run.id,
                progress=_event_progress(payload, run.progress),
            )
        elif event.type == "metrics":
            metrics = dict(run.metrics)
            metrics.update(_flatten_metrics(payload))
            project.repository.update_run(run.id, metrics=metrics)
        elif event.type == "artifact":
            artifacts = dict(run.artifacts)
            kind = str(payload.get("kind") or f"artifact-{event.seq}")
            path = payload.get("path")
            artifacts[kind] = str(path) if path is not None else _json_safe(payload)
            checkpoint = str(path) if kind in {"best", "last"} and path else None
            project.repository.update_run(
                run.id,
                artifacts=artifacts,
                checkpoint_path=checkpoint,
            )
            if run.kind is RunKind.DEPLOY:
                self._update_deployment_artifact(run.id, kind, path)
        elif event.type == "warning" and run.kind is RunKind.DEPLOY:
            self._update_deployment_warning(run.id, payload)
        elif event.type == "error":
            if str(payload.get("scope", "")).casefold() == "image":
                image_id = payload.get("image_id")
                if image_id:
                    project.repository.set_ai_status(str(image_id), "failed")
                self._job_failures[run.id] = self._job_failures.get(run.id, 0) + 1
            else:
                update: dict[str, Any] = {
                    "status": RunStatus.FAILED,
                    "error": str(payload.get("message") or payload),
                }
                if run.kind is RunKind.TRAIN:
                    update["metrics"] = _metrics_with_training_end(
                        run,
                        "failed",
                        payload,
                    )
                project.repository.update_run(
                    run.id,
                    **update,
                )
                if run.kind is RunKind.DEPLOY:
                    self._update_deployment_status(run.id, "failed")
                self._clear_active(run.id)
        elif event.type == "cancelled":
            update = {
                "status": RunStatus.CANCELLED,
                "error": str(payload.get("message") or "用户取消"),
            }
            if run.kind is RunKind.TRAIN:
                update["metrics"] = _metrics_with_training_end(
                    run,
                    "cancelled",
                    payload,
                )
            project.repository.update_run(
                run.id,
                **update,
            )
            if run.kind is RunKind.DEPLOY:
                self._update_deployment_status(run.id, "cancelled")
            self._clear_active(run.id)
        elif event.type == "completed":
            result_payload = payload.get("result")
            if isinstance(result_payload, Mapping):
                metrics = result_payload.get("metrics")
                training_end = result_payload.get("training_end")
                artifacts = result_payload.get("artifacts")
                update: dict[str, Any] = {}
                if isinstance(metrics, Mapping) or isinstance(training_end, Mapping):
                    merged_metrics = dict(run.metrics)
                    if isinstance(metrics, Mapping):
                        merged_metrics.update(dict(metrics))
                    if isinstance(training_end, Mapping):
                        merged_metrics["training_end"] = dict(training_end)
                    update["metrics"] = merged_metrics
                if isinstance(artifacts, Mapping):
                    merged = dict(run.artifacts)
                    merged.update({str(key): str(value) for key, value in artifacts.items()})
                    update["artifacts"] = merged
                    update["checkpoint_path"] = merged.get("best") or merged.get("last")
                if update:
                    run = project.repository.update_run(run.id, **update)
            if run.kind is RunKind.TRAIN:
                checkpoint_found = False
                for role in ("best", "last"):
                    try:
                        self._checkpoint_from_run(project, run, role)
                    except ValidationError:
                        continue
                    checkpoint_found = True
                    break
                if not checkpoint_found:
                    project.repository.update_run(
                        run.id,
                        status=RunStatus.FAILED,
                        error="训练进程已结束，但没有生成有效的 best.pt 或 last.pt。",
                        metrics=_metrics_with_training_end(
                            run,
                            "failed",
                            payload,
                        ),
                    )
                    self._clear_active(run.id)
                    return None
            failures = self._job_failures.pop(run.id, 0)
            project.repository.update_run(
                run.id,
                status=(
                    RunStatus.COMPLETED_WITH_ERRORS
                    if failures
                    else RunStatus.COMPLETED
                ),
                progress=1.0,
            )
            if run.kind is RunKind.DEPLOY:
                self._complete_deployment_record(run.id, payload)
            self._clear_active(run.id)
        elif event.type == "status":
            stage = str(payload.get("stage", "")).casefold()
            if run.kind is RunKind.PREDICT and stage == "image_running":
                image_id = payload.get("image_id")
                if image_id:
                    image = project.repository.get_image(str(image_id))
                    if (
                        image.review_status.value != "verified"
                        and image.ai_status.value in {"queued", "running"}
                    ):
                        project.repository.set_ai_status(
                            image.id,
                            "running",
                            expected_revision=image.revision,
                            bump_revision=False,
                        )
            status = _stage_status(run.kind, stage)
            if status is not None and status is not run.status:
                project.repository.update_run(run.id, status=status)
            if run.kind is RunKind.DEPLOY:
                deployment_status = _deployment_stage_status(stage)
                if deployment_status is not None:
                    self._update_deployment_status(run.id, deployment_status)
        return result

    consume_job_event = handle_job_event
    import_job_event = handle_job_event

    def handle_process_finished(
        self,
        job_id: str,
        *,
        success: bool,
        exit_code: int,
        cancelled: bool = False,
        completed_epochs: int | None = None,
        requested_epochs: int | None = None,
    ) -> None:
        """Close runs when a worker dies before emitting a terminal protocol event."""

        if job_id in self._external_job_cancel_files:
            self._external_job_cancel_files.pop(job_id, None)
            return

        project = self.current_project
        if project is None or not job_id:
            return
        try:
            run = project.repository.get_run(job_id)
        except Exception:
            return
        terminal = {
            RunStatus.COMPLETED,
            RunStatus.COMPLETED_WITH_ERRORS,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
        if run.status in terminal:
            return
        status = RunStatus.CANCELLED if cancelled else RunStatus.FAILED
        error = (
            "用户取消"
            if cancelled
            else "worker 已正常退出但未发送 completed 事件"
            if success
            else f"worker 异常退出，代码 {exit_code}"
        )
        update: dict[str, Any] = {"status": status, "error": error}
        if run.kind is RunKind.TRAIN:
            update["metrics"] = _metrics_with_training_end(
                run,
                "cancelled" if cancelled else "failed",
                completed_epochs=completed_epochs,
                requested_epochs=requested_epochs,
            )
        project.repository.update_run(run.id, **update)
        if run.kind is RunKind.DEPLOY:
            self._update_deployment_status(
                run.id,
                "cancelled" if cancelled else "failed",
            )
        self._clear_active(run.id)

    # ------------------------------------------------------------------
    # Maix deployment

    def inspect_conversion_environment(
        self,
        target: str | None = None,
        *,
        check_mount: bool = True,
    ) -> Any:
        images = {
            "maixcam_pro": (MAIXCAM_PRO_IMAGE,),
            "maixcam2": (MAIXCAM2_IMAGE,),
        }.get(str(target), (MAIXCAM_PRO_IMAGE, MAIXCAM2_IMAGE))
        return inspect_docker_environment(
            docker_executable=self.settings.get("docker_executable", "docker"),
            required_images=images,
            check_mount=check_mount,
            mount_root=self.settings.get("conversion_workspace", r"C:\tmp\ai_biaozhu"),
        )

    inspect_docker_environment = inspect_conversion_environment

    @property
    def last_maix_target(self) -> str:
        value = str(self.settings.get("last_maix_target", "maixcam2"))
        return value if value in {"maixcam_pro", "maixcam2"} else "maixcam2"

    def get_last_maix_target(self) -> str:
        return self.last_maix_target

    def set_last_maix_target(self, target: str) -> str:
        value = str(target)
        if value not in {"maixcam_pro", "maixcam2"}:
            raise ValidationError("目标设备必须是 MaixCAM-Pro 或 MaixCAM2。")
        self.settings.set("last_maix_target", value)
        return value

    def docker_desktop_recovery_status(
        self,
        target: str | None = None,
        *,
        launch_requested: bool = False,
        elapsed_seconds: float = 0.0,
    ) -> dict[str, Any]:
        report = self.inspect_conversion_environment(target, check_mount=False)
        executable = find_docker_desktop_executable(
            self.settings.get("docker_desktop_executable")
        )
        status = assess_docker_desktop_recovery(
            report,
            desktop_executable=executable,
            launch_requested=launch_requested,
            elapsed_seconds=elapsed_seconds,
        )
        return status.to_dict()

    def start_docker_desktop(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        values = dict(payload or {})
        if not bool(values.get("confirmed")):
            raise ValidationError("启动 Docker Desktop 需要用户明确确认。")
        executable = find_docker_desktop_executable(
            self.settings.get("docker_desktop_executable")
        )
        if executable is None:
            raise ValidationError("未找到 Docker Desktop，请先安装或手动选择安装位置。")
        self.settings.set("docker_desktop_executable", str(executable))
        creationflags = 0
        if os.name == "nt":
            creationflags = int(
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        try:
            subprocess.Popen(  # noqa: S603 - fixed, discovered executable only
                [str(executable)],
                cwd=str(executable.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise ValidationError(f"启动 Docker Desktop 失败：{exc}") from exc
        return self.docker_desktop_recovery_status(
            str(values.get("target") or "maixcam2"),
            launch_requested=True,
            elapsed_seconds=0.0,
        )

    def import_converter_image(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not bool(payload.get("confirmed")):
            raise ValidationError("导入 Docker 镜像需要用户明确确认。")
        archive = Path(str(payload.get("path") or "")).resolve()
        if not archive.is_file():
            raise ValidationError(f"Docker 镜像归档不存在：{archive}")
        target = str(payload.get("target") or "")
        expected_images = {
            "maixcam_pro": [MAIXCAM_PRO_IMAGE],
            "maixcam2": [MAIXCAM2_IMAGE],
        }.get(target, [])
        job_id = f"docker-import-{uuid4().hex}"
        job_dir = (self.paths.cache / "docker-imports" / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=False)
        cancel_file = job_dir / "cancel.requested"
        manifest_path = job_dir / "job.json"
        write_json(
            manifest_path,
            {
                "job_id": job_id,
                "kind": "docker_environment",
                "archive_path": str(archive),
                "docker_executable": str(
                    self.settings.get("docker_executable", "docker")
                ),
                "expected_images": expected_images,
                "cancel_file": str(cancel_file),
            },
        )
        self._external_job_cancel_files[job_id] = cancel_file
        return self._worker_launch(
            Path(sys.executable).resolve(),
            "docker-import",
            manifest_path,
            job_id,
        )

    def pull_converter_image(self, payload: Mapping[str, Any] | str) -> dict[str, Any]:
        values = dict(payload) if isinstance(payload, Mapping) else {"target": payload}
        if not bool(values.get("confirmed")):
            raise ValidationError("下载 Docker 镜像需要用户明确确认。")
        target = str(values.get("target") or "")
        if target == "maixcam2":
            raise ValidationError(
                "Pulsar2 官方转换环境以 tar 归档发布，不能使用 docker pull；"
                "请从 Sipeed 官方文档指向的下载源获取归档，再使用“导入镜像”。"
            )
        if target != "maixcam_pro":
            raise ValidationError("请选择 MaixCAM-Pro 或 MaixCAM2 转换镜像。")
        image = MAIXCAM_PRO_IMAGE
        docker = str(self.settings.get("docker_executable", "docker"))
        return {
            "job_id": f"docker-pull-{uuid4().hex}",
            "program": docker,
            "arguments": ["pull", image],
            "working_directory": str(self.source_root),
            "environment": self._worker_environment(),
        }

    def start_maix_deploy(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Create a deployment manifest; the worker performs export and Docker work."""

        project = self._require_project()
        self._ensure_no_active_job()
        source_run = project.repository.get_run(str(values.get("run_id") or ""))
        if source_run.kind is not RunKind.TRAIN or source_run.status is not RunStatus.COMPLETED:
            raise ValidationError("部署只能使用当前项目中成功完成的训练运行。")
        checkpoint_kind = str(values.get("checkpoint_kind") or "best").casefold()
        checkpoint = self._checkpoint_from_run(
            project,
            source_run,
            checkpoint_kind,
            supplied=values.get("checkpoint"),
        )
        target = str(values.get("target") or "")
        if target not in {"maixcam_pro", "maixcam2"}:
            raise ValidationError("目标设备必须是 MaixCAM-Pro 或 MaixCAM2。")
        normalized = dict(values)
        try:
            width = int(values.get("input_width", 0))
            height = int(values.get("input_height", 0))
            camera_width = int(values.get("camera_width", width))
            camera_height = int(values.get("camera_height", height))
            max_det = int(values.get("max_det", 100))
            confidence = float(values.get("confidence", 0.35))
            iou = float(values.get("iou", 0.45))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("部署尺寸、阈值或最大检测数不是有效数字。") from exc
        if (
            width < 32
            or width > 4096
            or height < 32
            or height > 4096
            or width % 32
            or height % 32
        ):
            raise ValidationError("部署宽高必须是 32～4096 之间的 32 的倍数。")
        if not 1 <= camera_width <= 8192 or not 1 <= camera_height <= 8192:
            raise ValidationError("相机宽高必须位于 1～8192。")
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValidationError("部署置信度阈值必须位于 0～1。")
        if not isfinite(iou) or not 0.0 <= iou <= 1.0:
            raise ValidationError("部署 IoU 阈值必须位于 0～1。")
        if not 1 <= max_det <= 1000:
            raise ValidationError("部署最大检测数必须位于 1～1000。")
        raw_calibration_ids = values.get("calibration_image_ids", ())
        if not isinstance(raw_calibration_ids, Sequence) or isinstance(
            raw_calibration_ids, str | bytes
        ):
            raise ValidationError("校准图片 ID 必须是图片列表。")
        calibration_ids = [str(item) for item in raw_calibration_ids]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise ValidationError("校准图片列表包含重复图片。")
        minimum_calibration, maximum_calibration = (
            (20, 200) if target == "maixcam_pro" else (20, 100)
        )
        if not minimum_calibration <= len(calibration_ids) <= maximum_calibration:
            raise ValidationError(
                f"{target} 校准图片必须为 {minimum_calibration}–{maximum_calibration} 张。"
            )
        if "calibration_count" in values:
            try:
                calibration_count = int(values["calibration_count"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError("校准图片数量不是有效整数。") from exc
            if calibration_count != len(calibration_ids):
                raise ValidationError("校准图片选择数量与校准数量不一致。")
        if str(values.get("calibration_source") or "project_verified") != (
            "project_verified"
        ):
            raise ValidationError("校准图片必须来自当前项目的人工确认图片。")
        supplied_project_id = values.get("calibration_project_id")
        if supplied_project_id and str(supplied_project_id) != project.config.project_id:
            raise ValidationError("校准图片所属项目与当前项目不一致。")
        if str(values.get("quantization") or "int8").casefold() != "int8":
            raise ValidationError("首版 Maix 转换只支持 INT8。")
        if values.get("static_shape", True) is not True:
            raise ValidationError("Maix 部署只支持静态输入尺寸。")
        raw_outputs = values.get("package_outputs") or [
            "maixapp",
            "editable_project",
        ]
        if not isinstance(raw_outputs, Sequence) or isinstance(
            raw_outputs, str | bytes
        ):
            raise ValidationError("部署输出类型必须是数组。")
        output_aliases = {
            "full_app": "maixapp",
            "full-app": "maixapp",
            "maixapp": "maixapp",
            "editable_project": "editable_project",
            "editable-project": "editable_project",
            "maixvision_project": "editable_project",
        }
        package_outputs: list[str] = []
        for raw_output in raw_outputs:
            output = output_aliases.get(str(raw_output).strip().casefold())
            if output is None:
                raise ValidationError(f"未知部署输出类型：{raw_output}")
            if output not in package_outputs:
                package_outputs.append(output)
        if not package_outputs:
            raise ValidationError("至少选择一种部署输出：.maixapp 或可编辑工程文件夹。")
        cam2_npu_mode: str | None = None
        if target == "maixcam2":
            cam2_npu_mode = str(values.get("cam2_npu_mode") or "both").casefold()
            if cam2_npu_mode not in {"both", "npu2", "vnpu"}:
                raise ValidationError("MaixCAM2 必须选择 NPU2、VNPU 或同时生成。")
        normalized.update(
            {
                "input_width": width,
                "input_height": height,
                "camera_width": camera_width,
                "camera_height": camera_height,
                "confidence": confidence,
                "iou": iou,
                "max_det": max_det,
                "calibration_image_ids": calibration_ids,
                "calibration_count": len(calibration_ids),
                "calibration_source": "project_verified",
                "calibration_project_id": project.config.project_id,
                "quantization": "int8",
                "static_shape": True,
                "package_outputs": package_outputs,
            }
        )
        if cam2_npu_mode is not None:
            normalized["cam2_npu_mode"] = cam2_npu_mode
        else:
            normalized.pop("cam2_npu_mode", None)
        calibration_images = []
        for image_id in calibration_ids:
            image = project.repository.get_image(image_id)
            if image.review_status.value != "verified":
                raise ValidationError(f"校准图片尚未人工确认：{image.original_name}")
            calibration_images.append(image)
        selected_calibration_ids = {image.id for image in calibration_images}
        fallback_calibration_images = [
            image
            for image in project.list_images()
            if image.review_status.value == "verified"
            and image.id not in selected_calibration_ids
        ]
        python = self._select_worker_python(values.get("ml_environment"))
        output_root = Path(
            str(
                values.get("output_directory")
                or self.settings.get("deployment_output_directory")
                or project.deployments_dir
            )
        ).resolve()
        workspace_root = Path(
            str(
                values.get("conversion_workspace")
                or os.environ.get("AI_BIAOZHU_CONVERSION_ROOT")
                or r"C:\tmp\ai_biaozhu"
            )
        ).resolve()
        if not str(workspace_root).isascii():
            raise ValidationError("Docker 转换临时目录必须是短 ASCII 路径。")
        self.settings.set("conversion_workspace", str(workspace_root))
        self.settings.set("deployment_output_directory", str(output_root))
        self.settings.set("last_maix_target", target)
        if values.get("docker_executable"):
            self.settings.set("docker_executable", str(values["docker_executable"]))
        run_id = uuid4().hex
        run_dir = project.deployments_dir / run_id
        manifest_path = run_dir / "job.json"
        publish_dir = output_root / run_id
        destination = workspace_root / run_id
        if destination.exists():
            raise ValidationError(f"转换临时目录已存在，拒绝覆盖：{destination}")
        if publish_dir.exists():
            raise ValidationError(f"部署产物目录已存在，拒绝覆盖：{publish_dir}")
        if _is_within(publish_dir, destination) or _is_within(
            destination, publish_dir
        ):
            raise ValidationError("转换临时目录与部署产物目录不能相互包含。")
        run = project.repository.create_run(
            RunKind.DEPLOY,
            source_run.model_key,
            run_id=run_id,
            parameters={
                **_json_safe(normalized),
                "source_run_id": source_run.id,
                "checkpoint": str(checkpoint),
            },
        )
        try:
            publish_dir.mkdir(parents=True, exist_ok=False)
            run_dir.mkdir(parents=True, exist_ok=True)
            calibration_snapshot = _freeze_calibration_sources(
                project,
                run_dir=run_dir,
                selected=calibration_images,
                fallback=fallback_calibration_images,
                required_count=len(calibration_images),
            )
            (
                deployment_class_ids,
                checkpoint_class_names,
                deployment_class_names,
            ) = _deployment_category_names(project, source_run)
            manifest = {
                **_json_safe(normalized),
                "job_id": run.id,
                "kind": "deploy",
                "model_key": source_run.model_key.value,
                "checkpoint": str(checkpoint),
                "source_checkpoint": str(checkpoint),
                "source_run_id": source_run.id,
                "checkpoint_kind": checkpoint_kind,
                "output_dir": str(destination),
                "output_directory": str(publish_dir),
                "conversion_workspace": str(destination),
                "audit_dir": str(run_dir),
                "cleanup_workdir": True,
                "package_path": str(
                    publish_dir
                    / (
                        f"{source_run.model_key.value}-{target}-"
                        f"{checkpoint_kind}.maixapp"
                    )
                ),
                "editable_project_path": (
                    str(
                        publish_dir
                        / (
                            f"{source_run.model_key.value}-{target}-"
                            f"{checkpoint_kind}-editable"
                        )
                    )
                    if "editable_project" in package_outputs
                    else None
                ),
                "input_width": width,
                "input_height": height,
                "class_names": checkpoint_class_names,
                "checkpoint_class_names": checkpoint_class_names,
                "deployment_class_names": deployment_class_names,
                "class_ids": deployment_class_ids,
                "calibration_images": calibration_snapshot["selected"],
                "calibration_candidate_images": calibration_snapshot["candidates"],
                "calibration_source_snapshot_manifest": calibration_snapshot[
                    "manifest_path"
                ],
                "calibration_count": len(calibration_images),
                "legacy_yolov5_repo": (
                    str(self._legacy_yolov5_repository())
                    if get_model(source_run.model_key.value).backend
                    is ModelBackend.LEGACY_YOLOV5
                    else None
                ),
                "python_executable": str(python),
                "cancel_file": str(run_dir / "cancel.requested"),
                "package_size_warning_bytes": 30_000_000,
            }
            write_json(manifest_path, manifest)
            with project.repository.transaction():
                project.repository.create_deployment_package(
                    run.id,
                    package_id=run.id,
                    target=target,
                    checkpoint_role=checkpoint_kind,
                    npu_mode=cam2_npu_mode or "not_applicable",
                    status="queued",
                )
                project.repository.update_run(
                    run.id,
                    status=RunStatus.PREFLIGHT,
                    snapshot_path=str(manifest_path),
                )
            self._activate_job(run.id, "deploy")
            return self._worker_launch(python, "deploy", manifest_path, run.id)
        except Exception as exc:
            self._clear_active(run.id)
            with suppress(Exception):
                project.repository.update_run(
                    run.id,
                    status=RunStatus.FAILED,
                    error=f"创建部署任务失败：{exc}",
                )
            raise

    prepare_maix_deployment = start_maix_deploy
    export_maix = start_maix_deploy
    deploy_maix = start_maix_deploy

    def _deployment_package(self, run_id: str) -> Any | None:
        project = self.current_project
        if project is None:
            return None
        packages = project.repository.list_deployment_packages(run_id=run_id)
        return packages[0] if packages else None

    def _update_deployment_status(self, run_id: str, status: str) -> None:
        project = self.current_project
        package = self._deployment_package(run_id)
        if project is not None and package is not None and package.status != status:
            project.repository.update_deployment_package(package.id, status=status)

    def _update_deployment_artifact(
        self,
        run_id: str,
        kind: str,
        path: Any,
    ) -> None:
        if path is None:
            return
        project = self.current_project
        package = self._deployment_package(run_id)
        if project is None or package is None:
            return
        values: dict[str, Any] = {}
        if kind == "maix_model_package":
            values["model_package_path"] = str(path)
        elif kind == "maix_app_package":
            values["app_package_path"] = str(path)
        elif kind == "deployment_report":
            values["report_path"] = str(path)
        if values:
            project.repository.update_deployment_package(package.id, **values)

    def _update_deployment_warning(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        project = self.current_project
        package = self._deployment_package(run_id)
        if project is None or package is None:
            return
        message = str(payload.get("message") or payload)
        warnings = tuple((*package.warnings, message))
        size_items = payload.get("packages")
        zip_bytes = package.zip_bytes
        payload_bytes = package.payload_bytes
        if isinstance(size_items, list):
            mappings = [item for item in size_items if isinstance(item, Mapping)]
            if mappings:
                zip_bytes = max(int(item.get("zip_size", 0)) for item in mappings)
                payload_bytes = max(
                    int(item.get("unpacked_size", 0)) for item in mappings
                )
        project.repository.update_deployment_package(
            package.id,
            warnings=warnings,
            zip_bytes=zip_bytes,
            payload_bytes=payload_bytes,
        )

    def _complete_deployment_record(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        project = self.current_project
        package = self._deployment_package(run_id)
        if project is None or package is None:
            return
        values: dict[str, Any] = {
            # Conversion on the desktop is not equivalent to loading and
            # exercising the artifact on physical Maix hardware.
            "status": "needs_device_validation",
        }
        for source_key, target_key in (
            ("model_package_path", "model_package_path"),
            ("app_package_path", "app_package_path"),
        ):
            if payload.get(source_key):
                values[target_key] = str(payload[source_key])
        report_path = package.report_path
        if report_path and Path(report_path).is_file():
            try:
                report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = {}
            packages = report.get("packages") if isinstance(report, Mapping) else None
            if isinstance(packages, list):
                full_app = next(
                    (
                        item
                        for item in packages
                        if isinstance(item, Mapping) and item.get("kind") == "full-app"
                    ),
                    None,
                )
                if full_app is not None:
                    values["zip_bytes"] = int(full_app.get("zip_size", 0))
                    values["payload_bytes"] = int(full_app.get("unpacked_size", 0))
        project.repository.update_deployment_package(package.id, **values)

    def _append_process_log(self, raw: Mapping[str, Any]) -> None:
        project = self.current_project
        job_id = str(raw.get("job_id") or "")
        payload = raw.get("payload")
        if project is None or not job_id or not isinstance(payload, Mapping):
            return
        run_dir = project.runs_dir / job_id
        if not _is_within(run_dir, project.runs_dir):
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        message = str(payload.get("message") or "")
        if not message:
            return
        with (run_dir / "console.log").open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(message.rstrip("\r\n") + "\n")
            handle.flush()

    # ------------------------------------------------------------------
    # Internal helpers

    def _training_checkpoint_source(
        self,
        project: AnnotationProject,
        model: ModelKey,
        values: Mapping[str, Any],
    ) -> str | Path | None:
        source = str(values.get("start_from") or "official").casefold()
        if source in {"official", "pretrained", "官方预训练"}:
            # ``None`` is intentional: it makes the worker use WeightManager,
            # whose URL allowlist and SHA-256 lock prevent implicit trust.
            return None
        if source not in {"best", "last"}:
            raise ValidationError("训练起点必须是 official、best 或 last。")
        run_id = str(values.get("historical_run_id") or "")
        if not run_id:
            raise ValidationError(f"选择 {source}.pt 时必须指定历史训练运行。")
        historical = project.repository.get_run(run_id)
        if historical.model_key is not model:
            raise ValidationError("历史 checkpoint 的模型型号与本次选择不一致。")
        if historical.status is not RunStatus.COMPLETED:
            raise ValidationError("历史 checkpoint 必须来自成功完成的训练。")
        return self._checkpoint_from_run(project, historical, source)

    def _checkpoint_from_run(
        self,
        project: AnnotationProject,
        run: RunRecord,
        kind: str,
        *,
        supplied: Any = None,
    ) -> Path:
        role = kind.casefold()
        if role not in {"best", "last"}:
            raise ValidationError("checkpoint 只能选择 best.pt 或 last.pt。")
        recorded = run.artifacts.get(role) or run.artifacts.get(f"{role}.pt")
        if (
            recorded is None
            and run.checkpoint_path
            and Path(run.checkpoint_path).name.casefold() == f"{role}.pt"
        ):
            recorded = run.checkpoint_path
        if not recorded:
            raise ValidationError(f"运行 {run.id} 没有记录 {role}.pt。")
        checkpoint = Path(str(recorded)).resolve()
        if supplied:
            supplied_path = Path(str(supplied)).resolve()
            if supplied_path != checkpoint:
                raise ValidationError("界面提交的 checkpoint 与运行记录不一致。")
        if not _is_within(checkpoint, project.runs_dir):
            raise ValidationError("checkpoint 不属于当前项目 runs 目录。")
        if not checkpoint.is_file():
            raise ValidationError(f"checkpoint 文件不存在：{checkpoint}")
        return checkpoint

    def _discover_training_checkpoints(
        self,
        project: AnnotationProject,
        run: RunRecord,
    ) -> dict[str, Path]:
        """Find checkpoint files materialized before a worker could report them."""

        run_dir = (project.runs_dir / run.id).resolve()
        if not _is_within(run_dir, project.runs_dir) or not run_dir.is_dir():
            return {}
        candidates: dict[str, list[Path]] = {"best": [], "last": []}

        manifest_path = run_dir / "job.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if isinstance(manifest, Mapping):
                output_value = manifest.get("output_dir")
                run_name = str(manifest.get("run_name") or "model")
                if output_value:
                    output_dir = Path(str(output_value))
                    for role in candidates:
                        candidates[role].append(
                            output_dir / run_name / "weights" / f"{role}.pt"
                        )

        for role in candidates:
            candidates[role].append(
                run_dir / "training" / "model" / "weights" / f"{role}.pt"
            )
            with suppress(OSError):
                candidates[role].extend(
                    path
                    for path in run_dir.rglob(f"{role}.pt")
                    if path.parent.name.casefold() == "weights"
                )

        discovered: dict[str, Path] = {}
        for role, values in candidates.items():
            unique: dict[str, Path] = {}
            for value in values:
                try:
                    resolved = value.resolve()
                except OSError:
                    continue
                if (
                    _is_within(resolved, run_dir)
                    and resolved.is_file()
                    and resolved.name.casefold() == f"{role}.pt"
                ):
                    unique.setdefault(str(resolved).casefold(), resolved)
            if not unique:
                continue
            # Known manifest/conventional paths are inserted first.  If only
            # fallback paths exist, prefer the newest complete checkpoint.
            ordered = list(unique.values())
            preferred = ordered[0]
            if len(ordered) > 1:
                with suppress(OSError):
                    preferred = max(
                        ordered,
                        key=lambda path: path.stat().st_mtime_ns,
                    )
            discovered[role] = preferred
        return discovered

    def _successful_training_runs(
        self, project: AnnotationProject
    ) -> tuple[RunRecord, ...]:
        successful: list[RunRecord] = []
        for run in project.list_runs(kind=RunKind.TRAIN):
            if run.status is not RunStatus.COMPLETED:
                continue
            try:
                self._checkpoint_from_run(project, run, "best")
            except ValidationError:
                try:
                    self._checkpoint_from_run(project, run, "last")
                except ValidationError:
                    continue
            successful.append(run)
        return tuple(successful)

    def _resumable_prediction_run(
        self,
        project: AnnotationProject,
        *,
        source_run: RunRecord,
        checkpoint_kind: str,
        deduplicate: bool,
        dedup_iou: float,
        enabled: bool,
    ) -> RunRecord | None:
        if not enabled:
            return None
        for run in project.list_runs(kind=RunKind.PREDICT):
            run_deduplicate = bool(run.parameters.get("deduplicate", False))
            if run_deduplicate != deduplicate:
                continue
            if deduplicate:
                try:
                    run_dedup_iou = float(run.parameters.get("dedup_iou", 0.8))
                except (TypeError, ValueError, OverflowError):
                    continue
                if run_dedup_iou != dedup_iou:
                    continue
            if (
                run.model_key is source_run.model_key
                and run.parameters.get("source_run_id") == source_run.id
                and run.parameters.get("checkpoint_kind") == checkpoint_kind
                and run.status
                in {
                    RunStatus.CREATED,
                    RunStatus.INFERENCING,
                    RunStatus.IMPORTING,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                    RunStatus.COMPLETED_WITH_ERRORS,
                }
            ):
                return run
        return None

    def _select_worker_python(
        self,
        requested: Any = None,
        *,
        device: Any = "auto",
    ) -> Path:
        # A Nuitka standalone release contains its own ML runtime in the sibling
        # worker executable.  Requiring an external Conda installation before
        # reaching ``_worker_launch`` would defeat the clean-PC installer.
        if self._bundled_worker_executable() is not None:
            return Path(sys.executable).resolve()
        candidates: list[str | Path] = []
        if requested:
            candidates.append(str(requested))
        stored = self.settings.get("ml_python")
        if stored:
            candidates.append(str(stored))
        candidates.extend(candidate.python for candidate in self.discover_ml_environments())
        seen: set[str] = set()
        problems: list[str] = []
        for candidate in candidates:
            key = str(Path(candidate)).casefold()
            if key in seen:
                continue
            seen.add(key)
            report = self._environment_cache.get(key)
            if report is None:
                report = self._environment_inspector(candidate)
                self._environment_cache[key] = report
            if report.valid and (
                not _requires_cuda_device(device) or report.gpu_ready
            ):
                python = report.candidate.python.resolve()
                self.settings.set("ml_python", str(python))
                return python
            details = (*report.errors, *report.compatibility_errors)
            if (
                report.valid
                and _requires_cuda_device(device)
                and not report.gpu_ready
            ):
                details = (
                    *details,
                    f"所选设备 {device!r} 需要可用的 CUDA 12.8 GPU",
                )
            problems.append(f"{report.candidate.python}: {'；'.join(details) or '不兼容'}")
        raise ValidationError(
            "没有可用的锁定 ML 环境。请在“环境”面板创建或选择 Conda yolo。\n"
            + "\n".join(problems[:5])
        )

    def _legacy_yolov5_repository(self) -> Path:
        candidates = (
            self.source_root / "third_party" / "runtime" / "yolov5",
            Path(sys.executable).resolve().parent / "third_party" / "yolov5",
            self.paths.data / "third_party" / "yolov5",
        )
        for candidate in candidates:
            if (candidate / "train.py").is_file() and (candidate / "detect.py").is_file():
                return candidate.resolve()
        raise ValidationError(
            "缺少已锁定的传统 YOLOv5 v7.0 后端；请运行 scripts/fetch_yolov5.ps1。"
        )

    def _worker_launch(
        self,
        python: Path,
        action: str,
        manifest: Path,
        job_id: str,
    ) -> dict[str, Any]:
        worker_exe = self._bundled_worker_executable()
        bundled = worker_exe is not None
        if worker_exe is not None:
            program = worker_exe
            arguments = [action, "--manifest", str(manifest)]
        else:
            program = python
            arguments = [
                "-m",
                "ai_biaozhu.workers.main",
                action,
                "--manifest",
                str(manifest),
            ]
        return {
            "job_id": job_id,
            "program": str(program),
            "arguments": arguments,
            "working_directory": str(program.parent if bundled else self.source_root),
            "environment": self._worker_environment(bundled=bundled),
        }

    def _bundled_worker_executable(self) -> Path | None:
        worker_exe = Path(sys.executable).resolve().parent / "AI-Biaozhu-Worker.exe"
        return worker_exe if worker_exe.is_file() else None

    def _worker_environment(self, *, bundled: bool = False) -> dict[str, str]:
        environment = {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "YOLO_CONFIG_DIR": str(self.paths.yolo_config),
            "AI_BIAOZHU_MODELS_DIR": str(self.paths.models),
        }
        if bundled:
            environment["AI_BIAOZHU_STANDALONE"] = "1"
            # A frozen executable cannot safely act as ``python -m pip``.
            # Missing runtime modules still fail at their real import site,
            # while Ultralytics is prevented from attempting to mutate the
            # signed standalone installation or access the network.
            environment["YOLO_AUTOINSTALL"] = "false"
        else:
            current = os.environ.get("PYTHONPATH", "")
            source = str(self.source_root / "src")
            environment["PYTHONPATH"] = (
                source if not current else os.pathsep.join((source, current))
            )
        return environment

    def _activate_job(self, job_id: str, kind: str) -> None:
        self._active_job_id = job_id
        self._active_job_kind = kind
        self._last_event_seq.pop(job_id, None)
        self._job_failures.pop(job_id, None)

    def _clear_active(self, job_id: str) -> None:
        if self._active_job_id == job_id:
            self._active_job_id = None
            self._active_job_kind = None

    def _ensure_no_active_job(self) -> None:
        if self._active_job_id:
            raise JobAlreadyRunningError(f"任务仍在运行：{self._active_job_id}")

    def _append_event(
        self,
        project: AnnotationProject,
        event: ProtocolEvent,
    ) -> None:
        run_dir = project.runs_dir / event.job_id
        # Deployment metadata lives in deployments/, but the audit stream remains
        # under runs/ because model_runs is the common run registry.
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "events.jsonl").open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(event.to_json() + "\n")
            handle.flush()

    def _find_conda_executable(self) -> Path | None:
        configured = self.settings.get("conda_executable")
        candidates = [
            Path(str(configured)) if configured else None,
            Path(os.environ["CONDA_EXE"]) if os.environ.get("CONDA_EXE") else None,
            Path(r"E:\tools\anacondass\Scripts\conda.exe"),
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        ]
        try:
            located = subprocess.run(
                ["where.exe", "conda.exe"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if located.returncode == 0 and located.stdout.strip():
                candidates.append(Path(located.stdout.splitlines()[0].strip()))
        except (OSError, subprocess.SubprocessError):
            pass
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                resolved = candidate.resolve()
                self.settings.set("conda_executable", str(resolved))
                return resolved
        return None

    def _conda_aware_runner(self) -> Callable[..., subprocess.CompletedProcess[str]]:
        conda = self._find_conda_executable()

        def runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            values = list(command)
            if values and values[0] == "conda" and conda is not None:
                values[0] = str(conda)
            return subprocess.run(values, **kwargs)

        return runner


Controller = ApplicationController


def _training_split_counts(
    count: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    """Mirror the snapshot allocator's exact train/val/test capacities."""

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.0001:
        raise ValidationError("训练、验证和测试比例之和必须为 100%。")
    val_count = round(count * val_ratio)
    test_count = round(count * test_ratio)
    if count >= 2:
        val_count = max(1, val_count)
    if test_ratio > 0 and count >= 3:
        test_count = max(1, test_count)
    while val_count + test_count >= count:
        if test_count > 0 and (
            test_count / max(test_ratio, 1e-12)
            >= val_count / max(val_ratio, 1e-12)
        ):
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    return {
        "train": count - val_count - test_count,
        "val": val_count,
        "test": test_count,
    }


def _augmentation_values(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    rotation_enabled = bool(
        raw.get(
            "rotation_enabled",
            float(raw.get("rotation_probability", 0.0)) > 0,
        )
    )
    blur_enabled = bool(
        raw.get(
            "blur_enabled",
            float(raw.get("blur_probability", 0.0)) > 0,
        )
    )
    horizontal_enabled = bool(
        raw.get(
            "horizontal_flip_enabled",
            float(raw.get("fliplr", 0.0)) > 0,
        )
    )
    vertical_enabled = bool(
        raw.get(
            "vertical_flip_enabled",
            float(raw.get("flipud", 0.0)) > 0,
        )
    )
    return {
        "enabled": bool(raw.get("enabled", True)),
        "rotation_enabled": rotation_enabled,
        "rotation_degrees": float(raw.get("rotation_degrees", 0.0)),
        "rotation_probability": (
            float(raw.get("rotation_probability", 0.0))
            if rotation_enabled
            else 0.0
        ),
        "blur_enabled": blur_enabled,
        "blur_kernel": int(raw.get("blur_kernel", 3)),
        "blur_probability": (
            float(raw.get("blur_probability", 0.0)) if blur_enabled else 0.0
        ),
        "horizontal_flip_enabled": horizontal_enabled,
        "fliplr": float(raw.get("fliplr", 0.0)) if horizontal_enabled else 0.0,
        "vertical_flip_enabled": vertical_enabled,
        "flipud": float(raw.get("flipud", 0.0)) if vertical_enabled else 0.0,
    }


def _validate_augmentation(values: Mapping[str, Any]) -> None:
    if not 0 <= float(values["rotation_degrees"]) <= 30:
        raise ValidationError("随机旋转角度必须位于 0–30°。")
    if int(values["blur_kernel"]) not in {3, 5, 7}:
        raise ValidationError("随机模糊核只能选择 3、5 或 7。")
    for name in ("rotation_probability", "blur_probability", "fliplr", "flipud"):
        if not 0 <= float(values[name]) <= 1:
            raise ValidationError(f"{name} 必须位于 0–1。")


def _batch_value(value: Any) -> int | str:
    if isinstance(value, str):
        stripped = value.strip().casefold()
        if stripped == "auto":
            return "auto"
        try:
            return int(stripped)
        except ValueError as exc:
            raise ValidationError("batch 必须是 auto 或正整数。") from exc
    if isinstance(value, bool):
        raise ValidationError("batch 不能是布尔值。")
    return int(value)


def _device_value(value: Any) -> int | str:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.casefold() in {"auto", "自动"}:
        return "auto"
    if text.isdigit():
        return int(text)
    return text


def _requires_cuda_device(value: Any) -> bool:
    """Return whether a user-selected device explicitly requires CUDA."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 0
    text = str(value).strip().casefold()
    if text in {"", "auto", "自动", "cpu", "mps", "none", "-1"}:
        return False
    if text.isdigit() or text.startswith("cuda"):
        return True
    return bool(text) and all(part.strip().isdigit() for part in text.split(","))


def _event_progress(payload: Mapping[str, Any], fallback: float) -> float:
    raw = payload.get("overall_progress", payload.get("progress"))
    if raw is None:
        current = payload.get("current")
        total = payload.get("total")
        if current is not None and total:
            raw = float(current) / float(total)
    if raw is None:
        return fallback
    value = float(raw)
    if value > 1:
        value /= 100.0
    return min(1.0, max(0.0, value))


def _flatten_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("metrics")
    values = dict(nested) if isinstance(nested, Mapping) else {}
    values.update(
        {
            str(key): _json_safe(value)
            for key, value in payload.items()
            if key != "metrics"
        }
    )
    return values


def _metrics_with_training_end(
    run: RunRecord,
    reason: str,
    payload: Mapping[str, Any] | None = None,
    *,
    completed_epochs: int | None = None,
    requested_epochs: int | None = None,
) -> dict[str, Any]:
    """Persist a backend-neutral terminal result for non-success exits."""

    values = payload or {}
    completed_candidates = [completed_epochs, values.get("completed_epochs")]
    for key in ("completed_epochs", "epoch", "current_epoch"):
        completed_candidates.append(run.metrics.get(key))
    completed = max(
        (
            parsed
            for value in completed_candidates
            if (parsed := _optional_nonnegative_int(value)) is not None
        ),
        default=0,
    )
    requested = _optional_nonnegative_int(requested_epochs)
    if not requested:
        requested = _optional_nonnegative_int(values.get("requested_epochs"))
    if not requested:
        requested = _optional_nonnegative_int(run.parameters.get("epochs"))
    if not requested:
        requested = _optional_nonnegative_int(run.metrics.get("epochs"))
    requested = max(1, requested or completed or 1)
    patience = _optional_nonnegative_int(run.parameters.get("patience")) or 0
    terminal = resolve_training_end(
        completed_epochs=completed,
        requested_epochs=requested,
        patience=patience,
        cancelled=reason == "cancelled",
        failed=reason == "failed",
        monitor=str(run.parameters.get("early_stopping_monitor", "fitness")),
    )
    metrics = dict(run.metrics)
    metrics["training_end"] = terminal.to_dict()
    return metrics


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _stage_status(kind: RunKind, stage: str) -> RunStatus | None:
    if kind is RunKind.TRAIN:
        if "evaluat" in stage or "validat" in stage:
            return RunStatus.EVALUATING
        if stage:
            return RunStatus.TRAINING
    if kind is RunKind.PREDICT:
        if "import" in stage:
            return RunStatus.IMPORTING
        if stage:
            return RunStatus.INFERENCING
    if kind is RunKind.DEPLOY and stage:
        return RunStatus.EVALUATING
    return None


def _deployment_stage_status(stage: str) -> str | None:
    if not stage:
        return None
    if "export" in stage:
        return "exporting"
    if "validat" in stage or "calibration" in stage:
        return "validating"
    if "quantiz" in stage:
        return "quantizing"
    if "convert" in stage or "compil" in stage:
        return "compiling"
    if "packag" in stage:
        return "packaging"
    if stage in {"started", "preflight"}:
        return "queued"
    return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return str(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _deployment_category_names(
    project: AnnotationProject,
    source_run: RunRecord,
) -> tuple[list[str], list[str], list[str]]:
    """Keep checkpoint labels frozen while using current canonical deploy labels.

    Training snapshots persist both the stable category ID and the name that
    was embedded in the checkpoint.  A later full category rename must not
    rewrite the historical snapshot/checkpoint, but every newly generated
    deployment package should expose the current canonical name.
    """

    current_by_id = {
        category.id: category
        for category in project.repository.list_categories()
    }
    snapshot_root = (
        Path(source_run.snapshot_path).resolve()
        if source_run.snapshot_path
        else None
    )
    classes: list[Mapping[str, Any]] = []
    legacy_checkpoint_names: list[str] = []
    if snapshot_root is not None and _is_within(snapshot_root, project.runs_dir):
        manifest_path = snapshot_root / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_classes = payload.get("classes", [])
            if isinstance(raw_classes, list) and all(
                isinstance(item, Mapping) for item in raw_classes
            ):
                classes = sorted(
                    raw_classes,
                    key=lambda item: int(item.get("index", 0)),
                )
                legacy_checkpoint_names = [
                    str(item.get("name") or "").strip() for item in classes
                ]
            elif isinstance(raw_classes, list) and all(
                isinstance(item, str) for item in raw_classes
            ):
                legacy_checkpoint_names = [str(item).strip() for item in raw_classes]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            classes = []

        if not legacy_checkpoint_names:
            legacy_checkpoint_names = _read_legacy_snapshot_class_names(
                snapshot_root / "data.yaml"
            )

    if classes and all(str(item.get("id") or "").strip() for item in classes):
        class_ids: list[str] = []
        checkpoint_names: list[str] = []
        deployment_names: list[str] = []
        for item in classes:
            category_id = str(item.get("id") or "").strip()
            checkpoint_name = str(item.get("name") or "").strip()
            category = current_by_id.get(category_id)
            if category is None:
                raise ValidationError(
                    f"训练快照中的类别已从当前项目删除：{category_id or checkpoint_name}"
                )
            if not checkpoint_name:
                raise ValidationError("训练快照包含空类别名称，无法安全生成部署包。")
            class_ids.append(category_id)
            checkpoint_names.append(checkpoint_name)
            deployment_names.append(category.name)
        return class_ids, checkpoint_names, deployment_names

    if legacy_checkpoint_names:
        class_ids = []
        deployment_names = []
        seen_ids: set[str] = set()
        for checkpoint_name in legacy_checkpoint_names:
            if not checkpoint_name:
                raise ValidationError("训练快照包含空类别名称，无法安全生成部署包。")
            category = project.repository.resolve_category_name(checkpoint_name)
            if category is None:
                raise ValidationError(
                    "旧训练快照类别无法映射到当前项目："
                    f"{checkpoint_name}。请恢复类别名称或重新训练模型。"
                )
            if category.id in seen_ids:
                raise ValidationError(
                    "旧训练快照中的多个类别映射到同一当前类别，无法安全部署："
                    f"{checkpoint_name}"
                )
            seen_ids.add(category.id)
            class_ids.append(category.id)
            deployment_names.append(category.name)
        return class_ids, legacy_checkpoint_names, deployment_names

    # Very old runs may not contain a snapshot manifest or data.yaml at all.
    # Keeping the historical fallback is safe only while no category has ever
    # been canonically renamed.  Once an import alias exists, guessing current
    # names as checkpoint names would make the worker's name gate fail (or,
    # worse, mislabel a deployment), so stop with an actionable preflight error.
    categories = list(project.repository.list_categories(enabled_only=True))
    if any(
        project.repository.list_category_name_aliases(category.id)
        for category in categories
    ):
        raise ValidationError(
            "该历史模型缺少可验证的类别快照，且项目类别已完整重命名；"
            "无法安全推断 checkpoint 类别顺序。请使用含 manifest.json/data.yaml "
            "的训练记录，或用当前数据重新训练后再部署。"
        )
    names = [category.name for category in categories]
    return [category.id for category in categories], names, list(names)


def _read_legacy_snapshot_class_names(path: Path) -> list[str]:
    """Read ordered checkpoint labels from an old Ultralytics data.yaml."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return []
    if not isinstance(payload, Mapping):
        return []
    raw_names = payload.get("names")
    if isinstance(raw_names, list):
        return [str(item).strip() for item in raw_names]
    if isinstance(raw_names, Mapping):
        try:
            ordered = sorted(raw_names.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            return []
        return [str(value).strip() for _, value in ordered]
    return []


def _freeze_calibration_sources(
    project: AnnotationProject,
    *,
    run_dir: Path,
    selected: Sequence[Any],
    fallback: Sequence[Any],
    required_count: int,
) -> dict[str, Any]:
    """Create a verified, immutable calibration snapshot before worker launch."""

    if required_count <= 0:
        raise ValidationError("校准图片数量必须大于 0。")
    snapshot_dir = run_dir / "calibration-source-snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    extra_limit = min(20, max(5, required_count // 5))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sources = [
        *(("selected", image) for image in selected),
        *(("fallback", image) for image in fallback),
    ]
    for candidate_index, (source_role, image) in enumerate(sources, start=1):
        if len(accepted) >= required_count + extra_limit:
            break
        source = project.image_path(image).resolve()
        suffix = source.suffix.casefold()
        temporary: Path | None = None
        destination: Path | None = None
        try:
            if not source.is_file() or source.is_symlink():
                raise ValueError("文件不存在或是符号链接")
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                raise ValueError(f"不支持的图片格式 {suffix or '<无>'}")
            before_hash = sha256_file(source)
            output_index = len(accepted) + 1
            destination = snapshot_dir / (
                f"{output_index:04d}-{before_hash[:16]}{suffix}"
            )
            temporary = snapshot_dir / f".{destination.name}.copying"
            shutil.copy2(source, temporary)
            with PillowImage.open(temporary) as opened:
                opened.verify()
            copied_hash = sha256_file(temporary)
            after_hash = sha256_file(source)
            if before_hash != copied_hash or before_hash != after_hash:
                raise ValueError("复制期间文件内容发生变化")
            os.replace(temporary, destination)
            if sha256_file(destination) != before_hash:
                raise ValueError("发布后的 SHA-256 校验失败")
            accepted.append(
                {
                    "image_id": image.id,
                    "original_name": image.original_name,
                    "path": str(destination),
                    "sha256": before_hash,
                    "source_path": str(source),
                    "source_role": source_role,
                    "replacement": (
                        source_role == "fallback" and len(accepted) < required_count
                    ),
                    "candidate_index": candidate_index,
                }
            )
        except (OSError, ValueError) as exc:
            rejected.append(
                {
                    "image_id": getattr(image, "id", None),
                    "original_name": getattr(image, "original_name", None),
                    "source_path": str(source),
                    "source_role": source_role,
                    "reason": str(exc),
                }
            )
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if destination is not None:
                destination.unlink(missing_ok=True)
    if len(accepted) < required_count:
        details = "；".join(str(item["reason"]) for item in rejected[-3:])
        raise ValidationError(
            f"可用校准图片不足：需要 {required_count} 张，快照成功 {len(accepted)} 张"
            + (f"；最近失败：{details}" if details else "")
        )
    selected_records = accepted[:required_count]
    candidate_records = accepted[required_count:]
    manifest_path = run_dir / "calibration-source-snapshot.json"
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "project_id": project.config.project_id,
            "required_count": required_count,
            "selected": selected_records,
            "candidates": candidate_records,
            "rejected": rejected,
        },
    )
    return {
        "selected": selected_records,
        "candidates": candidate_records,
        "rejected": rejected,
        "manifest_path": str(manifest_path),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _powershell_executable() -> str:
    system_root = Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
    executable = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(executable if executable.is_file() else "powershell.exe")
