"""Validated image ingestion with copying, EXIF normalization, and hash deduplication."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from ai_biaozhu.core.domain import ImportFailure, ImportReport
from ai_biaozhu.core.exceptions import DataIntegrityError

from .repository import AnnotationRepository
from .utils import sha256_file, utc_now, write_json

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_CANONICAL_SUFFIX = {
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".png": ".png",
    ".bmp": ".bmp",
    ".webp": ".webp",
}
_SAVE_FORMAT = {
    ".jpg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".webp": "WEBP",
}


class ImageImporter:
    def __init__(
        self,
        repository: AnnotationRepository,
        *,
        project_root: Path,
        images_dir: Path,
        reports_dir: Path,
    ) -> None:
        self.repository = repository
        self.project_root = Path(project_root)
        self.images_dir = Path(images_dir)
        self.reports_dir = Path(reports_dir)

    def import_paths(
        self,
        paths: Iterable[Path | str],
        *,
        recursive: bool = True,
    ) -> ImportReport:
        candidates, initial_failures = _expand_paths(paths, recursive=recursive)
        imported = []
        duplicates: list[Path] = []
        failures = list(initial_failures)

        for source in candidates:
            try:
                record = self._import_one(source)
            except DuplicateImageError:
                duplicates.append(source)
            except Exception as exc:
                failures.append(ImportFailure(source, _safe_error(exc)))
            else:
                imported.append(record)

        report_path = self._write_report(
            requested=len(candidates) + len(initial_failures),
            imported=imported,
            duplicates=duplicates,
            failures=failures,
        )
        return ImportReport(
            requested=len(candidates) + len(initial_failures),
            imported=tuple(imported),
            duplicate_paths=tuple(duplicates),
            failures=tuple(failures),
            report_path=report_path,
        )

    def _import_one(self, source: Path):
        source = source.resolve()
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片格式：{source.suffix or '(无扩展名)'}")
        source_hash = sha256_file(source)

        # Decode before reporting a duplicate so a damaged file is never accepted
        # solely because it happens to match a stale database hash.
        normalized, width, height, save_options = _load_normalized_image(source)
        try:
            if self.repository.find_image_by_sha256(source_hash) is not None:
                raise DuplicateImageError

            image_id = uuid4().hex
            canonical_suffix = _CANONICAL_SUFFIX[suffix]
            final_path = self.images_dir / f"{image_id}{canonical_suffix}"
            temporary_path = (
                self.images_dir / f".{image_id}.importing{canonical_suffix}"
            )
            self.images_dir.mkdir(parents=True, exist_ok=True)
            try:
                normalized.save(
                    temporary_path,
                    format=_SAVE_FORMAT[canonical_suffix],
                    **save_options,
                )
                # Re-open the encoded file so a filesystem/encoder problem cannot
                # produce a database record that points at an unreadable image.
                with Image.open(temporary_path) as check:
                    check.load()
                    if check.size != (width, height):
                        raise OSError("保存后的图片尺寸与解码结果不一致")
                temporary_path.replace(final_path)
                relative_path = final_path.relative_to(self.project_root).as_posix()
                try:
                    return self.repository.add_image_record(
                        image_id=image_id,
                        relative_path=relative_path,
                        original_name=source.name,
                        source_path=str(source),
                        sha256=source_hash,
                        width=width,
                        height=height,
                    )
                except DataIntegrityError as exc:
                    final_path.unlink(missing_ok=True)
                    if self.repository.find_image_by_sha256(source_hash) is not None:
                        raise DuplicateImageError from exc
                    raise
            finally:
                temporary_path.unlink(missing_ok=True)
        finally:
            normalized.close()

    def _write_report(
        self,
        *,
        requested: int,
        imported: list,
        duplicates: list[Path],
        failures: list[ImportFailure],
    ) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now().replace(":", "").replace("-", "")
        path = self.reports_dir / f"import-{timestamp}-{uuid4().hex[:8]}.json"
        write_json(
            path,
            {
                "format": "ai-biaozhu-import-report",
                "created_at": utc_now(),
                "requested": requested,
                "imported": [
                    {
                        "image_id": record.id,
                        "original_name": record.original_name,
                        "relative_path": record.relative_path,
                        "sha256": record.sha256,
                        "width": record.width,
                        "height": record.height,
                    }
                    for record in imported
                ],
                "duplicates": [str(value) for value in duplicates],
                "failures": [
                    {"path": str(failure.path), "reason": failure.reason}
                    for failure in failures
                ],
            },
        )
        return path


class DuplicateImageError(Exception):
    pass


def _expand_paths(
    paths: Iterable[Path | str], *, recursive: bool
) -> tuple[list[Path], list[ImportFailure]]:
    candidates: list[Path] = []
    failures: list[ImportFailure] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            failures.append(ImportFailure(path, "文件或目录不存在"))
            continue
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for child in iterator:
                if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        candidates.append(resolved)
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return candidates, failures


def _load_normalized_image(
    source: Path,
) -> tuple[Image.Image, int, int, dict[str, object]]:
    try:
        with Image.open(source) as opened:
            opened.load()
            normalized = ImageOps.exif_transpose(opened).copy()
            suffix = _CANONICAL_SUFFIX[source.suffix.lower()]
            if (
                suffix == ".jpg"
                and normalized.mode not in ("L", "RGB", "CMYK")
                or suffix == ".bmp"
                and normalized.mode not in ("1", "L", "P", "RGB")
            ):
                normalized = normalized.convert("RGB")
            elif suffix == ".webp" and normalized.mode not in ("RGB", "RGBA"):
                normalized = normalized.convert(
                    "RGBA" if "A" in normalized.getbands() else "RGB"
                )

            options: dict[str, object] = {}
            if suffix == ".jpg":
                options.update(quality=95, subsampling=0)
            elif suffix == ".webp":
                options.update(quality=95, method=4)

            exif = normalized.getexif()
            exif.pop(274, None)  # orientation is now physically applied
            if exif and suffix in {".jpg", ".png", ".webp"}:
                options["exif"] = exif.tobytes()
            if opened.info.get("icc_profile"):
                options["icc_profile"] = opened.info["icc_profile"]
            width, height = normalized.size
            if width <= 0 or height <= 0:
                normalized.close()
                raise OSError("图片尺寸无效")
            return normalized, width, height, options
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"图片损坏或无法解码：{exc}") from exc


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, OSError | ValueError | DataIntegrityError | sqlite3.Error):
        return str(exc) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {exc}"
