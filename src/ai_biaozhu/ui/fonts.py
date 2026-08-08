"""Application font selection and diagnostics.

The application runs on Windows installations with different CJK font sets.
The optional Noto Sans SC asset is loaded only when it is actually bundled, so
source checkouts remain usable when the large font file is not present.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QRawFont
from PySide6.QtWidgets import QApplication

_BUNDLED_FONT_NAME = "NotoSansSC[wght].ttf"
_BUNDLED_FONT_SHA256 = (
    "763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a"
)
_REQUIRED_GLYPHS = "训练标注✓—"
_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Noto Sans SC",
    "DengXian",
    "Segoe UI",
)


@dataclass(frozen=True)
class FontDiagnostics:
    """The chosen family and glyph coverage recorded at application startup."""

    family: str
    source: str
    missing_glyphs: str
    bundled_font_loaded: bool

    @property
    def supports_required_glyphs(self) -> bool:
        return not self.missing_glyphs


def bundled_font_path() -> Path:
    """Return the optional packaged Noto font location without requiring it."""

    return Path(__file__).resolve().parent.parent / "assets" / "fonts" / _BUNDLED_FONT_NAME


def configure_application_font(
    application: QApplication, *, logger: logging.Logger | None = None
) -> FontDiagnostics:
    """Choose a 10pt UI font with verified Chinese and annotation glyphs."""

    log = logger or logging.getLogger(__name__)
    bundled_loaded = _load_bundled_font(log)
    available_families = set(QFontDatabase.families())

    selected_family = ""
    selected_source = "system fallback"
    selected_missing = _REQUIRED_GLYPHS
    for family in _FONT_CANDIDATES:
        if family not in available_families:
            continue
        missing = _missing_glyphs(family)
        if not selected_family:
            selected_family = family
            selected_source = "bundled" if family == "Noto Sans SC" and bundled_loaded else "system"
            selected_missing = missing
        if not missing:
            selected_family = family
            selected_source = "bundled" if family == "Noto Sans SC" and bundled_loaded else "system"
            selected_missing = ""
            break

    if not selected_family:
        selected_family = application.font().family()
        selected_missing = _missing_glyphs(selected_family)
        log.warning("No preferred UI font is installed; using Qt default family %s", selected_family)

    font = QFont(selected_family)
    font.setPointSize(10)
    application.setFont(font)
    diagnostics = FontDiagnostics(
        family=selected_family,
        source=selected_source,
        missing_glyphs=selected_missing,
        bundled_font_loaded=bundled_loaded,
    )
    if diagnostics.supports_required_glyphs:
        log.info(
            "UI font: %s (%s; bundled=%s); required glyph coverage verified",
            diagnostics.family,
            diagnostics.source,
            diagnostics.bundled_font_loaded,
        )
    else:
        log.warning(
            "UI font: %s (%s) is missing glyphs: %s; bundled Noto asset=%s",
            diagnostics.family,
            diagnostics.source,
            diagnostics.missing_glyphs,
            bundled_font_path(),
        )
    return diagnostics


def _load_bundled_font(log: logging.Logger) -> bool:
    path = bundled_font_path()
    if not path.is_file():
        log.info("Optional bundled CJK font is not present: %s", path)
        return False
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        log.warning("Unable to verify bundled CJK font %s: %s", path, exc)
        return False
    if digest != _BUNDLED_FONT_SHA256:
        log.error(
            "Bundled CJK font checksum mismatch: expected %s, got %s (%s)",
            _BUNDLED_FONT_SHA256,
            digest,
            path,
        )
        return False
    font_id = QFontDatabase.addApplicationFont(str(path))
    if font_id < 0:
        log.warning("Unable to load bundled CJK font: %s", path)
        return False
    families = QFontDatabase.applicationFontFamilies(font_id)
    if "Noto Sans SC" not in families:
        log.warning("Bundled font has unexpected family names %s: %s", families, path)
    return True


def _missing_glyphs(family: str) -> str:
    if not family:
        return _REQUIRED_GLYPHS
    raw_font = QRawFont.fromFont(QFont(family, 10))
    if not raw_font.isValid():
        return _REQUIRED_GLYPHS
    missing: list[str] = []
    for char in _REQUIRED_GLYPHS:
        # Some Qt/DirectWrite combinations incorrectly return False from
        # supportsCharacter() for a variable CJK glyph even though shaping
        # resolves a valid non-zero glyph index. Require both checks to fail.
        indexes = raw_font.glyphIndexesForString(char)
        if not raw_font.supportsCharacter(char) and (
            not indexes or indexes[0] == 0
        ):
            missing.append(char)
    return "".join(missing)
