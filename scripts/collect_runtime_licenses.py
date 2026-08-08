"""Collect license/notice files from distributions bundled by Nuitka."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

BASE_DISTRIBUTIONS = (
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
    "torch",
    "torchvision",
    "ultralytics",
    "numpy",
    "opencv-python",
    "albumentations",
    "onnx",
    "onnxruntime-gpu",
    "onnxsim",
    "onnxslim",
    "Pillow",
    "PyYAML",
    "platformdirs",
    "psutil",
    "pynvml",
    "GitPython",
    # The frozen build explicitly excludes IPython.  Legacy YOLOv5 receives
    # the small desktop-only compatibility shim in ai_biaozhu.ml instead.
    "matplotlib",
    "pandas",
    "requests",
    "scipy",
    "seaborn",
    "tensorboard",
    # The import package is ``thop`` but its installed distribution is named
    # ``ultralytics-thop``.
    "ultralytics-thop",
    "tqdm",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nuitka-report", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    distribution_names = set(BASE_DISTRIBUTIONS)
    if args.nuitka_report is not None:
        distribution_names.update(_read_nuitka_distributions(args.nuitka_report))

    resolved: dict[str, tuple[metadata.Distribution, list[str]]] = {}
    for requested_name in sorted(distribution_names, key=str.casefold):
        distribution = _find_distribution(requested_name)
        if distribution is None:
            raise SystemExit(f"Missing bundled distribution: {requested_name}")
        canonical_name = str(distribution.metadata.get("Name") or requested_name)
        key = re.sub(r"[-_.]+", "-", canonical_name).casefold()
        existing = resolved.get(key)
        if existing is not None:
            existing[1].append(requested_name)
            continue
        resolved[key] = (distribution, [requested_name])

    index: list[dict[str, object]] = []
    for distribution, requested_names in sorted(
        resolved.values(),
        key=lambda item: str(item[0].metadata.get("Name") or item[1][0]).casefold(),
    ):
        canonical_name = str(distribution.metadata.get("Name") or requested_names[0])
        destination = output / _safe_name(canonical_name)
        copied: list[dict[str, str]] = []
        for package_path in distribution.files or ():
            if not _is_notice_file(package_path.name):
                continue
            source = Path(distribution.locate_file(package_path))
            if not source.is_file():
                continue
            relative = Path(*(_safe_name(part) for part in package_path.parts))
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            copied.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _file_sha256(target),
                }
            )
        if not copied:
            destination.mkdir(parents=True, exist_ok=True)
            generated = destination / "DECLARED_LICENSE_METADATA.txt"
            generated.write_text(
                _declared_license_text(distribution, canonical_name),
                encoding="utf-8",
            )
            copied.append(
                {
                    "path": generated.name,
                    "sha256": _file_sha256(generated),
                }
            )
        index.append(
            {
                "name": canonical_name,
                "requested_names": sorted(requested_names, key=str.casefold),
                "version": distribution.version,
                "license_expression": distribution.metadata.get("License-Expression"),
                "declared_license": distribution.metadata.get("License"),
                "home_page": distribution.metadata.get("Home-page"),
                "project_urls": distribution.metadata.get_all("Project-URL") or [],
                "wheel_supplied_license_files": any(
                    item["path"] != "DECLARED_LICENSE_METADATA.txt" for item in copied
                ),
                "files": sorted(copied, key=lambda item: item["path"]),
            }
        )

    (output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _find_distribution(name: str) -> metadata.Distribution | None:
    """Resolve normal and setuptools-vendored distribution metadata.

    Conda's setuptools package keeps several dependency ``.dist-info``
    directories below ``setuptools/_vendor`` instead of placing them at the
    site-packages root.  Nuitka correctly records those vendored distribution
    names, while ``importlib.metadata.distribution`` cannot see them on the
    normal search path.  Read their real metadata and license files from the
    vendor directory rather than silently dropping them from the notice set.
    """

    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        pass

    requested_key = _canonical_distribution_name(name)
    try:
        setuptools_distribution = metadata.distribution("setuptools")
    except metadata.PackageNotFoundError:
        return None
    vendor_root = Path(
        setuptools_distribution.locate_file("setuptools/_vendor")
    ).resolve()
    if not vendor_root.is_dir():
        return None
    for candidate in metadata.distributions(path=[str(vendor_root)]):
        candidate_name = str(candidate.metadata.get("Name") or "")
        if _canonical_distribution_name(candidate_name) == requested_key:
            return candidate
    return None


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _read_nuitka_distributions(report: Path) -> set[str]:
    report = report.resolve()
    if not report.is_file():
        raise SystemExit(f"Nuitka report does not exist: {report}")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as exc:
        # Nuitka 4.1.3 writes ``encoding='utf8'`` in its report declaration.
        # Expat accepts the ASCII-only report until it reaches the first
        # non-ASCII path, then rejects the otherwise valid UTF-8 bytes.  This
        # happens for this project because its Windows source path contains
        # Chinese characters.  Normalize only that declaration and keep the
        # original parse error for every other kind of malformed report.
        report_bytes = report.read_bytes()
        normalized = report_bytes.replace(
            b"encoding='utf8'", b"encoding='utf-8'", 1
        ).replace(b'encoding="utf8"', b'encoding="utf-8"', 1)
        if normalized == report_bytes:
            raise SystemExit(
                f"Invalid Nuitka report XML: {report}: {exc}"
            ) from exc
        try:
            root = ET.fromstring(normalized)
        except ET.ParseError as normalized_exc:
            raise SystemExit(
                f"Invalid Nuitka report XML: {report}: {normalized_exc}"
            ) from normalized_exc
    result: set[str] = set()
    for element_name in ("distribution-usage", "included_metadata"):
        for element in root.iter(element_name):
            name = str(element.attrib.get("name") or "").strip()
            if name:
                result.add(name)
    return result


def _is_notice_file(name: str) -> bool:
    lowered = name.casefold()
    return any(
        token in lowered for token in ("license", "licence", "copying", "notice")
    )


def _safe_name(value: str) -> str:
    safe = "".join(
        character
        if character.isalnum() or character in {".", "-", "_"}
        else "_"
        for character in value
    )
    return "_" if safe in {"", ".", ".."} else safe


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_license_text(
    distribution: metadata.Distribution,
    canonical_name: str,
) -> str:
    lines = [
        "NO LICENSE/COPYING/NOTICE FILE WAS PRESENT IN THE INSTALLED DISTRIBUTION.",
        "This generated file records package metadata; it is not a substitute for",
        "the upstream license text. Review this component before release.",
        "",
        f"Name: {canonical_name}",
        f"Version: {distribution.version}",
        "License-Expression: "
        + str(distribution.metadata.get("License-Expression") or "NOT PROVIDED"),
        "License: " + str(distribution.metadata.get("License") or "NOT PROVIDED"),
        "Home-page: " + str(distribution.metadata.get("Home-page") or "NOT PROVIDED"),
    ]
    for project_url in distribution.metadata.get_all("Project-URL") or ():
        lines.append(f"Project-URL: {project_url}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
