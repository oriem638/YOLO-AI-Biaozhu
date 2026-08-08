"""Export a compact, verifiable inventory of every installed Python distribution."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packages = sorted(
        (_distribution_inventory(distribution) for distribution in metadata.distributions()),
        key=lambda item: (str(item["name"]).casefold(), str(item["version"])),
    )
    document = {
        "schema_version": 1,
        "purpose": (
            "Installed-environment audit. RECORD entries contain the hashes supplied "
            "by each installed wheel; artifact hashes are exported separately when "
            "the release wheelhouse is materialized."
        ),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "distributions": packages,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _distribution_inventory(distribution: metadata.Distribution) -> dict[str, Any]:
    name = str(distribution.metadata.get("Name") or "UNKNOWN")
    record_path = _record_path(distribution)
    entries: list[tuple[str, str, str]] = []
    missing_hashes = 0
    unsafe_paths: list[str] = []
    if record_path is not None:
        with record_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                path = row[0].replace("\\", "/")
                digest = row[1] if len(row) > 1 else ""
                size = row[2] if len(row) > 2 else ""
                pure = PurePosixPath(path)
                if pure.is_absolute() or ".." in pure.parts:
                    unsafe_paths.append(path)
                if not digest:
                    missing_hashes += 1
                entries.append((path, digest, size))
    entries.sort()
    record_root = hashlib.sha256()
    for path, digest, size in entries:
        record_root.update(path.encode("utf-8", errors="surrogateescape"))
        record_root.update(b"\0")
        record_root.update(digest.encode("ascii", errors="replace"))
        record_root.update(b"\0")
        record_root.update(size.encode("ascii", errors="replace"))
        record_root.update(b"\n")

    direct_url = _direct_url(distribution)
    return {
        "name": name,
        "version": distribution.version,
        "license_expression": distribution.metadata.get("License-Expression"),
        "direct_url": direct_url,
        "record": (
            {
                "path": f"{record_path.parent.name}/{record_path.name}",
                "sha256": _file_sha256(record_path),
                "entry_count": len(entries),
                "entries_root_sha256": record_root.hexdigest(),
                "entries_without_hash": missing_hashes,
                "unsafe_paths": unsafe_paths,
            }
            if record_path is not None
            else None
        ),
    }


def _record_path(distribution: metadata.Distribution) -> Path | None:
    for item in distribution.files or ():
        normalized = item.as_posix().casefold()
        if normalized.endswith(".dist-info/record"):
            candidate = Path(distribution.locate_file(item))
            return candidate.resolve() if candidate.is_file() else None
    return None


def _direct_url(distribution: metadata.Distribution) -> dict[str, Any] | None:
    try:
        text = distribution.read_text("direct_url.json")
    except (FileNotFoundError, OSError):
        return None
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"invalid_json_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if not isinstance(value, dict):
        return {"invalid_type": type(value).__name__}
    # Do not leak a local user path into a redistributable lock inventory.
    url = str(value.get("url") or "")
    if url.startswith("file:"):
        value["url"] = "file:<local-path-redacted>"
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _urlsafe_sha256(value: str) -> bytes | None:
    """Decode a RECORD sha256 entry; retained for lock-audit callers."""

    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256":
        return None
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))


if __name__ == "__main__":
    raise SystemExit(main())
