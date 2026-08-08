"""Create a pip ``--require-hashes`` lock from a materialized Windows wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--expected-requirements", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    wheels: dict[str, dict[str, object]] = {}
    artifacts: list[dict[str, object]] = []
    for path in sorted(args.wheelhouse.resolve().glob("*.whl")):
        name, version, build, tags = parse_wheel_filename(path.name)
        canonical = canonicalize_name(name)
        digest = _file_sha256(path)
        item = {
            "filename": path.name,
            "name": canonical,
            "version": str(version),
            "build": list(build) if build else None,
            "tags": sorted(str(tag) for tag in tags),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        artifacts.append(item)
        existing = wheels.get(canonical)
        if existing is not None and existing["version"] != str(version):
            raise SystemExit(
                f"wheelhouse contains multiple versions of {canonical}: "
                f"{existing['version']} and {version}"
            )
        if existing is None:
            wheels[canonical] = {"version": str(version), "hashes": []}
        hashes = wheels[canonical]["hashes"]
        assert isinstance(hashes, list)
        hashes.append(digest)

    expected = _read_expected(args.expected_requirements)
    missing = sorted(name for name in expected if name not in wheels)
    if missing:
        raise SystemExit("wheelhouse is missing exact requirements: " + ", ".join(missing))
    mismatches = sorted(
        f"{name}: expected {version}, found {wheels[name]['version']}"
        for name, version in expected.items()
        if str(wheels[name]["version"]) != version
    )
    if mismatches:
        raise SystemExit("wheelhouse version mismatches:\n" + "\n".join(mismatches))

    lines = [
        "# Generated from a materialized Windows x64 wheelhouse.",
        "# Install with: pip install --no-index --find-links <wheelhouse> \\",
        "#   --require-hashes -r requirements-win-64.lock",
    ]
    for name, item in sorted(wheels.items()):
        hashes = sorted(set(str(value) for value in item["hashes"]))
        continuation = " \\\n    ".join(f"--hash=sha256:{value}" for value in hashes)
        lines.append(f"{name}=={item['version']} \\\n    {continuation}")
    args.lock_output.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.manifest_output.resolve().write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_count": len(artifacts),
                "total_bytes": sum(int(item["size"]) for item in artifacts),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _read_expected(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        pins = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator == "=="
        ]
        if len(pins) != 1:
            raise SystemExit(f"requirement is not exactly pinned: {line}")
        expected[canonicalize_name(requirement.name)] = pins[0]
    return expected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
