"""Install, inspect, probe and uninstall a Windows release installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
import re
import stat
import subprocess
import time
import winreg
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

REQUIRED_FILES = (
    "AI-Biaozhu.exe",
    "AI-Biaozhu-Worker.exe",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_LICENSES/index.json",
    "model-seed/yolo26s.pt",
    "third_party/yolov5/.ai-biaozhu-yolov5-tag",
    "unins000.exe",
)
FORBIDDEN_SUFFIXES = {".pt", ".onnx", ".engine", ".pyc", ".pyo"}
MODEL_SEED_RELATIVE = "model-seed/yolo26s.pt"
MODEL_SEED_SIZE = 20_422_725
MODEL_SEED_SHA256 = (
    "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
)
FORBIDDEN_DIRECTORIES = {".git", "__pycache__"}
DEFAULT_APP_ID = "{4C9330ED-77CB-4F81-A467-06B4D6A8FB2B}"
MAINTENANCE_APP_NAME = "AI Biaozhu Maintenance 0.2"
ORIGINAL_APP_NAME = "AI Biaozhu"
APP_ID_PATTERN = re.compile(
    r"^\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}"
    r"-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}$"
)
UNINSTALL_KEY_ROOT = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
SANDBOX_NO_UNINSTALL_REGISTRY_PARAMETER = (
    "/AI-BIAOZHU-SANDBOX-NO-UNINSTALL-REGISTRY"
)
WINDOWS_ACCESS_DENIED_PATTERN = re.compile(
    r"(?m)^PermissionError:\s*\[WinError 5\][^\r\n]*?"
    r"['\"](?P<path>[A-Za-z]:\\[^'\"\r\n]+)['\"]\s*$",
    re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installer", required=True, type=Path)
    parser.add_argument("--standalone-root", required=True, type=Path)
    parser.add_argument("--ml-python", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--timeout", default=1800, type=float)
    parser.add_argument("--allow-sandbox-gui-skip", action="store_true")
    parser.add_argument("--sandbox-no-uninstall-registry", action="store_true")
    parser.add_argument(
        "--app-id",
        default=DEFAULT_APP_ID,
        help="Inno Setup AppId to verify; defaults to the maintenance-edition identity.",
    )
    return parser


def _clean_environment(runtime_root: Path) -> dict[str, str]:
    removed = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    env = {key: value for key, value in os.environ.items() if key.upper() not in removed}
    env.update(
        {
            "AI_BIAOZHU_STANDALONE": "1",
            "AI_BIAOZHU_MODELS_DIR": str(runtime_root / "models"),
            "YOLO_CONFIG_DIR": str(runtime_root / "yolo-config"),
            "PYTHONUTF8": "1",
        }
    )
    return env


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _is_current_user_local_appdata_path(
    candidate: str,
    *,
    local_appdata: str | Path | None = None,
) -> bool:
    """Return whether an absolute Windows path is inside current Local AppData."""

    candidate_path = PureWindowsPath(candidate)
    if not candidate_path.is_absolute():
        return False
    if local_appdata is None:
        local_appdata = os.environ.get("LOCALAPPDATA")
        if not local_appdata:
            local_appdata = Path.home() / "AppData" / "Local"
    root_path = PureWindowsPath(str(local_appdata))
    if not root_path.is_absolute():
        return False
    normalized_candidate = ntpath.normcase(ntpath.normpath(str(candidate_path)))
    normalized_root = ntpath.normcase(ntpath.normpath(str(root_path)))
    try:
        return ntpath.commonpath([normalized_candidate, normalized_root]) == normalized_root
    except ValueError:
        return False


def _is_known_codex_sandbox_gui_block(
    stderr: str,
    *,
    local_appdata: str | Path | None = None,
) -> bool:
    """Recognize only a WinError 5 targeting current-user Local AppData."""

    return any(
        _is_current_user_local_appdata_path(
            match.group("path"),
            local_appdata=local_appdata,
        )
        for match in WINDOWS_ACCESS_DENIED_PATTERN.finditer(stderr)
    )


def _gui_probe(
    executable: Path,
    *,
    smoke_root: Path,
    env: dict[str, str],
    seconds: float = 8,
) -> dict[str, Any]:
    gui_env = dict(env)
    isolated_profile = smoke_root / "gui-profile"
    isolated_profile.mkdir(parents=True)
    # Windows platformdirs can resolve the user's real Local AppData through
    # shell APIs even when APPDATA/LOCALAPPDATA are overridden.  Supplying an
    # explicit (deliberately absent) project prevents the application from
    # reopening and migrating the user's remembered real project during an
    # installer smoke run.  The startup warning is harmless in offscreen mode;
    # the main window and Qt event loop are still constructed and exercised.
    isolated_project = smoke_root / "gui-smoke-project-do-not-open-last"
    if isolated_project.exists():
        raise RuntimeError(f"GUI smoke project path unexpectedly exists: {isolated_project}")
    gui_env.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "APPDATA": str(isolated_profile / "Roaming"),
            "LOCALAPPDATA": str(isolated_profile / "Local"),
        }
    )
    stdout_path = smoke_root / "gui-stdout.log"
    stderr_path = smoke_root / "gui-stderr.log"
    started = time.monotonic()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            [str(executable), "--project", str(isolated_project)],
            cwd=smoke_root,
            env=gui_env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        deadline = started + seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)
        survived = process.poll() is None
        if survived:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    sandbox_blocked = not survived and _is_known_codex_sandbox_gui_block(stderr)
    return {
        "status": ("passed" if survived else "sandbox_blocked" if sandbox_blocked else "failed"),
        "survived_seconds": round(time.monotonic() - started, 3),
        "return_code": process.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "isolated_project": str(isolated_project),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _tree_signature(root: Path, *, installed: bool = False) -> dict[str, str]:
    if _is_reparse_point(root):
        raise RuntimeError(f"runtime root is a reparse point: {root}")
    signature: dict[str, str] = {}
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise RuntimeError(f"reparse point found: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if installed and relative.casefold() in {"unins000.exe", "unins000.dat"}:
            continue
        key = relative.casefold()
        if key in signature:
            raise RuntimeError(f"case-insensitive path collision: {relative}")
        signature[key] = _file_sha256(path)
    return signature


def _verify_model_seed(root: Path) -> None:
    seed = root / MODEL_SEED_RELATIVE
    _require(seed.is_file(), f"verified model seed missing: {seed}")
    _require(
        seed.stat().st_size == MODEL_SEED_SIZE
        and _file_sha256(seed) == MODEL_SEED_SHA256,
        "installed yolo26s.pt seed does not match its locked size/SHA-256",
    )


def _signature_sha256(signature: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_hash in sorted(signature.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _uninstall_key(app_id: str) -> str:
    normalized = app_id.strip()
    if not APP_ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"invalid Inno Setup AppId: {app_id!r}")
    return f"{UNINSTALL_KEY_ROOT}\\{normalized}_is1"


def _registered_installations(app_id: str = DEFAULT_APP_ID) -> list[dict[str, str]]:
    """Return existing registrations for a specific Inno Setup AppId.

    A smoke run must never overwrite or uninstall a user's real installation.
    Query all registry views before invoking an installer that uses
    ``UsePreviousAppDir=yes``.
    """

    uninstall_key = _uninstall_key(app_id)
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    roots = (("HKCU", winreg.HKEY_CURRENT_USER), ("HKLM", winreg.HKEY_LOCAL_MACHINE))
    views = (
        ("native", 0),
        ("64", winreg.KEY_WOW64_64KEY),
        ("32", winreg.KEY_WOW64_32KEY),
    )
    for root_name, root in roots:
        for view_name, view_flag in views:
            try:
                with winreg.OpenKey(
                    root,
                    uninstall_key,
                    0,
                    winreg.KEY_READ | view_flag,
                ) as key:
                    try:
                        install_location = str(winreg.QueryValueEx(key, "InstallLocation")[0])
                    except FileNotFoundError:
                        install_location = ""
            except FileNotFoundError:
                continue
            except OSError as exc:
                results.append(
                    {
                        "hive": root_name,
                        "view": view_name,
                        "install_location": "",
                        "probe_error": str(exc),
                    }
                )
                continue
            identity = (root_name, view_name, install_location)
            if identity not in seen:
                seen.add(identity)
                results.append(
                    {
                        "hive": root_name,
                        "view": view_name,
                        "install_location": install_location,
                    }
                )
    return results


def _actual_registrations(
    registrations: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [item for item in registrations if not item.get("probe_error")]


def _registry_probe_errors(
    registrations: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [item for item in registrations if item.get("probe_error")]


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _user_shell_folder(value_name: str, fallback: Path) -> Path:
    key_path = (
        r"Software\Microsoft\Windows\CurrentVersion"
        r"\Explorer\User Shell Folders"
    )
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            raw = str(winreg.QueryValueEx(key, value_name)[0])
    except OSError:
        return fallback
    return Path(os.path.expandvars(raw))


def _shortcut_paths() -> dict[str, Path]:
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    programs = _user_shell_folder(
        "Programs",
        appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs",
    )
    desktop = _user_shell_folder("Desktop", Path.home() / "Desktop")
    # Track both products.  A /NOICONS maintenance smoke install must neither
    # create its own shortcuts nor overwrite the original edition's shortcuts.
    return {
        "maintenance_start_menu": (
            programs / MAINTENANCE_APP_NAME / f"{MAINTENANCE_APP_NAME}.lnk"
        ),
        "maintenance_desktop": desktop / f"{MAINTENANCE_APP_NAME}.lnk",
        "original_start_menu": programs / ORIGINAL_APP_NAME / f"{ORIGINAL_APP_NAME}.lnk",
        "original_desktop": desktop / f"{ORIGINAL_APP_NAME}.lnk",
    }


def _shortcut_state() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, path in _shortcut_paths().items():
        if path.is_file():
            result[role] = {
                "path": str(path),
                "exists": True,
                "sha256": _file_sha256(path),
            }
        else:
            result[role] = {
                "path": str(path),
                "exists": False,
                "sha256": None,
            }
    return result


def main() -> int:
    args = build_parser().parse_args()
    started_at = datetime.now(UTC).isoformat()
    summary_path = args.summary.resolve() if args.summary is not None else None
    running_summary = {
        "status": "running",
        "started_at": started_at,
        "installer": str(args.installer.resolve()),
        "app_id": args.app_id,
        "sandbox_no_uninstall_registry": bool(
            args.sandbox_no_uninstall_registry
        ),
    }
    if summary_path is not None:
        _write_json(summary_path, running_summary)
    try:
        return _execute(args, started_at=started_at)
    except BaseException as exc:
        failed_summary = dict(running_summary)
        if summary_path is not None and summary_path.is_file():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    failed_summary.update(existing)
            except (json.JSONDecodeError, OSError):
                pass
        failed_summary.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "outer_error": f"{type(exc).__name__}: {exc}",
            }
        )
        if summary_path is not None:
            _write_json(summary_path, failed_summary)
        raise


def _execute(args: argparse.Namespace, *, started_at: str) -> int:
    installer = args.installer.resolve()
    standalone_root = args.standalone_root.resolve()
    ml_python = args.ml_python.resolve()
    _uninstall_key(args.app_id)
    _require(installer.is_file(), f"installer missing: {installer}")
    _require(standalone_root.is_dir(), f"standalone root missing: {standalone_root}")
    _require(ml_python.is_file(), f"ML Python missing: {ml_python}")
    _require(not _is_reparse_point(installer), f"installer is a reparse point: {installer}")
    _require(
        not _is_reparse_point(standalone_root),
        f"standalone root is a reparse point: {standalone_root}",
    )
    standalone_main = standalone_root / "AI-Biaozhu.exe"
    standalone_worker = standalone_root / "AI-Biaozhu-Worker.exe"
    _require(standalone_main.is_file(), f"standalone GUI missing: {standalone_main}")
    _require(standalone_worker.is_file(), f"standalone Worker missing: {standalone_worker}")
    _verify_model_seed(standalone_root)
    standalone_signature = _tree_signature(standalone_root)
    standalone_tree_sha256 = _signature_sha256(standalone_signature)
    installer_sha256 = _file_sha256(installer)
    standalone_hashes = {
        "AI-Biaozhu.exe": _file_sha256(standalone_main),
        "AI-Biaozhu-Worker.exe": _file_sha256(standalone_worker),
    }
    registrations_before = _registered_installations(args.app_id)
    actual_registrations_before = _actual_registrations(registrations_before)
    registry_errors_before = _registry_probe_errors(registrations_before)
    _require(
        args.sandbox_no_uninstall_registry or not actual_registrations_before,
        "refusing to replace an existing matching installation: "
        f"{actual_registrations_before}",
    )
    _require(
        args.sandbox_no_uninstall_registry or not registry_errors_before,
        f"uninstall registry could not be verified: {registry_errors_before}",
    )
    registry_validation = (
        "not_verified_sandbox_mode"
        if args.sandbox_no_uninstall_registry
        else "verified"
    )
    shortcuts_before = _shortcut_state()

    smoke_root = args.results_root.resolve() / ("install-smoke-" + uuid4().hex)
    install_root = smoke_root / "app"
    if args.sandbox_no_uninstall_registry:
        for registration in actual_registrations_before:
            location = registration.get("install_location")
            if location:
                _require(
                    not _paths_overlap(install_root, Path(location)),
                    "sandbox install root overlaps an existing installation: "
                    f"{location}",
                )
    smoke_root.mkdir(parents=True)
    install_log = smoke_root / "install.log"
    install: dict[str, Any] | None = None
    worker_help: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    completed: dict[str, Any] | None = None
    gui: dict[str, Any] | None = None
    uninstall: dict[str, Any] | None = None
    forbidden_files: list[str] = []
    forbidden_directories: list[str] = []
    installed_signature: dict[str, str] = {}
    installed_tree_sha256: str | None = None
    installed_hashes: dict[str, str] = {}
    registrations_during: list[dict[str, str]] = []
    registrations_after: list[dict[str, str]] = []
    shortcuts_during: dict[str, dict[str, Any]] = {}
    shortcuts_after: dict[str, dict[str, Any]] = {}
    failure: str | None = None
    cleanup_failure: str | None = None
    uninstall_log = smoke_root / "uninstall.log"
    try:
        install_command = [
            str(installer),
            "/SP-",
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/NOICONS",
            f"/DIR={install_root}",
            f"/LOG={install_log}",
        ]
        if args.sandbox_no_uninstall_registry:
            install_command.append(SANDBOX_NO_UNINSTALL_REGISTRY_PARAMETER)
        install = _run(
            install_command,
            cwd=smoke_root,
            timeout=args.timeout,
        )
        _require(install["return_code"] == 0, "silent installation failed")
        shortcuts_during = _shortcut_state()
        _require(
            shortcuts_during == shortcuts_before,
            "/NOICONS changed a Start Menu or desktop shortcut",
        )
        registrations_during = _registered_installations(args.app_id)
        actual_during = _actual_registrations(registrations_during)
        errors_during = _registry_probe_errors(registrations_during)
        if args.sandbox_no_uninstall_registry:
            _require(
                registrations_during == registrations_before,
                "sandbox installer changed an existing uninstall registry key",
            )
        else:
            _require(
                not errors_during and bool(actual_during),
                "normal installer did not create a verifiable uninstall registry key",
            )
            expected_install_root = install_root.resolve()
            _require(
                all(
                    item.get("install_location")
                    and Path(item["install_location"]).resolve()
                    == expected_install_root
                    for item in actual_during
                ),
                "normal installer registry points to a different installation",
            )
        missing = [name for name in REQUIRED_FILES if not (install_root / name).is_file()]
        _require(not missing, f"installed files missing: {missing}")

        for path in install_root.rglob("*"):
            relative = path.relative_to(install_root).as_posix()
            if path.is_dir() and path.name.casefold() in FORBIDDEN_DIRECTORIES:
                forbidden_directories.append(relative)
            elif (
                path.is_file()
                and path.suffix.casefold() in FORBIDDEN_SUFFIXES
                and relative.casefold() != MODEL_SEED_RELATIVE.casefold()
            ):
                forbidden_files.append(relative)
        _require(
            not forbidden_files,
            f"forbidden installed files: {forbidden_files[:10]}",
        )
        _require(
            not forbidden_directories,
            f"forbidden installed directories: {forbidden_directories[:10]}",
        )
        _verify_model_seed(install_root)
        installed_signature = _tree_signature(install_root, installed=True)
        installed_tree_sha256 = _signature_sha256(installed_signature)
        _require(
            installed_signature == standalone_signature,
            "installed runtime tree does not match the audited standalone tree",
        )
        installed_hashes = {
            "AI-Biaozhu.exe": _file_sha256(install_root / "AI-Biaozhu.exe"),
            "AI-Biaozhu-Worker.exe": _file_sha256(install_root / "AI-Biaozhu-Worker.exe"),
        }
        _require(
            installed_hashes == standalone_hashes,
            "installed entry-point hashes do not match the audited standalone build",
        )

        runtime_env = _clean_environment(smoke_root / "runtime")
        worker = install_root / "AI-Biaozhu-Worker.exe"
        worker_help = _run(
            [str(worker), "--help"],
            env=runtime_env,
            cwd=smoke_root,
            timeout=120,
        )
        _require(worker_help["return_code"] == 0, "installed worker --help failed")
        environment = _run(
            [
                str(worker),
                "environment",
                "--python",
                str(ml_python),
                "--job-id",
                "installed-env-smoke",
            ],
            env=runtime_env,
            cwd=smoke_root,
            timeout=300,
        )
        _require(environment["return_code"] == 0, "installed environment probe failed")
        events = []
        last_seq = -1
        for line in environment["stdout"].splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (
                isinstance(candidate, dict)
                and candidate.get("protocol_version") == "1.0"
                and candidate.get("job_id") == "installed-env-smoke"
                and isinstance(candidate.get("payload"), dict)
            ):
                continue
            seq = int(candidate.get("seq", -1))
            _require(seq > last_seq, "installed environment event sequence is invalid")
            last_seq = seq
            events.append(candidate)
        completed = events[-1] if events and events[-1].get("type") == "completed" else None
        _require(completed is not None, "installed environment completed event missing")
        _require(
            not any(event.get("type") in {"error", "cancelled"} for event in events),
            "installed environment protocol contains a failure event",
        )
        _require(bool(completed["payload"].get("gpu_ready")), "installed GPU probe failed")

        gui = _gui_probe(
            install_root / "AI-Biaozhu.exe",
            smoke_root=smoke_root,
            env=runtime_env,
        )
        _require(gui["status"] != "failed", "installed GUI exited unexpectedly")
        _require(
            gui["status"] == "passed"
            or args.allow_sandbox_gui_skip
            or args.sandbox_no_uninstall_registry,
            "installed GUI could not be verified in the current sandbox",
        )
    except Exception as exc:  # cleanup is mandatory after any installed-stage failure
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        uninstaller = install_root / "unins000.exe"
        if uninstaller.is_file():
            try:
                uninstall = _run(
                    [
                        str(uninstaller),
                        "/VERYSILENT",
                        "/SUPPRESSMSGBOXES",
                        "/NORESTART",
                        f"/LOG={uninstall_log}",
                    ],
                    cwd=smoke_root,
                    timeout=args.timeout,
                )
                if uninstall["return_code"] != 0:
                    cleanup_failure = "silent uninstall returned a non-zero exit code"
            except Exception as exc:  # preserve the original validation failure
                cleanup_failure = f"{type(exc).__name__}: {exc}"
        for _ in range(40):
            if not install_root.exists():
                break
            time.sleep(0.25)
        if install_root.exists() and cleanup_failure is None:
            cleanup_failure = "installation directory remains after uninstall"
        registrations_after = _registered_installations(args.app_id)
        actual_after = _actual_registrations(registrations_after)
        errors_after = _registry_probe_errors(registrations_after)
        if args.sandbox_no_uninstall_registry:
            if registrations_after != registrations_before and cleanup_failure is None:
                cleanup_failure = (
                    "sandbox install/uninstall changed the pre-existing "
                    "uninstall registry state"
                )
        else:
            if actual_after and cleanup_failure is None:
                cleanup_failure = f"uninstall registration remains: {actual_after}"
            if errors_after and cleanup_failure is None:
                cleanup_failure = (
                    f"uninstall registry cleanup could not be verified: {errors_after}"
                )
        shortcuts_after = _shortcut_state()
        if shortcuts_after != shortcuts_before and cleanup_failure is None:
            cleanup_failure = "shortcut state changed after /NOICONS install/uninstall"

    summary = {
        "status": (
            "failed"
            if failure or cleanup_failure
            else "partial_sandbox_validation"
            if args.sandbox_no_uninstall_registry
            else "passed"
            if gui is not None and gui["status"] == "passed"
            else "partial_sandbox_gui_unverified"
        ),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "smoke_root": str(smoke_root),
        "installer": str(installer),
        "app_id": args.app_id,
        "installer_sha256": installer_sha256,
        "sandbox_no_uninstall_registry": bool(
            args.sandbox_no_uninstall_registry
        ),
        "registry_validation": registry_validation,
        "standalone_root": str(standalone_root),
        "standalone_file_count": len(standalone_signature),
        "standalone_tree_sha256": standalone_tree_sha256,
        "standalone_entry_hashes": standalone_hashes,
        "installed_file_count": len(installed_signature),
        "installed_tree_sha256": installed_tree_sha256,
        "installed_entry_hashes": installed_hashes,
        "required_file_count": len(REQUIRED_FILES),
        "forbidden_file_count": len(forbidden_files),
        "forbidden_directory_count": len(forbidden_directories),
        "install": install,
        "worker_help": worker_help,
        "environment_completed": completed,
        "gui": gui,
        "uninstall": uninstall,
        "failure": failure,
        "cleanup_failure": cleanup_failure,
        "registrations_before": registrations_before,
        "registrations_during": registrations_during,
        "registrations_after": registrations_after,
        "shortcuts_before": shortcuts_before,
        "shortcuts_during": shortcuts_during,
        "shortcuts_after": shortcuts_after,
        "shortcuts_unchanged": (
            shortcuts_before == shortcuts_during == shortcuts_after
        ),
        "install_root_removed": not install_root.exists(),
    }
    summary_path = smoke_root / "result.json"
    _write_json(summary_path, summary)
    if args.summary is not None:
        _write_json(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failure or cleanup_failure:
        raise RuntimeError(failure or cleanup_failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
