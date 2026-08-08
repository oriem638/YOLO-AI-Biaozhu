from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import (
    collect_runtime_licenses,
    prepare_release,
    run_installer_smoke,
    run_standalone_cuda_smoke,
)


def test_nuitka_report_utf8_alias_supports_non_ascii_paths(tmp_path: Path) -> None:
    report = tmp_path / "报告.xml"
    report.write_bytes(
        b"<?xml version='1.0' encoding='utf8'?>\n"
        b"<nuitka-compilation-report>"
        b'<option value="C:/Users/Test/Desktop/'
        + "中文路径".encode()
        + b'" />'
        b'<distribution-usage name="example-dist" />'
        b"</nuitka-compilation-report>"
    )

    assert collect_runtime_licenses._read_nuitka_distributions(report) == {
        "example-dist"
    }


def test_installer_gui_probe_never_reopens_remembered_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 1

        def wait(self, timeout: float) -> int:
            del timeout
            return int(self.returncode or 0)

        def kill(self) -> None:
            self.returncode = 1

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return FakeProcess()

    monkeypatch.setattr(run_installer_smoke.subprocess, "Popen", fake_popen)
    executable = tmp_path / "AI-Biaozhu.exe"
    result = run_installer_smoke._gui_probe(
        executable,
        smoke_root=tmp_path / "smoke",
        env={"BASE": "1"},
        seconds=0,
    )

    isolated_project = tmp_path / "smoke" / "gui-smoke-project-do-not-open-last"
    assert captured["command"] == [
        str(executable),
        "--project",
        str(isolated_project),
    ]
    assert result["isolated_project"] == str(isolated_project)
    assert result["status"] == "passed"


def test_windows_build_embeds_per_monitor_v2_manifest() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = (
        project_root / "packaging" / "AI-Biaozhu.exe.manifest"
    ).read_text(encoding="utf-8")
    build_script = (project_root / "scripts" / "build_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert "PerMonitorV2" in manifest
    assert "longPathAware" in manifest
    assert "Set-GuiManifest" in build_script
    assert "-outputresource:$Executable;#1" in build_script


def test_compute_process_query_only_accepts_current_worker_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            0,
            "123, another.exe, 50\n456, AI-Biaozhu-Worker.exe, 148\n",
            "",
        )

    monkeypatch.setattr(run_standalone_cuda_smoke.subprocess, "run", fake_run)
    assert run_standalone_cuda_smoke._query_compute_processes(456) == [
        "456, AI-Biaozhu-Worker.exe, 148"
    ]
    assert run_standalone_cuda_smoke._query_compute_processes(999) == []


def test_standalone_smoke_overwrites_old_success_when_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"passed"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_standalone_cuda_smoke.py",
            "--worker",
            str(tmp_path / "missing-worker.exe"),
            "--weight-cache",
            str(tmp_path / "cache"),
            "--results-root",
            str(tmp_path / "results"),
            "--summary",
            str(summary),
        ],
    )
    with pytest.raises(RuntimeError, match="worker missing"):
        run_standalone_cuda_smoke.main()
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "missing-worker.exe" in result["error"]


def test_installed_tree_signature_excludes_only_inno_uninstaller_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "AI-Biaozhu.exe").write_bytes(b"MZ-runtime")
    (tmp_path / "unins000.exe").write_bytes(b"MZ-uninstaller")
    (tmp_path / "unins000.dat").write_bytes(b"metadata")
    signature = run_installer_smoke._tree_signature(tmp_path, installed=True)
    assert signature == {
        "ai-biaozhu.exe": run_installer_smoke._file_sha256(
            tmp_path / "AI-Biaozhu.exe"
        )
    }


def test_installer_smoke_supports_an_isolated_test_app_id() -> None:
    test_app_id = "{7BCEB14E-9F20-41F0-BAD6-2B05FB0DB84A}"
    assert run_installer_smoke._uninstall_key(test_app_id).endswith(
        r"\{7BCEB14E-9F20-41F0-BAD6-2B05FB0DB84A}_is1"
    )
    with pytest.raises(ValueError, match="invalid Inno Setup AppId"):
        run_installer_smoke._uninstall_key("not-a-guid")


def test_runtime_tree_signature_detects_same_size_content_changes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "runtime.dll").write_bytes(b"AAAA")
    (second / "runtime.dll").write_bytes(b"BBBB")
    assert run_installer_smoke._tree_signature(
        first
    ) != run_installer_smoke._tree_signature(second)


def test_installer_smoke_model_seed_requires_exact_locked_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / run_installer_smoke.MODEL_SEED_RELATIVE
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"verified-seed")
    monkeypatch.setattr(run_installer_smoke, "MODEL_SEED_SIZE", seed.stat().st_size)
    monkeypatch.setattr(
        run_installer_smoke,
        "MODEL_SEED_SHA256",
        run_installer_smoke._file_sha256(seed),
    )
    run_installer_smoke._verify_model_seed(tmp_path)
    seed.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="size/SHA-256"):
        run_installer_smoke._verify_model_seed(tmp_path)


def test_standalone_protocol_rejects_foreign_or_repeated_events() -> None:
    foreign = json.dumps(
        {
            "protocol_version": "1.0",
            "job_id": "other",
            "seq": 0,
            "type": "completed",
            "payload": {},
        }
    )
    with pytest.raises(RuntimeError, match="foreign|belongs"):
        run_standalone_cuda_smoke._parse_protocol_events(
            foreign,
            expected_job_id="expected",
        )
    repeated = "\n".join(
        [
            json.dumps(
                {
                    "protocol_version": "1.0",
                    "job_id": "expected",
                    "seq": 1,
                    "type": "status",
                    "payload": {},
                }
            ),
            json.dumps(
                {
                    "protocol_version": "1.0",
                    "job_id": "expected",
                    "seq": 1,
                    "type": "completed",
                    "payload": {},
                }
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="sequence"):
        run_standalone_cuda_smoke._parse_protocol_events(
            repeated,
            expected_job_id="expected",
        )


def test_installer_smoke_overwrites_old_success_when_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "installer-summary.json"
    summary.write_text('{"status":"passed"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_installer_smoke.py",
            "--installer",
            str(tmp_path / "missing-installer.exe"),
            "--standalone-root",
            str(tmp_path / "missing-standalone"),
            "--ml-python",
            sys.executable,
            "--results-root",
            str(tmp_path / "results"),
            "--summary",
            str(summary),
        ],
    )
    with pytest.raises(RuntimeError, match="installer missing"):
        run_installer_smoke.main()
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "missing-installer.exe" in result["outer_error"]


def test_installer_sandbox_mode_is_explicit_even_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "installer-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_installer_smoke.py",
            "--installer",
            str(tmp_path / "missing-installer.exe"),
            "--standalone-root",
            str(tmp_path / "missing-standalone"),
            "--ml-python",
            sys.executable,
            "--results-root",
            str(tmp_path / "results"),
            "--summary",
            str(summary),
            "--sandbox-no-uninstall-registry",
        ],
    )
    with pytest.raises(RuntimeError, match="installer missing"):
        run_installer_smoke.main()
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["sandbox_no_uninstall_registry"] is True


def test_installer_smoke_path_overlap_protects_existing_installation(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "installed-app"
    assert run_installer_smoke._paths_overlap(existing, existing)
    assert run_installer_smoke._paths_overlap(existing / "nested", existing)
    assert run_installer_smoke._paths_overlap(existing, existing / "nested")
    assert not run_installer_smoke._paths_overlap(
        tmp_path / "isolated-smoke",
        existing,
    )


def test_gui_sandbox_classifier_accepts_winerror_5_in_current_local_appdata() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        "  File \"app.py\", line 1, in <module>\n"
        "PermissionError: [WinError 5] Access is denied: "
        "'C:\\Users\\Test User\\AppData\\Local\\AI-Biaozhu'\n"
    )
    assert run_installer_smoke._is_known_codex_sandbox_gui_block(
        stderr,
        local_appdata=r"C:\Users\Test User\AppData\Local",
    )


@pytest.mark.parametrize(
    "stderr",
    [
        (
            "PermissionError: [Errno 13] Permission denied: "
            "'C:\\Users\\Test\\AppData\\Local\\AI-Biaozhu'\n"
        ),
        (
            "FileNotFoundError: [WinError 3] The system cannot find the path: "
            "'C:\\Users\\Test\\AppData\\Local\\AI-Biaozhu'\n"
        ),
        (
            "PermissionError: [WinError 5] Access is denied: "
            "'C:\\Users\\Test\\Documents\\AI-Biaozhu'\n"
        ),
        (
            "PermissionError: [WinError 5] Access is denied: "
            "'C:\\Users\\Other\\AppData\\Local\\AI-Biaozhu'\n"
        ),
        (
            "PermissionError: [WinError 5] Access is denied\n"
            "diagnostic: C:\\Users\\Test\\AppData\\Local\\AI-Biaozhu\n"
        ),
        (
            "PermissionError: [WinError 5] Access is denied: "
            "'C:\\Users\\Test\\AppData\\Local-Backup\\AI-Biaozhu'\n"
        ),
    ],
)
def test_gui_sandbox_classifier_rejects_unrelated_gui_failures(stderr: str) -> None:
    assert not run_installer_smoke._is_known_codex_sandbox_gui_block(
        stderr,
        local_appdata=r"C:\Users\Test\AppData\Local",
    )


def test_release_attempt_invalidates_old_success_metadata(tmp_path: Path) -> None:
    report = tmp_path / prepare_release.RELEASE_REPORT_NAME
    checksums = tmp_path / prepare_release.RELEASE_CHECKSUM_NAME
    artifact = tmp_path / "AI-Biaozhu-0.1.0-source.zip"
    report.write_text('{"status":"passed"}\n', encoding="utf-8")
    checksums.write_text("old\n", encoding="utf-8")
    artifact.write_bytes(b"keep")
    prepare_release._invalidate_release_metadata(tmp_path)
    assert not report.exists()
    assert not checksums.exists()
    assert artifact.read_bytes() == b"keep"


def test_release_model_seed_allowlist_accepts_only_locked_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "yolo26s.pt"
    payload = b"locked-seed"
    seed.write_bytes(payload)
    monkeypatch.setattr(prepare_release, "MODEL_SEED_SIZE", len(payload))
    monkeypatch.setattr(
        prepare_release,
        "MODEL_SEED_SHA256",
        prepare_release._file_sha256(seed),
    )
    prepare_release._verify_model_seed_file(seed)
    seed.write_bytes(b"tampered")
    with pytest.raises(prepare_release.ReleaseError, match="size/SHA-256"):
        prepare_release._verify_model_seed_file(seed)


def test_source_copy_rejects_any_extra_pt_beside_locked_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "bundled_models"
    source.mkdir()
    payload = b"locked-seed"
    (source / "yolo26s.pt").write_bytes(payload)
    (source / "unexpected.pt").write_bytes(payload)
    monkeypatch.setattr(prepare_release, "MODEL_SEED_SIZE", len(payload))
    monkeypatch.setattr(
        prepare_release,
        "MODEL_SEED_SHA256",
        prepare_release._file_sha256(source / "yolo26s.pt"),
    )
    with pytest.raises(prepare_release.ReleaseError, match="forbidden"):
        prepare_release._copy_allowed_directory(source, tmp_path / "copy")


def test_release_main_invalidates_old_pass_before_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    report = output / prepare_release.RELEASE_REPORT_NAME
    checksums = output / prepare_release.RELEASE_CHECKSUM_NAME
    report.write_text('{"status":"passed"}\n', encoding="utf-8")
    checksums.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_release.py",
            "--project-root",
            str(project),
            "--output",
            str(output),
            "--source-only",
        ],
    )
    with pytest.raises(prepare_release.ReleaseError, match="validation report"):
        prepare_release.main()
    assert not report.exists()
    assert not checksums.exists()


def test_maintenance_release_cleanup_preserves_original_release_files(
    tmp_path: Path,
) -> None:
    original = tmp_path / "AI-Biaozhu-0.1.4-source.zip"
    maintenance = tmp_path / "AI-Biaozhu-Maintenance-0.2.0-source.zip"
    original_report = tmp_path / "release-report.json"
    maintenance_report = tmp_path / prepare_release.RELEASE_REPORT_NAME
    original.write_bytes(b"original")
    maintenance.write_bytes(b"maintenance")
    original_report.write_text("original\n", encoding="utf-8")
    maintenance_report.write_text("maintenance\n", encoding="utf-8")

    prepare_release._clean_generated_output(tmp_path)

    assert original.read_bytes() == b"original"
    assert original_report.read_text(encoding="utf-8") == "original\n"
    assert not maintenance.exists()
    assert not maintenance_report.exists()


def test_release_evidence_rejects_failed_cuda_summary(tmp_path: Path) -> None:
    standalone = tmp_path / "AI-Biaozhu.dist"
    standalone.mkdir()
    (standalone / "AI-Biaozhu.exe").write_bytes(b"MZ-gui")
    (standalone / "AI-Biaozhu-Worker.exe").write_bytes(b"MZ-worker")
    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"MZ-setup")
    gpu = tmp_path / "gpu.json"
    gpu.write_text('{"status":"failed"}\n', encoding="utf-8")
    onnx = tmp_path / "onnx.json"
    onnx.write_text('{"status":"passed"}\n', encoding="utf-8")
    installed = tmp_path / "installed.json"
    installed.write_text('{"status":"passed"}\n', encoding="utf-8")
    with pytest.raises(prepare_release.ReleaseError, match="CUDA validation did not pass"):
        prepare_release._verify_release_evidence(
            standalone=standalone,
            installer=installer,
            gpu_validation=gpu,
            onnx_validations={
                "YOLO26n": onnx,
                "YOLOv8n": onnx,
                "YOLO11n": onnx,
                "YOLOv5n": onnx,
            },
            installer_validation=installed,
        )


def test_release_evidence_binds_all_families_and_current_runtime(
    tmp_path: Path,
) -> None:
    standalone = tmp_path / "AI-Biaozhu.dist"
    standalone.mkdir()
    gui = standalone / "AI-Biaozhu.exe"
    worker = standalone / "AI-Biaozhu-Worker.exe"
    gui.write_bytes(b"MZ-gui-current")
    worker.write_bytes(b"MZ-worker-current")
    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"MZ-installer-current")
    smoke = tmp_path / "smoke"
    weights = smoke / "runs" / "model" / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    last = weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    gpu = tmp_path / "gpu.json"
    gpu.write_text(
        json.dumps(
            {
                "status": "passed",
                "worker": str(worker),
                "worker_sha256": prepare_release._file_sha256(worker),
                "device": 0,
                "gpu_memory_gb_max": 0.1,
                "smoke_dir": str(smoke),
                "checkpoint_paths": {"best": str(best), "last": str(last)},
                "checkpoint_sha256": {
                    "best": prepare_release._file_sha256(best),
                    "last": prepare_release._file_sha256(last),
                },
                "train": {
                    "return_code": 0,
                    "event_types": ["status", "completed"],
                    "nvidia_smi_observations": [{"pid": 1}],
                },
                "predict": {
                    "return_code": 0,
                    "event_types": ["prediction", "completed"],
                    "nvidia_smi_observations": [{"pid": 2}],
                },
            }
        ),
        encoding="utf-8",
    )
    target_evidence = {
        "maixcam_pro": {
            "return_code": 0,
            "numeric_validations": [{"ok": True}, {"ok": True}],
            "device_validation": "required",
        },
        "maixcam2": {
            "return_code": 0,
            "numeric_validations": [{"ok": True}, {"ok": True}],
            "device_validation": "required",
        },
    }
    onnx_files: dict[str, Path] = {}
    for model_key in ("YOLO26n", "YOLOv8n", "YOLO11n", "YOLOv5n"):
        path = tmp_path / f"{model_key}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "model_key": model_key,
                    "checkpoint": str(best),
                    "checkpoint_sha256": prepare_release._file_sha256(best),
                    "targets": target_evidence,
                }
            ),
            encoding="utf-8",
        )
        onnx_files[model_key] = path
    tree_hash, file_count = prepare_release._runtime_tree_digest(standalone)
    entries = {
        "AI-Biaozhu.exe": prepare_release._file_sha256(gui),
        "AI-Biaozhu-Worker.exe": prepare_release._file_sha256(worker),
    }
    installed = tmp_path / "installed.json"
    installed.write_text(
        json.dumps(
            {
                "status": "passed",
                "sandbox_no_uninstall_registry": False,
                "registry_validation": "verified",
                "installer": str(installer),
                "installer_sha256": prepare_release._file_sha256(installer),
                "standalone_root": str(standalone),
                "standalone_tree_sha256": tree_hash,
                "installed_tree_sha256": tree_hash,
                "standalone_file_count": file_count,
                "installed_file_count": file_count,
                "standalone_entry_hashes": entries,
                "installed_entry_hashes": entries,
                "install": {"return_code": 0},
                "worker_help": {"return_code": 0},
                "uninstall": {"return_code": 0},
                "environment_completed": {
                    "type": "completed",
                    "payload": {"gpu_ready": True},
                },
                "gui": {"status": "passed"},
                "failure": None,
                "cleanup_failure": None,
                "install_root_removed": True,
                "registrations_before": [],
                "registrations_during": [
                    {
                        "hive": "HKCU",
                        "view": "native",
                        "install_location": str(tmp_path / "installed-app"),
                    }
                ],
                "registrations_after": [],
                "shortcuts_unchanged": True,
                "forbidden_file_count": 0,
                "forbidden_directory_count": 0,
            }
        ),
        encoding="utf-8",
    )
    result = prepare_release._verify_release_evidence(
        standalone=standalone,
        installer=installer,
        gpu_validation=gpu,
        onnx_validations=onnx_files,
        installer_validation=installed,
    )
    assert set(result["real_onnx_gates"]) == set(onnx_files)
    assert result["installer_smoke"]["standalone_tree_sha256"] == tree_hash

    partial = json.loads(installed.read_text(encoding="utf-8"))
    partial.update(
        {
            "status": "partial_sandbox_validation",
            "sandbox_no_uninstall_registry": True,
            "registry_validation": "not_verified_sandbox_mode",
            "registrations_during": [],
            "gui": {"status": "sandbox_blocked"},
        }
    )
    installed.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(
        prepare_release.ReleaseError,
        match="explicit partial acceptance",
    ):
        prepare_release._verify_release_evidence(
            standalone=standalone,
            installer=installer,
            gpu_validation=gpu,
            onnx_validations=onnx_files,
            installer_validation=installed,
        )
    partial_result = prepare_release._verify_release_evidence(
        standalone=standalone,
        installer=installer,
        gpu_validation=gpu,
        onnx_validations=onnx_files,
        installer_validation=installed,
        allow_partial_installer_validation=True,
    )
    assert (
        partial_result["installer_smoke"]["status"]
        == "partial_sandbox_validation"
    )
    assert partial_result["installer_smoke"]["gui_status"] == "sandbox_blocked"
    assert prepare_release._release_completion_state(
        source_only=False,
        validation_evidence=partial_result,
    ) == ("partial", False, True)
    assert prepare_release._release_completion_state(
        source_only=False,
        validation_evidence=result,
    ) == ("passed", True, False)


def test_source_mirror_refuses_invalid_ownership_marker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source", encoding="utf-8")
    target = tmp_path / "mirror"
    target.mkdir()
    marker = target / prepare_release.MIRROR_MARKER
    marker.write_text('{"schema_version":1}\n', encoding="utf-8")
    keep = target / "keep.txt"
    keep.write_text("owned by someone else", encoding="utf-8")
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"archive")
    with pytest.raises(prepare_release.ReleaseError, match="ownership marker"):
        prepare_release._publish_source_mirror(
            source,
            target,
            replace=True,
            source_archive=archive,
            version="0.1.0",
        )
    assert keep.is_file()
