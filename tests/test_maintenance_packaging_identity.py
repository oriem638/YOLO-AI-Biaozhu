from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE_GUID = "4C9330ED-77CB-4F81-A467-06B4D6A8FB2B"
ORIGINAL_GUID = "147EE884-BF66-4DFE-BF0D-2D275C1A62AB"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_maintenance_version_and_runtime_identity_are_independent() -> None:
    project = tomllib.loads(_text("pyproject.toml"))["project"]
    assert project["version"] == "0.2.3"

    paths = _text("src/ai_biaozhu/app_paths.py")
    assert 'APP_NAME = "AI标注-维护版-0.2"' in paths
    assert 'APP_AUTHOR = "AI-Biaozhu-Maintenance"' in paths
    assert 'APP_NAME = "AI标注"\n' not in paths
    assert 'APP_AUTHOR = "AI-Biaozhu"\n' not in paths

    app = _text("src/ai_biaozhu/app.py")
    assert 'setOrganizationName("AI-Biaozhu-Maintenance")' in app
    assert 'setApplicationName("AI标注-维护版-0.2")' in app
    assert 'f"AI 数据集标注与训练 维护版 {__version__}"' in app


def test_maintenance_installer_cannot_upgrade_or_overwrite_original() -> None:
    installer = _text("packaging/ai_biaozhu.iss")
    assert MAINTENANCE_GUID in installer
    assert ORIGINAL_GUID not in installer
    assert '#define MyAppName "AI Biaozhu Maintenance 0.2"' in installer
    assert "DefaultDirName={autopf}\\AI-Biaozhu-Maintenance-0.2" in installer
    assert "AI-Biaozhu-Maintenance-Setup-" in installer
    assert "DefaultDirName={autopf}\\AI-Biaozhu\n" not in installer


def test_release_outputs_and_smoke_identity_are_maintenance_specific() -> None:
    release = _text("scripts/prepare_release.py")
    assert 'RELEASE_STEM = "AI-Biaozhu-Maintenance"' in release
    assert '.ai-biaozhu-maintenance-source-mirror.json' in release
    assert 'path.name.startswith(f"{RELEASE_STEM}-")' in release

    wrapper = _text("scripts/prepare_release.ps1")
    assert '"..\\$projectLeaf-outputs"' in wrapper
    assert '"..\\..\\outputs"' not in wrapper

    smoke = _text("scripts/run_installer_smoke.py")
    assert f'DEFAULT_APP_ID = "{{{MAINTENANCE_GUID}}}"' in smoke
    assert 'MAINTENANCE_APP_NAME = "AI Biaozhu Maintenance 0.2"' in smoke
    assert 'ORIGINAL_APP_NAME = "AI Biaozhu"' in smoke
