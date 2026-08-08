from __future__ import annotations

import json
import zipfile

import pytest

import ai_biaozhu.deploy.package as package_module
from ai_biaozhu.deploy.maix import Cam2NpuMode, MaixTarget
from ai_biaozhu.deploy.mud import build_mud
from ai_biaozhu.deploy.package import (
    DeploymentArtifact,
    DeploymentCancelled,
    _size_warning,
    build_deployment_package,
    validate_deployment_class_names,
)


def test_minimal_pro_package_uses_allowlist_and_size_is_warning(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(package_module, "SIZE_WARNING_BYTES", 5)
    model = tmp_path / "model.cvimodel"
    model.write_bytes(b"123456")
    unrelated = tmp_path / "best.pt"
    unrelated.write_bytes(b"must-not-be-packaged")
    source_onnx = tmp_path / "source.onnx"
    source_onnx.write_bytes(b"onnx")
    output = tmp_path / "detector.maixapp"
    result = build_deployment_package(
        package_path=output,
        target=MaixTarget.MAIXCAM_PRO,
        model_key="YOLO26n",
        model_artifacts=[DeploymentArtifact(model, "model.cvimodel")],
        class_names=["cat", "dog"],
        input_height=224,
        input_width=320,
        source_run_id="train-run-42",
        checkpoint_role="best",
        source_checkpoint=unrelated,
        source_onnx=source_onnx,
        converter_image="sipeed/maixcam-tpumlir:v3.4",
        tool_versions={"docker": "28.3.2", "tpu_mlir": "v3.4"},
        maixpy_version="4.12.5",
        maixcdk_commit="deadbeef",
    )
    assert output.is_file()
    assert result.warnings
    assert result.model_package_path.is_file()
    assert result.report_path.is_file()
    assert result.sha256_path.is_file()
    with zipfile.ZipFile(result.app_package_path) as archive:
        names = set(archive.namelist())
        assert names == set(result.app_files)
        assert "best.pt" not in names
        assert "deployment-report.json" not in names
        assert not any(name.endswith(".onnx") for name in names)
        manifest = archive.read("app.yaml").decode()
        assert "files:" in manifest
        assert "config.json" in manifest
        assert "models/model.mud" in names
    with zipfile.ZipFile(result.model_package_path) as archive:
        assert set(archive.namelist()) == {"model.cvimodel", "model.mud"}
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["target"] == "maixcam_pro"
    assert report["source"]["run_id"] == "train-run-42"
    assert report["source"]["checkpoint_role"] == "best"
    assert report["source"]["checkpoint"]["sha256"]
    assert report["class_names"] == ["cat", "dog"]
    assert report["runtime"] == {
        "maixpy_min_version": "4.12.5",
        "maixpy_version": "4.12.5",
        "maixcdk_commit": "deadbeef",
    }
    assert report["converter"]["tool_versions"]["docker"] == "28.3.2"
    assert report["payload"][0]["sha256"]
    expected_files = {
        "model-only": {"model.cvimodel", "model.mud"},
        "full-app": {
            "app.yaml",
            "main.py",
            "config.json",
            "models/model.cvimodel",
            "models/model.mud",
        },
    }
    for package in report["packages"]:
        records = package["files"]
        assert {record["path"] for record in records} == expected_files[package["kind"]]
        assert records == report["package_files"][package["kind"]]
        assert all(record["size"] >= 0 and record["sha256"] for record in records)


def test_cam2_package_requires_and_includes_both_models(tmp_path) -> None:
    npu = tmp_path / "model_npu.axmodel"
    vnpu = tmp_path / "model_vnpu.axmodel"
    npu.write_bytes(b"npu")
    vnpu.write_bytes(b"vnpu")
    output = tmp_path / "cam2.zip"
    result = build_deployment_package(
        package_path=output,
        target=MaixTarget.MAIXCAM2,
        model_key="YOLO11s",
        model_artifacts=[
            (npu, "model_npu.axmodel"),
            (vnpu, "model_vnpu.axmodel"),
        ],
        class_names=["object"],
        input_height=640,
        input_width=640,
        cam2_npu_mode=Cam2NpuMode.BOTH,
    )
    assert {
        "models/model_npu.axmodel",
        "models/model_vnpu.axmodel",
    }.issubset(result.app_files)
    with zipfile.ZipFile(result.model_package_path) as archive:
        mud = archive.read("model.mud").decode()
    assert "model_npu = model_npu.axmodel" in mud
    assert "model_vnpu = model_vnpu.axmodel" in mud
    assert "input_cache_flush = false" in mud
    assert "output_cache_inval = true" in mud

    with zipfile.ZipFile(result.app_package_path) as archive:
        config = json.loads(archive.read("config.json"))
        main_py = archive.read("main.py").decode()
    assert config["ai_isp_mode"] == "system"
    assert "get_aiisp_workmode" in main_py
    assert "FPS {smoothed_fps:.1f}" in main_py
    compile(main_py, "main.py", "exec")


def test_traditional_yolov5_mud_contains_anchors() -> None:
    mud = build_mud(
        target=MaixTarget.MAIXCAM_PRO,
        model_key="YOLOv5n",
        class_names=["object"],
        input_height=224,
        input_width=320,
        anchors=[
            10,
            13,
            16,
            30,
            33,
            23,
            30,
            61,
            62,
            45,
            59,
            119,
            116,
            90,
            156,
            198,
            373,
            326,
        ],
    )
    assert "model_type = yolov5" in mud
    assert "anchors =" in mud
    assert "labels = object" in mud


def test_oversize_rejection_keeps_final_paths_unpublished(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(package_module, "SIZE_WARNING_BYTES", 1)
    model = tmp_path / "model.cvimodel"
    model.write_bytes(b"model")
    output = tmp_path / "rejected.maixapp"
    try:
        build_deployment_package(
            package_path=output,
            target=MaixTarget.MAIXCAM_PRO,
            model_key="YOLO26n",
            model_artifacts=[(model, "model.cvimodel")],
            class_names=["object"],
            input_height=224,
            input_width=320,
            allow_oversize=False,
            oversize_confirmation=lambda _: False,
        )
    except DeploymentCancelled:
        pass
    else:
        raise AssertionError("expected DeploymentCancelled")
    assert not output.exists()
    assert not (tmp_path / "rejected-model.maixapp").exists()


def test_cam2_single_mode_package_and_mud(tmp_path) -> None:
    vnpu = tmp_path / "model_vnpu.axmodel"
    vnpu.write_bytes(b"vnpu")
    result = build_deployment_package(
        package_path=tmp_path / "vnpu.zip",
        target=MaixTarget.MAIXCAM2,
        model_key="YOLO11n",
        model_artifacts=[(vnpu, "model_vnpu.axmodel")],
        class_names=["object"],
        input_height=640,
        input_width=640,
        cam2_npu_mode=Cam2NpuMode.VNPU,
    )
    with zipfile.ZipFile(result.model_package_path) as archive:
        mud = archive.read("model.mud").decode()
        assert set(archive.namelist()) == {"model_vnpu.axmodel", "model.mud"}
    assert "model_vnpu = model_vnpu.axmodel" in mud
    assert "model_npu =" not in mud
    with zipfile.ZipFile(result.app_package_path) as archive:
        config = json.loads(archive.read("config.json"))
    assert config["ai_isp_mode"] == "required"


def test_selectable_maixapp_and_editable_project_preserve_alias_mapping(tmp_path) -> None:
    model = tmp_path / "model.cvimodel"
    model.write_bytes(b"model")
    result = build_deployment_package(
        package_path=tmp_path / "ball.maixapp",
        editable_project_path=tmp_path / "ball-editable",
        package_outputs=["maixapp", "editable_project"],
        target=MaixTarget.MAIXCAM_PRO,
        model_key="YOLO26n",
        model_artifacts=[(model, "model.cvimodel")],
        class_names=["小钢球"],
        deployment_class_names=["BALL"],
        input_height=224,
        input_width=320,
    )

    assert result.model_package_path is None
    assert result.app_package_path == tmp_path / "ball.maixapp"
    assert result.app_package_path.is_file()
    assert result.editable_project_path == tmp_path / "ball-editable"
    assert result.editable_project_path.is_dir()
    assert {item.kind for item in result.artifacts} == {
        "maixapp",
        "editable-project",
        "deployment-report",
        "sha256-manifest",
    }
    expected = {
        "app.yaml",
        "main.py",
        "config.json",
        "models/model.cvimodel",
        "models/model.mud",
    }
    assert {
        path.relative_to(result.editable_project_path).as_posix()
        for path in result.editable_project_path.rglob("*")
        if path.is_file()
    } == expected
    config = json.loads(
        (result.editable_project_path / "config.json").read_text(encoding="utf-8")
    )
    assert config["checkpoint_class_names"] == ["小钢球"]
    assert config["class_names"] == ["BALL"]
    mud = (result.editable_project_path / "models" / "model.mud").read_text(
        encoding="utf-8"
    )
    assert "labels = BALL" in mud
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["class_names"] == ["小钢球"]
    assert report["checkpoint_class_names"] == ["小钢球"]
    assert report["deployment_class_names"] == ["BALL"]


def test_editable_project_can_be_the_only_output(tmp_path) -> None:
    model = tmp_path / "model.cvimodel"
    model.write_bytes(b"model")
    result = build_deployment_package(
        package_path=tmp_path / "unused.maixapp",
        editable_project_path=tmp_path / "project",
        package_outputs=["editable_project"],
        target=MaixTarget.MAIXCAM_PRO,
        model_key="YOLO26n",
        model_artifacts=[(model, "model.cvimodel")],
        class_names=["ball"],
        input_height=224,
        input_width=320,
    )
    assert result.app_package_path is None
    assert result.model_package_path is None
    assert result.editable_project_path.is_dir()
    assert not (tmp_path / "unused.maixapp").exists()


@pytest.mark.parametrize(
    "aliases",
    [[], [""], ["BALL", "BALL"], ["bad,name"], ["bad\nname"]],
)
def test_deployment_aliases_are_strict(aliases) -> None:
    with pytest.raises(ValueError):
        validate_deployment_class_names(["one", "two"], aliases)


@pytest.mark.parametrize(
    ("unpacked_bytes", "warned"),
    [
        (29_999_999, False),
        (30_000_000, False),
        (30_000_001, True),
    ],
)
def test_decimal_30mb_unpacked_boundary(
    tmp_path,
    unpacked_bytes: int,
    warned: bool,
) -> None:
    root = tmp_path / str(unpacked_bytes)
    root.mkdir()
    payload = root / "model.cvimodel"
    with payload.open("wb") as handle:
        handle.truncate(unpacked_bytes)
    archive = tmp_path / f"{unpacked_bytes}.zip"
    archive.write_bytes(b"")

    warning = _size_warning(
        "model-only",
        archive,
        root,
        ["model.cvimodel"],
    )

    assert (warning is not None) is warned
    if warning is not None:
        assert warning.unpacked_size == unpacked_bytes
