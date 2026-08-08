from __future__ import annotations

import json
import subprocess
import tarfile

import pytest
from PIL import Image

from ai_biaozhu.deploy.maix import (
    MAIXCAM2_IMAGE,
    MAIXCAM_PRO_IMAGE,
    Cam2NpuMode,
    MaixConversionRequest,
    MaixTarget,
    build_conversion_plan,
    execute_conversion_plan,
    expected_output_tensors,
)


def _request(tmp_path, target, model_key="YOLO26n"):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    for index in range(20):
        (calibration / f"{index:02d}.jpg").write_bytes(b"calibration")
    output = tmp_path / "output"
    return MaixConversionRequest(
        target=target,
        model_key=model_key,
        onnx_path=onnx,
        output_dir=output,
        calibration_dir=calibration,
        class_names=("person",),
        input_height=224,
        input_width=320,
        calibration_count=20,
    )


def test_maixcam_pro_plan_uses_cv181x_int8_and_expected_nodes(tmp_path) -> None:
    request = _request(tmp_path, MaixTarget.MAIXCAM_PRO, "YOLOv8n")
    plan = build_conversion_plan(request, perform_gate=False)
    flattened = [" ".join(command) for command in plan.commands]
    assert plan.image == MAIXCAM_PRO_IMAGE
    assert plan.output_models == ("model.cvimodel",)
    assert any("--processor cv181x" in command for command in flattened)
    assert any("--quantize INT8" in command for command in flattened)
    assert any("--test_result /output/detector_top_outputs.npz" in command for command in flattened)
    assert any("--test_reference /output/detector_top_outputs.npz" in command for command in flattened)
    assert any("--tolerance 0.99,0.99" in command for command in flattened)
    assert any("--tolerance 0.9,0.6" in command for command in flattened)
    assert expected_output_tensors(request.target, request.model_key) == (
        "/model.22/dfl/conv/Conv_output_0",
        "/model.22/Sigmoid_output_0",
    )


def test_maixcam2_plan_generates_npu1_npu2_pulsar_configs(tmp_path) -> None:
    request = _request(tmp_path, MaixTarget.MAIXCAM2, "YOLO11s")
    plan = build_conversion_plan(request, perform_gate=False)
    assert plan.image == MAIXCAM2_IMAGE
    assert plan.output_models == ("model_npu.axmodel", "model_vnpu.axmodel")
    plan.materialize(prepare_inputs=False)
    configs = [
        json.loads((request.output_dir / item.relative_path).read_text(encoding="utf-8"))
        for item in plan.generated_files
    ]
    assert {config["npu_mode"] for config in configs} == {"NPU1", "NPU2"}
    assert all(config["compiler"]["check"] == 3 for config in configs)
    assert all("quantization" not in config for config in configs)
    assert all(
        config["quant"]["input_configs"][0]["calibration_dataset"]
        == "datasets/train.tar"
        for config in configs
    )
    assert all(
        config["input_processors"][0]
        == {
            "tensor_name": "images",
            "tensor_format": "RGB",
            "tensor_layout": "NCHW",
            "src_format": "RGB",
            "src_dtype": "U8",
            "src_layout": "NHWC",
            "csc_mode": "NoCSC",
        }
        for config in configs
    )
    assert all(
        output["dst_perm"] == [0, 2, 3, 1]
        for config in configs
        for output in config["output_processors"]
    )
    commands = [" ".join(command) for command in plan.commands]
    assert all("--input ./export.onnx" in command for command in commands)
    assert all("--output_dir ./tmp" in command for command in commands)
    assert all("--output_name" not in command for command in commands)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (Cam2NpuMode.NPU2, ("model_npu.axmodel",)),
        (Cam2NpuMode.VNPU, ("model_vnpu.axmodel",)),
        (
            Cam2NpuMode.BOTH,
            ("model_npu.axmodel", "model_vnpu.axmodel"),
        ),
    ],
)
def test_maixcam2_npu_modes_only_build_selected_artifacts(tmp_path, mode, expected) -> None:
    original = _request(tmp_path, MaixTarget.MAIXCAM2, "YOLO26n")
    request = MaixConversionRequest(
        target=original.target,
        model_key=original.model_key,
        onnx_path=original.onnx_path,
        output_dir=original.output_dir,
        calibration_dir=original.calibration_dir,
        class_names=original.class_names,
        input_height=original.input_height,
        input_width=original.input_width,
        cam2_npu_mode=mode,
    )
    plan = build_conversion_plan(request, perform_gate=False)
    assert plan.output_models == expected
    assert len(plan.commands) == len(expected)


def test_conversion_runner_is_injectable_and_checks_outputs(tmp_path) -> None:
    request = _request(tmp_path, MaixTarget.MAIXCAM_PRO)
    plan = build_conversion_plan(request, perform_gate=False)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if "model_deploy.py" in command:
            (request.output_dir / "model.cvimodel").write_bytes(b"model")
        return subprocess.CompletedProcess(command, 0, "", "")

    execute_conversion_plan(plan, runner=runner)
    assert len(calls) == 3


def test_cam2_runner_copies_compiled_axmodel_for_each_mode(tmp_path) -> None:
    request = _request(tmp_path, MaixTarget.MAIXCAM2, "YOLO11n")
    plan = build_conversion_plan(request, perform_gate=False)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        temporary = request.output_dir / "tmp"
        temporary.mkdir(parents=True)
        (temporary / "compiled.axmodel").write_bytes(
            f"model-{len(calls)}".encode()
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    execute_conversion_plan(plan, runner=runner, materialize=False)
    assert (request.output_dir / "model_npu.axmodel").read_bytes() == b"model-1"
    assert (request.output_dir / "model_vnpu.axmodel").read_bytes() == b"model-2"


def test_cam2_materialization_extracts_simplifies_and_archives_calibration(
    tmp_path,
) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    preferred = expected_output_tensors(MaixTarget.MAIXCAM2, "YOLOv8n")
    input_value = helper.make_tensor_value_info(
        "images", onnx.TensorProto.FLOAT, [1, 3, 224, 320]
    )
    feature_shapes = ((28, 40), (14, 20), (7, 10))
    initializers = []
    nodes = []
    role_channels = (64, 1, 4)
    for role_index, (name, channels) in enumerate(
        zip(preferred, role_channels, strict=True)
    ):
        flattened = []
        for scale_index, stride in enumerate((8, 16, 32)):
            weight_name = f"weight_{role_index}_{scale_index}"
            conv_name = f"conv_{role_index}_{scale_index}"
            shape_name = f"shape_{role_index}_{scale_index}"
            flat_name = f"flat_{role_index}_{scale_index}"
            initializers.extend(
                (
                    helper.make_tensor(
                        weight_name,
                        onnx.TensorProto.FLOAT,
                        [channels, 3, 1, 1],
                        [0.0] * (channels * 3),
                    ),
                    helper.make_tensor(
                        shape_name,
                        onnx.TensorProto.INT64,
                        [3],
                        [1, channels, -1],
                    ),
                )
            )
            nodes.extend(
                (
                    helper.make_node(
                        "Conv",
                        ["images", weight_name],
                        [conv_name],
                        name=f"head_{role_index}_{scale_index}",
                        strides=[stride, stride],
                    ),
                    helper.make_node(
                        "Reshape",
                        [conv_name, shape_name],
                        [flat_name],
                        name=f"flatten_{role_index}_{scale_index}",
                    ),
                )
            )
            flattened.append(flat_name)
        nodes.append(
            helper.make_node(
                "Concat",
                flattened,
                [name],
                name=f"output_{role_index}",
                axis=2,
            )
        )
    outputs = [
        helper.make_tensor_value_info(
            name,
            onnx.TensorProto.FLOAT,
            [1, channels, sum(height * width for height, width in feature_shapes)],
        )
        for name, channels in zip(preferred, role_channels, strict=True)
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "cam2-test",
            [input_value],
            outputs,
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    source = tmp_path / "model.onnx"
    onnx.save(model, source)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    for index in range(20):
        Image.new("RGB", (8, 8), (index, 0, 0)).save(
            calibration / f"{index:02d}.jpg"
        )
    request = MaixConversionRequest(
        target=MaixTarget.MAIXCAM2,
        model_key="YOLOv8n",
        onnx_path=source,
        output_dir=tmp_path / "output",
        calibration_dir=calibration,
        class_names=("person",),
        input_height=224,
        input_width=320,
        calibration_count=20,
    )
    plan = build_conversion_plan(request)
    paths = plan.materialize()
    assert request.output_dir / "export.onnx" in paths
    assert (request.output_dir / "export.onnx").is_file()
    with tarfile.open(request.output_dir / "datasets" / "train.tar") as archive:
        assert len(archive.getmembers()) == 20


def test_target_specific_calibration_range(tmp_path) -> None:
    request = _request(tmp_path, MaixTarget.MAIXCAM2)
    invalid = MaixConversionRequest(
        **{
            field: getattr(request, field)
            for field in request.__dataclass_fields__
            if field != "calibration_count"
        },
        calibration_count=101,
    )
    with pytest.raises(ValueError, match="20 到 100"):
        build_conversion_plan(invalid, perform_gate=False)
