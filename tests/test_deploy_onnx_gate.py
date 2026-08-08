from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ai_biaozhu.deploy.maix import (
    MaixTarget,
    expected_output_tensors,
    resolve_output_tensors,
)
from ai_biaozhu.deploy.onnx_gate import inspect_onnx, inspect_onnx_numerics


def _value_info(name, shape):
    dimensions = [SimpleNamespace(dim_value=value, dim_param="") for value in shape]
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dimensions))
        ),
    )


def _model(
    shape,
    outputs=("/wanted",),
    opset=17,
    tensor_shapes=None,
    graph_output_shape=(1, 84, 8400),
):
    dimensions = [
        SimpleNamespace(dim_value=value, dim_param="")
        for value in shape
    ]
    input_value = SimpleNamespace(
        name="images",
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dimensions))
        ),
    )
    graph = SimpleNamespace(
        input=[input_value],
        initializer=[],
        output=[_value_info("output0", graph_output_shape)],
        node=[SimpleNamespace(output=list(outputs))],
        value_info=[
            _value_info(name, value)
            for name, value in (tensor_shapes or {}).items()
        ],
    )
    return SimpleNamespace(
        graph=graph,
        opset_import=[SimpleNamespace(domain="", version=opset)],
    )


def test_onnx_gate_accepts_static_opset17_and_intermediate_tensor(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not-a-real-onnx")
    report = inspect_onnx(
        path,
        expected_tensors=["/wanted"],
        loader=lambda _: _model((1, 3, 224, 320)),
        checker=lambda _: None,
    )
    assert report.ok
    assert report.input_shape == (1, 3, 224, 320)
    assert report.sha256


def test_onnx_gate_rejects_dynamic_wrong_opset_and_missing_output(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"x")
    model = _model((1, 3, 0, 320), outputs=("/other",), opset=18)
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
    report = inspect_onnx(
        path,
        expected_tensors=["/wanted"],
        loader=lambda _: model,
    )
    codes = {issue.code for issue in report.errors}
    assert {"opset", "dynamic_shape", "output_tensors"}.issubset(codes)


def test_onnx_gate_shape_inference_failure_and_dynamic_output_are_hard_errors(
    tmp_path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"x")

    def fail_shape_inference(model):
        del model
        raise RuntimeError("cannot infer")

    inference_report = inspect_onnx(
        path,
        loader=lambda _: _model((1, 3, 640, 640)),
        shape_inferer=fail_shape_inference,
    )
    assert "shape_inference_failed" in {
        issue.code for issue in inference_report.errors
    }
    dynamic_report = inspect_onnx(
        path,
        loader=lambda _: _model(
            (1, 3, 640, 640),
            graph_output_shape=(1, None, 8400),
        ),
    )
    assert "dynamic_output_shape" in {
        issue.code for issue in dynamic_report.errors
    }


def test_output_discovery_uses_unique_name_and_semantic_contract(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"x")
    names = (
        "/custom/detect/concat_output",
        "/custom/detect/concat_1_output",
        "/custom/detect/concat_2_output",
    )
    shapes = {
        names[0]: (1, 64, 8400),
        names[1]: (1, 1, 8400),
        names[2]: (1, 4, 8400),
    }
    report = inspect_onnx(
        path,
        loader=lambda _: _model((1, 3, 640, 640), names, tensor_shapes=shapes),
    )
    selected = resolve_output_tensors(
        report,
        target=MaixTarget.MAIXCAM2,
        model_key="YOLOv8n",
        class_names=("object",),
    )
    assert set(selected) == set(names)


def test_output_discovery_fails_closed_and_allows_advanced_override(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"x")
    names = (
        "/custom/detect/concat_output",
        "/custom/detect/concat_1_output",
        "/custom/detect/concat_2_output",
        "/alternative/detect/concat_output",
    )
    report = inspect_onnx(
        path,
        loader=lambda _: _model(
            (1, 3, 640, 640),
            names,
            tensor_shapes={
                names[0]: (1, 64, 8400),
                names[1]: (1, 1, 8400),
                names[2]: (1, 4, 8400),
                names[3]: (1, 64, 8400),
            },
        ),
    )
    with pytest.raises(ValueError, match="Netron"):
        resolve_output_tensors(
            report,
            target=MaixTarget.MAIXCAM2,
            model_key="YOLOv8n",
            class_names=("object",),
        )
    assert resolve_output_tensors(
        report,
        target=MaixTarget.MAIXCAM2,
        model_key="YOLOv8n",
        class_names=("object",),
        override=names[:3],
    ) == names[:3]


@pytest.mark.parametrize(
    ("target", "model_key", "class_names", "shapes"),
    [
        (
            MaixTarget.MAIXCAM_PRO,
            "YOLOv5n",
            ("cat", "dog"),
            (
                (1, 21, 80, 80),
                (1, 21, 40, 40),
                (1, 21, 20, 20),
            ),
        ),
        (
            MaixTarget.MAIXCAM_PRO,
            "YOLOv8n",
            ("cat", "dog"),
            ((1, 1, 4, 8400), (1, 2, 8400)),
        ),
        (
            MaixTarget.MAIXCAM2,
            "YOLO11n",
            ("cat", "dog"),
            (
                (1, 64, 8400),
                (1, 2, 8400),
                (None, 4, 8400),
            ),
        ),
        (
            MaixTarget.MAIXCAM2,
            "YOLO26n",
            ("cat", "dog"),
            (
                (1, 4, 80, 80),
                (1, 4, 40, 40),
                (1, 4, 20, 20),
                (1, 2, 80, 80),
                (1, 2, 40, 40),
                (1, 2, 20, 20),
            ),
        ),
    ],
)
def test_output_semantics_accept_each_decoder_contract(
    tmp_path,
    target,
    model_key,
    class_names,
    shapes,
) -> None:
    path = tmp_path / f"{model_key}.onnx"
    path.write_bytes(b"x")
    names = expected_output_tensors(target, model_key)
    report = inspect_onnx(
        path,
        loader=lambda _: _model(
            (1, 3, 640, 640),
            names,
            tensor_shapes=dict(zip(names, shapes, strict=True)),
        ),
    )
    assert resolve_output_tensors(
        report,
        target=target,
        model_key=model_key,
        class_names=class_names,
    ) == names


def test_official_and_override_nodes_cannot_bypass_output_semantics(tmp_path) -> None:
    path = tmp_path / "identity.onnx"
    path.write_bytes(b"x")
    names = expected_output_tensors(MaixTarget.MAIXCAM2, "YOLOv8n")
    report = inspect_onnx(
        path,
        loader=lambda _: _model(
            (1, 3, 640, 640),
            names,
            tensor_shapes={name: (1, 3, 640, 640) for name in names},
        ),
    )
    with pytest.raises(ValueError, match="语义不匹配"):
        resolve_output_tensors(
            report,
            target=MaixTarget.MAIXCAM2,
            model_key="YOLOv8n",
            class_names=("object",),
        )

    valid_shapes = {
        names[0]: (1, 64, 8400),
        names[1]: (1, 1, 8400),
        names[2]: (1, 4, 8400),
    }
    valid_report = inspect_onnx(
        path,
        loader=lambda _: _model(
            (1, 3, 640, 640),
            names,
            tensor_shapes=valid_shapes,
        ),
    )
    with pytest.raises(ValueError, match="语义不匹配"):
        resolve_output_tensors(
            valid_report,
            target=MaixTarget.MAIXCAM2,
            model_key="YOLOv8n",
            class_names=("object",),
            override=(names[1], names[0], names[2]),
        )


def test_onnx_runtime_and_pytorch_numeric_parity_passes() -> None:
    sample = np.full((1, 3, 32, 32), 0.25, dtype=np.float32)

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 32, 32])]

        def get_outputs(self):
            return [SimpleNamespace(name="output0")]

        def run(self, names, feeds):
            assert names == ["output0"]
            return [feeds["images"] * 2]

    report = inspect_onnx_numerics(
        "model.onnx",
        input_array=sample,
        pytorch_outputs=[sample * 2],
        session_factory=lambda *args, **kwargs: Session(),
    )
    assert report.ok
    assert report.outputs[0].finite
    assert report.parity_outputs[0].passed
    assert report.parity_outputs[0].cosine_similarity > 0.999


def test_onnx_numeric_parity_fails_closed_on_drift() -> None:
    sample = np.ones((1, 3, 32, 32), dtype=np.float32)

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 32, 32])]

        def get_outputs(self):
            return [SimpleNamespace(name="output0")]

        def run(self, names, feeds):
            return [feeds["images"]]

    report = inspect_onnx_numerics(
        "model.onnx",
        input_array=sample,
        pytorch_outputs=[sample * 3],
        session_factory=lambda *args, **kwargs: Session(),
    )
    assert not report.ok
    assert "pytorch_onnx_parity" in {issue.code for issue in report.errors}
