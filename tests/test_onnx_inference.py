"""
Tests for ONNX Runtime inference integration and TEST_FIXTURE semantics.
Verifies that a synthetic ONNX test fixture validates the execution graph and runtime
WITHOUT ever falsely claiming to be a REAL_MODEL or production-ready checkpoint.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from affine import Affine

from satquery.core.raster_io import RasterData
from satquery.change_detection.models.base import (
    ModelArtifactStatus,
    RuntimeStatus,
    ValidationStatus,
    ModelStatus,
)
from satquery.change_detection.models.changeformer import ChangeFormerAdapter


def create_synthetic_changeformer_onnx(output_path: Path) -> Path:
    """
    Builds a minimal, syntactically valid bi-temporal ONNX compute graph.
    The graph takes two (1, 3, 256, 256) image tensors and produces a (1, 2, 256, 256) logits tensor.
    This validates ONNX runtime execution, tensor dimensions, and sliding window tiling
    without containing any genuine learned ChangeFormer weights.
    """
    import onnx
    from onnx import helper, TensorProto

    # Node 1: Subtract T1 from T2
    diff_node = helper.make_node("Sub", ["t2", "t1"], ["diff"])
    # Node 2: Absolute difference
    abs_node = helper.make_node("Abs", ["diff"], ["abs_diff"])
    # Node 3: Reduce mean across channel axis (1)
    reduce_node = helper.make_node("ReduceMean", ["abs_diff"], ["no_change_ch"], axes=[1], keepdims=1)
    # Node 4: Invert to create two-class logits [no_change, change]
    scale_node = helper.make_node("Neg", ["no_change_ch"], ["change_ch"])
    concat_node = helper.make_node("Concat", ["no_change_ch", "change_ch"], ["logits"], axis=1)

    graph = helper.make_graph(
        [diff_node, abs_node, reduce_node, scale_node, concat_node],
        "synthetic_changeformer_fixture",
        [
            helper.make_tensor_value_info("t1", TensorProto.FLOAT, [1, 3, 256, 256]),
            helper.make_tensor_value_info("t2", TensorProto.FLOAT, [1, 3, 256, 256]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 256, 256]),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def make_dummy_raster(width: int = 256, height: int = 256, fill_val: float = 0.2) -> RasterData:
    arr = np.full((3, height, width), fill_val, dtype=np.float32)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": width,
        "height": height,
        "count": 3,
        "crs": "EPSG:32643",
        "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3100000.0),
    }
    return RasterData(array=arr, meta=meta)


def test_synthetic_onnx_fixture_does_not_promote_to_real_model(tmp_path: Path):
    """
    CRITICAL ARCHITECTURAL TEST:
    A synthetic ONNX fixture validates ONNX Runtime execution but MUST NOT
    be classified as REAL_MODEL or REAL_CHECKPOINT.
    """
    onnx_file = tmp_path / "synthetic_changeformer.onnx"
    create_synthetic_changeformer_onnx(onnx_file)

    adapter = ChangeFormerAdapter(
        weights_path=onnx_file,
        patch_size=256,
        stride=192,
        is_test_fixture=True,
    )

    assert adapter.is_available() is True

    t1 = make_dummy_raster(width=256, height=256, fill_val=0.1)
    t2 = make_dummy_raster(width=256, height=256, fill_val=0.8)

    pred = adapter.predict(t1, t2)

    # 1. Provenance honesty
    assert pred.artifact_status == ModelArtifactStatus.TEST_FIXTURE
    assert pred.runtime_status == RuntimeStatus.INFERENCE_SUCCESS
    assert pred.validation_status == ValidationStatus.UNVALIDATED

    # 2. Composite model_status mapping MUST demote test fixtures to CLASSICAL_ALGORITHM
    assert pred.model_status == ModelStatus.CLASSICAL_ALGORITHM
    assert pred.model_status != ModelStatus.REAL_MODEL

    # 3. Production readiness MUST be False
    assert pred.is_production_ready is False
    assert pred.is_fallback is False

    # 4. Numerical output sanity
    assert pred.change_mask.shape == (256, 256)
    assert pred.probability_map.shape == (256, 256)
    assert np.all((pred.probability_map >= 0.0) & (pred.probability_map <= 1.0))


def test_missing_weights_visibly_falls_back():
    """
    Ensures that when learned model weights are missing, fallback is transparently
    visible in both model_status (HEURISTIC_FALLBACK) and fallback_reason.
    """
    adapter = ChangeFormerAdapter(
        weights_path=Path("non_existent_weights_xyz123.onnx"),
        fallback_on_missing=True,
    )

    assert adapter.is_available() is False

    t1 = make_dummy_raster(width=100, height=100, fill_val=0.2)
    t2 = make_dummy_raster(width=100, height=100, fill_val=0.7)

    pred = adapter.predict(t1, t2)

    assert pred.is_fallback is True
    assert pred.model_status == ModelStatus.HEURISTIC_FALLBACK
    assert pred.artifact_status == ModelArtifactStatus.UNAVAILABLE
    assert pred.fallback_reason is not None
    assert "ChangeFormer checkpoint unavailable" in pred.fallback_reason
    assert pred.is_production_ready is False
