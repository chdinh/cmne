import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

import cmne  # noqa: E402


@pytest.fixture(scope="module")
def onnx_file(small_model, tmp_path_factory):
    return cmne.export_onnx(small_model, tmp_path_factory.mktemp("onnx") / "m.onnx")


def test_onnx_matches_torch(small_model, onnx_file, rng):
    pred = cmne.OnnxPredictor(onnx_file, providers=["CPUExecutionProvider"])
    assert pred.config == small_model.config
    for batch in (1, 5):
        w = rng.normal(size=(batch, 10, small_model.config.n_sources)).astype(np.float32)
        np.testing.assert_allclose(pred.predict(w), small_model.predict(w), rtol=1e-4, atol=1e-5)


def test_apply_cmne_with_onnx(small_model, onnx_file, rng):
    x = rng.normal(size=(small_model.config.n_sources, 40))
    a = cmne.apply_cmne(x, small_model).cmne
    b = cmne.apply_cmne(x, cmne.OnnxPredictor(onnx_file, num_threads=1)).cmne
    # float32 kernel differences compound through the recursive chain
    np.testing.assert_allclose(a, b, atol=1e-2)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9999
