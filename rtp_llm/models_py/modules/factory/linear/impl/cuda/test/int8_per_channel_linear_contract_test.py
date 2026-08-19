#!/usr/bin/env python3
"""Direct contract test for CudaInt8PerChannelLinear W8A8 fast path.

Mirrors ``fp8_deepgemm_linear_quantized_input_contract_test.py``: loads
``int8_per_channel_linear.py`` with tiny dependency stubs (no compiled
RTP-LLM .so, no PPU INT8 kernels) and verifies only the Python-side
contract added for the true W8A8 INT8 GEMM path -- ``quantize_input`` /
``forward_quantized`` reuse, ``out`` buffer honouring, N-D input reshape,
and the dequant fallback. Real DeepGEMM numerics stay in the Bazel tests.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


def _install_stubs(calls):
    int8_gemm = types.ModuleType("rtp_llm.models_py.modules.dsv4.int8_gemm")

    def has_int8_dense_gemm():
        return True

    def quantize_per_token_int8(x_2d):
        calls.append(("quant", tuple(x_2d.shape), x_2d.dtype))
        M, K = x_2d.shape
        return (
            torch.ones((M, K), dtype=torch.int8, device=x_2d.device),
            torch.ones((M, 1), dtype=torch.float32, device=x_2d.device),
        )

    def int8_dense_gemm(x_i8, x_scale, w_i8, w_scale, out=None):
        calls.append(("int8_gemm", x_i8, x_scale, w_i8, w_scale))
        M = x_i8.shape[0]
        N = w_i8.shape[0]
        if out is None:
            out = torch.empty((M, N), dtype=torch.bfloat16, device=x_i8.device)
        out.fill_(3.0)
        return out

    int8_gemm.has_int8_dense_gemm = has_int8_dense_gemm
    int8_gemm.quantize_per_token_int8 = quantize_per_token_int8
    int8_gemm.int8_dense_gemm = int8_dense_gemm

    linear_pkg = types.ModuleType("rtp_llm.models_py.modules.factory.linear")

    class LinearBase(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    linear_pkg.LinearBase = LinearBase

    ops_mod = types.ModuleType("rtp_llm.ops")
    ops_mod.HWKernelConfig = object

    sys.modules["rtp_llm.models_py.modules.dsv4.int8_gemm"] = int8_gemm
    sys.modules["rtp_llm.models_py.modules.factory.linear"] = linear_pkg
    sys.modules["rtp_llm.ops"] = ops_mod


def _load_module(calls):
    _install_stubs(calls)
    path = Path(__file__).resolve().parents[1] / "int8_per_channel_linear.py"
    spec = importlib.util.spec_from_file_location("int8_per_channel_contract", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _QuantConfig:
    def get_method(self):
        return "INT8_PER_CHANNEL_COMPRESSED"


class Int8PerChannelContractTest(unittest.TestCase):
    def _make_linear(self, mod, *, bias=False):
        weight = torch.ones((3, 4), dtype=torch.int8)
        scale = torch.full((3, 1), 2.0, dtype=torch.float32)
        b = torch.ones((3,), dtype=torch.bfloat16) if bias else None
        return mod.CudaInt8PerChannelLinear(
            weight, scale, bias=b, quant_config=_QuantConfig()
        )

    def test_init_normalises_scale_and_shape(self):
        mod = _load_module([])
        layer = self._make_linear(mod)
        self.assertEqual((layer.N, layer.K), (3, 4))
        self.assertEqual(tuple(layer.weight_scale.shape), (3, 1))
        self.assertEqual(layer.weight_scale.dtype, torch.float32)
        self.assertTrue(layer._int8_gemm_ok)

    def test_forward_uses_quantize_then_int8_gemm(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        x = torch.zeros((2, 4), dtype=torch.bfloat16)

        out = layer(x)

        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertEqual(calls[0][0], "quant")
        self.assertEqual(calls[0][1], (2, 4))
        self.assertEqual(calls[1][0], "int8_gemm")
        self.assertEqual(calls[1][1].dtype, torch.int8)
        self.assertEqual(tuple(calls[1][1].shape), (2, 4))
        self.assertEqual(calls[1][2].dtype, torch.float32)
        self.assertEqual(tuple(calls[1][2].shape), (2, 1))

    def test_forward_nd_input_reshapes_and_restores(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        x = torch.zeros((2, 5, 4), dtype=torch.bfloat16)

        out = layer(x)

        # N-D collapses to [M=2*5, K=4] for the 2D GEMM, then restores leading.
        self.assertEqual(tuple(out.shape), (2, 5, 3))
        self.assertEqual(calls[0][1], (10, 4))

    def test_forward_quantized_reuses_supplied_quant_tuple(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod, bias=True)
        x_i8 = torch.ones((2, 4), dtype=torch.int8)
        x_scale = torch.ones((2, 1), dtype=torch.float32)

        out = layer.forward_quantized(x_i8, x_scale)

        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "int8_gemm")
        self.assertIs(calls[0][1], x_i8)
        self.assertIs(calls[0][2], x_scale)
        # GEMM stub fills 3.0; bias of ones adds to 4.0.
        self.assertTrue(torch.all(out == torch.tensor(4.0, dtype=out.dtype)))

    def test_forward_quantized_respects_out_buffer(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        x_i8 = torch.ones((2, 4), dtype=torch.int8)
        x_scale = torch.ones((2, 1), dtype=torch.float32)
        out = torch.empty((2, 3), dtype=torch.bfloat16)

        got = layer.forward_quantized(x_i8, x_scale, out=out)

        self.assertIs(got, out)
        self.assertTrue(torch.all(out == torch.tensor(3.0, dtype=out.dtype)))

    def test_forward_falls_back_to_dequant_when_unavailable(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        # Simulate a platform without the INT8 kernel.
        layer._int8_gemm_ok = False
        x = torch.full((2, 4), 0.5, dtype=torch.bfloat16)

        out = layer(x)

        # No INT8 GEMM path was taken.
        self.assertEqual(calls, [])
        # Reference: F.linear(x, weight.to(fp32) * scale).
        w = (layer.weight.to(torch.float32) * layer.weight_scale).to(torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertTrue(torch.allclose(out, ref))

    def test_forward_latches_to_dequant_on_gemm_failure(self):
        calls = []
        mod = _load_module(calls)
        layer = self._make_linear(mod)
        int8_gemm = sys.modules["rtp_llm.models_py.modules.dsv4.int8_gemm"]

        def boom(*args, **kwargs):
            raise RuntimeError("kernel rejects this shape")

        int8_gemm.int8_dense_gemm = boom
        x = torch.zeros((2, 4), dtype=torch.bfloat16)

        # First call raises internally, latches off, returns via dequant.
        out1 = layer(x)
        self.assertFalse(layer._int8_gemm_ok)
        self.assertEqual(tuple(out1.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
