import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    _use_ppu_reuse_fp32_tp_reduce,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import output_proj


class OutputProjectionAllReduceTest(unittest.TestCase):
    def test_prefill_consumes_independent_all_reduce_result(self):
        attn = SimpleNamespace(tp_size=2)
        out = torch.zeros((2, 3), dtype=torch.float32)
        reduced = torch.full_like(out, 7.0)

        with patch(
            "rtp_llm.models_py.distributed.collective_torch.all_reduce",
            return_value=reduced,
        ) as all_reduce:
            AttentionFP8._prefill_output_all_reduce(attn, out)

        self.assertTrue(torch.equal(out, reduced))
        self.assertTrue(all_reduce.call_args.kwargs["inplace"])

    def test_prefill_fp32_reduce_casts_result_back_to_bf16(self):
        attn = SimpleNamespace(tp_size=4, _prefill_tp_fp32_reduce=True)
        out = torch.zeros((2, 3), dtype=torch.bfloat16)
        reduced = torch.full((2, 3), 7.25, dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.distributed.collective_torch.all_reduce",
            return_value=reduced,
        ) as all_reduce:
            AttentionFP8._prefill_output_all_reduce(attn, out)

        self.assertTrue(torch.equal(out, reduced.to(torch.bfloat16)))
        self.assertEqual(all_reduce.call_args.args[0].dtype, torch.float32)
        self.assertTrue(all_reduce.call_args.kwargs["inplace"])

    def test_prefill_fp32_reduce_requires_ppu_reuse_opt_in(self):
        def enabled(name, options):
            provider = SimpleNamespace(name=name, execution_options=options)
            return _use_ppu_reuse_fp32_tp_reduce(provider)

        opts = {"DSV4_PPU_PREFIX_REUSE": "1", "DSV4_PPU_TP_FP32_REDUCE": "1"}
        self.assertFalse(enabled("cuda", opts))
        self.assertFalse(enabled("m890p-dsv4-candidate", {"DSV4_PPU_TP_FP32_REDUCE": "1"}))
        self.assertFalse(enabled("m890p-dsv4-candidate", {"DSV4_PPU_PREFIX_REUSE": "1"}))
        self.assertTrue(enabled("m890p-dsv4-candidate", opts))

    def test_decode_consumes_independent_all_reduce_result(self):
        local_out = torch.zeros((1, 1, 3), dtype=torch.float32)
        reduced = torch.full_like(local_out, 5.0)
        attn = SimpleNamespace(
            rope_head_dim=2,
            head_dim=2,
            n_groups=1,
            n_heads=1,
            o_lora_rank=1,
            wo_a=None,
            wo_a_w=torch.empty(1),
            wo_a_s=torch.empty(1),
            wo_b=object(),
            tp_size=2,
            _lin=lambda _layer, _x: local_out,
        )
        o = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
        freqs = torch.zeros((1, 1), dtype=torch.float32)

        with (
            patch.object(output_proj, "apply_rotary_emb_batched"),
            patch.object(
                output_proj,
                "dequantize_fp8_weight",
                return_value=torch.ones((1, 2), dtype=torch.float32),
            ),
            patch(
                "rtp_llm.models_py.distributed.collective_torch.all_reduce",
                return_value=reduced,
            ) as all_reduce,
        ):
            result = output_proj.decode_output_proj(attn, o, freqs, 1, 1)

        self.assertIs(result, local_out)
        self.assertTrue(torch.equal(result, reduced))
        self.assertTrue(all_reduce.call_args.kwargs["inplace"])


if __name__ == "__main__":
    unittest.main()
