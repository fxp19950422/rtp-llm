"""CPU integration of the real V4 FP8 wrapper and AtomicWeight splitter."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import V4PerBlockFp8Weight
from rtp_llm.models_py.modules.dsv4.shared_tp import shared_fp8_tp_slice
from rtp_llm.utils.model_weight import CkptWeightInfo, W, concat_0, identity


class V4SharedTpLoaderTest(unittest.TestCase):
    def test_pair_preserves_config_and_atomic_split_uses_shared_rank(self):
        for size in (1, 4):
            for rank in range(size):
                cfg = SimpleNamespace(shared_tp_size=size, shared_tp_rank=rank)
                src = AttnAtomicWeight(
                    W.v4_shared_w13_w,
                    [CkptWeightInfo("layers.{i}.ffn.shared_experts.w1.weight", identity),
                     CkptWeightInfo("layers.{i}.ffn.shared_experts.w3.weight", identity)],
                    concat_0, config=cfg,
                )
                parent = V4PerBlockFp8Weight.__new__(V4PerBlockFp8Weight)
                kernel, scale = parent._v4_build_pair(src, W.v4_shared_w13_w, W.v4_shared_w13_s)
                load_cfg = LoadConfig.model_construct(
                    tp_size=4, tp_rank=0, ep_size=1, ep_rank=0, dp_size=1, dp_rank=0,
                    ffn_tp_size=1, ffn_tp_rank=0, hidden_size=128, head_num=1,
                    head_num_kv=1, size_per_head=128, moe_pure_tp_mode=True, bit=8,
                )
                self.assertIs(kernel.config, cfg)
                self.assertIs(scale.config, cfg)
                self.assertEqual(
                    [ckpt.name for ckpt in scale.weights],
                    ["layers.{i}.ffn.shared_experts.w1.scale",
                     "layers.{i}.ffn.shared_experts.w3.scale"],
                )
                for weight, is_scale, shape in (
                    (kernel, False, (1024, 128)), (scale, True, (8, 1)),
                ):
                    data = torch.arange(shape[0] * shape[1], dtype=torch.int64)
                    raw = (data * 19 + data // 131).to(torch.uint8).reshape(shape).view(
                        torch.float8_e8m0fnu if is_scale else torch.float8_e4m3fn
                    )
                    actual = weight._split(raw, load_cfg)[weight.name]
                    expected = shared_fp8_tp_slice(
                        raw, projection="w13", tp_size=size, tp_rank=rank, is_scale=is_scale
                    )
                    with self.subTest(size=size, rank=rank, scale=is_scale):
                        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
