import unittest

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import V4PerBlockFp8Weight
from rtp_llm.models.deepseek_v4 import DeepSeekV4MtpWeight
from rtp_llm.utils.model_weight import W


def _mtp_weight_descriptor() -> DeepSeekV4MtpWeight:
    weight = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
    weight._num_layers = 1
    weight._compress_ratios = [0]
    weight._num_hash_layers = 0
    weight._hidden_size = 4096
    weight._size_per_head = 512
    weight._head_num = 64
    weight._head_num_kv = 1
    weight.expert_num_ = 256
    weight._moe_align_size = 128
    weight.enable_fp32_lm_head = False
    return weight


class DeepSeekV4MtpQuantWeightTest(unittest.TestCase):
    def test_projection_scales_are_owned_by_fp8_composites(self):
        info = _mtp_weight_descriptor()._get_weight_info().to_quant_weight_info(
            Fp8BlockWiseQuantConfig(is_quanted=True)
        )
        by_name = {weight.name: weight for weight in info.weights}

        self.assertEqual(len(by_name), len(info.weights), "duplicate global W keys")
        for weight_name, scale_name in (
            (W.v4_mtp_e_proj_w, W.v4_mtp_e_proj_s),
            (W.v4_mtp_h_proj_w, W.v4_mtp_h_proj_s),
        ):
            projection = by_name[weight_name]
            self.assertIsInstance(projection, V4PerBlockFp8Weight)
            self.assertNotIn(scale_name, by_name)
            self.assertEqual(projection.scale.name, scale_name)
            self.assertEqual(projection.scale.data_type, torch.float8_e8m0fnu)


if __name__ == "__main__":
    unittest.main()
