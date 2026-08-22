import unittest

import torch

from rtp_llm.models_py.modules.dsv4 import utils
from rtp_llm.models_py.modules.dsv4.fp8 import attention


class PpuDeepGemmLayoutCompatTest(unittest.TestCase):
    def setUp(self) -> None:
        if utils.get_mn_major_tma_aligned_packed_ue8m0_tensor is not None:
            self.skipTest("upstream DeepGEMM layout package is available")

    def test_plain_block_scale_fallback(self) -> None:
        scale = torch.ones((2, 4), dtype=torch.float8_e8m0fnu)
        result = utils._repack_v4_fp8_scale_to_int32(scale)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(tuple(result.shape), (2, 4))
        self.assertTrue(result.is_contiguous())

    def test_wo_a_uses_plain_grouped_block_grid(self) -> None:
        weight = torch.zeros((256, 256), dtype=torch.float8_e4m3fn)
        scale = torch.ones((2, 2), dtype=torch.float8_e8m0fnu)
        weight_stacked, scale_stacked = attention._prepare_wo_a_stacked(
            weight, scale, 2, 128, 256
        )
        self.assertEqual(tuple(weight_stacked.shape), (2, 128, 256))
        self.assertEqual(scale_stacked.dtype, torch.float32)
        self.assertEqual(tuple(scale_stacked.shape), (2, 1, 2))
        self.assertTrue(scale_stacked.is_contiguous())


if __name__ == "__main__":
    unittest.main()
