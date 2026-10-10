import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.platforms.ppu.models.dsv4.ppu_tp_moe import PpuTPMoE


class PpuTPMoEFp32ReduceTest(unittest.TestCase):
    def test_opt_in_reduces_in_fp32_and_returns_bf16(self):
        local = torch.ones((2, 3), dtype=torch.bfloat16)
        reduce = Mock(return_value=torch.full((2, 3), 2.25, dtype=torch.float32))
        moe = SimpleNamespace(_tp_fp32_reduce=True, _reduce=reduce)

        result = PpuTPMoE._reduce_local(moe, local)

        self.assertEqual(reduce.call_args.args[0].dtype, torch.float32)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, torch.full_like(local, 2.25)))

    def test_default_uses_existing_bf16_path(self):
        local = torch.ones((2, 3), dtype=torch.bfloat16)
        reduce = Mock(return_value=local)
        moe = SimpleNamespace(_tp_fp32_reduce=False, _reduce=reduce)

        result = PpuTPMoE._reduce_local(moe, local)

        self.assertIs(result, local)
        self.assertIs(reduce.call_args.args[0], local)


if __name__ == "__main__":
    unittest.main()
