import unittest

import torch


class PpuInt8QuantTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA-compatible device is unavailable")

    def test_matches_dynamic_per_token_reference(self):
        from rtp_llm.models_py.kernels.cuda.ppu_int8_quant import (
            per_token_quant_int8_triton,
        )

        torch.manual_seed(11)
        for rows, hidden_size in ((1, 1536), (16, 5120), (31, 12288)):
            x = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
            output_q, output_s = per_token_quant_int8_triton(x)

            x_float = x.float()
            ref_s = (x_float.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-10)
            ref_q = torch.round(x_float / ref_s).clamp(-128, 127).to(torch.int8)

            torch.testing.assert_close(output_s, ref_s, rtol=2e-3, atol=1e-6)
            torch.testing.assert_close(output_q, ref_q, rtol=0, atol=1)


if __name__ == "__main__":
    unittest.main()
