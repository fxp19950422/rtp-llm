import unittest

import torch


class SiluMulQuantInt8GlmTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA-compatible device is unavailable")

    def test_matches_up_gate_reference(self):
        from rtp_llm.ops.compute_ops import silu_mul_quant_int8_glm

        torch.manual_seed(7)
        for rows, hidden_size in ((1, 128), (2, 128), (16, 128), (31, 12288)):
            up = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(up)
            input_tensor = torch.cat((up, gate), dim=-1).contiguous()
            output_q = torch.empty_like(up, dtype=torch.int8)
            output_s = torch.empty(
                rows, 1, device=input_tensor.device, dtype=torch.float32
            )

            silu_mul_quant_int8_glm(
                input_tensor,
                output_q,
                output_s,
                hidden_size,
                1e-12,
                -128.0,
                127.0,
            )

            activated = torch.nn.functional.silu(gate.float()) * up.float()
            ref_s = (activated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(
                min=1e-12
            )
            ref_q = (activated / ref_s).clamp(-128, 127).to(torch.int8)

            torch.testing.assert_close(output_s, ref_s, rtol=2e-3, atol=1e-6)
            torch.testing.assert_close(output_q, ref_q, rtol=0, atol=1)


if __name__ == "__main__":
    unittest.main()
