import os
from typing import Any, Callable
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch
import torch
from rtp_llm.models_py.modules.dsv4.fp8 import compressor
import unittest
from rtp_llm.platforms.ppu.models.dsv4.ppu_provider import M890PDsv4Provider


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires PPU M890P",
)
class PrefillCompressorWorkspaceTest(unittest.TestCase):
    def test_projection_matches_original_in_union_workspace(self):
        with torch.inference_mode():
            N = 32769
            x = torch.randn(N, 4096, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
            buf = torch.empty(N, 8192, device="cuda", dtype=torch.bfloat16)
            outputs = []
            obj = SimpleNamespace(
                _state_pool_3d=object(),
                _kv_pool_view=object(),
                _kv_eb=256,
                overlap=1,
                head_dim=512,
                _wkv_wgate_fused=weight,
                _cp_ctx=None,
                _bf16_fp32_linear=partial(
                    M890PDsv4Provider.run_bf16_fp32_linear, None, None
                ),
                _launch=lambda kv, score, meta, **kw: outputs.append(
                    torch.cat([kv, score], dim=-1)
                ),
            )
            meta = SimpleNamespace(is_batched=True)
            for flag in ["0", "1"]:
                with patch.dict(
                    os.environ, DSV4_PREFILL_REUSE_COMPRESSOR_WORKSPACE=flag
                ):
                    compressor.CompressorFP8.forward(
                        obj,
                        x,
                        0,
                        meta=meta,
                        workspace=SimpleNamespace(prefill_q=lambda n: buf),
                    )
            torch.cuda.synchronize()
            print(
                "diff",
                int((outputs[0] != outputs[1]).sum()),
                "maxdiff",
                float((outputs[0] - outputs[1]).abs().max()),
                flush=True,
            )
            assert torch.equal(outputs[0], outputs[1])
            print(
                "PASS full versus workspace projection, rows=32769 hidden=4096 fused=2048",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
