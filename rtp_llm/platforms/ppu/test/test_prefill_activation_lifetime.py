import gc, weakref
from types import SimpleNamespace
import torch
import unittest
from unittest.mock import patch
from rtp_llm.models_py.modules.dsv4 import block


class PrefillActivationLifetimeTest(unittest.TestCase):
    def test_attention_activation_released_before_ffn(self):
        refs = []

        def pre(x):
            v = x.clone()
            refs.append(weakref.ref(v))
            return v, None, None

        def ffn_pre(x):
            gc.collect()
            assert (
                refs[0]() is None
            ), "attention activation still held before FFN allocation"
            return x.clone(), None, None

        class Attn:
            def __call__(self, x, *a, **kw):
                return x

        obj = SimpleNamespace(
            tp_size=4,
            tp_rank=0,
            attn=Attn(),
            attn_norm=None,
            ffn_norm=None,
            _prefill_fast_hc_impls=lambda: (
                pre,
                ffn_pre,
                lambda out, res, p, c: res,
                lambda out, res, p, c: res,
            ),
            _sync_after_first_cp_prefill_attention=lambda: None,
            ffn=lambda x, ids: x,
        )
        x = torch.ones(3, 4)
        with patch.object(
            block, "_prefill_fast_norm", side_effect=lambda norm, x, **kw: x
        ):
            out = block.Block._forward_prefill_fast_fp8(
                obj, x, torch.ones(3, dtype=torch.int64), torch.arange(3), None, None
            )
        assert out is x
        print("PASS activation reference released before FFN HC-pre")


if __name__ == "__main__":
    unittest.main()
