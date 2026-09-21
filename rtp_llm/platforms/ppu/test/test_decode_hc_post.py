"""Opt-in HC post geometry preserves in-place and Graph semantics."""
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit


class HCPostPolicyTest(unittest.TestCase):
    def test_provider_option(self):
        for threads in (128, 256):
            provider = PpuDecodeProvider({**DECODE_EXECUTION_OPTIONS,
                                         'DSV4_PPU_MTP_HC_POST_THREADS': str(threads)})
            with patch('rtp_llm.platforms.ppu.models.dsv4.ppu_hc.PpuHCUnit') as unit:
                provider.build_hc_unit()
                self.assertEqual(unit.call_args.kwargs['post_threads'], threads)
        self.assertEqual(PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)._hc_post_threads, 128)
        with self.assertRaisesRegex(ValueError, 'HC_POST_THREADS'):
            PpuDecodeProvider({**DECODE_EXECUTION_OPTIONS, 'DSV4_PPU_MTP_HC_POST_THREADS': '64'})

    def test_shape_guard_and_empty(self):
        unit = object.__new__(PpuHCUnit)
        unit._post_threads = 256
        unit.dim = 4096
        unit._post_kernel = Mock()
        for rows, hc, expected in ((12, 4, 256), (320, 4, 256), (321, 4, 128), (12, 2, 128)):
            residual = torch.empty(rows, hc, 4096, device='meta')
            unit._post_operator(None, residual, None, None, hc_mult=hc)
            self.assertEqual(unit._post_kernel.call_args.kwargs['n_thr'], expected)
        empty = torch.empty(0, 4, 4096, device='meta')
        unit._post_kernel.reset_mock()
        self.assertIs(unit._post_operator(None, empty, None, None, hc_mult=4), empty)
        unit._post_kernel.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_name() == 'ZW-M890P',
                     'requires PPU M890P')
class HCPostGPUValidation(unittest.TestCase):
    @torch.inference_mode()
    def test_exact_inplace_graph_all_buckets(self):
        import importlib
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401
        post_fn = importlib.import_module(
            'rtp_llm.models_py.3rdparty.tile_kernels.mhc.post_kernel').mhc_post_fwd
        buckets = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64, 80)
        for rows in sorted(set(buckets) | {b*4 for b in buckets}):
            with self.subTest(rows=rows):
                source = torch.randn(rows, 1, 4, 4096, device='cuda', dtype=torch.bfloat16)
                x = torch.randn(rows, 1, 4096, device='cuda', dtype=torch.bfloat16)
                post = torch.rand(rows, 1, 4, 1, device='cuda')
                comb = torch.rand(rows, 1, 4, 4, device='cuda')
                residual = torch.empty_like(source)

                def run():
                    residual.copy_(source)
                    return post_fn(x, residual, post, comb, out=residual, enable_pdl=False, n_thr=256)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = run()
                torch.cuda.current_stream().wait_stream(stream)
                self.assertEqual(out.data_ptr(), residual.data_ptr())
                for _ in range(3):
                    source.normal_()
                    x.normal_()
                    expected = post_fn(x, source, post, comb, enable_pdl=False)
                    residual.fill_(float('nan'))
                    graph.replay()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    self.assertTrue(bool(out.isfinite().all()))
                torch.cuda.synchronize()
                del graph


if __name__ == '__main__':
    unittest.main()
