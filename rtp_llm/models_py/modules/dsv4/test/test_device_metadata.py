"""GPU metadata equivalence, stable storage and capture/replay (not EP8 acceptance)."""
import os
import pickle
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4 import runtime_config
from rtp_llm.models_py.modules.dsv4.device_metadata import (
    device_metadata_enabled, device_start_positions,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    allocate_decode_metadata_fp8, update_decode_metadata_in_place_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImplFP8, DSv4DecodeFmhaImplConfigFP8,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies._expected_m import expected_m_for_tokens
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    SWA_KV, CSA_KV, HCA_KV, INDEXER_KV, CSA_STATE, HCA_STATE, INDEXER_STATE,
)


class DeviceMetadataTest(unittest.TestCase):
    def setUp(self):
        self.saved_values = runtime_config._VALUES.copy()
        runtime_config._VALUES.clear()

    def tearDown(self):
        runtime_config._VALUES.clear()
        runtime_config._VALUES.update(self.saved_values)

    def test_switches_independent_and_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(device_metadata_enabled(False))
            self.assertFalse(device_metadata_enabled(True))
        runtime_config._VALUES.clear()
        with mock.patch.dict(os.environ, {"DSV4_DECODE_DEVICE_METADATA": "1",
                                          "DSV4_VERIFY_DEVICE_METADATA": "0"}):
            self.assertTrue(device_metadata_enabled(False))
            self.assertFalse(device_metadata_enabled(True))

    def test_missing_device_mirror_fails_closed(self):
        with self.assertRaises(RuntimeError):
            device_start_positions(SimpleNamespace(sequence_lengths_plus_1_device=torch.ones(4)), False)
        with self.assertRaises(RuntimeError):
            device_start_positions(SimpleNamespace(), True)

    def test_expected_m_is_hint_not_capacity(self):
        for tokens in (0, 1, 4, 8, 16, 32, 80, 160, 320):
            self.assertEqual(expected_m_for_tokens(tokens, 8, 6, 256),
                             max(1, (tokens * 48 + 255) // 256))
        self.assertEqual(expected_m_for_tokens(320, 8, 6, 256), 60)
        self.assertEqual(expected_m_for_tokens(8, 8, 6, 256), 2)
        for args in ((-1, 8, 6, 256), (8, 0, 6, 256), (8, 8, 6, 0)):
            with self.assertRaises(ValueError):
                expected_m_for_tokens(*args)

    def test_scheduler_config_pickle_preserves_timeout(self):
        from rtp_llm.ops import FIFOSchedulerConfig
        config = FIFOSchedulerConfig()
        self.assertEqual(config.dp_fake_wait_ms, 10)
        config.dp_fake_wait_ms = 1
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.dp_fake_wait_ms, 1)
        self.assertIn("dp_fake_wait_ms: 1", restored.to_string())

    def test_paged_host_device_metadata_equivalence(self):
        self.assertTrue(torch.cuda.is_available())
        specs = {
            SWA_KV: (132, 16384, 2), CSA_KV: (32, 128, 256),
            INDEXER_KV: (32, 128, 256), HCA_KV: (1, 128, 256),
            CSA_STATE: (256, 256, 4), HCA_STATE: (256, 256, 4),
            INDEXER_STATE: (256, 256, 4),
        }

        def buffers(value):
            result = {}
            for key, item in vars(value).items():
                if isinstance(item, torch.Tensor):
                    result[key] = item
                elif isinstance(item, dict):
                    result.update({(key, tag): tensor for tag, tensor in item.items()
                                   if isinstance(tensor, torch.Tensor)})
            return result

        for phase, q_len in (("decode", 1), ("verify", 4), ("draft_prefill", 4)):
            for bucket in (2, 4):
                with self.subTest(phase=phase, bucket=bucket):
                    config = DSv4DecodeFmhaImplConfigFP8(
                        max_batch_size=bucket, q_len=q_len, window_size=128,
                        head_dim=512, max_seq_len=32768, compress_ratios=[0, 4, 128],
                        index_topk=512, paged_pool_specs=specs, group_tags=list(specs),
                    )
                    arms = []
                    for enabled in (False, True):
                        host = torch.zeros(bucket, dtype=torch.int32).pin_memory()
                        positions = host.to("cuda")
                        inputs = {tag: SimpleNamespace(
                            is_prefill=q_len > 1, is_target_verify=phase == "verify",
                            sequence_lengths=host, prefix_lengths=host,
                            sequence_lengths_plus_1_device=positions + 1,
                            prefix_lengths_device=positions,
                            kv_cache_kernel_block_id_device=torch.ones(
                                (bucket, columns), dtype=torch.int32, device="cuda"),
                        ) for tag, (_, _, columns) in specs.items()}
                        runtime_config._VALUES.clear()
                        with mock.patch.dict(os.environ, {
                            "DSV4_DECODE_DEVICE_METADATA": str(int(enabled)),
                            "DSV4_VERIFY_DEVICE_METADATA": str(int(enabled)),
                        }):
                            impl = DSv4DecodeFmhaImplFP8(config, torch.device("cuda"), inputs)
                        self.assertEqual(impl.cuda_graph_requires_host_metadata, not enabled)
                        pointers = {k: v.data_ptr() for k, v in buffers(impl.metadata).items()}
                        arms.append((inputs, impl, pointers))
                    boundaries = (0, 126, 127, 254, 16382)
                    for turn, active in enumerate((bucket, 1, bucket - 1, 1, bucket)):
                        expected = torch.tensor([
                            boundaries[(turn + row) % len(boundaries)] if row < active else 0
                            for row in range(bucket)
                        ], dtype=torch.int32)
                        for enabled, (inputs, impl, pointers) in zip((False, True), arms):
                            for index, (tag, attn) in enumerate(inputs.items()):
                                attn.prefix_lengths_device.copy_(expected)
                                attn.sequence_lengths_plus_1_device.copy_(expected + 1)
                                attn.sequence_lengths.copy_(expected)
                                if enabled:
                                    attn.sequence_lengths.fill_(17)  # Must not read host mirror.
                                columns = specs[tag][2]
                                table = (torch.arange(bucket * columns, device="cuda", dtype=torch.int32)
                                         .view(bucket, columns) + turn + index) % 8 + 1
                                table[active:].zero_()
                                attn.kv_cache_kernel_block_id_device.copy_(table)
                            impl.prepare_cuda_graph(inputs)
                            self.assertEqual(pointers, {
                                k: v.data_ptr() for k, v in buffers(impl.metadata).items()})
                            torch.testing.assert_close(impl.metadata.start_pos.cpu(), expected,
                                                       rtol=0, atol=0)
                        reference, candidate = [buffers(impl.metadata) for _, impl, _ in arms]
                        self.assertEqual(reference.keys(), candidate.keys())
                        for key in reference:
                            torch.testing.assert_close(reference[key], candidate[key],
                                                       rtol=0, atol=0, msg=str(key))

    def test_gpu_metadata_equivalence_and_graph_replay(self):
        # Fail, do not silently skip, if this GPU test is run without a device.
        self.assertTrue(torch.cuda.is_available())
        device = torch.device("cuda")
        for q_len in (1, 4):
            for bucket in (2, 4, 8):
                config = DSv4DecodeFmhaImplConfigFP8(
                    max_batch_size=bucket, q_len=q_len, window_size=16,
                    head_dim=512, max_seq_len=512, compress_ratios=[1, 4, 128],
                    index_topk=8,
                )
                host = torch.zeros(bucket, dtype=torch.int32)
                attn = SimpleNamespace(
                    is_prefill=q_len > 1, is_target_verify=q_len > 1,
                    sequence_lengths=host, prefix_lengths=host,
                    sequence_lengths_plus_1_device=torch.ones(bucket, dtype=torch.int32, device=device),
                    prefix_lengths_device=torch.zeros(bucket, dtype=torch.int32, device=device),
                )
                runtime_config._VALUES.clear()
                with mock.patch.dict(os.environ, {"DSV4_DECODE_DEVICE_METADATA": "1",
                                                  "DSV4_VERIFY_DEVICE_METADATA": "1"}):
                    impl = DSv4DecodeFmhaImplFP8(config, device, attn)
                self.assertFalse(impl.cuda_graph_requires_host_metadata)
                reference = allocate_decode_metadata_fp8(
                    max_batch_size=bucket, q_len=q_len, window_size=16, head_dim=512,
                    max_seq_len=512, compress_ratios=[1, 4, 128], index_topk=8, device=device,
                )
                pointers = {k: v.data_ptr() for k, v in vars(impl.metadata).items()
                            if isinstance(v, torch.Tensor)}
                # Captured consumer reads stable metadata; prepare stays outside graph.
                observed = torch.empty_like(impl.metadata.start_pos)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    observed.copy_(impl.metadata.start_pos)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    observed.copy_(impl.metadata.start_pos)
                for active in (bucket, 1, bucket - 1, bucket):
                    positions = torch.tensor(
                        [(i * 128 + active) % (512 - q_len) if i < active else 0
                         for i in range(bucket)], dtype=torch.int32, device=device,
                    )
                    attn.sequence_lengths_plus_1_device.copy_(positions + 1)
                    attn.prefix_lengths_device.copy_(positions)
                    # Poison host mirrors: fast path must not read them.
                    host.fill_(100)
                    impl.prepare_cuda_graph(attn)
                    update_decode_metadata_in_place_fp8(reference, positions, forbid_realloc=True)
                    graph.replay()
                    torch.testing.assert_close(observed, reference.start_pos, rtol=0, atol=0)
                    for key, value in vars(impl.metadata).items():
                        expected = getattr(reference, key)
                        if isinstance(value, torch.Tensor):
                            self.assertEqual(value.data_ptr(), pointers[key])
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)
                        elif isinstance(value, dict):
                            for subkey, tensor in value.items():
                                if isinstance(tensor, torch.Tensor):
                                    torch.testing.assert_close(tensor, expected[subkey], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
