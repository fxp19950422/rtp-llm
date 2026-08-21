import pickle
from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v4 import DeepSeekV4
from rtp_llm.models.dsv4_kv_cache import (
    CSA_KV_TAG,
    CSA_STATE_TAG,
    DSV4_FP4_INDEXER_ENTRY_BYTES,
    DSV4_FP8_INDEXER_ENTRY_BYTES,
    DSV4_FP8_KV_ENTRY_BYTES,
    DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
    DSV4_HCA_STATE_POOL_BLOCKS,
    DSV4_SPECULATIVE_C128_STATE_RING_ENTRIES,
    DSV4_SPECULATIVE_C4_MAX_QUERY_LEN,
    DSV4_SPECULATIVE_C4_STATE_RING_ENTRIES,
    DSV4_SWA_WINDOW_ENTRIES,
    DSV4_TOKENS_PER_BLOCK,
    Dsv4IndexerCacheMode,
    Dsv4StateRingMode,
    HCA_KV_TAG,
    HCA_STATE_TAG,
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    SWA_KV_TAG,
    apply_dsv4_explicit_pool_blocks,
    build_dsv4_kv_cache_spec_descs,
    validate_dsv4_speculative_target_query_len,
)
from rtp_llm.ops import (
    CacheEvictPolicy,
    CacheMemoryPlacement,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    DataType,
    HybridAttentionType,
    KvCacheDataType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)

# CSA / HCA / SWA / SWA / CSA -- covers every routing branch plus a repeat.
LAYER_COMPRESS_RATIOS = [4, 128, 0, 0, 4]
HEAD_DIM = 512
INDEXER_HEAD_DIM = 128
FRAMEWORK_DEFAULT_TOKENS_PER_BLOCK = 64
_DEFAULT_INDEXER_CACHE_MODE = object()


class Dsv4KvCacheSpecTest(TestCase):
    def _build(
        self,
        fp8_kv=True,
        fixed_pool_use_host_memory=False,
        state_ring_mode=Dsv4StateRingMode.NORMAL,
        indexer_cache_mode=_DEFAULT_INDEXER_CACHE_MODE,
    ):
        kwargs = dict(
            layer_num=len(LAYER_COMPRESS_RATIOS),
            layer_compress_ratios=LAYER_COMPRESS_RATIOS,
            fp8_kv=fp8_kv,
            head_dim=HEAD_DIM,
            indexer_head_dim=INDEXER_HEAD_DIM,
            fixed_pool_use_host_memory=fixed_pool_use_host_memory,
            state_ring_mode=state_ring_mode,
        )
        if indexer_cache_mode is not _DEFAULT_INDEXER_CACHE_MODE:
            kwargs["indexer_cache_mode"] = indexer_cache_mode
        return build_dsv4_kv_cache_spec_descs(**kwargs)

    def _by_tag(self, layer_descs):
        return {desc.tag: desc for descs in layer_descs for desc in descs}

    def test_layer_routing_tags(self):
        layer_descs = self._build()
        self.assertEqual(
            [[desc.tag for desc in descs] for descs in layer_descs],
            [
                [
                    CSA_KV_TAG,
                    INDEXER_KV_TAG,
                    INDEXER_STATE_TAG,
                    CSA_STATE_TAG,
                    SWA_KV_TAG,
                ],
                [HCA_KV_TAG, HCA_STATE_TAG, SWA_KV_TAG],
                [SWA_KV_TAG],
                [SWA_KV_TAG],
                [
                    CSA_KV_TAG,
                    INDEXER_KV_TAG,
                    INDEXER_STATE_TAG,
                    CSA_STATE_TAG,
                    SWA_KV_TAG,
                ],
            ],
        )

    def test_layers_past_compress_ratios_are_swa_only(self):
        layer_descs = build_dsv4_kv_cache_spec_descs(
            layer_num=len(LAYER_COMPRESS_RATIOS) + 2,
            layer_compress_ratios=LAYER_COMPRESS_RATIOS,
            fp8_kv=True,
            head_dim=HEAD_DIM,
            indexer_head_dim=INDEXER_HEAD_DIM,
        )
        self.assertEqual([desc.tag for desc in layer_descs[-1]], [SWA_KV_TAG])
        self.assertEqual([desc.tag for desc in layer_descs[-2]], [SWA_KV_TAG])

    def test_real_checkpoint_schedule_ignores_zero_mtp_tail(self):
        ratios = [0, 0] + [4 if layer % 2 == 0 else 128 for layer in range(2, 43)]
        ratios.append(0)
        layer_descs = build_dsv4_kv_cache_spec_descs(
            layer_num=43,
            layer_compress_ratios=ratios,
            fp8_kv=True,
            head_dim=HEAD_DIM,
            indexer_head_dim=INDEXER_HEAD_DIM,
        )

        self.assertEqual(len(layer_descs), 43)
        self.assertEqual([desc.tag for desc in layer_descs[0]], [SWA_KV_TAG])
        self.assertEqual([desc.tag for desc in layer_descs[1]], [SWA_KV_TAG])
        self.assertEqual(layer_descs[2][0].tag, CSA_KV_TAG)
        self.assertEqual(layer_descs[3][0].tag, HCA_KV_TAG)
        self.assertEqual(layer_descs[42][0].tag, CSA_KV_TAG)

    def test_rejects_non_integer_ratio_types(self):
        for ratio in (4.5, True, "4"):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(
                TypeError, "must be integers"
            ):
                build_dsv4_kv_cache_spec_descs(
                    layer_num=1,
                    layer_compress_ratios=[ratio],
                    fp8_kv=True,
                    head_dim=HEAD_DIM,
                    indexer_head_dim=INDEXER_HEAD_DIM,
                )

    def test_rejects_unknown_main_ratio(self):
        with self.assertRaisesRegex(ValueError, "main-layer compression ratio"):
            build_dsv4_kv_cache_spec_descs(
                layer_num=1,
                layer_compress_ratios=[8],
                fp8_kv=True,
                head_dim=HEAD_DIM,
                indexer_head_dim=INDEXER_HEAD_DIM,
            )

    def test_rejects_nonzero_mtp_tail(self):
        with self.assertRaisesRegex(ValueError, "trailing MTP/draft"):
            build_dsv4_kv_cache_spec_descs(
                layer_num=1,
                layer_compress_ratios=[4, 128],
                fp8_kv=True,
                head_dim=HEAD_DIM,
                indexer_head_dim=INDEXER_HEAD_DIM,
            )

    def test_cache_types_and_state_flags(self):
        by_tag = self._by_tag(self._build())
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertEqual(by_tag[tag].cache_type, KVCacheSpecType.OPAQUE_KV, tag)
            self.assertFalse(by_tag[tag].is_state_cache, tag)
            self.assertEqual(
                by_tag[tag].entry_count_mode,
                OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
                tag,
            )
            self.assertEqual(by_tag[tag].dtype, DataType.TYPE_UINT8, tag)
            self.assertEqual(by_tag[tag].entry_dtype, DataType.TYPE_UINT8, tag)
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            self.assertEqual(by_tag[tag].cache_type, KVCacheSpecType.OPAQUE_STATE, tag)
            self.assertTrue(by_tag[tag].is_state_cache, tag)
            self.assertEqual(
                by_tag[tag].entry_count_mode, OpaqueBlockEntryCountMode.STATE_RING, tag
            )
            self.assertTrue(by_tag[tag].state_ring_include_gen_num_per_cycle, tag)
            self.assertEqual(
                by_tag[tag].reuse.evict_policy, CacheEvictPolicy.INDEPENDENT
            )
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG):
            self.assertEqual(by_tag[tag].dtype, DataType.TYPE_FP32, tag)
            self.assertEqual(by_tag[tag].entry_dtype, DataType.TYPE_FP32, tag)
        self.assertEqual(by_tag[SWA_KV_TAG].dtype, DataType.TYPE_UINT8)

    def test_explicit_normal_mode_preserves_default_state_ring_abi(self):
        default_by_tag = self._by_tag(self._build())
        normal_by_tag = self._by_tag(
            self._build(state_ring_mode=Dsv4StateRingMode.NORMAL)
        )
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            default = default_by_tag[tag]
            normal = normal_by_tag[tag]
            self.assertEqual(normal.entry_count_mode, default.entry_count_mode, tag)
            self.assertEqual(normal.explicit_entry_count, default.explicit_entry_count, tag)
            self.assertEqual(
                normal.state_ring_include_gen_num_per_cycle,
                default.state_ring_include_gen_num_per_cycle,
                tag,
            )
            self.assertEqual(normal.compression_ratio, default.compression_ratio, tag)
            self.assertEqual(normal.state_ring_overlap, default.state_ring_overlap, tag)

    def test_speculative_target_verify_uses_explicit_double_state_rings(self):
        by_tag = self._by_tag(
            self._build(
                state_ring_mode=Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY
            )
        )
        expected_entries = {
            INDEXER_STATE_TAG: DSV4_SPECULATIVE_C4_STATE_RING_ENTRIES,
            CSA_STATE_TAG: DSV4_SPECULATIVE_C4_STATE_RING_ENTRIES,
            HCA_STATE_TAG: DSV4_SPECULATIVE_C128_STATE_RING_ENTRIES,
            SWA_KV_TAG: DSV4_SPECULATIVE_C128_STATE_RING_ENTRIES,
        }
        for tag, entries in expected_entries.items():
            desc = by_tag[tag]
            self.assertEqual(
                desc.entry_count_mode, OpaqueBlockEntryCountMode.EXPLICIT, tag
            )
            self.assertEqual(desc.explicit_entry_count, entries, tag)
            self.assertFalse(desc.state_ring_include_gen_num_per_cycle, tag)

        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertEqual(
                by_tag[tag].entry_count_mode,
                OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
                tag,
            )

    def test_rejects_implicit_or_invalid_state_ring_mode(self):
        for mode in ("speculative_target_verify", True, 1, None):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                TypeError, "state_ring_mode must be a Dsv4StateRingMode"
            ):
                self._build(state_ring_mode=mode)

    def test_speculative_target_query_len_contract(self):
        for query_len in (1, 2, 3, 5, DSV4_SPECULATIVE_C4_MAX_QUERY_LEN):
            with self.subTest(query_len=query_len):
                self.assertEqual(
                    validate_dsv4_speculative_target_query_len(query_len), query_len
                )

        for query_len in (0, -1, 10, 32):
            with self.subTest(query_len=query_len), self.assertRaisesRegex(
                ValueError, "outside the safe C4 state-ring range"
            ):
                validate_dsv4_speculative_target_query_len(query_len)

        for query_len in (True, 3.0, "3", None):
            with self.subTest(query_len=query_len), self.assertRaisesRegex(
                TypeError, "query_len must be an integer"
            ):
                validate_dsv4_speculative_target_query_len(query_len)

    def test_fp8_entry_elems(self):
        by_tag = self._by_tag(self._build(fp8_kv=True))
        self.assertEqual(by_tag[CSA_KV_TAG].entry_elems, DSV4_FP8_KV_ENTRY_BYTES)
        self.assertEqual(by_tag[HCA_KV_TAG].entry_elems, DSV4_FP8_KV_ENTRY_BYTES)
        self.assertEqual(by_tag[SWA_KV_TAG].entry_elems, DSV4_FP8_KV_ENTRY_BYTES)
        self.assertEqual(
            by_tag[INDEXER_KV_TAG].entry_elems, DSV4_FP8_INDEXER_ENTRY_BYTES
        )
        self.assertEqual(by_tag[INDEXER_STATE_TAG].entry_elems, 4 * INDEXER_HEAD_DIM)
        self.assertEqual(by_tag[CSA_STATE_TAG].entry_elems, 4 * HEAD_DIM)
        self.assertEqual(by_tag[HCA_STATE_TAG].entry_elems, 2 * HEAD_DIM)

    def test_default_and_explicit_follow_kv_descriptors_serialize_identically(self):
        default_descs = self._build()
        explicit_descs = self._build(
            indexer_cache_mode=Dsv4IndexerCacheMode.FOLLOW_KV
        )

        self.assertEqual(
            pickle.dumps(default_descs, protocol=pickle.HIGHEST_PROTOCOL),
            pickle.dumps(explicit_descs, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def test_default_non_fp8_preserves_legacy_descriptor_serialization(self):
        default_descs = self._build(fp8_kv=False)
        explicit_compat_descs = self._build(
            fp8_kv=False,
            indexer_cache_mode=Dsv4IndexerCacheMode.FOLLOW_KV,
        )
        self.assertEqual(
            self._by_tag(default_descs)[INDEXER_KV_TAG].entry_elems,
            INDEXER_HEAD_DIM * 2,
        )
        self.assertEqual(
            pickle.dumps(default_descs, protocol=pickle.HIGHEST_PROTOCOL),
            pickle.dumps(explicit_compat_descs, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def test_explicit_fp8_is_independent_of_main_kv_dtype(self):
        for fp8_kv in (True, False):
            with self.subTest(fp8_kv=fp8_kv):
                by_tag = self._by_tag(
                    self._build(
                        fp8_kv=fp8_kv,
                        indexer_cache_mode=Dsv4IndexerCacheMode.FP8,
                    )
                )
                self.assertEqual(
                    by_tag[INDEXER_KV_TAG].entry_elems,
                    DSV4_FP8_INDEXER_ENTRY_BYTES,
                )

    def test_explicit_fp4_changes_only_indexer_kv_entry_geometry(self):
        fp8_descs = self._build(indexer_cache_mode=Dsv4IndexerCacheMode.FP8)
        fp4_descs = self._build(indexer_cache_mode=Dsv4IndexerCacheMode.FP4)
        fp8_by_tag = self._by_tag(fp8_descs)
        fp4_by_tag = self._by_tag(fp4_descs)

        self.assertEqual(
            fp4_by_tag[INDEXER_KV_TAG].entry_elems,
            DSV4_FP4_INDEXER_ENTRY_BYTES,
        )
        self.assertNotEqual(
            pickle.dumps(fp8_by_tag[INDEXER_KV_TAG]),
            pickle.dumps(fp4_by_tag[INDEXER_KV_TAG]),
        )
        for tag in fp8_by_tag.keys() - {INDEXER_KV_TAG}:
            self.assertEqual(
                pickle.dumps(fp8_by_tag[tag]),
                pickle.dumps(fp4_by_tag[tag]),
                tag,
            )

    def test_indexer_mode_is_independent_of_main_kv_dtype(self):
        fp8_main = self._by_tag(
            self._build(fp8_kv=True, indexer_cache_mode=Dsv4IndexerCacheMode.FP4)
        )
        non_fp8_main = self._by_tag(
            self._build(fp8_kv=False, indexer_cache_mode=Dsv4IndexerCacheMode.FP4)
        )
        self.assertEqual(
            fp8_main[INDEXER_KV_TAG].entry_elems,
            DSV4_FP4_INDEXER_ENTRY_BYTES,
        )
        self.assertEqual(
            non_fp8_main[INDEXER_KV_TAG].entry_elems,
            DSV4_FP4_INDEXER_ENTRY_BYTES,
        )
        self.assertNotEqual(
            fp8_main[CSA_KV_TAG].entry_elems,
            non_fp8_main[CSA_KV_TAG].entry_elems,
        )

    def test_rejects_implicit_or_invalid_indexer_cache_mode(self):
        for mode in ("fp4", True, 1, None):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                TypeError,
                "indexer_cache_mode must be a Dsv4IndexerCacheMode",
            ):
                self._build(indexer_cache_mode=mode)

    def test_fp4_mode_preserves_normal_and_speculative_state_rings(self):
        for state_ring_mode in (
            Dsv4StateRingMode.NORMAL,
            Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY,
        ):
            with self.subTest(state_ring_mode=state_ring_mode):
                fp8_by_tag = self._by_tag(
                    self._build(
                        state_ring_mode=state_ring_mode,
                        indexer_cache_mode=Dsv4IndexerCacheMode.FP8,
                    )
                )
                fp4_by_tag = self._by_tag(
                    self._build(
                        state_ring_mode=state_ring_mode,
                        indexer_cache_mode=Dsv4IndexerCacheMode.FP4,
                    )
                )
                for tag in (
                    INDEXER_STATE_TAG,
                    CSA_STATE_TAG,
                    HCA_STATE_TAG,
                    SWA_KV_TAG,
                ):
                    self.assertEqual(
                        pickle.dumps(fp8_by_tag[tag]),
                        pickle.dumps(fp4_by_tag[tag]),
                        tag,
                    )

    def test_fp4_indexer_production_block_split_literal_oracle(self):
        # Independent SGLang-fork geometry oracle: do not construct expected
        # values from production constants or from the production writer.
        raw_tokens_per_page = 256
        compression_ratio = 4
        packed_bytes_per_entry = 64
        scale_bytes_per_entry = 4
        entries = raw_tokens_per_page // compression_ratio
        data_bytes = entries * packed_bytes_per_entry
        scale_bytes = entries * scale_bytes_per_entry
        physical_bytes = data_bytes + scale_bytes

        self.assertEqual(entries, 64)
        self.assertEqual((data_bytes, scale_bytes, physical_bytes), (4096, 256, 4352))
        self.assertEqual(physical_bytes // entries, 68)

        # Raw token 255 completes compressed slot 63 in page 0; raw token 259
        # completes slot 0 in page 1.  Data and raw UE8M0 scales occupy
        # separate physical slabs inside the same cache block.
        for raw_token, expected_page, expected_slot in (
            (3, 0, 0),
            (255, 0, 63),
            (259, 1, 0),
        ):
            compressed_position = (raw_token + 1) // compression_ratio - 1
            page = compressed_position // entries
            slot = compressed_position % entries
            self.assertEqual((page, slot), (expected_page, expected_slot))

            data_begin = slot * packed_bytes_per_entry
            data_end = data_begin + packed_bytes_per_entry
            scale_begin = data_bytes + slot * scale_bytes_per_entry
            scale_end = scale_begin + scale_bytes_per_entry
            self.assertLessEqual(data_end, data_bytes)
            self.assertGreaterEqual(scale_begin, data_bytes)
            self.assertLessEqual(scale_end, physical_bytes)

        last_data_end = entries * packed_bytes_per_entry
        last_scale_end = data_bytes + entries * scale_bytes_per_entry
        guard_begin = physical_bytes
        self.assertEqual(last_data_end, 4096)
        self.assertEqual(last_scale_end, guard_begin)

    def test_non_fp8_entry_elems(self):
        by_tag = self._by_tag(self._build(fp8_kv=False))
        self.assertEqual(by_tag[CSA_KV_TAG].entry_elems, HEAD_DIM * 2)
        self.assertEqual(by_tag[HCA_KV_TAG].entry_elems, HEAD_DIM * 2)
        self.assertEqual(by_tag[SWA_KV_TAG].entry_elems, HEAD_DIM * 2)
        # FOLLOW_KV is the compatibility-only default and preserves the old
        # non-FP8 indexer width for existing public callers.
        self.assertEqual(by_tag[INDEXER_KV_TAG].entry_elems, INDEXER_HEAD_DIM * 2)
        # Fixed-state entry sizes do not depend on the KV dtype.
        self.assertEqual(by_tag[INDEXER_STATE_TAG].entry_elems, 4 * INDEXER_HEAD_DIM)
        self.assertEqual(by_tag[CSA_STATE_TAG].entry_elems, 4 * HEAD_DIM)
        self.assertEqual(by_tag[HCA_STATE_TAG].entry_elems, 2 * HEAD_DIM)

    def test_compression_ratio_and_state_ring_overlap(self):
        by_tag = self._by_tag(self._build())
        self.assertEqual(by_tag[CSA_KV_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[INDEXER_KV_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[HCA_KV_TAG].compression_ratio, 128)
        self.assertEqual(by_tag[INDEXER_STATE_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[CSA_STATE_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[HCA_STATE_TAG].compression_ratio, 128)
        self.assertEqual(by_tag[SWA_KV_TAG].compression_ratio, DSV4_SWA_WINDOW_ENTRIES)

        self.assertEqual(by_tag[INDEXER_STATE_TAG].state_ring_overlap, 1)
        self.assertEqual(by_tag[CSA_STATE_TAG].state_ring_overlap, 1)
        self.assertEqual(by_tag[HCA_STATE_TAG].state_ring_overlap, 0)
        self.assertEqual(by_tag[SWA_KV_TAG].state_ring_overlap, 0)

    def test_576_alignment_only_on_584_byte_pools(self):
        fp8_by_tag = self._by_tag(self._build(fp8_kv=True))
        for tag in (CSA_KV_TAG, HCA_KV_TAG, SWA_KV_TAG):
            self.assertEqual(fp8_by_tag[tag].entry_elems, DSV4_FP8_KV_ENTRY_BYTES, tag)
            self.assertEqual(
                fp8_by_tag[tag].block_stride_bytes_alignment,
                DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
                tag,
            )
        for tag in (
            INDEXER_KV_TAG,
            INDEXER_STATE_TAG,
            CSA_STATE_TAG,
            HCA_STATE_TAG,
        ):
            self.assertEqual(fp8_by_tag[tag].block_stride_bytes_alignment, 0, tag)

        non_fp8_by_tag = self._by_tag(self._build(fp8_kv=False))
        for desc in non_fp8_by_tag.values():
            self.assertEqual(desc.block_stride_bytes_alignment, 0, desc.tag)

    def test_block_stride_alignment_min_entries_only_on_state_pools(self):
        by_tag = self._by_tag(self._build())
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            self.assertEqual(
                by_tag[tag].block_stride_alignment_min_entries,
                DSV4_SWA_WINDOW_ENTRIES,
                tag,
            )
        # Compressed pools must keep 0 here: with 128 the 576-byte alignment
        # above would be skipped (they hold fewer than 128 entries per block).
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertEqual(by_tag[tag].block_stride_alignment_min_entries, 0, tag)

    def test_cp_policy_per_pool(self):
        by_tag = self._by_tag(self._build())
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG):
            cp = by_tag[tag].cp
            self.assertIsNotNone(cp, tag)
            self.assertTrue(cp.align_payload, tag)
            self.assertTrue(cp.scale_seq_size, tag)
            self.assertEqual(cp.prefill_slice_layout, CpPrefillSliceLayout.PAYLOAD, tag)
            self.assertEqual(cp.slice, CpBlockSliceMode.PAYLOAD_BYTES, tag)

        swa_cp = by_tag[SWA_KV_TAG].cp
        self.assertIsNotNone(swa_cp)
        self.assertTrue(swa_cp.align_payload)
        self.assertTrue(swa_cp.scale_seq_size)
        self.assertEqual(swa_cp.prefill_slice_layout, CpPrefillSliceLayout.BLOCK_STRIDE)
        self.assertEqual(swa_cp.slice, CpBlockSliceMode.EQUAL_BYTES)

        # Compressed pools carry no cp policy at all.
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertIsNone(by_tag[tag].cp, tag)

    def test_hca_state_capacity_reuse_and_tail(self):
        by_tag = self._by_tag(self._build())
        hca_state = by_tag[HCA_STATE_TAG]
        self.assertIsNotNone(hca_state.capacity)
        self.assertEqual(
            hca_state.capacity.explicit_block_num, DSV4_HCA_STATE_POOL_BLOCKS
        )
        self.assertTrue(hca_state.capacity.charge_to_paged_budget)
        self.assertFalse(hca_state.reuse.enable_prefix_reuse)
        self.assertIsNotNone(hca_state.tail)
        self.assertEqual(hca_state.tail.active_tail_blocks, 1)
        self.assertFalse(hca_state.tail.validate_tail_blocks)

        for tag in (
            CSA_KV_TAG,
            HCA_KV_TAG,
            INDEXER_KV_TAG,
            INDEXER_STATE_TAG,
            CSA_STATE_TAG,
            SWA_KV_TAG,
        ):
            self.assertIsNone(by_tag[tag].capacity, tag)
            self.assertIsNone(by_tag[tag].tail, tag)

    def test_no_memory_placement_by_default(self):
        for desc in self._by_tag(self._build()).values():
            self.assertIsNone(desc.memory, desc.tag)

    def test_host_pinned_fixed_pools_leave_paged_budget(self):
        by_tag = self._by_tag(self._build(fixed_pool_use_host_memory=True))
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            desc = by_tag[tag]
            self.assertIsNotNone(desc.memory, tag)
            self.assertEqual(
                desc.memory.placement, CacheMemoryPlacement.HOST_PINNED, tag
            )
            self.assertIsNotNone(desc.capacity, tag)
            self.assertFalse(desc.capacity.charge_to_paged_budget, tag)
        # hca_state keeps its explicit sizing while leaving the HBM budget.
        self.assertEqual(
            by_tag[HCA_STATE_TAG].capacity.explicit_block_num,
            DSV4_HCA_STATE_POOL_BLOCKS,
        )
        # Compressed pools stay on device.
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertIsNone(by_tag[tag].memory, tag)
            self.assertIsNone(by_tag[tag].capacity, tag)

    def test_explicit_pool_blocks_helper(self):
        layer_descs = self._build()
        apply_dsv4_explicit_pool_blocks(layer_descs, SWA_KV_TAG, 512)
        swa = self._by_tag(layer_descs)[SWA_KV_TAG]
        self.assertEqual(swa.capacity.explicit_block_num, 512)
        self.assertTrue(swa.capacity.charge_to_paged_budget)

    def test_explicit_pool_blocks_helper_keeps_host_pool_off_budget(self):
        layer_descs = self._build(fixed_pool_use_host_memory=True)
        apply_dsv4_explicit_pool_blocks(layer_descs, SWA_KV_TAG, 512)
        swa = self._by_tag(layer_descs)[SWA_KV_TAG]
        self.assertEqual(swa.capacity.explicit_block_num, 512)
        self.assertFalse(swa.capacity.charge_to_paged_budget)

    def test_rejects_zero_layers(self):
        with self.assertRaises(ValueError):
            build_dsv4_kv_cache_spec_descs(
                layer_num=0,
                layer_compress_ratios=[],
                fp8_kv=True,
                head_dim=HEAD_DIM,
                indexer_head_dim=INDEXER_HEAD_DIM,
            )


class Dsv4PostBuildModelConfigTest(TestCase):
    def _model_config(self, tokens_per_block=FRAMEWORK_DEFAULT_TOKENS_PER_BLOCK):
        config = ModelConfig()
        config.num_layers = len(LAYER_COMPRESS_RATIOS)
        config.attn_config.size_per_head = HEAD_DIM
        config.attn_config.indexer_head_dim = INDEXER_HEAD_DIM
        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_config.layer_compress_ratios = LAYER_COMPRESS_RATIOS
        config.attn_config.tokens_per_block = tokens_per_block
        config.attn_config.kernel_tokens_per_block = tokens_per_block
        return config

    def test_post_build_enables_independent_pools(self):
        config = self._model_config()

        DeepSeekV4._post_build_model_config(config)

        self.assertTrue(
            config.hybrid_attention_config.enable_independent_kv_cache_pools
        )
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            [HybridAttentionType.NONE] * config.num_layers,
        )
        self.assertEqual(len(config.kv_cache_spec_descs), config.num_layers)
        self.assertEqual(
            [desc.tag for desc in config.kv_cache_spec_descs[1]],
            [HCA_KV_TAG, HCA_STATE_TAG, SWA_KV_TAG],
        )

    def test_post_build_promotes_default_block_size(self):
        config = self._model_config()

        DeepSeekV4._post_build_model_config(config)

        self.assertEqual(config.attn_config.tokens_per_block, DSV4_TOKENS_PER_BLOCK)
        self.assertEqual(
            config.attn_config.kernel_tokens_per_block, DSV4_TOKENS_PER_BLOCK
        )

    def test_post_build_keeps_explicit_block_size(self):
        config = self._model_config(tokens_per_block=128)

        DeepSeekV4._post_build_model_config(config)

        self.assertEqual(config.attn_config.tokens_per_block, 128)
        self.assertEqual(config.attn_config.kernel_tokens_per_block, 128)

    def test_post_build_does_not_override_existing_descs(self):
        config = self._model_config()
        sentinel = KVCacheSpecDesc()
        sentinel.tag = "sentinel"
        sentinel.cache_type = KVCacheSpecType.MHA
        config.kv_cache_spec_descs = [[sentinel]] * config.num_layers

        DeepSeekV4._post_build_model_config(config)

        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "sentinel")
        self.assertEqual(
            config.attn_config.tokens_per_block, FRAMEWORK_DEFAULT_TOKENS_PER_BLOCK
        )


if __name__ == "__main__":
    main()
