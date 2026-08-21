"""Independent scalar oracle for DeepSeek-V4 state-ring geometry.

This test intentionally imports no production helpers.  It models the current
write-all/read-later compressor order with position IDs, then checks physical
slot/page arithmetic from test-owned constants.
"""

import unittest


TOKENS_PER_BLOCK = 256
NORMAL_BASE = "normal_base"
SPECULATIVE = "speculative_target_verify"
RING_ENTRIES = {
    NORMAL_BASE: {4: 8, 128: 128},
    SPECULATIVE: {4: 16, 128: 256},
}
HISTORY_WINDOWS = {4: 8, 128: 128}
QUERY_LENS = (1, 2, 3, 5, 9, 10, 32)


def _first_write_before_read_hazard(window: int, ring: int, query_len: int):
    """Return (reader, required_position, observed_position), or None."""
    boundary = window - 1
    ring_storage = {}

    # History available immediately before the target-verify batch.
    for position in range(boundary - window + 1, boundary):
        ring_storage[position % ring] = position

    # Production ordering: save all query states before any compressor read.
    for position in range(boundary, boundary + query_len):
        ring_storage[position % ring] = position

    for reader in range(boundary, boundary + query_len):
        for required in range(reader - window + 1, reader + 1):
            observed = ring_storage.get(required % ring)
            if observed != required:
                return reader, required, observed
    return None


def _normal_dynamic_ring_entries(window: int, query_len: int) -> int:
    """Existing STATE_RING ABI: add gen=q_len-1, then round up to even."""
    raw_entries = window + query_len - 1
    return (raw_entries + 1) & ~1


def _state_slot(block_table, position: int, entries_per_block: int) -> int:
    if position < 0:
        raise ValueError("position must be non-negative")
    if entries_per_block <= 0:
        raise ValueError("entries_per_block must be positive")
    block_row = position // TOKENS_PER_BLOCK
    if block_row >= len(block_table):
        raise IndexError("position has no allocated block-table row")
    block_id = block_table[block_row]
    if block_id <= 0:
        return -1
    return block_id * entries_per_block + position % entries_per_block


class Dsv4StateRingGeometryOracleTest(unittest.TestCase):
    def test_unexpanded_normal_base_collides_for_multi_token_batch(self):
        for ratio in (4, 128):
            window = HISTORY_WINDOWS[ratio]
            ring = RING_ENTRIES[NORMAL_BASE][ratio]
            self.assertIsNone(
                _first_write_before_read_hazard(window, ring, 1), ratio
            )
            for query_len in QUERY_LENS[1:]:
                with self.subTest(ratio=ratio, query_len=query_len):
                    self.assertIsNotNone(
                        _first_write_before_read_hazard(window, ring, query_len)
                    )

    def test_normal_dynamic_abi_expands_with_generation_count(self):
        expected = {
            1: {4: 8, 128: 128},
            2: {4: 10, 128: 130},
            3: {4: 10, 128: 130},
            5: {4: 12, 128: 132},
            9: {4: 16, 128: 136},
            10: {4: 18, 128: 138},
            32: {4: 40, 128: 160},
        }
        for query_len in QUERY_LENS:
            for ratio in (4, 128):
                with self.subTest(query_len=query_len, ratio=ratio):
                    entries = _normal_dynamic_ring_entries(
                        HISTORY_WINDOWS[ratio], query_len
                    )
                    self.assertEqual(entries, expected[query_len][ratio])
                    self.assertIsNone(
                        _first_write_before_read_hazard(
                            HISTORY_WINDOWS[ratio], entries, query_len
                        )
                    )

    def test_speculative_geometry_is_safe_through_c4_limit(self):
        for query_len in (1, 2, 3, 5, 9):
            for ratio in (4, 128):
                with self.subTest(ratio=ratio, query_len=query_len):
                    self.assertIsNone(
                        _first_write_before_read_hazard(
                            HISTORY_WINDOWS[ratio],
                            RING_ENTRIES[SPECULATIVE][ratio],
                            query_len,
                        )
                    )

    def test_c4_makes_query_len_10_and_32_unsupported(self):
        for query_len in (10, 32):
            with self.subTest(query_len=query_len):
                self.assertIsNotNone(
                    _first_write_before_read_hazard(
                        HISTORY_WINDOWS[4],
                        RING_ENTRIES[SPECULATIVE][4],
                        query_len,
                    )
                )
                # C128 alone still fits; the combined mode is unsafe because
                # every target can contain C4 layers.
                self.assertIsNone(
                    _first_write_before_read_hazard(
                        HISTORY_WINDOWS[128],
                        RING_ENTRIES[SPECULATIVE][128],
                        query_len,
                    )
                )

    def test_exact_first_collision_counterexamples(self):
        self.assertEqual(_first_write_before_read_hazard(8, 8, 2), (7, 0, 8))
        self.assertEqual(_first_write_before_read_hazard(8, 8, 3), (7, 0, 8))
        self.assertEqual(
            _first_write_before_read_hazard(128, 128, 2), (127, 0, 128)
        )
        self.assertEqual(
            _first_write_before_read_hazard(128, 128, 3), (127, 0, 128)
        )
        self.assertEqual(_first_write_before_read_hazard(8, 16, 10), (7, 0, 16))

    def test_speculative_geometry_is_mode_selected_not_query_len_selected(self):
        for query_len in QUERY_LENS:
            with self.subTest(query_len=query_len):
                self.assertEqual(RING_ENTRIES[SPECULATIVE], {4: 16, 128: 256})

    def test_physical_slot_stride_and_page_boundaries(self):
        block_table = [11, 17]
        for mode in (NORMAL_BASE, SPECULATIVE):
            for ratio in (4, 128):
                entries = RING_ENTRIES[mode][ratio]
                with self.subTest(mode=mode, ratio=ratio):
                    for position in (0, entries - 1, entries, 255, 256, 257):
                        block_row = position // TOKENS_PER_BLOCK
                        expected = (
                            block_table[block_row] * entries + position % entries
                        )
                        self.assertEqual(
                            _state_slot(block_table, position, entries), expected
                        )

                    # One physical block id advances exactly one E-entry row.
                    self.assertEqual(
                        _state_slot([12], 0, entries)
                        - _state_slot([11], 0, entries),
                        entries,
                    )
                    self.assertEqual(
                        _state_slot(block_table, 255, entries),
                        11 * entries + 255 % entries,
                    )
                    self.assertEqual(
                        _state_slot(block_table, 256, entries), 17 * entries
                    )

    def test_slot_oracle_rejects_invalid_or_unallocated_inputs(self):
        with self.assertRaises(ValueError):
            _state_slot([11], -1, 8)
        with self.assertRaises(ValueError):
            _state_slot([11], 0, 0)
        with self.assertRaises(IndexError):
            _state_slot([11], 256, 8)
        self.assertEqual(_state_slot([0], 0, 8), -1)
        self.assertEqual(_state_slot([-1], 0, 8), -1)


if __name__ == "__main__":
    unittest.main()
