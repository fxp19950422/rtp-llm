import os
import sys
import unittest

import torch

from rtp_llm.cpp.models.context_parallel.test import (
    libth_context_parallel_py_wrapper_test as cp_test,
)


class TestContextParallelLoadBalanceSplit(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(
            os.environ.get("TEST_SRCDIR", ".")
            + "/rtp_llm/rtp_llm/cpp/models/context_parallel/test"
        )
        self.fake_input_tokens = [
            45231,
            12089,
            58934,
            2341,
            61234,
            9876,
            34567,
            51234,
            8901,
            43210,
            15678,
            29345,
            48901,
            5234,
            62345,
            18765,
            38901,
            54321,
            7890,
            41234,
            26789,
            52345,
            13456,
            39012,
            55678,
            4567,
            47890,
            21345,
            59012,
            11234,
            35678,
            49012,
        ]

    def test_basic_split(self):
        """Test basic load balance split with simple parameters"""
        # Test with 16 tokens, 2 ranks, chunk size 8
        total_tokens = self.fake_input_tokens[:16]
        input_tokens = []
        shuffle_indices = []
        cp_rank = 0
        cp_size = 2
        cp_chunk_size = 8
        cp_padding_size = 0
        expect_shuffle_indices = [0, 1, 2, 3, 12, 13, 14, 15]
        expect_input_tokens = [total_tokens[i] for i in expect_shuffle_indices]

        result, input_tokens, shuffle_indices = (
            cp_test.context_parallel_load_balance_split(
                total_tokens,
                input_tokens,
                shuffle_indices,
                cp_rank,
                cp_size,
                cp_chunk_size,
                cp_padding_size,
            )
        )
        self.assertTrue(result)
        self.assertEqual(len(input_tokens), cp_chunk_size)
        self.assertEqual(len(shuffle_indices), cp_chunk_size)
        self.assertEqual(input_tokens, expect_input_tokens)
        self.assertEqual(shuffle_indices, expect_shuffle_indices)

    def test_with_padding(self):
        """Test load balance split with padding"""
        # Test with padding
        total_tokens = self.fake_input_tokens[:14]  # 14 tokens
        input_tokens = []
        shuffle_indices = []
        cp_rank = 0
        cp_size = 2
        cp_chunk_size = 8
        cp_padding_size = 2
        expect_shuffle_indices = [0, 1, 2, 3, 12, 13, 14, 15]
        expect_input_tokens = [total_tokens[i] for i in expect_shuffle_indices[0:6]] + [
            0,
            0,
        ]

        result, input_tokens, shuffle_indices = (
            cp_test.context_parallel_load_balance_split(
                total_tokens,
                input_tokens,
                shuffle_indices,
                cp_rank,
                cp_size,
                cp_chunk_size,
                cp_padding_size,
            )
        )
        self.assertTrue(result)
        self.assertEqual(len(input_tokens), cp_chunk_size)
        self.assertEqual(len(shuffle_indices), cp_chunk_size)
        self.assertEqual(input_tokens, expect_input_tokens)
        self.assertEqual(shuffle_indices, expect_shuffle_indices)

    def test_multiple_ranks(self):
        """Test load balance split across multiple ranks"""
        total_tokens = self.fake_input_tokens[:32]
        cp_size = 4
        cp_chunk_size = 8
        cp_padding_size = 0

        expect_shuffle_indices = [
            [0, 1, 2, 3, 28, 29, 30, 31],
            [4, 5, 6, 7, 24, 25, 26, 27],
            [8, 9, 10, 11, 20, 21, 22, 23],
            [12, 13, 14, 15, 16, 17, 18, 19],
        ]
        expect_input_tokens = [
            [total_tokens[i] for i in indices] for indices in expect_shuffle_indices
        ]

        # Test each rank
        for cp_rank in range(cp_size):
            input_tokens = []
            shuffle_indices = []
            result, input_tokens, shuffle_indices = (
                cp_test.context_parallel_load_balance_split(
                    total_tokens,
                    input_tokens,
                    shuffle_indices,
                    cp_rank,
                    cp_size,
                    cp_chunk_size,
                    cp_padding_size,
                )
            )
            self.assertTrue(result)
            self.assertEqual(input_tokens, expect_input_tokens[cp_rank])
            self.assertEqual(shuffle_indices, expect_shuffle_indices[cp_rank])


class TestGenerateQKVRestoreIndices(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(
            os.environ.get("TEST_SRCDIR", ".")
            + "/rtp_llm/rtp_llm/cpp/models/context_parallel/test"
        )

    def test_basic_restore_indices(self):
        """Test basic QKV restore indices generation"""
        # Create chunk lengths for 2 ranks
        prefill_cp_chunk_lengths = torch.tensor([8, 8], dtype=torch.int32)
        cp_size = 2

        expect_restore_indices = torch.tensor(
            [
                0,
                1,
                2,
                3,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                12,
                13,
                14,
                15,
            ],
            dtype=torch.int32,
        )

        restore_indices = cp_test.generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )
        self.assertTrue(torch.equal(restore_indices, expect_restore_indices))

    def test_different_chunk_sizes(self):
        """Test with different chunk sizes across ranks"""
        # Create uneven chunk lengths
        prefill_cp_chunk_lengths = torch.tensor([2, 4, 8, 2], dtype=torch.int32)
        cp_size = 4

        restore_indices = cp_test.generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )
        # rank0:[0,7,8,9,22,23,24,25,26,27,52,53,54,55,56,63]
        # rank1:[1,6,10,11,20,21,28,29,30,31,48,49,50,51,57,62]
        # rank2:[2,5,12,13,18,19,32,33,34,35,44,45,46,47,58,61]
        # rank3:[3,4,14,15,16,17,36,37,38,39,40,41,42,43,59,60]
        # gather all rank position_ids and then argsort
        expect_restore_indices = torch.tensor(
            [
                0,
                16,
                32,
                48,
                49,
                33,
                17,
                1,
                2,
                3,
                18,
                19,
                34,
                35,
                50,
                51,
                52,
                53,
                36,
                37,
                20,
                21,
                4,
                5,
                6,
                7,
                8,
                9,
                22,
                23,
                24,
                25,
                38,
                39,
                40,
                41,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                42,
                43,
                44,
                45,
                26,
                27,
                28,
                29,
                10,
                11,
                12,
                13,
                14,
                30,
                46,
                62,
                63,
                47,
                31,
                15,
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(restore_indices, expect_restore_indices))


class TestGenerateQKVPaddingMask(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(
            os.environ.get("TEST_SRCDIR", ".")
            + "/rtp_llm/rtp_llm/cpp/models/context_parallel/test"
        )

    def test_with_padding(self):
        """Test padding mask with actual padding"""
        # Some ranks have padding
        prefill_cp_chunk_lengths = torch.tensor([4, 8, 2, 6], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 2, 0, 2], dtype=torch.int32)
        cp_size = 2
        padding_mask = cp_test.generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )
        expect_padding_mask = torch.ones(
            cp_size * torch.sum(prefill_cp_chunk_lengths), dtype=torch.int32
        )
        expect_padding_mask[22:24] = 0
        expect_padding_mask[38:] = 0
        self.assertTrue(torch.equal(expect_padding_mask, padding_mask))


class TestRestoreContextParallelOutputs(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA-compatible device")
    def test_decode_only_preserves_device_tokens_between_mtp_draft_steps(self):
        combo_tokens = torch.tensor([11, 12], dtype=torch.int32, device="cuda")
        hidden_states = torch.arange(8, dtype=torch.float32, device="cuda").reshape(
            2, 4
        )

        prepared_tokens, prepared_hidden = (
            cp_test.handle_context_parallel_decode_inputs(combo_tokens, hidden_states)
        )

        self.assertTrue(prepared_tokens.is_cuda)
        self.assertEqual(prepared_tokens.data_ptr(), combo_tokens.data_ptr())
        self.assertTrue(torch.equal(prepared_tokens, combo_tokens))
        self.assertTrue(torch.equal(prepared_hidden, hidden_states))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA-compatible device")
    def test_target_verify_shuffles_and_pads_device_tokens(self):
        combo_tokens = torch.tensor(
            [10, 11, 12, 13, 14, 15], dtype=torch.int32, device="cuda"
        )

        rank0_tokens = cp_test.handle_context_parallel_target_verify_inputs(
            combo_tokens, 0
        )
        rank1_tokens = cp_test.handle_context_parallel_target_verify_inputs(
            combo_tokens, 1
        )

        self.assertTrue(torch.equal(rank0_tokens.cpu(), torch.tensor([10, 0])))
        self.assertTrue(torch.equal(rank1_tokens.cpu(), torch.tensor([11, 0])))

    def test_mtp_hidden_states_follow_token_shuffle_and_zero_padding(self):
        global_hidden = torch.arange(22, dtype=torch.float32).reshape(11, 2)
        selected = cp_test.select_context_parallel_hidden_states(
            global_hidden,
            1,
            torch.tensor([1, 6, 4], dtype=torch.int32),
            torch.tensor([4, 2], dtype=torch.int32),
            torch.tensor([0, 1, 5, 6, 1, 2], dtype=torch.int32),
        )

        expected = torch.stack(
            (
                global_hidden[0],
                global_hidden[1],
                global_hidden[2],
                global_hidden[6],
                torch.zeros(2),
                global_hidden[8],
                global_hidden[9],
            )
        )
        self.assertTrue(torch.equal(selected, expected))

    def test_decode_only_preserves_hidden_rows(self):
        decode_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        restored = cp_test.restore_context_parallel_outputs(
            decode_hidden,
            torch.empty((0, 2)),
            torch.empty((0,), dtype=torch.int32),
            torch.empty((0,), dtype=torch.bool),
        )
        self.assertTrue(torch.equal(restored, decode_hidden))

    def test_mixed_decode_and_prefill_restores_only_valid_prefill_rows(self):
        decode_hidden = torch.tensor([[10.0], [20.0]])
        gathered_prefill_hidden = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        restore_indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        padding_mask = torch.tensor([True, True, False, True])
        restored = cp_test.restore_context_parallel_outputs(
            decode_hidden,
            gathered_prefill_hidden,
            restore_indices,
            padding_mask,
        )
        expected = torch.tensor([[10.0], [20.0], [2.0], [0.0], [1.0]])
        self.assertTrue(torch.equal(restored, expected))

    def test_no_padding(self):
        """Test when there is no padding"""
        prefill_cp_chunk_lengths = torch.tensor([8, 8, 8], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 0, 0], dtype=torch.int32)
        cp_size = 4

        padding_mask = cp_test.generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )

        expect_padding_mask = torch.ones(
            cp_size * torch.sum(prefill_cp_chunk_lengths), dtype=torch.bool
        )
        self.assertTrue(torch.equal(expect_padding_mask, padding_mask))


if __name__ == "__main__":
    unittest.main()
