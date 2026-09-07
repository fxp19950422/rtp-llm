import copy
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)

GROUP_TAGS = ["full", "aux"]
HIDDEN_SIZE = 4
TOKENS_PER_BLOCK = 8


class TaggedBlockTableModel:
    """Small graph-safe model whose output exposes both tag-local block tables."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        full_id = attention_inputs["full"].kv_cache_kernel_block_id_device[0, 0]
        aux_id = attention_inputs["aux"].kv_cache_kernel_block_id_device[0, 0]
        signature = (full_id + 16 * aux_id).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class TaggedSequenceLengthModel:
    """Expose the cumulative lengths used by a tagged captured graph."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        full_inputs = inputs.attention_inputs["full"]
        signature = torch.stack(
            (
                full_inputs.cu_seqlens_device[-1],
                full_inputs.cu_kv_seqlens_device[-1],
                full_inputs.input_lengths_device.sum(),
                full_inputs.prefix_lengths_device.sum(),
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class TargetVerifyTailSensitiveModel:
    """Make stale fixed-capacity token rows observable in live output rows."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        hidden_signature = inputs.input_hiddens.sum()
        token_signature = inputs.input_ids.sum().to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + hidden_signature + token_signature)


class DeviceOnlyLengthImpl:
    cuda_graph_requires_host_metadata = False

    def __init__(self):
        self.metadata_addresses = None

    def support_cuda_graph(self):
        return True

    def prepare_cuda_graph(self, attn_inputs):
        # A device-only backend must not need the host mirror refreshed.
        entries = sorted(attn_inputs.items()) if isinstance(attn_inputs, dict) else [("flat", attn_inputs)]
        addresses = tuple((tag, tuple(getattr(attn, name).data_ptr() for name in (
            "input_lengths_device", "prefix_lengths_device",
            "cu_seqlens_device", "cu_kv_seqlens_device"))) for tag, attn in entries)
        if self.metadata_addresses is None:
            self.metadata_addresses = addresses
        else:
            assert addresses == self.metadata_addresses, "captured metadata addresses changed"


class DeviceOnlySequenceLengthModel(TaggedSequenceLengthModel):
    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return DeviceOnlyLengthImpl()


class DeviceOnlyPerRowLengthModel(DeviceOnlySequenceLengthModel):
    def forward(self, inputs, fmha_impl=None):
        attn = inputs.attention_inputs["full"]
        rows = torch.stack((attn.cu_seqlens_device[1:],
                            attn.cu_kv_seqlens_device[1:],
                            attn.input_lengths_device,
                            attn.prefix_lengths_device), dim=1)
        return PyModelOutputs(inputs.input_hiddens + rows.to(
            inputs.input_hiddens.dtype).repeat_interleave(4, dim=0))


class NumericalStatusModel:
    numerical_status_scope = "origin_row"

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        live_rows = inputs.input_ids.shape[0]
        inputs.numerical_status.values[:live_rows].bitwise_or_(inputs.input_ids)
        return PyModelOutputs(inputs.input_hiddens[:, :HIDDEN_SIZE])


class CallableNumericalStatusModel(NumericalStatusModel):
    def __init__(self) -> None:
        self.scope_calls = 0

    def numerical_status_scope(self) -> str:
        self.scope_calls += 1
        return "origin_row"


class BatchNumericalStatusModel(NumericalStatusModel):
    numerical_status_scope = "batch"

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        flag = inputs.input_ids.ne(0).any().to(torch.int32).reshape(1)
        inputs.numerical_status.values.bitwise_or_(flag)
        return PyModelOutputs(inputs.input_hiddens)


def _tag_attention_inputs(
    common: PyAttentionInputs, tags: list[str], values: dict[str, int]
) -> dict[str, PyAttentionInputs]:
    tagged = {}
    for tag in tags:
        tag_inputs = copy.copy(common)
        host_blocks = torch.full_like(
            common.kv_cache_kernel_block_id, values[tag], device="cpu"
        ).pin_memory()
        device_blocks = host_blocks.cuda()
        tag_inputs.kv_cache_kernel_block_id = host_blocks
        tag_inputs.kv_cache_kernel_block_id_device = device_blocks
        tag_inputs.kv_cache_block_id = host_blocks
        tag_inputs.kv_cache_block_id_device = device_blocks
        tagged[tag] = tag_inputs
    return tagged


def _build_common_inputs(
    attention_inputs: PyAttentionInputs,
    tags: list[str],
    values: dict[str, int],
    batch_size: int,
    token_count: int,
    block_count: int,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )

    attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attention_inputs.padding_offset = torch.zeros(
        token_count, dtype=torch.int32, device="cuda"
    )
    attention_inputs.total_tokens = token_count
    attention_inputs.kv_cache_kernel_block_id = torch.zeros(
        (batch_size, block_count), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_kernel_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id.cuda()
    )
    attention_inputs.kv_cache_block_id = attention_inputs.kv_cache_kernel_block_id
    attention_inputs.kv_cache_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id_device
    )
    inputs.attention_inputs = _tag_attention_inputs(attention_inputs, tags, values)
    return inputs


def _build_decode_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 2,
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = False
    attention_inputs.is_target_verify = False
    attention_inputs.prefix_lengths = torch.empty(
        0, dtype=torch.int32
    ).pin_memory()
    attention_inputs.input_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="cuda"
    )
    attention_inputs.decode_cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    attention_inputs.cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.zeros_like(
        attention_inputs.cu_seqlens_device
    )
    attention_inputs.context_total_kv_length = batch_size
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=batch_size,
        token_count=batch_size,
        block_count=1,
    )


def _build_prefill_inputs(
    tags: list[str], values: dict[str, int], seq_len: int = 4
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = True
    attention_inputs.is_target_verify = False
    attention_inputs.input_lengths = torch.tensor(
        [seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = torch.tensor(
        [0, seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device.clone()
    attention_inputs.context_total_kv_length = seq_len
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=1,
        token_count=seq_len,
        block_count=1,
    )


def _build_target_verify_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 1,
    query_len: int = 5,
    prefix_len: int = 11,
    is_prefill: bool = True,
) -> PyModelInputs:
    token_count = batch_size * query_len

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = is_prefill
    attention_inputs.is_target_verify = True
    attention_inputs.input_lengths = torch.full(
        (batch_size,), query_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.empty(
        0, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = (
        attention_inputs.prefix_lengths.cuda() + 1
    )

    cu_q = torch.arange(
        0, token_count + 1, query_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens = cu_q
    attention_inputs.cu_seqlens_device = cu_q.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.arange(
        0,
        batch_size * (query_len + prefix_len) + 1,
        query_len + prefix_len,
        dtype=torch.int32,
        device="cuda",
    )
    attention_inputs.decode_cu_seqlens = torch.arange(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.decode_cu_seqlens_device = (
        attention_inputs.decode_cu_seqlens.cuda()
    )

    attention_inputs.context_total_kv_length = batch_size * (
        query_len + prefix_len
    )

    block_count = (
        prefix_len + query_len + TOKENS_PER_BLOCK - 1
    ) // TOKENS_PER_BLOCK
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=batch_size,
        token_count=token_count,
        block_count=block_count,
    )


class TestCudaGraphTaggedCache(unittest.TestCase):
    def _assert_replay_signature(
        self, runner: CudaGraphRunner, inputs: PyModelInputs, expected: int
    ) -> None:
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected_output = torch.full_like(output.hidden_states, expected)
        torch.testing.assert_close(output.hidden_states, expected_output)

    def test_decode_tag_validation_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1}),
            18,
        )
        self._assert_replay_signature(
            runner,
            _build_decode_inputs(GROUP_TAGS, {"full": 5, "aux": 3}),
            53,
        )

        self.assertFalse(runner.canRun(_build_decode_inputs(["full"], {"full": 2})))
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(
                    ["full", "aux", "extra"],
                    {"full": 2, "aux": 1, "extra": 9},
                )
            )
        )
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(["full", "wrong"], {"full": 2, "wrong": 1})
            )
        )

    def test_prefill_tagged_capture_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TaggedBlockTableModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}),
            33,
        )
        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(GROUP_TAGS, {"full": 4, "aux": 3}),
            52,
        )

    def test_duplicate_capture_tag_is_rejected(self) -> None:
        runner = CudaGraphRunner()
        with self.assertRaisesRegex(
            RuntimeError, "duplicate CUDA graph KV cache tag=full"
        ):
            runner.init_decode(
                TaggedBlockTableModel(),
                HIDDEN_SIZE,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                [1],
                ["full", "full"],
            )

    def test_target_verify_validates_exact_tag_set(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
            True,
        )

        valid = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertTrue(runner.canRun(valid))

        missing = _build_target_verify_inputs(
            ["full"], {"full": 2}, batch_size=2, query_len=1, prefix_len=1
        )
        self.assertFalse(runner.canRun(missing))

        wrong = _build_target_verify_inputs(
            ["full", "wrong"],
            {"full": 2, "wrong": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertFalse(runner.canRun(wrong))

        non_prefill = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
            is_prefill=False,
        )
        self.assertFalse(runner.canRun(non_prefill))

    def test_target_verify_clears_rounded_batch_sequence_lengths(self) -> None:
        query_len = 5
        prefix_len = 11
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedSequenceLengthModel(),
            HIDDEN_SIZE,
            64,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            GROUP_TAGS,
            True,
            query_len,
        )

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                inputs = _build_target_verify_inputs(
                    GROUP_TAGS,
                    {"full": 2, "aux": 1},
                    batch_size=batch_size,
                    query_len=query_len,
                    prefix_len=prefix_len,
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), 4)

                output = runner.forward(inputs)
                torch.cuda.synchronize()
                total_query_length = batch_size * query_len
                total_kv_length = batch_size * (query_len + prefix_len)
                expected_signature = torch.tensor(
                    [
                        total_query_length,
                        total_kv_length,
                        total_query_length,
                        batch_size * prefix_len,
                    ],
                    dtype=output.hidden_states.dtype,
                    device=output.hidden_states.device,
                )
                torch.testing.assert_close(
                    output.hidden_states,
                    expected_signature.unsqueeze(0).expand_as(output.hidden_states),
                )

    def test_device_only_target_verify_lengths(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(DeviceOnlySequenceLengthModel(), HIDDEN_SIZE, 64,
                           TOKENS_PER_BLOCK, TOKENS_PER_BLOCK, [4], GROUP_TAGS, True, 4)
        for batch_size, prefix in ((4, 11), (1, 31), (3, 7), (1, 0), (4, 15)):
            inputs = _build_target_verify_inputs(
                GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=batch_size,
                query_len=4, prefix_len=prefix)
            self.assertTrue(runner.canRun(inputs))
            output = runner.forward(inputs)
            expected = torch.tensor([batch_size * 4, batch_size * (4 + prefix),
                                     batch_size * 4, batch_size * prefix],
                                    dtype=output.hidden_states.dtype, device="cuda")
            torch.testing.assert_close(output.hidden_states, expected.expand_as(output.hidden_states))

    def test_device_only_verify_reuses_live_cumulative_prefix(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(DeviceOnlyPerRowLengthModel(), HIDDEN_SIZE, 64,
                           TOKENS_PER_BLOCK, TOKENS_PER_BLOCK, [1, 2, 4],
                           GROUP_TAGS, True, 4)
        # Decreasing/repeated buckets and different prefixes across a KV block
        # boundary must preserve every live scan entry, not only its final sum.
        for prefixes in ([7, 8, 15, 16], [0], [31, 7, 8], [1], [8, 15], [0, 0, 0, 0]):
            with self.subTest(prefixes=prefixes):
                bs = len(prefixes)
                inputs = _build_target_verify_inputs(
                    GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=bs,
                    query_len=4, prefix_len=max(prefixes))
                prefix = torch.tensor(prefixes, dtype=torch.int32)
                cu_q = torch.arange(bs + 1, dtype=torch.int32) * 4
                cu_kv = torch.cat((torch.zeros(1, dtype=torch.int32), (prefix + 4).cumsum(0)))
                tagged = inputs.attention_inputs
                for attn in tagged.values():
                    attn.prefix_lengths = prefix.pin_memory()
                    attn.cu_kv_seqlens_device = cu_kv.to(device="cuda", dtype=torch.int32)
                inputs.attention_inputs = tagged
                self.assertTrue(runner.canRun(inputs))
                # CPU dispatch profiling sees ATen calls made by the actual C++
                # runner; building the oracle is deliberately outside this scope.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    output = runner.forward(inputs)
                expected = torch.stack((cu_q[1:], cu_kv[1:],
                                        torch.full_like(prefix, 4), prefix), dim=1)
                expected = expected.to(device="cuda", dtype=output.hidden_states.dtype).repeat_interleave(4, dim=0)
                torch.testing.assert_close(output.hidden_states, expected, atol=0, rtol=0)
                self.assertNotIn("aten::cumsum", {event.key for event in profile.key_averages()},
                                 "graph prepare must reuse producer cumulative lengths")

    def test_device_only_draft_prefill_reuses_cumulative_tail(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(DeviceOnlySequenceLengthModel(), 4, 64,
                            TOKENS_PER_BLOCK, TOKENS_PER_BLOCK, [2, 4],
                            HIDDEN_SIZE, GROUP_TAGS, 4)
        for query_len, prefix_len in ((4, 7), (1, 31), (3, 8), (1, 0), (2, 15)):
            with self.subTest(query_len=query_len, prefix_len=prefix_len):
                inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, query_len)
                tagged = inputs.attention_inputs
                for attn in tagged.values():
                    attn.prefix_lengths = torch.tensor([prefix_len], dtype=torch.int32).pin_memory()
                    attn.cu_kv_seqlens_device = torch.tensor([0, query_len + prefix_len], dtype=torch.int32, device="cuda")
                inputs.attention_inputs = tagged
                self.assertTrue(runner.canRun(inputs))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    output = runner.forward(inputs)
                expected = torch.tensor([query_len, query_len + prefix_len, query_len, prefix_len],
                                        dtype=output.hidden_states.dtype, device="cuda")
                torch.testing.assert_close(output.hidden_states, expected.expand_as(output.hidden_states), atol=0, rtol=0)
                self.assertNotIn("aten::cumsum", {event.key for event in profile.key_averages()})

    def test_device_only_verify_rejects_missing_live_cumulative_lengths(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(DeviceOnlySequenceLengthModel(), HIDDEN_SIZE, 64,
                           TOKENS_PER_BLOCK, TOKENS_PER_BLOCK, [4], GROUP_TAGS, True, 4)
        inputs = _build_target_verify_inputs(GROUP_TAGS, {"full": 0, "aux": 0}, query_len=4)
        tagged = inputs.attention_inputs
        for attn in tagged.values():
            attn.cu_kv_seqlens_device = torch.empty(0, dtype=torch.int32, device="cuda")
        inputs.attention_inputs = tagged
        self.assertTrue(runner.canRun(inputs))
        with self.assertRaisesRegex(RuntimeError, "producer's live cumulative lengths"):
            runner.forward(inputs)

    def test_target_verify_clears_token_tail_across_a_b_b_replays(self) -> None:
        query_len = 4
        runner = CudaGraphRunner()
        runner.init_decode(
            TargetVerifyTailSensitiveModel(),
            HIDDEN_SIZE,
            64,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [8],
            GROUP_TAGS,
            True,
            query_len,
        )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            poison = _build_target_verify_inputs(
                GROUP_TAGS,
                {"full": 7, "aux": 11},
                batch_size=8,
                query_len=query_len,
                prefix_len=11,
            )
            poison.input_ids.fill_(3)
            poison.input_hiddens.fill_(5)
            self.assertTrue(runner.canRun(poison))
            self.assertEqual(runner.getCurrentRealGraphSize(), 8)
            poison_output = runner.forward(poison).hidden_states.clone()

            replay_outputs = []
            for _ in range(2):
                clean = _build_target_verify_inputs(
                    GROUP_TAGS,
                    {"full": 13, "aux": 17},
                    batch_size=2,
                    query_len=query_len,
                    prefix_len=5,
                )
                clean.input_ids.zero_()
                clean.input_hiddens.zero_()
                self.assertTrue(runner.canRun(clean))
                self.assertEqual(runner.getCurrentRealGraphSize(), 8)
                replay_outputs.append(runner.forward(clean).hidden_states.clone())

        stream.synchronize()
        self.assertTrue(torch.isfinite(poison_output).all().item())
        self.assertTrue(torch.all(poison_output != 0).item())
        expected = torch.zeros(
            (2 * query_len, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        torch.testing.assert_close(replay_outputs[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_outputs[1], expected, rtol=0, atol=0)
        torch.testing.assert_close(replay_outputs[0], replay_outputs[1], rtol=0, atol=0)

    def test_numerical_status_uses_fixed_per_graph_storage_and_replay_reset(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            NumericalStatusModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2, 4],
            GROUP_TAGS,
        )

        first = _build_decode_inputs(GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=1)
        first.input_ids.fill_(7)
        self.assertTrue(runner.canRun(first))
        first_output = runner.forward(first)
        torch.cuda.synchronize()
        self.assertEqual(first_output.numerical_status.live_rows, 1)
        self.assertEqual(first_output.numerical_status.values.numel(), 2)
        torch.testing.assert_close(
            first_output.numerical_status.values,
            torch.tensor([7, 0], dtype=torch.int32, device="cuda"),
        )
        with self.assertRaises((AttributeError, TypeError)):
            first_output.numerical_status = first_output.numerical_status
        first_address = first_output.numerical_status.values.data_ptr()

        clean = _build_decode_inputs(GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=1)
        clean.input_ids.zero_()
        self.assertTrue(runner.canRun(clean))
        clean_output = runner.forward(clean)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            clean_output.numerical_status.values,
            torch.zeros(2, dtype=torch.int32, device="cuda"),
        )
        self.assertEqual(clean_output.numerical_status.values.data_ptr(), first_address)

        larger = _build_decode_inputs(GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=3)
        larger.input_ids.copy_(torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda"))
        self.assertTrue(runner.canRun(larger))
        larger_output = runner.forward(larger)
        torch.cuda.synchronize()
        self.assertEqual(larger_output.numerical_status.live_rows, 3)
        torch.testing.assert_close(
            larger_output.numerical_status.values,
            torch.tensor([1, 0, 2, 0], dtype=torch.int32, device="cuda"),
        )
        self.assertNotEqual(larger_output.numerical_status.values.data_ptr(), first_address)

    def test_batch_numerical_status_uses_scalar_storage(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            BatchNumericalStatusModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )

        failed = _build_decode_inputs(GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=1)
        failed.input_ids.fill_(1)
        self.assertTrue(runner.canRun(failed))
        failed_output = runner.forward(failed)
        torch.cuda.synchronize()
        self.assertEqual(failed_output.numerical_status.values.shape, (1,))
        torch.testing.assert_close(
            failed_output.numerical_status.values,
            torch.ones(1, dtype=torch.int32, device="cuda"),
        )

        clean = _build_decode_inputs(GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=1)
        clean.input_ids.zero_()
        self.assertTrue(runner.canRun(clean))
        clean_output = runner.forward(clean)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            clean_output.numerical_status.values,
            torch.zeros(1, dtype=torch.int32, device="cuda"),
        )

    def test_status_scope_callable_is_resolved_once_before_graph_runner(self) -> None:
        model = CallableNumericalStatusModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )
        self.assertEqual(model.scope_calls, 1)

    def test_prefill_status_capacity_uses_captured_input_ids_not_hidden_hold(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            NumericalStatusModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 0, "aux": 0}, seq_len=3
        )
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        # The hidden capture hold is 2 * max_seq_len (=16) rows, whereas the
        # graph-key input_ids view and its status capacity are exactly 4.
        self.assertEqual(output.numerical_status.values.numel(), 4)
        self.assertEqual(output.numerical_status.live_rows, 3)

    def test_fixed_mtp_prefill_status_keeps_fixed_input_capacity(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            NumericalStatusModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            HIDDEN_SIZE,
            GROUP_TAGS,
            2,
            2,
        )

        inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 0, "aux": 0}, seq_len=2
        )
        inputs.input_hiddens = torch.zeros(
            (2, HIDDEN_SIZE * 2), dtype=torch.bfloat16, device="cuda"
        )
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(output.numerical_status.live_rows, 2)
        self.assertEqual(output.numerical_status.values.numel(), 4)

    def test_numerical_status_source_fence_orders_100_cross_stream_replays(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            NumericalStatusModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )

        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        observed = torch.empty(100, dtype=torch.int32, device="cuda")
        expected = torch.empty_like(observed)
        for iteration in range(100):
            value = iteration % 7 + 1
            stream = streams[iteration % len(streams)]
            with torch.cuda.stream(stream):
                inputs = _build_decode_inputs(
                    GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=1
                )
                inputs.input_ids.fill_(value)
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward_with_numerical_status_snapshot(inputs)
                observed[iteration].copy_(output.numerical_status.values[0])
                expected[iteration].fill_(value)

        # The source fence and D2D snapshots must make one tail wait sufficient
        # even though every replay changes streams.
        torch.cuda.synchronize()
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
