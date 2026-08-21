import unittest

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    NumericalStatusScope,
    PyModelInputs,
    PyModelOutputs,
    run_numerical_status_scenario,
    run_scenario,
)


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    def __init__(self) -> None:
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []
        self.seen_numerical_status: list[tuple[str, int | None, int]] = []
        self.input_status_replacement_succeeded = False
        self.values_replacement_succeeded = False

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        first_inputs = (
            next(iter(attention_inputs.values()))
            if isinstance(attention_inputs, dict)
            else attention_inputs
        )
        self.seen_input_lengths.append(first_inputs.input_lengths.tolist())

        status = inputs.numerical_status
        status_numel = (
            status.values.numel()
            if status.scope != NumericalStatusScope.NONE
            else None
        )
        self.seen_numerical_status.append(
            (status.scope.name, status_numel, status.live_rows)
        )
        try:
            inputs.numerical_status = status
        except (AttributeError, TypeError):
            pass
        else:
            self.input_status_replacement_succeeded = True
        try:
            status.values = torch.empty(0, device=inputs.input_ids.device)
        except (AttributeError, TypeError):
            pass
        else:
            self.values_replacement_succeeded = True
        if status.scope == NumericalStatusScope.ORIGIN_ROW:
            # The owner fields are read-only, while in-place producer updates
            # to the fixed-address int32 storage remain supported.
            status.values.bitwise_or_(inputs.input_ids.to(dtype=torch.int32))
        if getattr(self, "throw_after_status_on_call", None) == self.forward_calls:
            status.values.bitwise_or_(torch.full_like(status.values, 16))
            raise RuntimeError("injected failure after numerical status producer")

        assert self.kv_cache is not None
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            tag_inputs = (
                attention_inputs[layer_cache.tag]
                if isinstance(attention_inputs, dict)
                else attention_inputs
            )
            if (
                tag_inputs.cache_store_inputs is not None
                and tag_inputs.cache_store_writer is not None
            ):
                tag_inputs.cache_store_writer.write(
                    tag_inputs.cache_store_inputs, layer_cache
                )

        hidden_states = torch.zeros(
            (inputs.input_ids.numel(), 1),
            dtype=torch.float16,
            device=inputs.input_ids.device,
        )
        return PyModelOutputs(hidden_states)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


class OriginRowStatusModel(CacheStoreForwardModel):
    numerical_status_scope = "origin_row"


class CallableOriginRowStatusModel(CacheStoreForwardModel):
    def __init__(self) -> None:
        super().__init__()
        self.scope_getter_calls = 0
        self.scope_calls = 0

    @property
    def numerical_status_scope(self):
        self.scope_getter_calls += 1

        def resolve() -> str:
            self.scope_calls += 1
            return "origin_row"

        return resolve


class ThrowingOriginRowStatusModel(OriginRowStatusModel):
    def __init__(self) -> None:
        super().__init__()
        self.throw_after_status_on_call = 2


def _blocks_by_key(result: dict) -> dict[str, dict]:
    return {
        block["key"]: block
        for record in result["records"]
        for block in record["blocks"]
    }


def _record_for_request(result: dict, request_id: int) -> dict:
    matches = [
        record
        for record in result["records"]
        if record["request_id"] == str(request_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for request {request_id}, got {len(matches)}"
        )
    return matches[0]


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result["records"]), 2)
        blocks = _blocks_by_key(result)

        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 2)
        self.assertEqual(len(linear_blocks), 4)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})

    def test_micro_batch_slices_request_metadata_with_block_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(result["records"]), 3)

        expected = {
            201: ([2101], [16]),
            202: ([2201, 2202], [32, 48]),
            203: ([2301], [64]),
        }
        base = result["base_addresses"]["default"]
        for request_id, (token_keys, offsets) in expected.items():
            record = _record_for_request(result, request_id)
            self.assertEqual(len(record["blocks"]), len(token_keys))
            self.assertEqual(
                sorted(block["address"] - base for block in record["blocks"]),
                offsets,
            )
            for token_key in token_keys:
                self.assertTrue(
                    any(
                        f"_token_id_str_{token_key}_" in block["key"]
                        for block in record["blocks"]
                    )
                )

    def test_micro_batch_plan_false_keeps_fake_status_none_without_opt_in(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch_disabled")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(
            model.seen_numerical_status,
            [("NONE", None, 0), ("NONE", None, 0)],
        )
        self.assertFalse(result["status_ring_configured"])
        self.assertFalse(result["eager_status_allocated"])

    def test_micro_batch_plan_false_assigns_status_only_to_real_input(self) -> None:
        model = OriginRowStatusModel()
        run_scenario(model, "micro_batch_disabled")

        self.assertEqual(
            model.seen_numerical_status,
            [("ORIGIN_ROW", 4, 4), ("NONE", None, 0)],
        )
        self.assertFalse(model.input_status_replacement_succeeded)
        self.assertFalse(model.values_replacement_succeeded)

    def test_micro_batch_plan_true_uses_exact_disjoint_real_slices(self) -> None:
        model = OriginRowStatusModel()
        run_scenario(model, "micro_batch")

        self.assertEqual(
            model.seen_numerical_status,
            [("ORIGIN_ROW", 6, 6), ("ORIGIN_ROW", 2, 2)],
        )

    def test_eager_status_exact_shape_epoch_and_outstanding_leases(self) -> None:
        model = CallableOriginRowStatusModel()
        result = run_numerical_status_scenario(
            model,
            [33, 1, 1, 1, 1],
            hold_leases=True,
        )

        self.assertEqual(model.scope_calls, 1)
        self.assertEqual(model.scope_getter_calls, 1)
        self.assertEqual(result["numel"], [33, 1, 1, 1, 1])
        self.assertEqual(result["live_rows"], [33, 1, 1, 1, 1])
        self.assertEqual(result["epochs"], [1, 2, 3, 4, 5])
        self.assertEqual(len(set(result["addresses"])), 5)
        self.assertEqual(result["values"][0], list(range(1, 34)))
        self.assertEqual(result["values"][1:], [[1], [1], [1], [1]])
        self.assertEqual(
            [entry[1:] for entry in model.seen_numerical_status],
            [(33, 33), (1, 1), (1, 1), (1, 1), (1, 1)],
        )

    def test_unleased_small_slot_resizes_then_reuses_without_consumer(self) -> None:
        model = OriginRowStatusModel()
        result = run_numerical_status_scenario(
            model,
            [1, 33, 1, 1],
            read_values=False,
        )

        self.assertNotEqual(result["addresses"][0], result["addresses"][1])
        self.assertEqual(result["addresses"][1:], [result["addresses"][1]] * 3)
        self.assertEqual(result["numel"], [1, 33, 1, 1])
        self.assertEqual(result["values"], [])

    def test_zero_local_rows_keep_exact_empty_status_and_epoch(self) -> None:
        model = OriginRowStatusModel()
        result = run_numerical_status_scenario(
            model,
            [0, 1, 0],
            hold_leases=True,
        )

        self.assertEqual(result["numel"], [0, 1, 0])
        self.assertEqual(result["live_rows"], [0, 1, 0])
        self.assertEqual(result["epochs"], [1, 2, 3])
        self.assertEqual(result["values"], [[], [1], []])

    def test_consumed_fence_orders_aux_stream_before_slot_reuse(self) -> None:
        model = OriginRowStatusModel()
        result = run_numerical_status_scenario(
            model,
            [4] * 8,
            consume_on_aux_stream=True,
        )

        self.assertEqual(result["epochs"], list(range(1, 9)))
        self.assertEqual(result["values"], [list(range(1, 5))] * 8)
        self.assertEqual(result["addresses"], [result["addresses"][0]] * 8)

    def test_exception_records_source_fence_before_cross_stream_retry(self) -> None:
        model = ThrowingOriginRowStatusModel()
        result = run_numerical_status_scenario(
            model,
            [4, 4, 4],
            expected_throw_index=1,
        )

        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(result["epochs"], [1, 2])
        self.assertEqual(result["values"], [list(range(1, 5))] * 2)

    def test_context_parallel_publishes_original_lengths_not_local_chunk(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        # CP turns the six-token request into a four-token rank-local chunk for
        # attention, while CacheStore must still publish three two-token blocks.
        self.assertEqual(model.seen_input_lengths, [[4]])
        record = _record_for_request(result, 301)
        self.assertEqual(len(record["blocks"]), 3)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32, 48],
        )
        self.assertEqual(
            sorted(
                token_key
                for token_key in (3102, 3104, 3106)
                if any(
                    f"_token_id_str_{token_key}_" in block["key"]
                    for block in record["blocks"]
                )
            ),
            [3102, 3104, 3106],
        )

    def test_mtp_writer_uses_selected_sub_config_for_real_write(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "mtp_sub_config")

        record = _record_for_request(result, 401)
        self.assertEqual(len(record["blocks"]), 2)
        base = result["base_addresses"]["draft"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [32, 64],
        )
        self.assertEqual({block["length"] for block in record["blocks"]}, {32})
        self.assertTrue(
            all("model_id_7_" in block["key"] for block in record["blocks"])
        )
        self.assertTrue(all("_tag_draft" in block["key"] for block in record["blocks"]))


if __name__ == "__main__":
    unittest.main()
