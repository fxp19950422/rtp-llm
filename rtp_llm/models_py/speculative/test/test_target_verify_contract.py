import ast
from pathlib import Path
import unittest
from unittest import TestCase
from rtp_llm.models.dsv4_contracts import Dsv4StateRingMode
from rtp_llm.models_py.speculative.target_verify_contract import (
    NumericalFailureTransaction,
    TargetVerifyContract,
    TargetVerifyMode,
    TargetVerifyRole,
    map_verify_rows,
    validate_dsv4_speculative_target_query_len,
)


def oracle_verify_rows(batch_size, query_len, accepted, base, padded_batch_size):
    request_ids, token_indices = [], []
    for request in range(padded_batch_size):
        for token in range(query_len):
            request_ids.append(request if request < batch_size else -1)
            token_indices.append(token if request < batch_size else -1)
    return (
        tuple(request_ids),
        tuple(token_indices),
        tuple(request * query_len for request in range(batch_size)),
        tuple(accepted),
        tuple(prefix + length for prefix, length in zip(base, accepted)),
    )


class TargetVerifyContractTest(TestCase):
    def test_legacy_cache_module_reexports_t13_leaf(self):
        source = Path(__file__).parents[3].joinpath("models", "dsv4_kv_cache.py")
        tree = ast.parse(source.read_text())
        imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "rtp_llm.models.dsv4_contracts"
            for alias in node.names
        }
        self.assertTrue(
            {"Dsv4StateRingMode", "validate_dsv4_speculative_target_query_len"}
            <= imports
        )

    def test_query_boundary_and_explicit_ring_mode(self):
        contract = TargetVerifyContract(
            TargetVerifyRole.TARGET_VERIFY,
            TargetVerifyMode.MTP,
            3,
            Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY,
        )
        self.assertEqual(contract.query_len, 3)
        with self.assertRaises(ValueError):
            TargetVerifyContract(
                TargetVerifyRole.TARGET_VERIFY,
                TargetVerifyMode.MTP,
                10,
                Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY,
            )

    def test_b1_b2_rows_accept_reject_padding_and_next_prefix(self):
        b1 = map_verify_rows(1, 3, (2,), (10,))
        self.assertEqual(b1.request_ids, (0, 0, 0))
        self.assertEqual(b1.token_indices, (0, 1, 2))
        self.assertEqual(b1.row_offsets, (0,))
        self.assertEqual(b1.accepted_lengths, (2,))
        self.assertEqual(b1.next_prefix_lengths, (12,))
        self.assertEqual(
            (b1.request_ids, b1.token_indices, b1.row_offsets, b1.accepted_lengths, b1.next_prefix_lengths),
            oracle_verify_rows(1, 3, (2,), (10,), 1),
        )

        b2 = map_verify_rows(2, 3, (0, 3), (5, 20), padded_batch_size=3)
        self.assertEqual(b2.request_ids, (0, 0, 0, 1, 1, 1, -1, -1, -1))
        self.assertEqual(b2.token_indices[-3:], (-1, -1, -1))
        self.assertEqual(b2.row_offsets, (0, 3))
        self.assertEqual(b2.accepted_lengths, (0, 3))
        self.assertEqual(b2.next_prefix_lengths, (5, 23))
        self.assertEqual(
            (b2.request_ids, b2.token_indices, b2.row_offsets, b2.accepted_lengths, b2.next_prefix_lengths),
            oracle_verify_rows(2, 3, (0, 3), (5, 20), 3),
        )

    def test_unsupported_query_lengths_rejected(self):
        self.assertEqual(validate_dsv4_speculative_target_query_len(1), 1)
        self.assertEqual(validate_dsv4_speculative_target_query_len(9), 9)
        for query_len in (10, 32):
            with self.assertRaises(ValueError):
                map_verify_rows(1, query_len, (0,), (0,))
        for query_len in (True, 3.0, "3"):
            with self.assertRaises(TypeError):
                map_verify_rows(1, query_len, (0,), (0,))

    def test_contract_matrix_is_closed(self):
        TargetVerifyContract(TargetVerifyRole.NORMAL, TargetVerifyMode.DISABLED, 0, Dsv4StateRingMode.NORMAL)
        for role, mode, ring in (
            (TargetVerifyRole.NORMAL, TargetVerifyMode.MTP, Dsv4StateRingMode.NORMAL),
            (TargetVerifyRole.NORMAL, TargetVerifyMode.DISABLED, Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY),
            (TargetVerifyRole.TARGET_VERIFY, TargetVerifyMode.DISABLED, Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY),
        ):
            with self.assertRaises(ValueError):
                TargetVerifyContract(role, mode, 3, ring)

    def test_prefix_and_length_types_are_strict(self):
        for values in ((True,), (-1,), (1.5,), ("1",)):
            with self.assertRaises(ValueError):
                map_verify_rows(1, 3, (0,), values)
        for values in ((True,), (1.5,), ("1",)):
            with self.assertRaises(ValueError):
                map_verify_rows(1, 3, values, (0,))
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            mapped = map_verify_rows(np.int64(1), np.int64(3), (np.int64(1),), (np.int64(7),))
            self.assertEqual(mapped.next_prefix_lengths, (8,))

    def test_numerical_failure_rolls_back_transaction(self):
        events = []
        tx = NumericalFailureTransaction()
        self.assertFalse(
            tx.run(
                lambda: events.append("prepare"),
                lambda: False,
                lambda: events.append("commit"),
                lambda: events.append("rollback"),
            )
        )
        self.assertEqual(events, ["prepare", "rollback"])
        self.assertEqual(tx.state, "rolled_back")

    def test_transaction_success_failures_and_repeat(self):
        tx = NumericalFailureTransaction()
        events = []
        self.assertTrue(
            tx.run(
                lambda: events.append("p"),
                lambda: True,
                lambda: events.append("c"),
                lambda: events.append("r"),
            )
        )
        self.assertEqual(tx.state, "committed")
        self.assertEqual(events, ["p", "c"])
        with self.assertRaises(RuntimeError):
            tx.run(lambda: None, lambda: True, lambda: None, lambda: None)

        for phase in ("prepare", "verify", "commit"):
            tx = NumericalFailureTransaction()
            events = []
            def action(name):
                def run():
                    events.append(name)
                    if name == phase:
                        raise RuntimeError(name)
                    return True
                return run
            with self.assertRaisesRegex(RuntimeError, phase):
                tx.run(action("prepare"), action("verify"), action("commit"), lambda: events.append("rollback"))
            self.assertEqual(tx.state, "rolled_back")
            self.assertEqual(events[-1], "rollback")

        tx = NumericalFailureTransaction()
        rollback_calls = []
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            tx.run(
                lambda: (_ for _ in ()).throw(RuntimeError("prepare")),
                lambda: True,
                lambda: None,
                lambda: (rollback_calls.append("rollback"), (_ for _ in ()).throw(RuntimeError("rollback")))[1],
            )
        self.assertEqual(tx.state, "rollback_failed")
        self.assertEqual(rollback_calls, ["rollback"])


if __name__ == "__main__":
    unittest.main()
