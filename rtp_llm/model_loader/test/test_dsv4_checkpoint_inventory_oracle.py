"""Hermetic stdlib tests for the independent DSV4 checkpoint oracle."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from rtp_llm.model_loader.test.dsv4_checkpoint_inventory import (
    _validate_routed_fp4,
    current_rss_bytes,
    expected_checkpoint_mapping,
    expert_owners,
    inspect_tensor,
    repeated_plan_rss_growth,
    validate_inventory,
)


def _config():
    return {
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "num_nextn_predict_layers": 1,
        "compress_ratios": [0, 0, 4, *([128, 4] * 20), 0],
    }


class Dsv4CheckpointInventoryOracleTest(unittest.TestCase):
    def test_exact_flash_inventory(self):
        config = _config()
        mapping = expected_checkpoint_mapping(config)
        self.assertEqual(len(mapping), 69187)
        self.assertEqual(sum(name.startswith("mtp.0.") for name in mapping), 1575)
        index = {
            "metadata": {"total_size": 159609485896},
            "weight_map": {name: "model.safetensors" for name in mapping},
        }
        self.assertEqual(validate_inventory(config, index), mapping)

    def test_missing_and_unexpected_are_fatal(self):
        config = _config()
        names = expected_checkpoint_mapping(config)
        indexed = dict.fromkeys(names, "model.safetensors")
        indexed.pop("layers.0.ffn.experts.255.w3.scale")
        indexed["layers.0.unexpected.weight"] = "model.safetensors"
        with self.assertRaisesRegex(ValueError, "missing=.*unexpected="):
            validate_inventory(config, {"weight_map": indexed})

    def test_ep_ownership_is_lossless(self):
        for ep_size in (1, 2, 4, 8):
            owners = expert_owners(256, ep_size)
            self.assertEqual(len(owners), ep_size)
            self.assertEqual(
                [expert for rank in owners for expert in rank], list(range(256))
            )
        with self.assertRaises(ValueError):
            expert_owners(255, 8)

    def test_tp_packed_axis_reassembles_without_reencoding(self):
        packed_row = bytes(range(256)) * 8
        for tp_size in (1, 2, 4, 8):
            width = len(packed_row) // tp_size
            shards = [
                packed_row[rank * width : (rank + 1) * width]
                for rank in range(tp_size)
            ]
            self.assertEqual(b"".join(shards), packed_row)

    def test_safetensors_header_slice_and_hash(self):
        name = "layers.0.ffn.experts.0.w1.weight"
        payload = bytes(range(251)) * 20
        entry = {
            "dtype": "I8",
            "shape": [2048, 2048],
            "data_offsets": [0, len(payload)],
        }
        header = json.dumps({name: entry}, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (root / "model.safetensors").open("wb") as stream:
                stream.write(struct.pack("<Q", len(header)))
                stream.write(header)
                stream.write(payload)
            metadata = inspect_tensor(
                root, {"weight_map": {name: "model.safetensors"}}, name
            )
        self.assertEqual(metadata.dtype, "I8")
        self.assertEqual(metadata.shape, (2048, 2048))
        self.assertEqual(
            metadata.slice_sha256, hashlib.sha256(payload[:4096]).hexdigest()
        )
        _validate_routed_fp4(metadata, _config())

    def test_repeated_plan_has_bounded_rss(self):
        self.assertGreater(current_rss_bytes(), 0)
        self.assertLessEqual(
            repeated_plan_rss_growth(_config(), iterations=4), 32 << 20
        )


if __name__ == "__main__":
    unittest.main()
