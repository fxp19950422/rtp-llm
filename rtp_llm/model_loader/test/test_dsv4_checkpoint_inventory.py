import hashlib
import json
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.models.deepseek_v4 import DeepSeekV4MtpWeight, DeepSeekV4Weight
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
        "compress_ratios": [
            0,
            0,
            4,
            *([128, 4] * 20),
            0,
        ],
    }


class TestDsv4CheckpointInventory(unittest.TestCase):
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
        index_names = dict.fromkeys(names, "model.safetensors")
        index_names.pop("layers.0.ffn.experts.255.w3.scale")
        index_names["layers.0.unexpected.weight"] = "model.safetensors"
        with self.assertRaisesRegex(ValueError, "missing=.*unexpected="):
            validate_inventory(config, {"weight_map": index_names})

    def test_production_descriptors_consume_each_source_exactly_once(self):
        config = _config()
        expected = expected_checkpoint_mapping(config)
        load_config = LoadConfig.model_construct(
            ep_size=1,
            ep_rank=0,
            phy2log=None,
        )

        def descriptor(cls, num_layers, ratios, num_hash_layers):
            weight = cls.__new__(cls)
            weight._num_layers = num_layers
            weight._compress_ratios = ratios
            weight._num_hash_layers = num_hash_layers
            weight._hidden_size = 4096
            weight._size_per_head = 512
            weight._head_num = 64
            weight._head_num_kv = 1
            weight.expert_num_ = 256
            weight._moe_align_size = 64
            weight.enable_fp32_lm_head = False
            return weight._get_weight_info().to_quant_weight_info(
                Fp8BlockWiseQuantConfig(is_quanted=True)
            )

        def sources(weight_info):
            counts = Counter()
            for layer_id, weights in enumerate(weight_info.layer_weights):
                for weight in weights:
                    for component in weight.get_components():
                        counts.update(component.get_tensor_names(layer_id, load_config))
            for weight in weight_info.weights:
                counts.update(weight.get_tensor_names(None, load_config))
            return counts

        main_info = descriptor(
            DeepSeekV4Weight,
            43,
            [int(value) for value in config["compress_ratios"][:43]],
            3,
        )
        mtp_info = descriptor(DeepSeekV4MtpWeight, 1, [0], 0)
        actual = sources(main_info) + sources(mtp_info)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(
            [name for name, count in actual.items() if count != 1],
            [],
        )

    def test_ep_ownership_is_lossless_for_supported_sizes(self):
        for ep_size in (1, 2, 4, 8):
            owners = expert_owners(256, ep_size)
            self.assertEqual(len(owners), ep_size)
            self.assertEqual(
                [expert for rank in owners for expert in rank], list(range(256))
            )
        owners = expert_owners(256, 8)
        self.assertEqual(owners[0][0], 0)
        self.assertIn(31, owners[0])
        self.assertEqual(owners[1][0], 32)
        self.assertEqual(owners[-1][-1], 255)
        with self.assertRaises(ValueError):
            expert_owners(255, 8)

    def test_tp_packed_axis_shards_reassemble_without_reencoding(self):
        # Synthetic packed FP4 rows: TP slicing is byte-aligned and a pure
        # contiguous partition.  Reassembly must preserve every packed byte.
        packed_row = bytes(range(256)) * 8
        for tp_size in (1, 2, 4, 8):
            shard_width = len(packed_row) // tp_size
            shards = [
                packed_row[rank * shard_width : (rank + 1) * shard_width]
                for rank in range(tp_size)
            ]
            self.assertEqual(b"".join(shards), packed_row)

    def test_safetensors_original_slice_metadata_and_hash(self):
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
            index = {"weight_map": {name: "model.safetensors"}}
            metadata = inspect_tensor(root, index, name)
        self.assertEqual(metadata.dtype, "I8")
        self.assertEqual(metadata.shape, (2048, 2048))
        self.assertEqual(
            metadata.slice_sha256, hashlib.sha256(payload[:4096]).hexdigest()
        )
        _validate_routed_fp4(metadata, _config())

    def test_repeated_plan_has_bounded_rss(self):
        self.assertGreater(current_rss_bytes(), 0)
        self.assertLessEqual(repeated_plan_rss_growth(_config(), iterations=4), 32 << 20)


class TestLoadConfigExpertPartition(unittest.TestCase):
    @staticmethod
    def _partition(ep_size, ep_rank, *, expert_num=256, phy2log=None):
        load_config = LoadConfig.model_construct(
            ep_size=ep_size,
            ep_rank=ep_rank,
            phy2log=phy2log,
        )
        return load_config.get_selected_experts(0, expert_num)

    def test_ep_1_2_4_8_boundaries(self):
        for ep_size in (1, 2, 4, 8):
            actual = [
                self._partition(ep_size, rank) for rank in range(ep_size)
            ]
            self.assertEqual(
                actual, [list(values) for values in expert_owners(256, ep_size)]
            )

    def test_eplb_physical_slots_preserve_duplicates(self):
        physical = list(range(256)) + list(range(8))
        parts = [
            self._partition(8, rank, phy2log=[physical]) for rank in range(8)
        ]
        self.assertEqual([value for part in parts for value in part], physical)

    def test_zero_experts_remains_a_valid_empty_partition(self):
        for ep_size in (1, 8):
            for rank in range(ep_size):
                self.assertEqual(
                    self._partition(ep_size, rank, expert_num=0),
                    [],
                )

    def test_empty_phy2log_uses_logical_expert_range(self):
        self.assertEqual(
            self._partition(8, 0, phy2log=[]),
            list(range(32)),
        )

    def test_invalid_partition_fails_instead_of_dropping_experts(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._partition(8, 0, expert_num=255)
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            self._partition(8, 8)
        with self.assertRaisesRegex(ValueError, "ep_size"):
            self._partition(0, 0)
        load_config = LoadConfig.model_construct(
            ep_size=8,
            ep_rank=0,
            phy2log=[list(range(256))],
        )
        with self.assertRaisesRegex(ValueError, "layer_id"):
            load_config.get_selected_experts(1, 256)


if __name__ == "__main__":
    unittest.main()
