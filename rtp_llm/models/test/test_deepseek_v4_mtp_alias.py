from types import SimpleNamespace
import unittest

import torch

from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Mtp
from rtp_llm.utils.model_weight import W


def _config(**overrides):
    values = {
        "vocab_size": 129280,
        "hidden_size": 4096,
        "data_type": torch.bfloat16,
        "enable_fp32_lm_head": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DeepSeekV4MtpAliasTest(unittest.TestCase):
    def _target(self):
        target = object.__new__(DeepSeekV4)
        target.model_config = _config()
        return target

    def test_aliases_full_vocabulary_matrices(self):
        self.assertEqual(
            DeepSeekV4Mtp.speculative_weight_alias_names(
                self._target(), _config()
            ),
            (W.embedding, W.lm_head),
        )

    def test_rejects_incompatible_storage_semantics(self):
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            DeepSeekV4Mtp.speculative_weight_alias_names(
                self._target(), _config(vocab_size=1)
            )

    def test_rejects_non_v4_owner(self):
        with self.assertRaisesRegex(TypeError, "target owner"):
            DeepSeekV4Mtp.speculative_weight_alias_names(object(), _config())


if __name__ == "__main__":
    unittest.main()
