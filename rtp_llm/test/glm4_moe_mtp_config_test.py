import unittest
from types import SimpleNamespace

from rtp_llm.models.glm4_moe import _create_mtp_moe_config


class Glm4MoeMtpConfigTest(unittest.TestCase):
    def test_draft_uses_independent_auto_strategy(self):
        target_config = SimpleNamespace(
            moe_strategy="int8_per_channel_pure_cp",
            nested=SimpleNamespace(value=1),
        )

        draft_config = _create_mtp_moe_config(target_config)

        self.assertEqual(target_config.moe_strategy, "int8_per_channel_pure_cp")
        self.assertEqual(draft_config.moe_strategy, "auto")
        self.assertIsNot(draft_config, target_config)
        self.assertIsNot(draft_config.nested, target_config.nested)


if __name__ == "__main__":
    unittest.main()
