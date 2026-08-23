from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import SpeculativeType


class _DraftModel:
    kwargs = None

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        return ()

    @classmethod
    def from_config(cls, **kwargs):
        cls.kwargs = kwargs
        return object()


class ModelFactoryProposeLoadPolicyTest(unittest.TestCase):
    def test_propose_model_inherits_force_cpu_load_policy(self):
        score_config = SimpleNamespace(max_seq_len=8192, gen_num_per_cycle=1)
        draft_config = SimpleNamespace(
            model_type="fake_mtp", max_seq_len=0, gen_num_per_cycle=0
        )
        engine_config = SimpleNamespace(
            runtime_config=SimpleNamespace(
                warm_up=False, model_warm_up=False, max_generate_batch_size=1
            ),
            sp_config=SimpleNamespace(
                type=SpeculativeType.MTP, gen_num_per_cycle=1
            ),
            parallelism_config=object(),
            hw_kernel_config=object(),
            kv_cache_config=object(),
            fmha_config=object(),
            moe_config=object(),
            device_resource_config=object(),
            load_config=SimpleNamespace(
                load_method="scratch",
                force_cpu_load_weights=True,
                loader_recycle_handles=False,
                moe_pure_tp_preshard=False,
            ),
        )

        with patch.object(ModelFactory, "get_model_cls", return_value=_DraftModel):
            result = ModelFactory.get_sp_model(
                score_config, draft_config, engine_config
            )

        self.assertIsNotNone(result)
        self.assertTrue(_DraftModel.kwargs["force_cpu_load_weights"])


if __name__ == "__main__":
    unittest.main()
