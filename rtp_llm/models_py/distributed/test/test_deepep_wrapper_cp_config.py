import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed import deepep_wrapper
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPWrapper,
    DeepepWrapperConfig,
    init_deepep_wrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import CPRotateMethod, MoeConfig, ParallelismConfig, SpeculativeType


class DeepEPWrapperCPConfigTest(TestCase):
    def test_low_latency_capacity_uses_attention_tp_view_in_cp_mode(self):
        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = 2
        parallelism_config.dp_size = 4
        parallelism_config.ep_size = 8
        parallelism_config.world_size = 8
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

        moe_config = MoeConfig()
        moe_config.use_deepep_low_latency = True
        moe_config.ll_num_max_token = 32

        model_config = ModelConfig()
        model_config.hidden_size = 5120
        model_config.expert_num = 160
        model_config.moe_k = 8
        model_config.quant_config = SimpleNamespace(
            is_quanted=lambda: True,
            get_method=lambda: "INT8_PER_CHANNEL_COMPRESSED",
        )

        engine_config = SimpleNamespace(
            hw_kernel_config=None,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            runtime_config=SimpleNamespace(max_generate_batch_size=32),
            sp_config=SimpleNamespace(type=SpeculativeType.NONE),
        )

        adapter = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=model_config.quant_config,
        )
        self.assertEqual(adapter.tp_size, 1)
        router_capacity = DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
            moe_config.ll_num_max_token,
            adapter.tp_size,
            model_config.quant_config,
        )

        with (
            patch.object(DeepEPWrapper, "supported", return_value=True),
            patch.object(DeepEPWrapper, "create") as create,
        ):
            init_deepep_wrapper(engine_config, model_config)

        initialized_config = create.call_args.args[0]
        self.assertEqual(initialized_config.tp_size, 2)
        self.assertEqual(
            initialized_config.ll_num_max_token_per_rank,
            router_capacity,
        )

    def test_low_latency_capacity_honors_environment_override(self):
        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = 16
        parallelism_config.dp_size = 1
        parallelism_config.ep_size = 16
        parallelism_config.world_size = 16
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

        moe_config = MoeConfig()
        moe_config.use_deepep_low_latency = True
        moe_config.ll_num_max_token = 16

        model_config = ModelConfig()
        model_config.hidden_size = 7168
        model_config.expert_num = 160
        model_config.moe_k = 8
        model_config.quant_config = SimpleNamespace(
            is_quanted=lambda: True,
            get_method=lambda: "INT8_PER_CHANNEL_COMPRESSED",
        )
        engine_config = SimpleNamespace(
            hw_kernel_config=None,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            runtime_config=SimpleNamespace(max_generate_batch_size=16),
            sp_config=SimpleNamespace(type=SpeculativeType.NONE),
        )

        with (
            patch.dict(os.environ, {"DEEPEP_LL_NUM_MAX_TOKEN": "4096"}),
            patch.object(DeepEPWrapper, "supported", return_value=True),
            patch.object(DeepEPWrapper, "create") as create,
        ):
            init_deepep_wrapper(engine_config, model_config)

        initialized_config = create.call_args.args[0]
        expected_capacity = DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
            4096, 1, model_config.quant_config
        )
        self.assertEqual(
            initialized_config.ll_num_max_token_per_rank,
            expected_capacity,
        )

    def test_pd_decode_capacity_is_aligned_to_the_int8_kernel_bucket(self):
        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = 8
        parallelism_config.dp_size = 1
        parallelism_config.ep_size = 8
        parallelism_config.world_size = 8
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

        moe_config = MoeConfig()
        moe_config.use_deepep_low_latency = True

        model_config = ModelConfig()
        model_config.hidden_size = 7168
        model_config.expert_num = 160
        model_config.moe_k = 8
        model_config.quant_config = SimpleNamespace(
            is_quanted=lambda: True,
            get_method=lambda: "INT8_PER_CHANNEL_COMPRESSED",
        )
        engine_config = SimpleNamespace(
            hw_kernel_config=None,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            runtime_config=SimpleNamespace(max_generate_batch_size=8),
            sp_config=SimpleNamespace(type=SpeculativeType.NONE),
        )

        with (
            patch.dict(os.environ, {"DEEPEP_LL_NUM_MAX_TOKEN": "8"}),
            patch.object(DeepEPWrapper, "supported", return_value=True),
            patch.object(DeepEPWrapper, "create") as create,
        ):
            init_deepep_wrapper(engine_config, model_config)

        initialized_config = create.call_args.args[0]
        self.assertEqual(initialized_config.ll_num_max_token_per_rank, 16)

    def test_normal_buffer_size_can_be_bounded_for_chunked_prefill(self):
        wrapper = DeepEPWrapper.__new__(DeepEPWrapper)
        wrapper._config = SimpleNamespace(use_deepep_internode=False, local_rank=0)
        wrapper._use_accl_ep = False

        with (
            patch.dict(
                os.environ,
                {"DEEPEP_NORMAL_NUM_NVL_BYTES": "1000000000"},
            ),
            patch.object(deepep_wrapper, "DeepEPBuffer") as buffer,
        ):
            wrapper._init_normal_buffer(object())

        self.assertEqual(buffer.call_args.kwargs["num_nvl_bytes"], 1000000000)


if __name__ == "__main__":
    import unittest

    unittest.main()
