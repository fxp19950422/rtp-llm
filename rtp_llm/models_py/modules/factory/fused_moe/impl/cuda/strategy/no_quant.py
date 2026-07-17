"""CUDA strategies without quantization"""

import logging
from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)

logger = logging.getLogger(__name__)


class CudaNoQuantPureCPStrategy(MoeStrategy):
    """Unquantized PureCP strategy for GLM MTP draft execution."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(config.moe_strategy in ("no_quant_pure_cp", "auto"))
        checker.check(config.dp_size == 1)
        checker.check(resolver.is_cp_equal_ep(config))
        checker.check(config.ep_size > 1)
        checker.check(config.parallelism_config.prefill_cp_config.is_enabled())
        checker.check(resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
            PureCpRouterNoQuant,
        )

        return StrategyAttributes(
            router_class=PureCpRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class CudaNoQuantEpLowLatencyStrategy(MoeStrategy):
    """Graph-safe BF16 DeepEP low-latency strategy used by MTP draft decode."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(
            config.moe_strategy == "no_auant_ep_low_latency"
            or config.moe_strategy == "auto"
        )
        if quant_method is None and config.enable_cuda_graph:
            logger.info(
                "BF16 EP Low Latency MoE with CUDA Graph enabled. "
                "DeepGEMM masked execution has no host synchronization or fallback, "
                "DeepEP LL uses static buffers, and TP=1 bypasses the post-combine all_gather."
            )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutorNoQuant,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouterNoQuant,
            executor_class=DeepGemmMaskedExecutorNoQuant,
            quant_config=quant_config,
        )


class CudaNoQuantCppStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "no_auant_cpp" or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=PureTpRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=quant_config,
        )


class CudaNoQuantDpNormalStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy and dp normal mode"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "no_auant_dp_normal" or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepepNormalRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=quant_config,
        )
