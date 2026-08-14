"""Per-channel INT8 (W8A8) quantized weight modules.

DeepSeek-V4-Flash-W8A8-INT8 stores each quantized projection as an int8
``.weight`` plus an FP32 per-output-channel ``.scale`` (the compressed-tensors
"int-quantized" scheme; activations are per-token dynamic and carry no static
scale). This mirrors the FP8 per-block path in ``per_block_fp8_quant_weight.py``
but is simpler: the scale is a plain per-channel FP32 vector, so there is no
UE8M0 / TMA-packing round trip.

Only the ``v4.*`` dense linears are handled here (attention, indexer outer
projection, shared expert, MTP / DSpARK projections). They all replicate across
TP (``sp_id``) — DSV4 keeps attention replicated and shards routed experts via
EP, so there is no output-/input-axis scale split to reason about. The stacked
routed MoE experts are handled separately.
"""

import copy
from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.quant_config import (
    Int8PerChannelCompressedQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, is_v4_weight, sp_id

# ``v4.*`` weight key -> its ``.scale`` companion key. Same dense set the FP8
# path quantizes (see ``_V4_FP8_WEIGHT_LIST``); the routed MoE experts are
# stacked and handled by a dedicated module, not here.
_V4_INT8_WEIGHT_LIST: Dict[str, str] = {
    W.v4_attn_wq_a_w: W.v4_attn_wq_a_s,
    W.v4_attn_wq_b_w: W.v4_attn_wq_b_s,
    W.v4_attn_wkv_w: W.v4_attn_wkv_s,
    W.v4_attn_wo_a_w: W.v4_attn_wo_a_s,
    W.v4_attn_wo_b_w: W.v4_attn_wo_b_s,
    W.v4_indexer_wq_b_w: W.v4_indexer_wq_b_s,
    W.v4_shared_w1_w: W.v4_shared_w1_s,
    W.v4_shared_w2_w: W.v4_shared_w2_s,
    W.v4_shared_w3_w: W.v4_shared_w3_s,
    W.v4_shared_w13_w: W.v4_shared_w13_s,
    W.v4_mtp_e_proj_w: W.v4_mtp_e_proj_s,
    W.v4_mtp_h_proj_w: W.v4_mtp_h_proj_s,
    W.v4_dspark_main_proj_w: W.v4_dspark_main_proj_s,
}


def gemm_per_channel_int8_v4_tp_strategy() -> Dict[str, Any]:
    """TP split strategy for the v4 int8 dense weights + their scales.

    Every v4 dense weight and its per-channel scale replicate (``sp_id``):
    attention is replicated across TP and routed experts shard via EP, so no
    v4 dense linear is axis-split.
    """
    strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    for w_key, s_key in _V4_INT8_WEIGHT_LIST.items():
        strategy[w_key] = sp_id
        strategy[s_key] = sp_id
    return strategy


class W8A8Int8PerChannelAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = gemm_per_channel_int8_v4_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class W8A8Int8PerChannelAttnAtomicWeight(
    AttnAtomicWeight, W8A8Int8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Int8PerChannelMlaAttnAtomicWeight(
    MlaAttnAtomicWeight, W8A8Int8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Int8PerChannelFfnAtomicWeight(
    FfnAtomicWeight, W8A8Int8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Int8PerChannelMoeAtomicWeight(
    MoeAtomicWeight, W8A8Int8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w8a8_int8_per_channel_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> W8A8Int8PerChannelAtomicWeight:
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Int8PerChannelMlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Int8PerChannelAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Int8PerChannelMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Int8PerChannelFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Int8PerChannelAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class V4PerChannelInt8Weight(CompositeWeight, QuantWeight):
    """V4 per-channel INT8 weight: int8 ``.weight`` + FP32 per-channel ``.scale``.

    The atoms declared by ``DeepSeekV4Weight`` carry only the ``.weight`` ckpt
    key; this wrapper derives the ``.scale`` companion from the same stem, the
    way ``V4PerBlockFp8Weight`` does for FP8, and keeps both byte-exact from the
    checkpoint (no dequant/requant, no scale re-layout).
    """

    V4_W_SUFFIX = ".weight"
    V4_S_SUFFIX = ".scale"

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, Int8PerChannelCompressedQuantConfig
        ):
            return False
        if not is_v4_weight(src_weight_info):
            return False
        return src_weight_info.name in _V4_INT8_WEIGHT_LIST

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        scale_name = _V4_INT8_WEIGHT_LIST[src_weight_info.name]
        kernel, scale = self._v4_build_pair(
            src_weight_info, src_weight_info.name, scale_name
        )
        sub_weights = {kernel.name: kernel, scale.name: scale}
        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = sub_weights[kernel.name]
        self.scale = sub_weights[scale.name]

    def _v4_build_pair(
        self,
        src_weight_info: WeightModule,
        weight_key: str,
        scale_key: str,
    ):
        """Build the int8 kernel + FP32 per-channel scale atom pair from the
        V4 ``.weight`` / ``.scale`` ckpt suffix convention."""
        weight_infos = []
        scale_infos = []
        for ckpt in src_weight_info.weights:
            base_ckpt = ckpt.name
            assert base_ckpt.endswith(self.V4_W_SUFFIX), (
                f"expected V4 int8 weight ckpt key to end with "
                f"'{self.V4_W_SUFFIX}', got {base_ckpt}"
            )
            stem = base_ckpt[: -len(self.V4_W_SUFFIX)]
            weight_infos.append(CkptWeightInfo(stem + self.V4_W_SUFFIX, ckpt.merge_fun))
            scale_infos.append(CkptWeightInfo(stem + self.V4_S_SUFFIX, ckpt.merge_fun))

        kernel = create_w8a8_int8_per_channel_weight(
            src_weight_info,
            weight_key,
            weight_infos,
            src_weight_info.process_fun,
            data_type=torch.int8,
            config=getattr(src_weight_info, "config", None),
        )
        scale = create_w8a8_int8_per_channel_weight(
            src_weight_info,
            scale_key,
            scale_infos,
            src_weight_info.process_fun,
            data_type=torch.float32,
            config=getattr(src_weight_info, "config", None),
        )
        return [kernel, scale]

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # int8 weight + FP32 per-channel scale are used as-is by the dsv4 python
        # forward (dequant happens in the INT8 GEMM); nothing to re-layout.
        return CompositeWeight._postprocess(self, tensor, device, load_config)
