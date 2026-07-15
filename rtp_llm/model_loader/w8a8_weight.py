import copy
from typing import Any, Dict

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import (
    W,
    ffn_sp_0,
    ffn_sp_0_w13,
    ffn_sp_neg1,
    sp_0,
    sp_0_w13,
    sp_head_gemm_a8,
    sp_head_s_gemm_a8,
    sp_id,
    sp_neg1,
)


def gemm_int8_gpt_style_tp_strategy():
    gemm_a8_weight_tp_strategy: Dict[str, Any] = {
        W.attn_qkv_w: sp_head_gemm_a8,
        W.attn_qkv_s: sp_head_s_gemm_a8,
        W.attn_o_w: sp_neg1,
        W.attn_o_s: sp_id,
        W.attn_o_smoother: sp_0,
        W.attn_o_shift: sp_0,
        W.ffn_w1: ffn_sp_0,
        W.ffn_s1: ffn_sp_0,
        W.ffn_w3: ffn_sp_0,
        W.ffn_s3: ffn_sp_0,
        W.ffn_w13: ffn_sp_0_w13,
        W.ffn_s13: ffn_sp_0_w13,
        W.ffn_w2: ffn_sp_neg1,
        W.ffn_s2: sp_id,
    }
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(gemm_a8_weight_tp_strategy)
    return tp_strategy


def compressed_int8_per_channel_tp_strategy():
    """TP strategy for compressed-tensors INT8 per-channel weights.

    Unlike SmoothQuant (which stores transposed [in, out] weights),
    compressed-tensors stores weights in standard [out, in] format.
    Therefore w13 must be split along dim 0 (out_features) using sp_0_w13
    instead of ffn_sp_0_w13 (which splits dim -1, designed for transposed format).
    """
    gemm_a8_weight_tp_strategy: Dict[str, Any] = {
        W.attn_qkv_w: sp_head_gemm_a8,
        W.attn_qkv_s: sp_head_s_gemm_a8,
        W.attn_o_w: sp_neg1,
        W.attn_o_s: sp_id,
        W.attn_o_smoother: sp_0,
        W.attn_o_shift: sp_0,
        W.ffn_w1: ffn_sp_0,
        W.ffn_s1: ffn_sp_0,
        W.ffn_w3: ffn_sp_0,
        W.ffn_s3: ffn_sp_0,
        W.ffn_w13: sp_0_w13,
        W.ffn_s13: sp_0_w13,
        W.ffn_w2: ffn_sp_neg1,
        W.ffn_s2: sp_id,
    }
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(gemm_a8_weight_tp_strategy)
    return tp_strategy


class W8A8Int8AtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = gemm_int8_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class CompressedInt8PerChannelAtomicWeight(AtomicWeight):
    """Atomic weight for compressed-tensors INT8 per-channel format (non-transposed [out, in])."""

    gpt_style_tp_strategy = compressed_int8_per_channel_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class CompressedInt8PerChannelAttnAtomicWeight(
    AttnAtomicWeight, CompressedInt8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class CompressedInt8PerChannelFfnAtomicWeight(
    FfnAtomicWeight, CompressedInt8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class CompressedInt8PerChannelMoeAtomicWeight(
    MoeAtomicWeight, CompressedInt8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_compressed_int8_per_channel_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> CompressedInt8PerChannelAtomicWeight:
    """Factory for compressed-tensors INT8 per-channel weights with correct TP strategy."""
    if isinstance(src_weight_info, AttnAtomicWeight):
        return CompressedInt8PerChannelAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return CompressedInt8PerChannelMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return CompressedInt8PerChannelFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return CompressedInt8PerChannelAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class W8A8Int8AttnAtomicWeight(AttnAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Int8MlaAttnAtomicWeight(MlaAttnAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Int8FfnAtomicWeight(FfnAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Int8MoeAtomicWeight(MoeAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w8a8_int8_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> W8A8Int8AtomicWeight:
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Int8MlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Int8AttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Int8MoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Int8FfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Int8AtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


def gemm_fp8_per_tensor_gpt_style_tp_strategy():
    gemm_a8_weight_tp_strategy: Dict[str, Any] = {
        W.attn_qkv_w: sp_head_gemm_a8,
        W.attn_qkv_s: sp_id,
        W.attn_o_w: sp_neg1,
        W.attn_o_s: sp_id,
        W.attn_o_smoother: sp_id,
        W.attn_o_shift: sp_id,
        W.ffn_w1: ffn_sp_0,
        W.ffn_s1: sp_id,
        W.ffn_w3: ffn_sp_0,
        W.ffn_s3: sp_id,
        W.ffn_w13: sp_0_w13,
        W.ffn_s13: sp_id,
        W.ffn_w2: ffn_sp_neg1,
        W.ffn_s2: sp_id,
        W.pre_ln_static_quant: sp_id,
        W.pre_ln_static_quant_reciprocal: sp_id,
        W.attention_output_static_quant: sp_id,
        W.attention_output_static_quant_reciprocal: sp_id,
        W.post_ln_static_quant: sp_id,
        W.post_ln_static_quant_reciprocal: sp_id,
        W.ffn_intermediate_weight2_static_quant: sp_id,
        W.ffn_intermediate_weight2_static_quant_reciprocal: sp_id,
        W.ffn_intermediate_weight3_static_quant: sp_id,
        W.ffn_intermediate_weight3_static_quant_reciprocal: sp_id,
        W.post_ffn_ln_static_quant: sp_id,
        W.post_ffn_ln_static_quant_reciprocal: sp_id,
    }
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(gemm_a8_weight_tp_strategy)
    return tp_strategy


class W8A8Fp8AtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = gemm_fp8_per_tensor_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class W8A8Fp8AttnAtomicWeight(AttnAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8MlaAttnAtomicWeight(MlaAttnAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8FfnAtomicWeight(FfnAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8MoeAtomicWeight(MoeAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w8a8_fp8_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> W8A8Fp8AtomicWeight:
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Fp8MlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Fp8AttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Fp8MoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Fp8FfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Fp8AtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")
