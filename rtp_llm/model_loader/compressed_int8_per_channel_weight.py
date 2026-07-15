"""Loader for compressed-tensors INT8 per-channel symmetric W8A8 weights.

Used by GLM-4.7-INT8-W8A8 (llm-compressor v0.12.2):
  - weights: int8 symmetric per-channel (.weight + .weight_scale)
  - input_activations: dynamic per-token int8
  - ignore: lm_head, mlp.gate (router)

The compressed-tensors format stores weights in standard PyTorch [out, in]
layout. Unlike SmoothQuant (.qweight in [in, out] requiring transpose), no
transpose is needed here. We reuse the existing C++ INT8 per-token GEMM kernel
(same as smooth_quant) via `create_w8a8_int8_weight`.

On PPU, Dense layers keep INT8 weights and use deep_gemm's gemm_int8_int8_bf16_nt
for hardware-accelerated INT8 GEMM. On non-PPU devices, weights are dequantized
to BF16 at loading time: bf16_weight = (int8_weight.float() * scale).bfloat16()
"""

import logging
import re
from typing import Any, List, Optional

import torch

from rtp_llm.config.quant_config import (
    CompressedInt8PerChannelQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.w8a8_weight import (
    CompressedInt8PerChannelAtomicWeight,
    create_compressed_int8_per_channel_weight,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    concat_0,
    identity,
    stack_,
    stack_moe_w1,
)

# Checkpoint suffixes for compressed-tensors INT8 per-channel format.
# INT8 kernel is stored directly as standard .weight (same suffix as BF16 weights).
W_SUFFIX = ".weight"
QS_SUFFIX = ".weight_scale"  # per-channel scale (FP32)


def _matches_any(name: str, patterns) -> bool:
    """Check if a checkpoint key matches any ignore pattern."""
    for pat in patterns:
        if pat.startswith("re:"):
            if re.search(pat[3:], name):
                return True
        else:
            if pat in name:
                return True
    return False


def _ensure_1d(ts: List[torch.Tensor], allow_empty: bool = False) -> torch.Tensor:
    """Load per-channel scale ensuring 1D [N] output."""
    result = identity(ts, allow_empty)
    if result is not None and result.dim() == 2 and result.shape[-1] == 1:
        return result.squeeze(-1)
    return result


class CompressedInt8PerChannelWeight(CompositeWeight, QuantWeight):
    """Load compressed-tensors INT8 per-channel symmetric weights."""

    w8a8_weight_list = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.ffn_w1: W.ffn_s1,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w3: W.ffn_s3,
        W.ffn_w13: W.ffn_s13,
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, CompressedInt8PerChannelQuantConfig
        ):
            return False
        name = src_weight_info.name
        if name not in cls.w8a8_weight_list:
            return False
        if src_weight_info.weight_style in [
            WeightStyle.TRT_ENGINE,
            WeightStyle.RTP_SMOOTH_LLM_STYLE,
        ]:
            return False
        if hasattr(src_weight_info, "weights") and src_weight_info.weights:
            ckpt_name = src_weight_info.weights[0].name
            if _matches_any(ckpt_name, quant_config.ignore_patterns):
                return False
        return True

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        self._src_weight_name = src_weight_info.name

        kernel: WeightModule = None
        scale: WeightModule = None

        if src_weight_info.name == W.attn_qkv_w:
            kernel, scale = self._get_qkv_quant_weight(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            kernel, scale = self._get_attn_out_quant_weight(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]:
            kernel, scale = self._get_ffn_quant_weight(src_weight_info)
        elif src_weight_info.name == W.moe_w1:
            kernel, scale = self._get_moe_w1_quant_weight(src_weight_info)
        elif src_weight_info.name == W.moe_w2:
            kernel, scale = self._get_moe_w2_quant_weight(src_weight_info)
        else:
            raise ValueError(
                f"Unsupported weight name for CompressedInt8PerChannelWeight: "
                f"{src_weight_info.name}"
            )

        sub_weights = {kernel.name: kernel}
        if scale is not None:
            sub_weights[scale.name] = scale
        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale

    _MOE_WEIGHT_NAMES = {W.moe_w1, W.moe_w2}

    def _split(self, tensor, load_config):
        """Override to fix MoE scale TP split."""
        split_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensor = tensor.get(name)

            if (
                self._src_weight_name in self._MOE_WEIGHT_NAMES
                and self.scale is not None
                and name == self.scale.name
                and load_config.tp_size > 1
                and load_config.moe_pure_tp_mode
            ):
                tp = load_config.tp_size
                tp_rank = load_config.tp_rank

                raw = (
                    sub_tensor
                    if isinstance(sub_tensor, torch.Tensor)
                    else sub_tensor[name]
                )

                if self._src_weight_name == W.moe_w1:
                    E = raw.shape[0]
                    raw_3d = raw.reshape(E, 2, -1)
                    split_size = raw_3d.shape[2] // tp
                    split_3d = torch.split(raw_3d, split_size, dim=2)[tp_rank]
                    result = split_3d.reshape(E, -1).contiguous().clone()
                else:  # W.moe_w2
                    # w2 scale is per-output-channel [E, hidden]; output (hidden) is NOT
                    # split in TP — only input (inter) is split. Return unchanged.
                    result = raw.contiguous().clone()

                split_tensors[name] = {name: result}
            else:
                sub_tensors = sub_weight._split(sub_tensor, load_config)
                if isinstance(sub_weight, AtomicWeight) and isinstance(
                    sub_tensors, dict
                ):
                    split_tensors.update(sub_tensors)
                else:
                    split_tensors.update({name: sub_tensors})
        return split_tensors

    def _postprocess(
        self,
        tensor,
        device: str,
        load_config: LoadConfig,
    ):
        """Post-process weights after TP split and device placement."""
        processed = super()._postprocess(tensor, device, load_config)

        kernel_name = self.kernel.name
        scale_name = self.scale.name if self.scale else None

        if kernel_name not in processed:
            return processed
        if scale_name is None or scale_name not in processed:
            return processed

        kernel = processed[kernel_name]
        scale = processed[scale_name]

        # MoE path: keep INT8 + scale
        if self._src_weight_name in self._MOE_WEIGHT_NAMES:
            if scale.dim() == 3 and scale.shape[1] == 2:
                E = scale.shape[0]
                scale = scale.reshape(E, -1)
            elif scale.dim() == 3 and scale.shape[-1] == 1:
                scale = scale.squeeze(-1)
            elif scale.dim() == 2 and scale.shape[-1] == 1:
                scale = scale.squeeze(-1)
            return {kernel_name: kernel, scale_name: scale}

        # Dense path
        kernel = processed[kernel_name]
        scale = processed[scale_name]

        if scale.dim() == 2:
            scale = scale.squeeze()
        if scale.dim() == 0:
            scale = scale.unsqueeze(0)

        from rtp_llm.device.device_type import DeviceType, get_device_type

        if get_device_type() == DeviceType.Ppu:
            # Keep INT8 + scale for deep_gemm INT8 GEMM
            return {kernel_name: kernel, scale_name: scale}

        # Non-PPU: dequantize INT8 -> BF16, transpose
        if kernel.dtype == torch.int8:
            if kernel.dim() == 2:
                s = scale.unsqueeze(-1)
                dequant = (kernel.float() * s).to(torch.bfloat16)
            else:
                dequant = kernel.to(torch.bfloat16)
        else:
            dequant = kernel

        if dequant.dim() == 2:
            dequant = dequant.t().contiguous()

        return {kernel_name: dequant}

    def _get_qkv_quant_weight(self, src_weight_info):
        weights = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3

        if len(weights) == 3:
            qkv_w_list = [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + W_SUFFIX, identity)
                for w in weights
            ]
            qkv_s_list = [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QS_SUFFIX, _ensure_1d)
                for w in weights
            ]
            kernel = create_compressed_int8_per_channel_weight(
                src_weight_info,
                W.attn_qkv_w,
                qkv_w_list,
                concat_0,
                data_type=torch.int8,
                config=src_weight_info.config,
            )
            scale = create_compressed_int8_per_channel_weight(
                src_weight_info,
                W.attn_qkv_s,
                qkv_s_list,
                concat_0,
                data_type=torch.float32,
                config=src_weight_info.config,
            )
        else:
            qkv_name = weights[0].name[: -len(W_SUFFIX)]
            kernel = create_compressed_int8_per_channel_weight(
                src_weight_info,
                W.attn_qkv_w,
                [CkptWeightInfo(qkv_name + W_SUFFIX, identity)],
                identity,
                data_type=torch.int8,
                config=src_weight_info.config,
            )
            scale = create_compressed_int8_per_channel_weight(
                src_weight_info,
                W.attn_qkv_s,
                [CkptWeightInfo(qkv_name + QS_SUFFIX, _ensure_1d)],
                identity,
                data_type=torch.float32,
                config=src_weight_info.config,
            )
        return [kernel, scale]

    def _get_attn_out_quant_weight(self, src_weight_info):
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_compressed_int8_per_channel_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + W_SUFFIX, identity)],
            identity,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale = create_compressed_int8_per_channel_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + QS_SUFFIX, _ensure_1d)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_ffn_quant_weight(self, src_weight_info):
        weights = src_weight_info.weights
        ffn_w_name = src_weight_info.name

        if ffn_w_name == W.ffn_w13:
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]
            kernel = create_compressed_int8_per_channel_weight(
                src_weight_info,
                W.ffn_w13,
                [
                    CkptWeightInfo(w1_name + W_SUFFIX, identity),
                    CkptWeightInfo(w3_name + W_SUFFIX, identity),
                ],
                concat_0,
                data_type=torch.int8,
                config=src_weight_info.config,
            )
            scale = create_compressed_int8_per_channel_weight(
                src_weight_info,
                W.ffn_s13,
                [
                    CkptWeightInfo(w1_name + QS_SUFFIX, _ensure_1d),
                    CkptWeightInfo(w3_name + QS_SUFFIX, _ensure_1d),
                ],
                concat_0,
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale]

        w_name = weights[0].name[: -len(W_SUFFIX)]
        if ffn_w_name == W.ffn_w1:
            w, s = W.ffn_w1, W.ffn_s1
        elif ffn_w_name == W.ffn_w3:
            w, s = W.ffn_w3, W.ffn_s3
        else:
            w, s = W.ffn_w2, W.ffn_s2

        kernel = create_compressed_int8_per_channel_weight(
            src_weight_info,
            w,
            [CkptWeightInfo(w_name + W_SUFFIX, identity)],
            identity,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale = create_compressed_int8_per_channel_weight(
            src_weight_info,
            s,
            [CkptWeightInfo(w_name + QS_SUFFIX, _ensure_1d)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_moe_w1_quant_weight(self, src_weight_info):
        weights = src_weight_info.weights
        kernel = create_compressed_int8_per_channel_weight(
            src_weight_info,
            W.moe_w1,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + W_SUFFIX, identity)
                for w in weights
            ],
            stack_moe_w1,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale = create_compressed_int8_per_channel_weight(
            src_weight_info,
            W.moe_s1,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QS_SUFFIX, _ensure_1d)
                for w in weights
            ],
            stack_moe_w1,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_moe_w2_quant_weight(self, src_weight_info):
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_compressed_int8_per_channel_weight(
            src_weight_info,
            W.moe_w2,
            [CkptWeightInfo(w_name + W_SUFFIX, identity)],
            stack_,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale = create_compressed_int8_per_channel_weight(
            src_weight_info,
            W.moe_s2,
            [CkptWeightInfo(w_name + QS_SUFFIX, _ensure_1d)],
            stack_,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]
