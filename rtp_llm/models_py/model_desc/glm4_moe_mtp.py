import logging
from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


def apply_mtp_final_norm(norm, hidden_states, residual):
    hidden_states, _ = norm(hidden_states, residual)
    return hidden_states


def _dequantize_int8_to_bf16(weights_dict):
    """Dequantize INT8 per-channel weights to BF16 in-place.

    Called when the MTP draft model uses the BF16 (NoQuant) path instead of
    INT8 W8A8.  Converts INT8 kernel + FP32 scale -> BF16 for every quantized
    weight type (MoE expert GEMMs, attention QKV/O projections, and shared
    expert FFN layers).

    The conversion is:  bf16_weight = int8_weight.float() * scale.unsqueeze(-1).
    Dense 2D weights are then transposed from checkpoint [out, in] layout to
    RTP-LLM's internal [in, out] layout; 3D MoE weights keep their layout.

    Weights that are already BF16 (or absent) are left untouched.
    """
    # weight key -> companion scale key
    weight_scale_pairs = [
        (W.moe_w1, W.moe_s1),
        (W.moe_w2, W.moe_s2),
        (W.attn_qkv_w, W.attn_qkv_s),
        (W.attn_o_w, W.attn_o_s),
        (W.ffn_w1, W.ffn_s1),
        (W.ffn_w2, W.ffn_s2),
        (W.ffn_w3, W.ffn_s3),
        (W.ffn_w13, W.ffn_s13),
    ]

    for w_key, s_key in weight_scale_pairs:
        weight = weights_dict.get(w_key)
        if weight is None:
            continue
        scale = weights_dict.get(s_key)
        if weight.dtype == torch.int8:
            if scale is None:
                raise ValueError(
                    f"MTP draft INT8 weight {w_key} is missing scale {s_key}"
                )
            # Broadcast scale to match weight dimensions
            if scale.dim() < weight.dim():
                scale_expanded = scale.unsqueeze(-1)
            else:
                scale_expanded = scale
            dequant = (weight.float() * scale_expanded.to(torch.float32)).to(
                torch.bfloat16
            )
            if weight.dim() == 2:
                dequant = dequant.t().contiguous()
            weights_dict[w_key] = dequant
            logging.info(
                "MTP draft dequantize: %s INT8 -> BF16 (shape=%s)",
                w_key,
                tuple(weight.shape),
            )
        # LinearFactory treats a remaining scale as a quantized-weight signal.
        weights_dict.pop(s_key, None)

    # Safety check: eh_proj should already be BF16 (not in w8a8_weight_list)
    eh_proj = weights_dict.get(W.multi_tokens_predict_eh_proj)
    if eh_proj is not None and eh_proj.dtype == torch.int8:
        logging.warning(
            "MTP draft eh_proj weight is INT8 but no scale was loaded; "
            "this weight will be cast to BF16 directly, which may cause "
            "precision issues."
        )
        weights_dict[W.multi_tokens_predict_eh_proj] = eh_proj.to(torch.bfloat16)


class Glm4MoeMtpModel(GptModelBase):
    """GLM-4.7 MTP draft model backed by the appended checkpoint layer."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        layer_weights = weights.weights[0]

        # MTP draft model uses BF16 (NoQuant) path instead of INT8 W8A8.
        # The draft model has only 1 layer; BF16 eliminates 6 INT8 GEMM
        # quantization errors + 1 intermediate re-quantization that destroy
        # draft token quality and cause near-zero acceptance rate.
        # Main model (92 layers) keeps INT8 W8A8 for memory/throughput.
        _dequantize_int8_to_bf16(layer_weights)
        # The propose model owns a separate config, so switch it to NoQuant in
        # place. pybind ModelConfig cannot be copied or pickled.
        model_config.quant_config = None

        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.pre_fc_norm_embedding = RMSNorm(
            layer_weights[W.multi_tokens_predict_enorm], eps=model_config.layernorm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm(
            layer_weights[W.multi_tokens_predict_hnorm], eps=model_config.layernorm_eps
        )
        self.fc = LinearFactory.create_linear_from_weights(
            layer_weights,
            W.multi_tokens_predict_eh_proj,
            hw_kernel_config=py_hw_kernel_config,
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            layer_weights[W.multi_tokens_predict_final_ln_gamma],
            eps=model_config.layernorm_eps,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        inputs_embeds = self.embed_tokens(input_ids)
        last_hidden_states = inputs.input_hiddens
        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        # GLM-4.7 MTP fuses embed/hidden as eh_proj(cat([enorm(embed), hnorm(hidden)])),
        # matching SGLang glm4_moe_nextn. reverse_e_h_norm=False (GLM-4.7 order).
        if getattr(self.config, "reverse_e_h_norm", False):
            cat_hidden_states = torch.cat([h_norm, e_norm], -1)
        else:
            cat_hidden_states = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden_states)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual
        hidden_states = apply_mtp_final_norm(self.norm, hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Glm4MoeMtpModel",
    "apply_mtp_final_norm",
]
