"""DeepSeek-V4 MTP draft model.

Thin subclass of :class:`DeepSeekV4Model`.  The MTP draft is structurally
a single regular V4 ``Block`` plus four MTP-only fusion tensors
(``enorm`` / ``hnorm`` / ``e_proj`` / ``h_proj``).  We keep the Block
inside the inherited ``self.v4`` (its ``W.*`` keys are the same as the
main model — see ``DeepSeekV4MtpWeight._get_weight_info``), and host the
fusion modules at the model level so the only code unique to this class
is the ``e_proj(enorm(masked_embed)) + h_proj(hnorm(prev_hidden))`` step
that produces the layer-loop input.

The rest — initialize, prepare_fmha_impl, forward dispatch, mHC reduce,
final norm, ``_mtp_hidden_buffer`` accessor — falls through to
``DeepSeekV4Model`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    DeepSeekV4Model,
    Dsv4SharedRuntimeBufferStore,
)
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    DEFAULT_DSV4_CHUNK_TOKENS,
    dsv4_chunk_tokens_from_env,
)
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear
from rtp_llm.utils.model_weight import W

# ---------------------------------------------------------------------------
# Type-only import for the attention implementation (decode fmha metadata).
# Actual class is resolved lazily inside forward_draft_loop to avoid a hard
# import-time dep cycle.
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass  # forward reference only


class DeepSeekV4MtpModel(DeepSeekV4Model):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        Dsv4SharedRuntimeBufferStore.enable_mtp_hidden()
        # MTP overrides for V4Args. ``DeepSeekV4Mtp._create_config``
        # already sets ``num_layers=1`` and ``layer_compress_ratios=[0]``
        # on the ModelConfig; we additionally drop the hash-router count
        # so the lone draft layer always picks the noaux_tc path.
        self._v4_args.n_hash_layers = 0
        self._v4_args.compress_ratios = (
            [int(self._v4_args.compress_ratios[0])]
            if (self._v4_args.compress_ratios)
            else [0]
        )
        logging.info(
            "[DeepSeekV4MtpModel] V4Args: layers=%d hc_mult=%d compress_ratios=%s "
            "fp8_kv_cache=%s max_tokens_per_rank=%d",
            self._v4_args.n_layers,
            self._v4_args.hc_mult,
            list(self._v4_args.compress_ratios),
            self._v4_args.fp8_kv_cache,
            self._v4_args.max_tokens_per_rank,
        )

        # MTP-only fusion modules — populated in ``_load_extra_weights``.
        self.enorm: Optional[RMSNorm] = None
        self.hnorm: Optional[RMSNorm] = None
        self.e_proj = None
        self.h_proj = None
        self._mtp_fusion_chunk_logged = False

    def _resolve_mtp_last_hidden_token_capacity(self) -> Optional[int]:
        # Parent allocation is already gated by self._is_speculative.  Any
        # non-decode draft path, including PDFUSION, needs the CP prefill
        # per-request last-hidden handoff.
        if self._is_decode_role:
            return None
        return max(self._max_context_batch_size, self._max_generate_batch_size, 1)

    # ------------------------------------------------------------------
    # CUDA-graph gate — accept all cudagraph requests.  C++ MtpExecutor
    # creates two draft PyWrappedModel instances: ``draft_model_``
    # (q_len=1) and ``sp_prefill_draft_`` (q_len=gen+1).  The latter is
    # marked ``is_prefill=True`` even though it's functionally a
    # multi-token decode — the parent's default would reject it.
    # Both captures route through ``forward_decode`` + MTP's
    # ``_prepare_decode_hidden`` (``T = B*q_len``).  This requires
    # ``prepareCaptureInputs`` to size ``input_ids`` / ``input_hiddens``
    # and the output buffer to ``max_bs * num_tokens_per_bs`` (full
    # capacity), NOT the per-capture ``seq_len`` — handled in
    # ``cuda_graph_runner.cc`` / ``cuda_graph_prefill.cc`` for draft
    # prefill mode (num_tokens_per_bs != max_seq_len).
    # ------------------------------------------------------------------

    def _should_capture_cuda_graph(self, attn, is_target_verify: bool) -> bool:
        return True

    # ------------------------------------------------------------------
    # Weight loading hook (called by parent's _initialize_impl right
    # before ``del self.weight``).
    # ------------------------------------------------------------------

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        gw = weights.global_weights
        eps = float(self._v4_args.norm_eps)
        self.enorm = RMSNorm(gw[W.v4_mtp_enorm], eps)
        self.hnorm = RMSNorm(gw[W.v4_mtp_hnorm], eps)
        self.e_proj = _v4_fp8_linear(gw[W.v4_mtp_e_proj_w], gw[W.v4_mtp_e_proj_s])
        self.h_proj = _v4_fp8_linear(gw[W.v4_mtp_h_proj_w], gw[W.v4_mtp_h_proj_s])

    # ------------------------------------------------------------------
    # Hidden-state preparation overrides — splice the e/h fusion stage in
    # front of the inherited layer loop.
    # ------------------------------------------------------------------

    def _apply_proj(self, layer, x: torch.Tensor) -> torch.Tensor:
        """``_v4_fp8_linear``-built layers want 2D input; reshape N-D
        tensors round-trip so callers can keep their natural rank."""
        if x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def _mtp_fusion_chunk_tokens(self) -> int:
        return dsv4_chunk_tokens_from_env(
            "DSV4_MTP_FUSION_CHUNK_TOKENS",
            DEFAULT_DSV4_CHUNK_TOKENS,
            min_value=1,
        )

    def _build_fused_chunked(
        self,
        input_ids: torch.Tensor,
        pre_hc: torch.Tensor,
        positions: torch.Tensor,
        chunk_tokens: int,
    ) -> torch.Tensor:
        assert self.enorm is not None
        assert self.hnorm is not None
        assert self.e_proj is not None
        assert self.h_proj is not None

        T, hc, dim = pre_hc.shape
        # In-place reuse: pre_hc is either a fresh CP index_select tensor
        # (ContextParallelProcessorBase) or an allocBuf'd one-shot input
        # buffer (non-CP); neither has other live readers in this step.
        # The chunk loop reads pre_hc[start:end] into h_norm before writing
        # back to the same slice, so the per-chunk in-place copy is safe.
        # CUDA graph capture/replay is excluded by the caller guard.
        # Saves ~6 GiB peak at 1M ctx (T=125k, hc*dim=7*7168 bf16).
        fused = pre_hc
        if not self._mtp_fusion_chunk_logged:
            self._mtp_fusion_chunk_logged = True
            logging.info(
                "[DeepSeekV4MtpModel] chunked MTP fusion enabled: tokens=%d "
                "chunk_tokens=%d hc=%d dim=%d device=%s",
                T,
                chunk_tokens,
                hc,
                dim,
                pre_hc.device,
            )
        for start in range(0, T, chunk_tokens):
            end = min(start + chunk_tokens, T)
            input_ids_chunk = input_ids[start:end]
            positions_chunk = positions[start:end]
            # The checkpoint embedding is hidden-sharded under TP.  MTP's
            # fusion norms/projections consume the global hidden dimension,
            # so use the transformer's TP-aware embedding accessor.
            embed_chunk = self.v4._embed(input_ids_chunk)
            # TP all-gather may expose the global hidden tensor as a strided
            # view; the PPU RMSNorm kernel requires a contiguous input.
            embed_chunk = torch.where(
                positions_chunk.reshape(-1, 1) == 0,
                torch.zeros_like(embed_chunk),
                embed_chunk,
            ).contiguous()
            e_norm = self.enorm(embed_chunk)
            pre_hc_chunk = pre_hc[start:end]
            chunk_len = int(pre_hc_chunk.size(0))
            h_norm = self.hnorm(pre_hc_chunk.reshape(-1, dim)).view(chunk_len, hc, dim)
            fused_chunk = self._apply_proj(self.h_proj, h_norm)
            fused_chunk.add_(self._apply_proj(self.e_proj, e_norm).unsqueeze(1))
            fused[start:end].copy_(fused_chunk)
        return fused

    def _build_fused(
        self,
        input_ids: torch.Tensor,  # [T] int
        pre_hc: torch.Tensor,  # [T, hc, dim] bf16
        positions: torch.Tensor,  # [T] int (mask token at position 0)
    ) -> torch.Tensor:
        """``e_proj(enorm(masked_embed)) + h_proj(hnorm(prev_hidden))``.
        Returns ``[T, hc, dim]``."""
        assert self.enorm is not None
        assert self.hnorm is not None
        assert self.e_proj is not None
        assert self.h_proj is not None
        T, hc, dim = pre_hc.shape
        chunk_tokens = self._mtp_fusion_chunk_tokens()
        if T > chunk_tokens and not torch.cuda.is_current_stream_capturing():
            return self._build_fused_chunked(
                input_ids.reshape(-1), pre_hc, positions[:T], chunk_tokens
            )

        inputs_embeds = self.v4._embed(input_ids)  # [T, dim]
        # Suppress position-0 embedding (matches main-model "step 0 of a
        # brand-new request" behavior the official MTP impl relies on).
        # TP all-gather may expose the global hidden tensor as a strided view;
        # normalize only after materializing the kernel's layout contract.
        inputs_embeds = torch.where(
            positions.reshape(-1, 1) == 0,
            torch.zeros_like(inputs_embeds),
            inputs_embeds,
        ).contiguous()
        e_norm = self.enorm(inputs_embeds)  # [T, dim]
        h_norm = self.hnorm(pre_hc.reshape(-1, dim)).view(T, hc, dim)
        return self._apply_proj(self.h_proj, h_norm) + self._apply_proj(
            self.e_proj, e_norm
        ).unsqueeze(
            1
        )  # [T, hc, dim]

    def _pre_hc_from_inputs(self, inputs, T: int) -> torch.Tensor:
        pre_hc_in = inputs.input_hiddens
        if pre_hc_in is None or pre_hc_in.numel() == 0:
            raise RuntimeError(
                "DeepSeekV4MtpModel expected pre-hc hidden states in input_hiddens"
            )
        hc = int(self._v4_args.hc_mult)
        dim = int(self._v4_args.dim)
        pre_hc = pre_hc_in.reshape(-1, pre_hc_in.size(-1))
        if int(pre_hc.size(-1)) != hc * dim:
            raise RuntimeError(
                f"DeepSeekV4MtpModel expected hidden dim {hc * dim}, "
                f"got {pre_hc.size(-1)}"
            )
        # CP layout is handled before Python sees the tensors: C++
        # handleInputs splits input_hiddens with the same zigzag plan as
        # input_ids.  This method only trims CUDA graph capacity to real T.
        if pre_hc.size(0) < T:
            raise RuntimeError(
                f"DeepSeekV4MtpModel: input_hiddens has {pre_hc.size(0)} rows "
                f"but {T} tokens required"
            )
        return pre_hc[:T].view(T, hc, dim).to(device=self.v4.embed.weight.device)

    def _prepare_decode_hidden(
        self,
        input_ids: torch.Tensor,
        meta: Any,
    ) -> torch.Tensor:
        B = int(meta.batch_size)
        q_len = int(meta.q_len_per_req)
        T = B * q_len
        hc = int(self._v4_args.hc_mult)
        dim = int(self._v4_args.dim)
        pre_hc = self._pre_hc_from_inputs(self._cur_inputs, T)
        positions = meta.position_ids[:T]
        fused = self._build_fused(input_ids.reshape(-1), pre_hc, positions)
        return fused.view(B, q_len, hc, dim)

    def _prepare_prefill_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        T = int(input_ids.numel())
        pre_hc = self._pre_hc_from_inputs(self._cur_inputs, T)
        return self._build_fused(input_ids.reshape(-1), pre_hc, positions[:T])

    # ------------------------------------------------------------------
    # forward — delegate to parent, just stash ``inputs`` so the prepare
    # hooks can pull ``input_hiddens`` off it.
    # ------------------------------------------------------------------

    def forward(self, inputs, fmha_impl: Any = None):
        self._cur_inputs = inputs
        try:
            return super().forward(inputs, fmha_impl)
        finally:
            self._cur_inputs = None

    # ------------------------------------------------------------------
    # forward_draft_loop — N-1 step MTP draft loop unrolled for CUDA
    # graph capture.  Every operation is pure device-side: no .item(),
    # .cpu(), .synchronize(), print(), or logging inside the loop body.
    #
    # Called by CudaGraphRunner when is_draft_loop=True.  The method is
    # graph-captured once and replayed for every decode step.
    # ------------------------------------------------------------------

    def forward_draft_loop(
        self,
        inputs,       # PyModelInputs (capture or replay buffer)
        fmha_impl: Any = None,
        draft_loop_steps: int = 0,
    ):
        """Run *draft_loop_steps* iterations of
        ``forward → lm_head → argmax → update_input`` on a single CUDA
        stream, returning every step's draft tokens and probabilities
        packed into a single :class:`PyModelOutputs`.

        The output ``hidden_states`` layout is:
            ``[B * draft_loop_steps, vocab_size]``  (draft probs, fp32)
        and ``draft_tokens`` is:
            ``[B, draft_loop_steps]``  (int32 token ids)

        **CUDA-graph contract**: this entire method body must be
        host-sync-free.  The Python interpreter executes during *capture*
        only; during *replay* the recorded kernel stream is replayed
        without re-entering Python.
        """
        from rtp_llm.ops.compute_ops import PyModelOutputs
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            primary_attention_inputs,
        )

        if draft_loop_steps <= 0:
            # Fallback: single forward, same as regular path.
            self._cur_inputs = inputs
            try:
                return super().forward(inputs, fmha_impl)
            finally:
                self._cur_inputs = None

        device = self.v4.embed.weight.device
        head_w = self.v4.head_weight  # [vocab, dim]
        hc = int(self._v4_args.hc_mult)
        dim = int(self._v4_args.dim)

        # Resolve attention metadata from fmha_impl (graph path always
        # passes a pre-built impl whose .metadata is graph-safe).
        # Lazy import to avoid import-time dep cycle.
        from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImpl,
        )
        _graph_impl_types = (DSv4DecodeFmhaImpl,)
        try:
            from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
                DSv4DecodeFmhaImplFP8,
            )
            _graph_impl_types = (DSv4DecodeFmhaImpl, DSv4DecodeFmhaImplFP8)
        except ImportError:
            pass

        if isinstance(fmha_impl, _graph_impl_types):
            meta = fmha_impl.metadata
        else:
            raise RuntimeError(
                "forward_draft_loop requires a graph-path fmha_impl with "
                "pre-built metadata (DSv4DecodeFmhaImpl)."
            )

        B = meta.batch_size
        q_len = meta.q_len_per_req  # should be 1 for decode

        # Accumulators — pre-allocated on device.
        all_draft_token_ids = torch.empty(
            (B, draft_loop_steps), dtype=torch.int32, device=device
        )
        all_draft_probs = torch.empty(
            (B * draft_loop_steps, head_w.size(0)),
            dtype=torch.float32,
            device=device,
        )
        all_hidden_list: list = []  # [step] x [B*q_len, dim]

        # ---- Unrolled draft loop ----
        for step_i in range(draft_loop_steps):
            # 1. Full MTP forward (prepare_decode_hidden + layer loop).
            self._cur_inputs = inputs
            from rtp_llm.models_py.modules.dsv4.decode.forward import (
                forward_layers,
            )
            h = forward_layers(
                self.v4,
                self.kv_cache,
                inputs.input_ids,
                meta,
                prepare_hidden_fn=self._prepare_decode_hidden,
            )  # [B, q_len, dim]
            hidden = h.reshape(B * q_len, dim)  # [T, dim]

            # Capture per-step hidden for C++ maybeOverrideLastHidden.
            all_hidden_list.append(hidden)

            # 2. lm_head + sampling (pure device ops).
            logits = torch.mm(
                hidden.to(head_w.dtype), head_w.t()
            ).float()  # [T, vocab]

            probs = torch.softmax(logits, dim=-1)  # [T, vocab]
            draft_ids = probs.argmax(dim=-1).to(torch.int32)  # [T]

            # Store results for this step.
            all_draft_probs[step_i * B : (step_i + 1) * B] = probs
            all_draft_token_ids[:, step_i] = draft_ids.reshape(B)

            # 3. Update inputs for next iteration (all device-side).
            #    - combo_tokens = draft_ids
            #    - input_hiddens = current hidden (for MTP fusion)
            #    - sequence_lengths += 1
            if step_i < draft_loop_steps - 1:
                inputs.input_ids = draft_ids.reshape(B * q_len)
                # MTP fusion needs input_hiddens = [T, hc*dim] from the
                # MTP hidden buffer written during forward_layers.
                # The buffer is managed by the transformer and already
                # on device.
                mtp_buf = self.v4._mtp_hidden_buffer
                if mtp_buf is not None:
                    valid = min(B * q_len, mtp_buf.size(0))
                    inputs.input_hiddens = mtp_buf[:valid]

                # Advance sequence lengths on device (int32 add).
                if (
                    inputs.attention_inputs.sequence_lengths is not None
                    and inputs.attention_inputs.sequence_lengths.is_cuda
                ):
                    inputs.attention_inputs.sequence_lengths = (
                        inputs.attention_inputs.sequence_lengths + 1
                    ).to(torch.int32)

            self._cur_inputs = None

        # Pack outputs into PyModelOutputs.
        # hidden_states = all_draft_probs so C++ can extract logits.
        # draft_tokens = [B, draft_loop_steps] int32.
        # The C++ side will interpret these via the draft-loop protocol.
        all_hidden_cat = torch.cat(all_hidden_list, dim=0)  # [B*steps, dim]
        out = PyModelOutputs(all_hidden_cat)
        out.draft_tokens = all_draft_token_ids
        # Stash probs as a custom attribute for C++ retrieval.
        # CaptureMemoryHold will snapshot draft_tokens; probs travel via
        # the hidden_states channel reshaped by the caller.
        return out


__all__ = ["DeepSeekV4MtpModel"]
