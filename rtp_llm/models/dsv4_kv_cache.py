"""DeepSeek-V4 KV cache spec descriptors.

DSv4 does not have a single homogeneous KV cache.  Each layer routes to a set
of independent pools selected by its ``compress_ratios`` entry:

  ==========  ====================================================
  ratio       pools
  ==========  ====================================================
  4  (CSA)    ``csa_kv``, ``indexer_kv``, ``indexer_state``,
              ``csa_state``, ``swa_kv``
  128 (HCA)   ``hca_kv``, ``hca_state``, ``swa_kv``
  0  (SWA)    ``swa_kv``
  ==========  ====================================================

C++ turns the resulting per-layer desc lists into the cache topology through
``HybridPoolConfigCreator`` (``validateHybridPoolDescs`` ->
``buildLayerSpecsFromDescs`` -> ``populateGroupsFromLayerSpecs`` ->
``setupIndependentPoolSizes``), which is only reached when
``hybrid_attention_config.enable_independent_kv_cache_pools`` is set.

This module is the production twin of
``rtp_llm/cpp/cache/test/CacheConfigTestUtils.h`` (``makeDsv4Desc`` /
``setDsv4KvCacheSpecs`` / ``setDsv4ExplicitPoolBlocks``).  Keep the two in
lock step: the C++ header is the reference implementation that the cache unit
tests assert against.

pybind note: ``KVCacheSpecDesc``'s policy sub-structs are ``std::optional<...>``
bound with ``def_readwrite`` and ``pybind11/stl.h``, so reading ``desc.cp``
yields a *copy* rather than a reference.  Every helper below therefore builds a
sub-struct locally, fills it, and assigns it back onto the desc.  The same is
true of ``ModelConfig.kv_cache_spec_descs`` (a ``std::vector<std::vector<...>>``):
mutate the Python list first, then assign it once.
"""

from enum import Enum
from numbers import Integral
from typing import Optional, Sequence

from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheCpPolicyDesc,
    CacheEvictPolicy,
    CacheMemoryPlacement,
    CacheMemoryPolicyDesc,
    CacheReusePolicyDesc,
    CacheTailPolicyDesc,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    DataType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)

# Byte-exact FP8 entry sizes of the DSv4 MLA/indexer kernels.  The pools are
# declared as UINT8 so ``entry_elems`` is a byte count.
DSV4_FP8_KV_ENTRY_BYTES = 584
DSV4_FP8_INDEXER_ENTRY_BYTES = 132
# The FP4 indexer stores 128 E2M1 values as 64 packed bytes plus four raw
# UE8M0 scale bytes per compressed token.  The scale bytes are part of the
# same opaque entry and do not form a sidecar cache.
DSV4_FP4_INDEXER_ENTRY_BYTES = 68
# FlashMLA requires the FP8 KV block stride to be a multiple of 576 bytes.
DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576
# Sliding window length in entries; doubles as the HCA compression unit and as
# the minimum entry count that may pay for block-stride alignment padding.
DSV4_SWA_WINDOW_ENTRIES = 128
# HCA_STATE is a small fixed ring; it is sized explicitly instead of tracking
# the paged budget.
DSV4_HCA_STATE_POOL_BLOCKS = 256
# DSv4's physical KV block is 256 tokens (the CLI default is 64).
DSV4_TOKENS_PER_BLOCK = 256

# Target-verify writes the whole speculative batch before the compressor reads
# any historical state.  The speculative geometry therefore uses a double
# ring for both state-window classes.  C4 is the tighter bound: with 16 entries
# the current write-all/read-later ordering is collision-free through q_len=9.
DSV4_SPECULATIVE_C4_STATE_RING_ENTRIES = 16
DSV4_SPECULATIVE_C128_STATE_RING_ENTRIES = 256
DSV4_SPECULATIVE_C4_MAX_QUERY_LEN = 9

CSA_LAYER_COMPRESS_RATIO = 4
HCA_LAYER_COMPRESS_RATIO = 128

CSA_KV_TAG = "csa_kv"
HCA_KV_TAG = "hca_kv"
INDEXER_KV_TAG = "indexer_kv"
INDEXER_STATE_TAG = "indexer_state"
CSA_STATE_TAG = "csa_state"
HCA_STATE_TAG = "hca_state"
SWA_KV_TAG = "swa_kv"

_COMPRESSED_KV_KIND = "compressed_kv"
_FIXED_STATE_KIND = "fixed_state"
_SLIDING_WINDOW_KV_KIND = "sliding_window_kv"

# The four "fixed" pools that ``--dsv4_fixed_pool_use_memory`` may move to
# pinned host memory.
DSV4_FIXED_POOL_TAGS: tuple[str, ...] = (
    INDEXER_STATE_TAG,
    CSA_STATE_TAG,
    HCA_STATE_TAG,
    SWA_KV_TAG,
)


class Dsv4StateRingMode(Enum):
    """State-ring geometry selected explicitly by the runtime caller."""

    NORMAL = "normal"
    SPECULATIVE_TARGET_VERIFY = "speculative_target_verify"


class Dsv4IndexerCacheMode(Enum):
    """Indexer-cache representation selected explicitly by the caller."""

    FOLLOW_KV = "follow_kv"
    FP8 = "fp8"
    FP4 = "fp4"


def validate_dsv4_speculative_target_query_len(query_len: int) -> int:
    """Validate target-verify q_len for the current C4 state-ring ordering.

    State writers currently write all target-verify tokens before compressors
    read their historical windows.  A 16-entry C4 ring is safe for q_len 1..9;
    larger batches need a different geometry or write/read ordering.  Keep this
    helper pure so the T20/T30 runtime wiring can reject unsupported requests
    without environment-dependent mode inference.
    """
    if isinstance(query_len, bool) or not isinstance(query_len, Integral):
        raise TypeError(
            "DeepSeek-V4 speculative target query_len must be an integer: "
            f"value={query_len!r}, type={type(query_len).__name__}"
        )
    query_len = int(query_len)
    if not 1 <= query_len <= DSV4_SPECULATIVE_C4_MAX_QUERY_LEN:
        raise ValueError(
            "DeepSeek-V4 speculative target query_len is outside the safe "
            "C4 state-ring range: "
            f"query_len={query_len}, supported=1.."
            f"{DSV4_SPECULATIVE_C4_MAX_QUERY_LEN}"
        )
    return query_len


def _make_dsv4_desc(
    tag: str,
    kind: str,
    entry_elems: int,
    dtype: DataType,
    compression_ratio: int = 1,
) -> KVCacheSpecDesc:
    """Mirror of ``rtp_llm::test::makeDsv4Desc``."""
    desc = KVCacheSpecDesc()
    desc.tag = tag
    desc.dtype = dtype
    desc.entry_elems = entry_elems
    desc.entry_dtype = dtype

    if kind == _COMPRESSED_KV_KIND:
        desc.cache_type = KVCacheSpecType.OPAQUE_KV
        desc.is_state_cache = False
        desc.entry_count_mode = OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED
        desc.compression_ratio = compression_ratio
        if desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES:
            desc.block_stride_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES
        # Compressed pools deliberately carry no ``cp`` policy and leave
        # ``block_stride_alignment_min_entries`` at 0.  Their entry count is
        # kernel_block/ratio (64 entries for CSA, 2 for HCA); raising the
        # minimum to 128 would silently disable the 576-byte alignment above.
        return desc

    desc.cache_type = KVCacheSpecType.OPAQUE_STATE
    desc.is_state_cache = True
    desc.entry_count_mode = OpaqueBlockEntryCountMode.STATE_RING

    reuse = CacheReusePolicyDesc()
    reuse.evict_policy = CacheEvictPolicy.INDEPENDENT
    cp = CacheCpPolicyDesc()

    if tag in (INDEXER_STATE_TAG, CSA_STATE_TAG):
        desc.compression_ratio = CSA_LAYER_COMPRESS_RATIO
        desc.state_ring_overlap = 1
        cp.align_payload = True
        cp.prefill_slice_layout = CpPrefillSliceLayout.PAYLOAD
        cp.slice = CpBlockSliceMode.PAYLOAD_BYTES
    elif tag == HCA_STATE_TAG:
        desc.compression_ratio = HCA_LAYER_COMPRESS_RATIO
        cp.align_payload = True
        cp.prefill_slice_layout = CpPrefillSliceLayout.PAYLOAD
        cp.slice = CpBlockSliceMode.PAYLOAD_BYTES
        capacity = CacheCapacityPolicyDesc()
        capacity.explicit_block_num = DSV4_HCA_STATE_POOL_BLOCKS
        capacity.charge_to_paged_budget = True
        desc.capacity = capacity
        reuse.enable_prefix_reuse = False
        tail = CacheTailPolicyDesc()
        tail.active_tail_blocks = 1
        tail.validate_tail_blocks = False
        desc.tail = tail
    elif tag == SWA_KV_TAG:
        desc.compression_ratio = DSV4_SWA_WINDOW_ENTRIES
        cp.align_payload = True
        cp.prefill_slice_layout = CpPrefillSliceLayout.BLOCK_STRIDE
        cp.slice = CpBlockSliceMode.EQUAL_BYTES
        if desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES:
            desc.block_stride_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES

    desc.state_ring_include_gen_num_per_cycle = True
    cp.scale_seq_size = True
    desc.block_stride_alignment_min_entries = DSV4_SWA_WINDOW_ENTRIES
    desc.reuse = reuse
    desc.cp = cp
    return desc


def _is_host_resident(desc: KVCacheSpecDesc) -> bool:
    memory = desc.memory
    return (
        memory is not None
        and memory.placement is not None
        and memory.placement != CacheMemoryPlacement.DEVICE
    )


def _use_host_pinned_memory(desc: KVCacheSpecDesc) -> None:
    """Move a pool to pinned host memory.

    ``memory.placement`` and ``capacity.charge_to_paged_budget`` must move
    together: a host-resident pool that still charges the paged HBM budget
    trips ``checkGroupResidencyBudget`` in ``rtp_llm/cpp/cache/CacheConfig.h``.
    """
    memory = CacheMemoryPolicyDesc()
    memory.placement = CacheMemoryPlacement.HOST_PINNED
    desc.memory = memory

    capacity = desc.capacity
    if capacity is None:
        capacity = CacheCapacityPolicyDesc()
    capacity.charge_to_paged_budget = False
    desc.capacity = capacity


def _apply_dsv4_state_ring_mode(
    state_ring_mode: Dsv4StateRingMode,
    indexer_state: KVCacheSpecDesc,
    csa_state: KVCacheSpecDesc,
    hca_state: KVCacheSpecDesc,
    swa_kv: KVCacheSpecDesc,
) -> None:
    if state_ring_mode is Dsv4StateRingMode.NORMAL:
        return

    for desc, entry_count in (
        (indexer_state, DSV4_SPECULATIVE_C4_STATE_RING_ENTRIES),
        (csa_state, DSV4_SPECULATIVE_C4_STATE_RING_ENTRIES),
        (hca_state, DSV4_SPECULATIVE_C128_STATE_RING_ENTRIES),
        (swa_kv, DSV4_SPECULATIVE_C128_STATE_RING_ENTRIES),
    ):
        desc.entry_count_mode = OpaqueBlockEntryCountMode.EXPLICIT
        desc.explicit_entry_count = entry_count
        desc.state_ring_include_gen_num_per_cycle = False


def apply_dsv4_explicit_pool_blocks(
    layer_descs: Sequence[Sequence[KVCacheSpecDesc]],
    tag: str,
    block_num: int,
) -> None:
    """Pin one pool's block count, in place.

    Python twin of ``rtp_llm::test::setDsv4ExplicitPoolBlocks``.  Must run
    before ``layer_descs`` is assigned to ``model_config.kv_cache_spec_descs``:
    the pybind getter returns a copy, so mutating a read-back list is a no-op.

    Unlike the C++ test helper this never re-enables ``charge_to_paged_budget``
    on a host-resident pool, so the residency/budget pair stays consistent.
    """
    for descs in layer_descs:
        for desc in descs:
            if desc.tag != tag:
                continue
            capacity = desc.capacity
            if capacity is None:
                capacity = CacheCapacityPolicyDesc()
            capacity.explicit_block_num = block_num
            capacity.charge_to_paged_budget = block_num > 0 and not _is_host_resident(
                desc
            )
            desc.capacity = capacity


def build_dsv4_kv_cache_spec_descs(
    layer_num: int,
    layer_compress_ratios: Sequence[int],
    fp8_kv: bool,
    head_dim: int,
    indexer_head_dim: int,
    fixed_pool_use_host_memory: bool = False,
    state_ring_mode: Dsv4StateRingMode = Dsv4StateRingMode.NORMAL,
    indexer_cache_mode: Dsv4IndexerCacheMode = Dsv4IndexerCacheMode.FOLLOW_KV,
) -> list[list[KVCacheSpecDesc]]:
    """Build the per-layer DSv4 desc lists.

    Python twin of ``rtp_llm::test::setDsv4KvCacheSpecs``.  Layers past the end
    of ``layer_compress_ratios`` are treated as ratio 0 (SWA only).  Checkpoint
    configs may append MTP/draft ratios after the target-model layers; those
    trailing entries are accepted only when they are ratio 0 and are not used
    to build target-layer cache descriptors.

    One desc object per tag is shared across the layers that use it, exactly as
    the C++ helper does.  This is safe: ``kv_cache_spec_descs`` is a
    ``std::vector<std::vector<KVCacheSpecDesc>>`` bound through
    ``pybind11/stl.h``, so assignment deep-copies every desc into C++.

    Args:
        layer_num: target-model layer count (``model_config.num_layers``).
        layer_compress_ratios: ``attn_config.layer_compress_ratios``.
        fp8_kv: ``attn_config.kv_cache_dtype == KvCacheDataType.FP8``.
        head_dim: ``attn_config.size_per_head``.
        indexer_head_dim: ``attn_config.indexer_head_dim``.
        fixed_pool_use_host_memory: place the four fixed pools
            (``indexer_state`` / ``csa_state`` / ``hca_state`` / ``swa_kv``) in
            pinned host memory and take them off the paged HBM budget.
        state_ring_mode: explicit normal or speculative target-verify geometry.
            The default preserves the existing dynamic state-ring ABI.
        indexer_cache_mode: ``FOLLOW_KV`` preserves the legacy public default
            (FP8=132 bytes; non-FP8=``indexer_head_dim * 2``).  New runtime
            wiring must select explicit FP8 or FP4 geometry, which is
            independent of ``fp8_kv``.  No platform, environment, or
            checkpoint inference is performed here.
    """
    if not isinstance(state_ring_mode, Dsv4StateRingMode):
        raise TypeError(
            "DeepSeek-V4 state_ring_mode must be a Dsv4StateRingMode: "
            f"value={state_ring_mode!r}, type={type(state_ring_mode).__name__}"
        )
    if not isinstance(indexer_cache_mode, Dsv4IndexerCacheMode):
        raise TypeError(
            "DeepSeek-V4 indexer_cache_mode must be a Dsv4IndexerCacheMode: "
            f"value={indexer_cache_mode!r}, "
            f"type={type(indexer_cache_mode).__name__}"
        )
    if layer_num <= 0:
        raise ValueError(f"dsv4 kv cache descs require layer_num > 0, got {layer_num}")

    kv_entry_elems = DSV4_FP8_KV_ENTRY_BYTES if fp8_kv else head_dim * 2
    if indexer_cache_mode is Dsv4IndexerCacheMode.FOLLOW_KV:
        indexer_entry_elems = (
            DSV4_FP8_INDEXER_ENTRY_BYTES if fp8_kv else indexer_head_dim * 2
        )
    else:
        indexer_entry_elems = {
            Dsv4IndexerCacheMode.FP8: DSV4_FP8_INDEXER_ENTRY_BYTES,
            Dsv4IndexerCacheMode.FP4: DSV4_FP4_INDEXER_ENTRY_BYTES,
        }[indexer_cache_mode]

    csa_kv = _make_dsv4_desc(
        CSA_KV_TAG,
        _COMPRESSED_KV_KIND,
        kv_entry_elems,
        DataType.TYPE_UINT8,
        CSA_LAYER_COMPRESS_RATIO,
    )
    hca_kv = _make_dsv4_desc(
        HCA_KV_TAG,
        _COMPRESSED_KV_KIND,
        kv_entry_elems,
        DataType.TYPE_UINT8,
        HCA_LAYER_COMPRESS_RATIO,
    )
    indexer_kv = _make_dsv4_desc(
        INDEXER_KV_TAG,
        _COMPRESSED_KV_KIND,
        indexer_entry_elems,
        DataType.TYPE_UINT8,
        CSA_LAYER_COMPRESS_RATIO,
    )
    indexer_state = _make_dsv4_desc(
        INDEXER_STATE_TAG,
        _FIXED_STATE_KIND,
        4 * indexer_head_dim,
        DataType.TYPE_FP32,
    )
    csa_state = _make_dsv4_desc(
        CSA_STATE_TAG,
        _FIXED_STATE_KIND,
        4 * head_dim,
        DataType.TYPE_FP32,
    )
    hca_state = _make_dsv4_desc(
        HCA_STATE_TAG,
        _FIXED_STATE_KIND,
        2 * head_dim,
        DataType.TYPE_FP32,
    )
    swa_kv = _make_dsv4_desc(
        SWA_KV_TAG,
        _SLIDING_WINDOW_KV_KIND,
        kv_entry_elems,
        DataType.TYPE_UINT8,
    )

    _apply_dsv4_state_ring_mode(
        state_ring_mode,
        indexer_state,
        csa_state,
        hca_state,
        swa_kv,
    )

    if fixed_pool_use_host_memory:
        for desc in (indexer_state, csa_state, hca_state, swa_kv):
            _use_host_pinned_memory(desc)

    ratios: list[int] = []
    for ratio_id, ratio in enumerate(layer_compress_ratios):
        if isinstance(ratio, bool) or not isinstance(ratio, Integral):
            raise TypeError(
                "DeepSeek-V4 compression ratios must be integers: "
                f"index={ratio_id}, value={ratio!r}, type={type(ratio).__name__}"
            )
        ratios.append(int(ratio))
    supported_ratios = {
        0,
        CSA_LAYER_COMPRESS_RATIO,
        HCA_LAYER_COMPRESS_RATIO,
    }
    for layer_id, ratio in enumerate(ratios[:layer_num]):
        if ratio not in supported_ratios:
            raise ValueError(
                "unsupported DeepSeek-V4 main-layer compression ratio: "
                f"layer={layer_id}, ratio={ratio}, "
                f"supported={sorted(supported_ratios)}"
            )
    for trailing_id, ratio in enumerate(ratios[layer_num:], start=layer_num):
        if ratio != 0:
            raise ValueError(
                "DeepSeek-V4 trailing MTP/draft compression ratios must be 0: "
                f"index={trailing_id}, ratio={ratio}"
            )

    layer_descs: list[list[KVCacheSpecDesc]] = []
    for layer_id in range(layer_num):
        ratio = ratios[layer_id] if layer_id < len(ratios) else 0
        if ratio == CSA_LAYER_COMPRESS_RATIO:
            layer_descs.append([csa_kv, indexer_kv, indexer_state, csa_state, swa_kv])
        elif ratio == HCA_LAYER_COMPRESS_RATIO:
            layer_descs.append([hca_kv, hca_state, swa_kv])
        else:
            layer_descs.append([swa_kv])
    return layer_descs


def resolve_dsv4_tokens_per_block(
    tokens_per_block: int,
    framework_default: int = 64,
) -> Optional[int]:
    """Return the physical block size DSv4 should run with, or None to keep.

    ``HybridPoolConfigCreator::createHybridAttentionPoolConfig`` takes
    ``kv_cache_config.seq_size_per_block`` only when it differs from the
    framework default of 64, otherwise it falls back to
    ``attn_config.tokens_per_block``; and ``createBasicConfig`` (the warm-up
    path) zeroes ``seq_size_per_block`` entirely, so ``attn_config`` is the only
    channel that reaches both paths.  Promote the default to 256, but leave an
    explicit ``--seq_size_per_block`` alone so the two paths stay in agreement.
    """
    if tokens_per_block == framework_default:
        return DSV4_TOKENS_PER_BLOCK
    return None
