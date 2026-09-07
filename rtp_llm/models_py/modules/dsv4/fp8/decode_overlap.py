"""Default-off, target-only layer-local decode attention fork/join.

Two streams belong to one target model; layers join before reusing them.
This module proves host orchestration only. Native BLAS/PPU stream safety and
real pool-address disjointness still require the scheduled GPU gate.
"""
from __future__ import annotations

import dataclasses
import logging
import itertools
from collections.abc import Mapping

logger = logging.getLogger(__name__)
SWITCH = "DSV4_DECODE_ATTN_OVERLAP"


def target_class_identity(model) -> bool:
    """Unknown subclasses, MTP and DSpARK are excluded despite inheritance."""
    return type(model).__dict__.get("_dsv4_attention_overlap_target", False) is True


def eligible(*, target, graph_metadata, shape, compress_ratio, cp_active=False):
    return (
        target is True
        and graph_metadata is True
        and not cp_active
        and len(shape) == 3
        and int(shape[1]) in (1, 4)
        and 0 < int(shape[0]) * int(shape[1]) <= 64
        and int(compress_ratio) in (4, 128)
    )


def tensor_span(tensor):
    """Conservative occupied-byte interval, including stride holes/padding.

    data_ptr already incorporates storage_offset. Different storage wrappers
    pointing into the same physical allocation are therefore still detected.
    """
    shape = tuple(int(v) for v in tensor.shape)
    strides = tuple(int(v) for v in tensor.stride())
    if len(shape) != len(strides) or any(s < 0 for s in strides):
        raise RuntimeError("D1 cannot certify a negative/invalid tensor stride")
    if any(n == 0 for n in shape):
        return None
    start = int(tensor.data_ptr())
    itemsize = int(tensor.element_size())
    if start <= 0 or itemsize <= 0:
        raise RuntimeError("D1 requires real materialized storage")
    end = start + (1 + sum((n - 1) * s for n, s in zip(shape, strides))) * itemsize
    return str(tensor.device), start, end, shape, strides, itemsize


def tensor_regions(tensor):
    """Exact occupied runs, kept compact for block-strided pool views.

    Inner contiguous axes form one byte run; outer axes repeat it. Padding
    exposed in the raw pool tensor is occupied. Unexposed stride holes are not.
    Native accesses beyond the raw pool view remain a separate ABI check.
    """
    span = tensor_span(tensor)
    if span is None:
        return None
    device, start, end, shape, strides, itemsize = span
    width, outer = itemsize, []
    for size, stride in reversed(tuple(zip(shape, strides))):
        if size == 1 or stride == 0:
            continue
        byte_stride = stride * itemsize
        if not outer and byte_stride == width:
            width *= size
        else:
            outer.append((size, byte_stride))
    return {"device": device, "start": start, "end": end,
            "shape": shape, "strides": strides, "itemsize": itemsize,
            "run_bytes": width, "outer_axes": tuple(reversed(outer))}


def _simple_runs_overlap(a, b):
    axes_a, axes_b = a["outer_axes"], b["outer_axes"]
    if len(axes_a) > 1 or len(axes_b) > 1:
        return None
    count_a, stride_a = axes_a[0] if axes_a else (1, 0)
    count_b, stride_b = axes_b[0] if axes_b else (1, 0)
    if count_a == count_b == 1:
        return max(a["start"], b["start"]) < min(a["end"], b["end"])
    if stride_a and stride_b and stride_a != stride_b:
        return None
    stride = stride_a or stride_b
    delta = b["start"] - a["start"]
    # For some runs j,k: -width_b < delta+(k-j)*stride < width_a.
    low = max((-b["run_bytes"]-delta)//stride+1, -(count_a-1))
    high = min((a["run_bytes"]-delta-1)//stride, count_b-1)
    return low <= high


def _expanded_runs(spec):
    count = 1
    for size, _ in spec["outer_axes"]:
        count *= size
    if count > 1_000_000:
        raise RuntimeError("D1 cannot certify this irregular view in the bounded alias audit")
    runs = []
    for coordinates in itertools.product(*(range(n) for n, _ in spec["outer_axes"])):
        start = spec["start"] + sum(i*s for i, (_, s) in zip(coordinates, spec["outer_axes"]))
        runs.append((start, start+spec["run_bytes"]))
    return sorted(runs)


def regions_overlap(a, b):
    if a is None or b is None or a["device"] != b["device"]:
        return False
    if max(a["start"], b["start"]) >= min(a["end"], b["end"]):
        return False
    simple = _simple_runs_overlap(a, b)
    if simple is not None:
        return simple
    aa, bb = _expanded_runs(a), _expanded_runs(b)
    i = j = 0
    while i < len(aa) and j < len(bb):
        if max(aa[i][0], bb[j][0]) < min(aa[i][1], bb[j][1]):
            return True
        if aa[i][1] <= bb[j][0]:
            i += 1
        else:
            j += 1
    return False


def assert_disjoint_branches(writes, reads):
    """Reject actual cross-branch W/W and W/R byte overlap before any fork."""
    write_regions = {
        branch: [(name, tensor_regions(t)) for name, t in tensors]
        for branch, tensors in writes.items()
    }
    read_regions = {
        branch: [(name, tensor_regions(t)) for name, t in tensors]
        for branch, tensors in reads.items()
    }
    for branch, regions in write_regions.items():
        for other in writes:
            if other == branch:
                continue
            candidates = write_regions[other] + read_regions.get(other, [])
            for name, region in regions:
                for other_name, other_region in candidates:
                    if regions_overlap(region, other_region):
                        raise RuntimeError(
                            f"D1 cross-branch storage overlap: {branch}.{name} / "
                            f"{other}.{other_name}; no workspace rewrite in v1"
                        )
    return write_regions


def tensor_leaves(value):
    if hasattr(value, "record_stream") and hasattr(value, "is_cuda"):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_leaves(child)
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from tensor_leaves(getattr(value, field.name))


class DecodeAttentionOverlapContext:
    """Exactly two persistent side streams for a serialized target model."""

    def __init__(self, torch_module, device):
        self.torch = torch_module
        self.device = torch_module.device(device)
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("D1 streams must be created before graph capture")
        self.swa_stream = self.torch.cuda.Stream(device=self.device)
        self.compressor_stream = self.torch.cuda.Stream(device=self.device)
        self.active = False
        self.failed_join = False
        self.unjoined_tensors = None

    def new_layer(self):
        return DecodeAttentionLayerFork(self)


class DecodeAttentionLayerFork:
    def __init__(self, owner):
        self.owner = owner
        self.torch = owner.torch
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("D1 events must be created before graph capture")
        self.ready = self.torch.cuda.Event()
        self.swa_done = self.torch.cuda.Event()
        self.compressor_done = self.torch.cuda.Event()
        self.warmed_geometries = set()
        # CUDA Event objects allocate lazily. Materialize every event now.
        main = self.torch.cuda.current_stream(owner.device)
        self.ready.record(main)
        for stream, event in self._sides():
            with self.torch.cuda.stream(stream):
                stream.wait_event(self.ready)
                event.record(stream)
            main.wait_event(event)

    def _sides(self):
        return ((self.owner.swa_stream, self.swa_done),
                (self.owner.compressor_stream, self.compressor_done))

    def invalidate_rope(self):
        self.warmed_geometries.clear()

    def require_warm_capture(self, geometry):
        if (
            self.torch.cuda.is_current_stream_capturing()
            and tuple(geometry) not in self.warmed_geometries
        ):
            raise RuntimeError("D1 needs this graph geometry warmed before capture")

    def run(self, swa, compressor, indexer, *, swa_tensors, compressor_tensors):
        if self.owner.failed_join:
            raise RuntimeError("D1 context cannot be reused after a failed join")
        if self.owner.active:
            raise RuntimeError("D1 forbids cross-layer/model reentrant side-stream work")
        self.owner.active = True
        main = self.torch.cuda.current_stream(self.owner.device)
        primary_error = None
        # Keep every tensor owner alive until dependencies have joined.
        retained = (swa_tensors, compressor_tensors)
        try:
            self.ready.record(main)
            for stream, _ in self._sides():
                stream.wait_event(self.ready)
            with self.torch.cuda.stream(self.owner.swa_stream):
                for tensor in tensor_leaves(retained[0]):
                    if tensor.is_cuda:
                        tensor.record_stream(self.owner.swa_stream)
                swa()
            with self.torch.cuda.stream(self.owner.compressor_stream):
                for tensor in tensor_leaves(retained[1]):
                    if tensor.is_cuda:
                        tensor.record_stream(self.owner.compressor_stream)
                compressor()
            return indexer() if indexer is not None else None
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_errors = []
            # Attempt BOTH joins even if either branch or a join fails.
            for stream, event in self._sides():
                try:
                    with self.torch.cuda.stream(stream):
                        event.record(stream)
                    main.wait_event(event)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            self.owner.active = False
            if cleanup_errors:
                # A failed event dependency cannot certify completion. Keep
                # storage owners and fail closed on reuse until model teardown.
                self.owner.failed_join = True
                self.owner.unjoined_tensors = retained
                if primary_error is None:
                    raise cleanup_errors[0]
                logger.error("D1 join also failed while preserving original exception: %s",
                             cleanup_errors)


def create_decode_attention_overlap_context(*, is_target_model, device):
    # Importing/initializing a disabled or non-target model never creates streams.
    if is_target_model is not True:
        return None
    from rtp_llm.models_py.modules.dsv4.runtime_config import get_switch, parse_bool
    if not get_switch(SWITCH, False, parse_bool):
        return None
    import torch
    if str(device).split(":")[0] != "cuda":
        raise RuntimeError("D1 requires the CUDA/PPU target device")
    if torch.cuda.get_device_name(device) != "ZW-M890P":
        raise RuntimeError("D1 v1 is limited to the audited M890P provider")
    context = DecodeAttentionOverlapContext(torch, device)
    logger.info("DSV4_D1_TARGET_STREAMS switch=1 device=%s streams=2", device)
    return context
