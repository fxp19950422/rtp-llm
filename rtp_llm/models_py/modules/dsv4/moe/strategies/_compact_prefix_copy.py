"""CUDA-runtime compact-to-strided prefix copy for the PPU DeepEP path.

The producer owns a compact contiguous ``[E, C, D]`` BF16 tensor while
DeepEP combine owns a contiguous registered ``[E, LL, D]`` buffer.  A single
``cudaMemcpy2DAsync`` copies one compact expert slab per row with different
source and destination pitches, avoiding TensorIterator's elementwise
strided-copy kernel.  The operation is submitted to PyTorch's current stream
and performs no allocation or synchronization, so it can be captured in the
existing Decode CUDA graph.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Optional

import torch


_DEFAULT_CUDART = (
    "/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcudart.so"
)
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
_LOAD_LOCK = threading.Lock()
_CUDART: Optional[ctypes.CDLL] = None
_CUDA_MEMCPY_2D_ASYNC = None
_CUDA_GET_ERROR_STRING = None


def prepare_compact_prefix_copy() -> None:
    """Resolve and type the CUDA Runtime entry points before graph capture."""
    global _CUDART, _CUDA_MEMCPY_2D_ASYNC, _CUDA_GET_ERROR_STRING
    if _CUDA_MEMCPY_2D_ASYNC is not None:
        return
    with _LOAD_LOCK:
        if _CUDA_MEMCPY_2D_ASYNC is not None:
            return
        path = os.environ.get("DSV4_PPU_CUDART_PATH", _DEFAULT_CUDART)
        try:
            cudart = ctypes.CDLL(path)
        except OSError as exc:
            raise RuntimeError(f"failed to load CUDA Runtime from {path}: {exc}") from exc
        memcpy_2d_async = cudart.cudaMemcpy2DAsync
        memcpy_2d_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        memcpy_2d_async.restype = ctypes.c_int
        get_error_string = cudart.cudaGetErrorString
        get_error_string.argtypes = [ctypes.c_int]
        get_error_string.restype = ctypes.c_char_p
        # Publish the readiness sentinel last.  Readers intentionally use
        # ``_CUDA_MEMCPY_2D_ASYNC is not None`` as the lock-free fast path, so
        # observing it must imply that every other entry point and the owning
        # CDLL reference are already initialized.
        _CUDA_GET_ERROR_STRING = get_error_string
        _CUDART = cudart
        _CUDA_MEMCPY_2D_ASYNC = memcpy_2d_async


def compact_to_strided_prefix(
    compact: torch.Tensor,
    padded: torch.Tensor,
) -> None:
    """Copy ``compact[e, :, :]`` to ``padded[e, :C, :]`` asynchronously.

    This intentionally supports only the production BF16 rank-3 contract and
    fails closed for every other layout.  ``padded`` remains the tensor passed
    to DeepEP with ``zero_copy=True``.
    """
    if not isinstance(compact, torch.Tensor) or not isinstance(padded, torch.Tensor):
        raise TypeError("compact_to_strided_prefix requires two torch.Tensor values")
    if compact.dim() != 3 or padded.dim() != 3:
        raise ValueError("compact_to_strided_prefix requires rank-3 tensors")
    if compact.dtype != torch.bfloat16 or padded.dtype != torch.bfloat16:
        raise TypeError("compact_to_strided_prefix requires BF16 tensors")
    if not compact.is_cuda or not padded.is_cuda:
        raise ValueError("compact_to_strided_prefix requires CUDA/PPU tensors")
    if compact.device != padded.device:
        raise ValueError("compact and padded tensors must be on the same device")
    if not compact.is_contiguous() or not padded.is_contiguous():
        raise ValueError("compact and padded tensors must both be contiguous")
    if compact.requires_grad or padded.requires_grad:
        raise ValueError("compact_to_strided_prefix is inference-only")

    experts, compact_rows, width = compact.shape
    padded_experts, padded_rows, padded_width = padded.shape
    if experts <= 0 or compact_rows <= 0 or width <= 0:
        raise ValueError("compact tensor dimensions must all be positive")
    if experts != padded_experts or width != padded_width:
        raise ValueError("compact and padded expert/width dimensions must match")
    if compact_rows > padded_rows:
        raise ValueError("compact rows must not exceed padded rows")
    if torch.cuda.current_device() != compact.get_device():
        raise RuntimeError("compact tensor device must be the current CUDA device")

    element_bytes = compact.element_size()
    width_bytes = compact_rows * width * element_bytes
    dst_pitch = padded_rows * width * element_bytes
    compact_bytes = compact.numel() * element_bytes
    padded_bytes = padded.numel() * element_bytes
    compact_start = compact.data_ptr()
    padded_start = padded.data_ptr()
    if not (
        compact_start + compact_bytes <= padded_start
        or padded_start + padded_bytes <= compact_start
    ):
        raise ValueError("compact and padded storage must not overlap")

    prepare_compact_prefix_copy()
    assert _CUDA_MEMCPY_2D_ASYNC is not None
    assert _CUDA_GET_ERROR_STRING is not None
    stream = torch.cuda.current_stream(compact.device).cuda_stream
    status = _CUDA_MEMCPY_2D_ASYNC(
        ctypes.c_void_p(padded_start),
        ctypes.c_size_t(dst_pitch),
        ctypes.c_void_p(compact_start),
        ctypes.c_size_t(width_bytes),
        ctypes.c_size_t(width_bytes),
        ctypes.c_size_t(experts),
        ctypes.c_int(_CUDA_MEMCPY_DEVICE_TO_DEVICE),
        ctypes.c_void_p(stream),
    )
    if status != 0:
        raw_message = _CUDA_GET_ERROR_STRING(status)
        message = (
            raw_message.decode("utf-8", "replace")
            if raw_message is not None
            else "unknown CUDA Runtime error"
        )
        raise RuntimeError(f"cudaMemcpy2DAsync failed ({status}): {message}")
