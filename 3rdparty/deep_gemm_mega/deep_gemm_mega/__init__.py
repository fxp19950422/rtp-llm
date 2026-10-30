"""PPU MXFP4 MegaMoE, isolated from the official deep_gemm package."""

import os
from pathlib import Path

import torch  # Load the shared libraries required by the C++ extension first.

from . import _C

_root = Path(__file__).resolve().parent
_sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME") or "/usr/local/PPU_SDK"
os.environ.setdefault(
    "RTP_PPU_MEGA_LUT_PATH", str(_root / "configs" / "dsv4_flash_bm64")
)
_C.init(str(_root), _sdk)

from .mega import (
    SymmBuffer,
    fp4_fp4_mega_moe,
    get_symm_buffer_for_mega_moe,
    mega_moe_pre_dispatch,
    transform_weights_for_mega_moe,
)

__all__ = [
    "SymmBuffer",
    "fp4_fp4_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
    "transform_weights_for_mega_moe",
]
