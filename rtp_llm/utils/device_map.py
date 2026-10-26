import os

from rtp_llm.utils.gemm_utils.device_map import register_device

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

if "ACEXT_GEMM_CONFIG_DIR" not in os.environ:
    _MOE_LUT_PATH = os.path.join(CUR_PATH, "moe_luts")
    os.environ["ACEXT_GEMM_CONFIG_DIR"] = _MOE_LUT_PATH

_LUT_PATH = os.path.join(CUR_PATH, "luts")
register_device("PPU610", ["PPU-ZW610"], _LUT_PATH)
register_device("PPU810", ["PPU-ZW810"], _LUT_PATH)
register_device("PPU810E", ["PPU-ZW810E"], "")
