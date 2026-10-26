"""PPU backend entrypoint for the public model factories.

The implementation modules are public; this entrypoint only records their
backend hooks before the factories consume those slots.
"""

import logging
import os
import sys

_jit = os.environ.get("_JIT_CACHE_PATHS", "")
if _jit:
    for _p in _jit.split(":"):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from rtp_llm.device.device_type import DeviceType, get_device_type

    if get_device_type() == DeviceType.Ppu:
        from rtp_llm.models_py.modules.factory.attention import register as attn_register
        from rtp_llm.models_py.modules.factory.fused_moe import register as moe_register
        from rtp_llm.models_py.modules.factory.linear import register as linear_register

        attn_register.install()
        moe_register.install()
        linear_register.install()
        logging.info("[ppu] backend hooks installed")
except Exception as exc:  # noqa: BLE001
    logging.warning("[ppu] failed to install backend hooks: %s", exc, exc_info=True)

