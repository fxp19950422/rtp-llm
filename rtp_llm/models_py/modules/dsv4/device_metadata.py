"""Opt-in contract between DSV4 attention and the C++ graph runner.

Never advertise this capability for a backend that reads host length mirrors.
The legacy path is intentionally unchanged when both switches are off.
"""
from .runtime_config import get_switch, parse_bool


def device_metadata_enabled(multitoken: bool) -> bool:
    name = "DSV4_VERIFY_DEVICE_METADATA" if multitoken else "DSV4_DECODE_DEVICE_METADATA"
    return get_switch(name, False, parse_bool)


def device_start_positions(attn, multitoken: bool):
    """Read only fixed-address device mirrors; reject incomplete contracts."""
    if multitoken:
        value = getattr(attn, "prefix_lengths_device", None)
    else:
        value = getattr(attn, "sequence_lengths_plus_1_device", None)
    if value is None or not value.is_cuda or value.numel() == 0:
        raise RuntimeError("DSV4 device metadata requires populated CUDA length mirrors")
    return value if multitoken else value - 1
