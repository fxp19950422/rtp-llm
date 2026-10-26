import importlib
import logging


def _load_ppu_backend_entrypoint() -> None:
    """Register public PPU backends before any factory slot is initialized."""
    try:
        from rtp_llm.device.device_type import DeviceType, get_device_type

        if get_device_type() == DeviceType.Ppu:
            importlib.import_module("rtp_llm.models_py")
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ppu] early backend entrypoint load failed: %s", exc, exc_info=True)


_load_ppu_backend_entrypoint()
from typing import Any


def __getattr__(name: str) -> Any:
    """Preserve old top-level access without importing C++ ops eagerly."""
    if name == "_ft_pickler":
        module = importlib.import_module("rtp_llm._ft_pickler")
        globals()[name] = module
        return module

    if name == "enable_compile_monitor":
        from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

        globals()[name] = enable_compile_monitor
        return enable_compile_monitor

    ops = importlib.import_module("rtp_llm.ops")
    try:
        value = getattr(ops, name)
    except AttributeError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    globals()[name] = value
    return value

