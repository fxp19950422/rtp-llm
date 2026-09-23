"""Built-in platform manifests. Loading hooks must not import a device runtime.

Platforms supply deferred adapters to the existing factories and lightweight
model descriptors to ModuleRegistry; execution uses the selected real modules.
"""

from importlib import import_module

_BUILTIN_PLATFORMS = ("rtp_llm.platforms.ppu",)


def _uses_builtin_platforms():
    from rtp_llm.device.device_type import DeviceType, get_device_type

    return get_device_type() == DeviceType.Ppu


def register_backend_hooks():
    if not _uses_builtin_platforms():
        return
    for name in _BUILTIN_PLATFORMS:
        import_module(name).register_backend_hooks()


def register_modules(registry):
    if not _uses_builtin_platforms():
        return
    for name in _BUILTIN_PLATFORMS:
        import_module(name).register_modules(registry)
