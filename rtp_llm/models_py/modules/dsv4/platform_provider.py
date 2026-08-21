"""Platform-neutral construction provider for DeepSeek-V4 modules.

The built-in provider delegates to the existing constructors.  An optional
provider may be registered by a platform integration before the first model
or standalone transformer is constructed.  Registration is deliberately
process-global and immutable after construction starts so one model tree
cannot contain modules from different providers.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any, Callable, FrozenSet, Iterable, Optional, Protocol


class Dsv4ProviderCapability(str, Enum):
    """Construction points a provider must explicitly claim."""

    BLOCK = "block"
    TRANSFORMER = "transformer"


class Dsv4PlatformProvider(Protocol):
    """Interface implemented by DeepSeek-V4 construction providers."""

    name: str
    capabilities: FrozenSet[Dsv4ProviderCapability]

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...


class DefaultDsv4PlatformProvider:
    """Provider preserving the existing constructors and runtime behavior."""

    name = "cuda"
    capabilities = frozenset(
        {
            Dsv4ProviderCapability.BLOCK,
            Dsv4ProviderCapability.TRANSFORMER,
        }
    )

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return default_factory(*args, **kwargs)

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return default_factory(*args, **kwargs)


def _normalize_capabilities(
    capabilities: Iterable[Dsv4ProviderCapability],
) -> FrozenSet[Dsv4ProviderCapability]:
    try:
        return frozenset(
            Dsv4ProviderCapability(capability) for capability in capabilities
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid DSV4 provider capability: {error}") from error


def _validate_provider(provider: Dsv4PlatformProvider) -> None:
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("DSV4 platform provider must have a non-empty name")
    capabilities = _normalize_capabilities(getattr(provider, "capabilities", ()))
    methods = {
        Dsv4ProviderCapability.BLOCK: "build_block",
        Dsv4ProviderCapability.TRANSFORMER: "build_transformer",
    }
    for capability in capabilities:
        method_name = methods[capability]
        if not callable(getattr(provider, method_name, None)):
            raise TypeError(
                f"DSV4 provider {name!r} declares {capability.value!r} "
                f"but has no callable {method_name}"
            )


class Dsv4PlatformProviderRegistry:
    """Exactly-once provider registry with a construction-time freeze."""

    def __init__(
        self, default_provider: Optional[Dsv4PlatformProvider] = None
    ) -> None:
        self._lock = threading.Lock()
        self._default_provider = (
            default_provider
            if default_provider is not None
            else DefaultDsv4PlatformProvider()
        )
        _validate_provider(self._default_provider)
        self._registered_provider: Optional[Dsv4PlatformProvider] = None
        self._registration_consumed = False
        self._construction_started = False

    def register(self, provider: Dsv4PlatformProvider) -> None:
        """Register once, before the first construction-time resolution."""

        _validate_provider(provider)
        with self._lock:
            if self._construction_started:
                raise RuntimeError(
                    "cannot register a DSV4 platform provider after "
                    "construction started"
                )
            if self._registration_consumed:
                raise RuntimeError("a DSV4 platform provider is already registered")
            self._registered_provider = provider
            self._registration_consumed = True

    def _active_provider(self) -> Dsv4PlatformProvider:
        if self._registered_provider is not None:
            return self._registered_provider
        return self._default_provider

    def capabilities(self) -> FrozenSet[Dsv4ProviderCapability]:
        """Query active capabilities without freezing registration."""

        with self._lock:
            return _normalize_capabilities(self._active_provider().capabilities)

    def validate_capabilities(
        self, required: Iterable[Dsv4ProviderCapability]
    ) -> None:
        """Fail fast if the active provider lacks a required capability."""

        required_set = _normalize_capabilities(required)
        with self._lock:
            provider = self._active_provider()
            _validate_provider(provider)
            available = _normalize_capabilities(provider.capabilities)
            missing = required_set - available
            if missing:
                missing_names = ", ".join(sorted(cap.value for cap in missing))
                raise RuntimeError(
                    f"DSV4 provider {provider.name!r} lacks required capabilities: "
                    f"{missing_names}"
                )

    def resolve(
        self, required: Iterable[Dsv4ProviderCapability]
    ) -> Dsv4PlatformProvider:
        """Resolve for construction and permanently close registration."""

        required_set = _normalize_capabilities(required)
        with self._lock:
            provider = self._active_provider()
            _validate_provider(provider)
            available = _normalize_capabilities(provider.capabilities)
            missing = required_set - available
            if missing:
                missing_names = ", ".join(sorted(cap.value for cap in missing))
                raise RuntimeError(
                    f"DSV4 provider {provider.name!r} lacks required capabilities: "
                    f"{missing_names}"
                )
            self._construction_started = True
            return provider


_PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()


def register_dsv4_platform_provider(provider: Dsv4PlatformProvider) -> None:
    _PROVIDER_REGISTRY.register(provider)


def get_dsv4_platform_provider_capabilities() -> FrozenSet[Dsv4ProviderCapability]:
    return _PROVIDER_REGISTRY.capabilities()


def validate_dsv4_platform_provider_capabilities(
    required: Iterable[Dsv4ProviderCapability],
) -> None:
    _PROVIDER_REGISTRY.validate_capabilities(required)


def resolve_dsv4_platform_provider(
    required: Iterable[Dsv4ProviderCapability],
) -> Dsv4PlatformProvider:
    return _PROVIDER_REGISTRY.resolve(required)


__all__ = [
    "DefaultDsv4PlatformProvider",
    "Dsv4PlatformProvider",
    "Dsv4PlatformProviderRegistry",
    "Dsv4ProviderCapability",
    "get_dsv4_platform_provider_capabilities",
    "register_dsv4_platform_provider",
    "resolve_dsv4_platform_provider",
    "validate_dsv4_platform_provider_capabilities",
]
