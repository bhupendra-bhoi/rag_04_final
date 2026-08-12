"""Provider module with registry-based strategy architecture."""

from providers.base.strategy import ProviderStrategy
from providers.registry import provider_registry, ProviderRegistry
from providers.enums import ProviderType
from providers.config import ProviderConfig, BaseProviderConfig

# Eagerly import to trigger @register decorator; guarded for missing deps
try:
    from providers.groq.groq_provider import GroqProvider
except ImportError:
    pass

try:
    from providers.openrouter.openrouter_provider import OpenRouterProvider
except ImportError:
    pass


__all__ = [
    "ProviderStrategy",
    "ProviderRegistry",
    "provider_registry",
    "ProviderType",
    "ProviderConfig",
    "BaseProviderConfig",
    "GroqProvider",
    "OpenRouterProvider",
]
