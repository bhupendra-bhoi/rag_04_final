"""Provider strategy types."""

from enum import Enum


class ProviderType(str, Enum):
    """Available LLM provider strategies."""
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
