"""Core data structures and LLM interfaces for Multi-Objective Agent Optimization."""

from .agent import Agent, HarnessKnobs
from .archive import PopulationArchive
from .llm import LLMClient, make_llm_client
from .mutator import PromptMutator

__all__ = [
    "Agent",
    "HarnessKnobs",
    "PopulationArchive",
    "LLMClient",
    "make_llm_client",
    "GroqLLMClient",
    "PromptMutator",
]


def __getattr__(name: str):
    # Provider clients load on first use only, so `import core` never pulls in a provider SDK
    if name == "GroqLLMClient":
        from .groq_client import GroqLLMClient

        return GroqLLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
