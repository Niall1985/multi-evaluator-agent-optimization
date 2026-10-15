"""Provider-neutral LLM client interface and factory.

Provider modules are imported only inside the branch that uses them, so running against a local
OpenAI-compatible server never loads the Groq SDK and vice versa.

Provider selection, first match wins:
  1. the `provider` argument ("groq" | "openai")
  2. the LLM_PROVIDER env var
  3. "openai" if a base_url is given (argument or OPENAI_BASE_URL), else "groq"
Mock mode (`is_mock=True`) always uses GroqLLMClient's deterministic offline mock, whatever the provider.
"""
import os
from typing import Optional, Protocol, runtime_checkable

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = ("groq", "openai")
# How long an OpenAI-compatible call keeps retrying 5xx / connection errors: outlasts a Modal container
# replacement (cold start ~3-5 min), so a call waits for the new container instead of returning empty.
OPENAI_RETRY_WINDOW_SEC = 480.0


@runtime_checkable
class LLMClient(Protocol):
    """What the controller, mutator, evaluators, UI and runners use on any provider."""

    model: str
    is_mock: bool
    inter_call_delay: float
    failed_calls: int
    quota_exhausted_until: Optional[float]
    quota_exhausted_error: Optional[str]

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> str: ...


def resolve_provider(provider: Optional[str] = None, base_url: Optional[str] = None) -> str:
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
    name = (provider or os.getenv("LLM_PROVIDER") or ("openai" if base_url else "groq")).strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; expected one of {PROVIDERS}")
    return name


def make_llm_client(
    provider: Optional[str] = None,
    *,
    model: str = "openai/gpt-oss-120b",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    is_mock: Optional[bool] = None,
    inter_call_delay: float = 1.5,
    min_max_tokens: int = 0,
    reasoning_effort: Optional[str] = None,
    rpm: int = 0,
    enable_thinking: Optional[bool] = None,
    thinking_budget: Optional[int] = None,
) -> LLMClient:
    """Builds the client for the resolved provider.

    `reasoning_effort` is Groq-only; `rpm`, `enable_thinking` (chat_template_kwargs, None = server
    default) and `thinking_budget` (vLLM thinking_token_budget) apply to the OpenAI-compatible provider (Groq rate limiting is
    handled by GroqLLMClient's backoff and, in scripts/live_arc_run.py, its own limiter).
    """
    name = resolve_provider(provider, base_url)
    if name == "openai" and not is_mock:
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        if not base_url:
            raise ValueError("the openai provider needs base_url or OPENAI_BASE_URL, e.g. http://localhost:8080/v1")
        from core.openai_compat_client import OpenAICompatLLMClient

        return OpenAICompatLLMClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            rpm=rpm,
            inter_call_delay=inter_call_delay,
            min_max_tokens=min_max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            retry_window=OPENAI_RETRY_WINDOW_SEC,
        )

    from core.groq_client import GroqLLMClient

    return GroqLLMClient(
        api_key=api_key,
        model=model,
        inter_call_delay=inter_call_delay,
        is_mock=is_mock,
        reasoning_effort=reasoning_effort,
        min_max_tokens=min_max_tokens,
    )
