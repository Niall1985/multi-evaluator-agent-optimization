"""Second LLM provider: any OpenAI-compatible chat endpoint (llama.cpp `llama-server`, vLLM, Ollama, ...).

Deliberately small next to GroqLLMClient: no fallback models, no reasoning-effort handling (only an on/off
thinking switch), no quota or
token-per-minute bookkeeping. It has

  - a requests-per-minute cap (sliding 60 s window) and a minimum gap between calls,
  - exponential backoff with jitter on retryable failures (429, 5xx, connection errors, timeouts),
    honouring a Retry-After header when the server sends one,
  - an empty string on final failure, counted in `failed_calls` (never a mock answer: that would fake
    successes in real results).

llama.cpp: start `llama-server -m model.gguf`, then use base_url "http://<host>:<port>/v1"; the key is a
placeholder unless the server was started with --api-key. While the model is still loading the server
answers 503, which is retried like any other 5xx.
"""
import logging
import os
import random
import threading
import time
from collections import deque
from typing import Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class OpenAICompatLLMClient:
    """Drop-in for GroqLLMClient's `generate()` against an OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str = "local",
        api_key: Optional[str] = None,
        rpm: int = 0,
        inter_call_delay: float = 0.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        timeout: float = 600.0,
        min_max_tokens: int = 0,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "") or "sk-no-key-required"
        self.rpm = rpm  # 0 = no cap
        self.inter_call_delay = inter_call_delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_max_tokens = min_max_tokens
        # Qwen3-style templates think by default; when the reasoning parser is on and thinking uses up
        # max_tokens, `content` comes back empty. None = send nothing and keep the server's default.
        self.enable_thinking = enable_thinking
        # vLLM's `thinking_token_budget`: after this many reasoning tokens the server ends the thinking block
        # (needs --reasoning-parser; see deploy/modal_vllm.py --reasoning-config). None = no limit.
        self.thinking_budget = thinking_budget
        # The SDK's own retries are off so the backoff below is the only one
        self.client = OpenAI(base_url=base_url, api_key=self.api_key, max_retries=0, timeout=timeout)

        # Attributes the controller, UI and runner read on any client
        self.is_mock = False
        self.failed_calls = 0
        self.quota_exhausted_until: Optional[float] = None
        self.quota_exhausted_error: Optional[str] = None

        self._lock = threading.Lock()
        self._call_times: deque = deque()
        self._last_call = 0.0

    def _wait_for_slot(self) -> None:
        """Blocks until both the minimum gap and the RPM window allow another call, then claims it."""
        with self._lock:
            while True:
                now = time.monotonic()
                while self._call_times and now - self._call_times[0] >= 60.0:
                    self._call_times.popleft()
                wait = max(0.0, self._last_call + self.inter_call_delay - now)
                if self.rpm > 0 and len(self._call_times) >= self.rpm:
                    wait = max(wait, 60.0 - (now - self._call_times[0]))
                if wait <= 0:
                    break
                time.sleep(wait)
            self._last_call = now
            self._call_times.append(now)

    def _backoff(self, attempt: int, error: Exception) -> float:
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        try:
            return min(self.backoff_max, float(headers.get("retry-after", "")))
        except (TypeError, ValueError):
            delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
            return delay * random.uniform(0.5, 1.0)  # jitter so parallel callers don't retry in lockstep

    def _extra_body(self) -> Optional[dict]:
        body: dict = {}
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            body["thinking_token_budget"] = self.thinking_budget
        return body or None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=max(0.0, min(2.0, temperature)),
                    top_p=max(0.0, min(1.0, top_p)),
                    max_tokens=max(max_tokens, self.min_max_tokens),
                    extra_body=self._extra_body(),
                )
                return response.choices[0].message.content or ""
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                # APITimeoutError is a subclass of APIConnectionError
                last_error = e
                if attempt == self.max_retries:
                    break
                delay = self._backoff(attempt, e)
                logger.warning(f"{type(e).__name__} from {self.base_url} (attempt {attempt + 1}); retrying in {delay:.1f}s")
                time.sleep(delay)
            except APIStatusError as e:
                # Other 4xx (bad request, auth, unknown model): retrying sends the same request again
                last_error = e
                break

        with self._lock:  # generate() runs on several threads when tasks are evaluated in parallel
            self.failed_calls += 1
        logger.error(f"{self.base_url} failed; returning empty answer (failed calls so far: {self.failed_calls}). {last_error}")
        return ""
