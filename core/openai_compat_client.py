"""Second LLM provider: any OpenAI-compatible chat endpoint (llama.cpp `llama-server`, vLLM, Ollama, ...).

Deliberately small next to GroqLLMClient: no fallback models, no quota or token-per-minute bookkeeping;
thinking is an on/off switch plus a per-thread override of budget and Qwen3.8 reasoning_effort
(call_options). It has

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
from contextlib import contextmanager
from typing import Iterator, Optional

import httpx
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
        stream: bool = True,
        stream_read_timeout: float = 120.0,
        retry_window: float = 0.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "") or "sk-no-key-required"
        self.rpm = rpm  # 0 = no cap
        self.inter_call_delay = inter_call_delay
        self.max_retries = max_retries
        # Retryable failures (5xx, 503 "no upstreams", dropped connections) keep being retried, past max_retries,
        # until this many seconds have passed since the first attempt. Modal replaces a GPU container now and then
        # (two drains on 2026-10-15); the new one needs ~3-5 min to load, far longer than max_retries of backoff.
        self.retry_window = retry_window
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_max_tokens = min_max_tokens
        # Qwen3-style templates think by default; when the reasoning parser is on and thinking uses up
        # max_tokens, `content` comes back empty. None = send nothing and keep the server's default.
        self.enable_thinking = enable_thinking
        # vLLM's `thinking_token_budget`: after this many reasoning tokens the server ends the thinking block
        # (needs --reasoning-parser; see deploy/modal_vllm.py --reasoning-config). None = no limit.
        self.thinking_budget = thinking_budget
        # Streaming: a non-streamed answer sends no bytes for minutes, and on the Modal endpoint ~2-5 of 22
        # requests at a time never got their finished response (the server showed 17-20 running while the client
        # had 22 in flight) and sat out the full 600 s timeout. Streamed tokens keep the connection busy, and a
        # stall is caught by the per-read timeout (the gap between chunks) after `stream_read_timeout` seconds.
        self.stream = stream
        read_timeout = stream_read_timeout if stream else timeout
        # The SDK's own retries are off so the backoff below is the only one
        self.client = OpenAI(base_url=base_url, api_key=self.api_key, max_retries=0,
                             timeout=httpx.Timeout(timeout, read=read_timeout, connect=30.0))

        # Attributes the controller, UI and runner read on any client
        self.is_mock = False
        self.failed_calls = 0
        self.quota_exhausted_until: Optional[float] = None
        self.quota_exhausted_error: Optional[str] = None
        # Token accounting across all calls, and the last call's usage per thread (generate() runs on several
        # threads, so the caller reads its own call's numbers with last_call_info() right after generate()).
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                      "finish_length": 0}
        self._local = threading.local()

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

    @contextmanager
    def call_options(self, thinking_budget: Optional[int] = None,
                     reasoning_effort: Optional[str] = None) -> Iterator[None]:
        """Overrides the thinking budget and/or sets Qwen3.8's `reasoning_effort` (xhigh/medium/low, sent in
        chat_template_kwargs) for generate() calls made inside the block on this thread only."""
        previous = getattr(self._local, "options", {})
        self._local.options = {k: v for k, v in {"thinking_budget": thinking_budget,
                                                 "reasoning_effort": reasoning_effort}.items() if v is not None}
        try:
            yield
        finally:
            self._local.options = previous

    def _extra_body(self) -> Optional[dict]:
        options = getattr(self._local, "options", {})
        body: dict = {}
        template: dict = {}
        if self.enable_thinking is not None:
            template["enable_thinking"] = self.enable_thinking
        if options.get("reasoning_effort") and self.enable_thinking is not False:
            template["reasoning_effort"] = options["reasoning_effort"]
        if template:
            body["chat_template_kwargs"] = template
        budget = options.get("thinking_budget", self.thinking_budget)
        if budget is not None and self.enable_thinking is not False:
            body["thinking_token_budget"] = budget
        return body or None

    @property
    def total_tokens(self) -> int:
        return self.usage["prompt_tokens"] + self.usage["completion_tokens"]

    def last_call_info(self) -> dict:
        """Usage and finish_reason of the last generate() on this thread ({} if it failed or none ran)."""
        return dict(getattr(self._local, "info", {}) or {})

    def _record_usage(self, usage, finish_reason: Optional[str]) -> None:
        details = getattr(usage, "completion_tokens_details", None)
        info = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
            "finish_reason": finish_reason,
        }
        self._local.info = info
        with self._lock:
            self.usage["calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                self.usage[key] += info[key]
            self.usage["finish_length"] += int(info["finish_reason"] == "length")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> str:
        last_error: Optional[Exception] = None
        self._local.info = {}
        started = time.monotonic()
        attempt = -1
        while True:
            attempt += 1
            self._wait_for_slot()
            try:
                request = dict(
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
                if not self.stream:
                    response = self.client.chat.completions.create(**request)
                    self._record_usage(response.usage, response.choices[0].finish_reason)
                    return response.choices[0].message.content or ""
                return self._generate_streamed(request)
            except (RateLimitError, InternalServerError, APIConnectionError, httpx.TransportError) as e:
                # APITimeoutError is a subclass of APIConnectionError. A stall or a dropped connection while a stream
                # is being read surfaces as an httpx error (ReadTimeout, RemoteProtocolError), not an SDK one.
                last_error = e
                if attempt >= self.max_retries and time.monotonic() - started >= self.retry_window:
                    break
                delay = self._backoff(min(attempt, 10), e)
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

    def _generate_streamed(self, request: dict) -> str:
        """One streamed completion: the answer text, with usage from the final chunk (stream_options)."""
        parts = []
        finish_reason = None
        usage = None
        stream = self.client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}
        )
        with stream:
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                for choice in chunk.choices:
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        self._record_usage(usage, finish_reason)
        return "".join(parts)
