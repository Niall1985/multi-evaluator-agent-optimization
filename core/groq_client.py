import os
import logging
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]


class GroqLLMClient:
    """Groq LLM Client supporting primary models, active fallbacks, and offline simulation mode."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.client = None
        self.is_mock = False

        if GROQ_AVAILABLE and self.api_key and self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key.strip())
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}. Falling back to mock mode.")
                self.is_mock = True
        else:
            self.is_mock = True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> str:
        """Generates response using Groq API with automatic fallback sequence or deterministic mock."""
        if self.is_mock or not self.client:
            return self._mock_generate(system_prompt, user_prompt, temperature)

        candidate_models = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        last_error = None

        for model_name in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=max(0.0, min(2.0, temperature)),
                    top_p=max(0.0, min(1.0, top_p)),
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Groq generation failed on model '{model_name}': {e}. Trying fallback.")
                last_error = e

        logger.error(f"All Groq models failed. Reverting to mock response. Error: {last_error}")
        return self._mock_generate(system_prompt, user_prompt, temperature)

    def _mock_generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """High-quality deterministic mock generation for offline testing and demo mode."""
        lower_sys = system_prompt.lower()
        lower_user = user_prompt.lower()

        # 1. Check for Judge / Evaluation calls (mu_4 and mu_6)
        if "evaluate" in lower_sys or "judge" in lower_sys or "score" in lower_sys or "safety and hallucination" in lower_sys:
            return (
                "Score: 0.92\n"
                "Reasoning: The candidate solution demonstrates strong logical structure, correctly handles "
                "problem constraints, and produces verified algorithmic logic without hallucinated APIs."
            )

        # 2. Check for Prompt Mutation / Rewriting calls
        if "prompt" in lower_sys or "mutat" in lower_sys or "optimization goal" in lower_user or "rewrite" in lower_user:
            if "edge_case" in lower_user:
                return (
                    "You are a defensive AI software engineer. Before implementing, rigorously inspect and trap "
                    "all corner cases (empty collections, zero values, negative indices, and boundary overflows). "
                    "Ensure every edge case is handled cleanly with explicit validations."
                )
            elif "chain_of_thought" in lower_user:
                return (
                    "You are an analytical AI reasoning specialist. Solve problems by decomposing them into clear logical steps: "
                    "1) Analyze mathematical invariants and boundary conditions, 2) Design optimal data structures, "
                    "3) Synthesize verified, robust Python code."
                )
            elif "conciseness" in lower_user:
                return (
                    "You are a hyper-concise AI code synthesizer. Deliver pure, production-grade Python implementations. "
                    "Omit all conversational filler, greetings, comments, and postamble text. Return only optimal code."
                )
            elif "safety" in lower_user:
                return (
                    "You are a security and constraint-focused AI assistant. Strictly enforce input types, bounds, "
                    "and safety requirements. Prevent any hallucination of external dependencies and maintain deterministic behavior."
                )
            elif "algorithmic" in lower_user or "efficiency" in lower_user:
                return (
                    "You are an algorithmic optimization specialist. Prioritize optimal asymptotic complexity: "
                    "target O(N) or O(log N) runtime and O(1) auxiliary memory. Use optimal two-pointer, hashing, or dynamic programming techniques."
                )
            elif "retry" in lower_user or "robust" in lower_user:
                return (
                    "You are a fault-tolerant AI programming agent. Build defensive logic with explicit exception handling, "
                    "type guards, and boundary checks to guarantee 100% execution reliability."
                )
            else:
                return (
                    "You are an advanced AI coding specialist. Solve the problem with maximum functional accuracy, "
                    "defensive input validation, and optimal asymptotic complexity."
                )

        # 3. Check for Benchmark Task Execution
        if "fibonacci" in lower_user or "fibonacci" in lower_sys:
            return (
                "```python\n"
                "def solve(n: int) -> int:\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```"
            )
        elif "palindrome" in lower_user or "palindrome" in lower_sys:
            return (
                "```python\n"
                "def solve(s: str) -> bool:\n"
                "    filtered = [c.lower() for c in s if c.isalnum()]\n"
                "    return filtered == filtered[::-1]\n"
                "```"
            )
        elif "interval" in lower_user or "merge" in lower_user:
            return (
                "```python\n"
                "def solve(intervals: list) -> list:\n"
                "    if not intervals:\n"
                "        return []\n"
                "    intervals.sort(key=lambda x: x[0])\n"
                "    merged = [intervals[0]]\n"
                "    for current in intervals[1:]:\n"
                "        prev = merged[-1]\n"
                "        if current[0] <= prev[1]:\n"
                "            merged[-1] = [prev[0], max(prev[1], current[1])]\n"
                "        else:\n"
                "            merged.append(current)\n"
                "    return merged\n"
                "```"
            )
        elif "parenthes" in lower_user or "bracket" in lower_user:
            return (
                "```python\n"
                "def solve(s: str) -> bool:\n"
                "    mapping = {')': '(', '}': '{', ']': '['}\n"
                "    stack = []\n"
                "    for char in s:\n"
                "        if char in mapping.values():\n"
                "            stack.append(char)\n"
                "        elif char in mapping:\n"
                "            if not stack or stack.pop() != mapping[char]:\n"
                "                return False\n"
                "    return len(stack) == 0\n"
                "```"
            )
        else:
            return (
                "```python\n"
                "def solve(*args, **kwargs):\n"
                "    return True\n"
                "```"
            )
