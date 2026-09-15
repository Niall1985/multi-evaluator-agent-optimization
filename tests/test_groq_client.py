import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.groq_client import TRUNCATION_RETRY_MIN_TOKENS, GroqLLMClient


def _response(content, finish_reason):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class TestGroqClientTruncationRetry(unittest.TestCase):
    """Reasoning models can exhaust max_tokens on hidden reasoning and return empty content.

    Observed live with openai/gpt-oss-120b at the default harness budget of 1024 tokens:
    finish_reason='length', 1024 completion tokens, content=''. The client must retry with a
    larger budget instead of handing '' to the evaluators.
    """

    def _client(self, responses):
        c = GroqLLMClient(api_key="x", is_mock=False, inter_call_delay=0)
        c.is_mock = False
        create = mock.Mock(side_effect=responses)
        c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return c, create

    def test_empty_truncated_content_retries_once_with_bigger_budget(self):
        c, create = self._client([_response("", "length"), _response("def solve(grid): return grid", "stop")])
        out = c.generate("sys", "user", max_tokens=1024)
        self.assertEqual(out, "def solve(grid): return grid")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].kwargs["max_tokens"], 1024)
        self.assertEqual(create.call_args_list[1].kwargs["max_tokens"], max(TRUNCATION_RETRY_MIN_TOKENS, 4 * 1024))

    def test_budget_is_raised_only_once(self):
        c, create = self._client([_response("", "length"), _response("", "length")])
        self.assertEqual(c.generate("sys", "user", max_tokens=100), "")
        self.assertEqual(create.call_count, 2)  # no infinite retry loop

    def test_normal_and_truncated_nonempty_responses_are_returned_as_is(self):
        c, create = self._client([_response("partial code", "length")])
        self.assertEqual(c.generate("sys", "user", max_tokens=1024), "partial code")
        self.assertEqual(create.call_count, 1)
        c, create = self._client([_response("ok", "stop")])
        self.assertEqual(c.generate("sys", "user"), "ok")
        self.assertEqual(create.call_args.kwargs["max_tokens"], 1024)

    def test_exhausted_live_api_returns_empty_not_mock(self):
        """Regression: a failed live call used to fall back to the mock, which knows the real answers to the
        bundled ARC / canonical tasks - i.e. it faked successes inside live results."""
        from benchmarks.arc_challenge import get_arc_task

        arc_prompt = get_arc_task("3c9b0459").description
        c, create = self._client(Exception("Error code: 400 - tool_use_failed"))  # every attempt, every model
        with mock.patch("core.groq_client.time.sleep"):
            out = c.generate("You are a coding agent.", arc_prompt)
        self.assertEqual(out, "")
        self.assertNotIn("def solve", out)
        self.assertEqual(c.failed_calls, 1)
        self.assertGreater(create.call_count, 1)  # it did try the fallback chain first

        # Mock mode itself is untouched: same prompt still yields the canned ARC solution
        self.assertIn("def solve", GroqLLMClient(is_mock=True).generate("You are a coding agent.", arc_prompt))


if __name__ == "__main__":
    unittest.main()
