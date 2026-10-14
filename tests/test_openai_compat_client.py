"""OpenAI-compatible provider (llama.cpp etc.) and provider selection. Uses a local stub HTTP server, no
real model."""
import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import make_llm_client, resolve_provider
from core.openai_compat_client import OpenAICompatLLMClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _completion(content: str) -> dict:
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "local",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


class StubServer:
    """Answers POST /v1/chat/completions from a queue of (status, body) and records request bodies."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                stub.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status, body = stub.responses.pop(0) if stub.responses else (200, _completion("ok"))
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


class TestOpenAICompatClient(unittest.TestCase):
    def client(self, server, **kw):
        kw.setdefault("backoff_base", 0.01)
        return OpenAICompatLLMClient(base_url=server.base_url, model="local", **kw)

    def test_sends_chat_request_and_returns_content(self):
        server = StubServer([(200, _completion("def solve(): pass"))])
        try:
            out = self.client(server, min_max_tokens=3000).generate("sys", "user", temperature=3.0, max_tokens=512)
            self.assertEqual(out, "def solve(): pass")
            req = server.requests[0]
            self.assertEqual([m["role"] for m in req["messages"]], ["system", "user"])
            self.assertEqual(req["max_tokens"], 3000)  # floor applied
            self.assertEqual(req["temperature"], 2.0)  # clamped
        finally:
            server.close()

    def test_records_usage_and_finish_reason(self):
        truncated = _completion("partial")
        truncated["choices"][0]["finish_reason"] = "length"
        truncated["usage"] = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                              "completion_tokens_details": {"reasoning_tokens": 40}}
        server = StubServer([(200, truncated), (400, {"error": "bad"})])
        try:
            client = self.client(server)
            client.generate("sys", "user")
            self.assertEqual(client.last_call_info(), {"prompt_tokens": 100, "completion_tokens": 50,
                                                       "reasoning_tokens": 40, "finish_reason": "length"})
            self.assertEqual(client.total_tokens, 150)
            self.assertEqual(client.usage["finish_length"], 1)
            client.generate("sys", "user")  # failed call: no stale info from the previous one
            self.assertEqual(client.last_call_info(), {})
            self.assertEqual(client.usage["calls"], 1)
        finally:
            server.close()

    def test_thinking_switch_sent_only_when_set(self):
        server = StubServer([])
        try:
            self.client(server, enable_thinking=False).generate("s", "u")
            self.client(server).generate("s", "u")
            self.assertEqual(server.requests[0]["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("chat_template_kwargs", server.requests[1])
            self.client(server, enable_thinking=True, thinking_budget=512).generate("s", "u")
            self.assertEqual(server.requests[2]["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(server.requests[2]["thinking_token_budget"], 512)
        finally:
            server.close()

    def test_retries_503_loading_then_succeeds(self):
        loading = (503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        server = StubServer([loading, (429, {"error": {"message": "busy"}}), (200, _completion("done"))])
        try:
            c = self.client(server)
            self.assertEqual(c.generate("s", "u"), "done")
            self.assertEqual(len(server.requests), 3)
            self.assertEqual(c.failed_calls, 0)
        finally:
            server.close()

    def test_does_not_retry_4xx_and_returns_empty(self):
        server = StubServer([(400, {"error": {"message": "bad request"}})])
        try:
            c = self.client(server)
            self.assertEqual(c.generate("s", "u"), "")
            self.assertEqual(len(server.requests), 1)
            self.assertEqual(c.failed_calls, 1)
        finally:
            server.close()

    def test_gives_up_after_max_retries(self):
        server = StubServer([(500, {"error": {"message": "boom"}})] * 10)
        try:
            c = self.client(server, max_retries=2)
            self.assertEqual(c.generate("s", "u"), "")
            self.assertEqual(len(server.requests), 3)
        finally:
            server.close()

    def test_rpm_cap_blocks_until_window_frees(self):
        c = OpenAICompatLLMClient(base_url="http://127.0.0.1:9/v1", rpm=2)
        clock = [1000.0]
        slept = []

        def fake_sleep(s):
            slept.append(s)
            clock[0] += s

        with mock.patch("core.openai_compat_client.time.monotonic", lambda: clock[0]), \
                mock.patch("core.openai_compat_client.time.sleep", fake_sleep):
            c._wait_for_slot()
            c._wait_for_slot()
            self.assertEqual(slept, [])
            c._wait_for_slot()  # third call in the same minute waits for the first to age out
        self.assertAlmostEqual(sum(slept), 60.0)


class TestParallelTasks(unittest.TestCase):
    def test_map_tasks_keeps_order_and_runs_concurrently(self):
        from types import SimpleNamespace
        from controller import EvolutionController

        ctrl = SimpleNamespace(max_concurrency=8)
        tasks = [SimpleNamespace(id=f"t{i}") for i in range(16)]
        barrier = threading.Barrier(8, timeout=5)  # deadlocks unless 8 calls are in flight together

        def fn(task):
            barrier.wait()
            return task.id

        out = EvolutionController._map_tasks(ctrl, fn, tasks)
        self.assertEqual(out, [t.id for t in tasks])

    def test_controller_parallel_matches_sequential_in_mock_mode(self):
        from benchmarks.arc_challenge import ARC_BENCHMARK_ID, build_arc_evaluator_pool
        from controller import EvolutionController

        def seed_metrics(concurrency):
            ctrl = EvolutionController(is_mock=True, selected_task_id=ARC_BENCHMARK_ID, inter_call_delay=0,
                                       evaluator_pool_factory=build_arc_evaluator_pool, max_concurrency=concurrency)
            return {k: v for k, v in ctrl.history_records[0]["task_metrics"].items()}

        seq, par = seed_metrics(1), seed_metrics(8)
        self.assertEqual(seq.keys(), par.keys())
        for tid in seq:  # latency differs run to run; every other metric must match
            self.assertEqual({k: v for k, v in seq[tid].items() if "latency" not in k},
                             {k: v for k, v in par[tid].items() if "latency" not in k})


class TestProviderSelection(unittest.TestCase):
    def test_resolution_order(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_provider(), "groq")
            self.assertEqual(resolve_provider(base_url="http://x/v1"), "openai")
            self.assertEqual(resolve_provider("groq", base_url="http://x/v1"), "groq")
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "http://x/v1"}, clear=True):
            self.assertEqual(resolve_provider(), "openai")
        with self.assertRaises(ValueError):
            resolve_provider("anthropic")

    def test_mock_mode_uses_offline_mock_for_any_provider(self):
        client = make_llm_client("openai", base_url="http://x/v1", is_mock=True)
        self.assertTrue(client.is_mock)
        self.assertIn("def solve", client.generate("You are a coding agent.", "Write solve(n) returning n."))

    def test_openai_provider_never_imports_groq(self):
        code = (
            "import sys; from core.llm import make_llm_client; "
            "c = make_llm_client('openai', base_url='http://127.0.0.1:9/v1'); "
            "import controller, evaluation.pool, benchmarks.arc_challenge; "
            "print(type(c).__name__, 'groq' in sys.modules, 'core.groq_client' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["OpenAICompatLLMClient", "False", "False"])


if __name__ == "__main__":
    unittest.main()
