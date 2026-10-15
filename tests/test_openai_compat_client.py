"""OpenAI-compatible provider (llama.cpp etc.) and provider selection. Uses a local stub HTTP server, no
real model."""
import json
import os
import subprocess
import sys
import threading
import time
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
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(req)
                status, body = stub.responses.pop(0) if stub.responses else (200, _completion("ok"))
                if status == 200 and req.get("stream"):
                    self._send_stream(body)
                    return
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_stream(self, body):
                """The completion as server-sent events: content in two deltas, the finish reason, then a usage
                chunk (stream_options.include_usage). `stall_after_first` seconds of silence after the first delta
                imitates a response that never arrives."""
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                choice = body["choices"][0]
                text = choice["message"]["content"]
                base = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "local"}

                def event(choices, usage=None):
                    self.wfile.write(f"data: {json.dumps({**base, 'choices': choices, 'usage': usage})}\n\n".encode())
                    self.wfile.flush()

                try:
                    half = len(text) // 2
                    event([{"index": 0, "delta": {"role": "assistant", "content": text[:half]}, "finish_reason": None}])
                    if body.get("stall_after_first"):
                        time.sleep(body["stall_after_first"])
                    event([{"index": 0, "delta": {"content": text[half:]}, "finish_reason": None}])
                    event([{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}])
                    if body.get("usage"):
                        event([], body["usage"])
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up on a stalled stream

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

    def test_streams_by_default_and_non_streaming_still_works(self):
        server = StubServer([(200, _completion("streamed answer")), (200, _completion("plain answer"))])
        try:
            self.assertEqual(self.client(server).generate("s", "u"), "streamed answer")
            self.assertTrue(server.requests[0]["stream"])
            self.assertEqual(server.requests[0]["stream_options"], {"include_usage": True})
            self.assertEqual(self.client(server, stream=False).generate("s", "u"), "plain answer")
            self.assertNotIn("stream", server.requests[1])
        finally:
            server.close()

    def test_stalled_stream_is_retried_after_the_read_timeout(self):
        stalled = _completion("never finishes")
        stalled["stall_after_first"] = 3
        server = StubServer([(200, stalled), (200, _completion("second try"))])
        try:
            c = self.client(server, stream_read_timeout=0.5)
            t0 = time.monotonic()
            self.assertEqual(c.generate("s", "u"), "second try")
            self.assertLess(time.monotonic() - t0, 2.5)  # gave up on the stall well before it would have ended
            self.assertEqual(len(server.requests), 2)
            self.assertEqual(c.failed_calls, 0)
        finally:
            server.close()

    def test_call_options_override_thinking_on_this_thread_only(self):
        server = StubServer([])
        try:
            c = self.client(server, enable_thinking=True, thinking_budget=4096)
            with c.call_options(thinking_budget=2048, reasoning_effort="low"):
                c.generate("s", "u")
                other = threading.Thread(target=c.generate, args=("s", "u"))
                other.start()
                other.join()
            c.generate("s", "u")
            inside, other_thread, after = server.requests
            self.assertEqual(inside["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "low"})
            self.assertEqual(inside["thinking_token_budget"], 2048)
            for req in (other_thread, after):
                self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": True})
                self.assertEqual(req["thinking_token_budget"], 4096)
            off = self.client(server, enable_thinking=False)
            with off.call_options(thinking_budget=2048, reasoning_effort="low"):
                off.generate("s", "u")  # thinking off: nothing thinking-related is sent
            self.assertEqual(server.requests[-1]["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("thinking_token_budget", server.requests[-1])
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

    def test_retry_window_outlasts_max_retries(self):
        unavailable = (503, {"error": "no upstreams available"})
        server = StubServer([unavailable] * 6 + [(200, _completion("new container"))])
        try:
            c = self.client(server, max_retries=1, retry_window=5.0)
            self.assertEqual(c.generate("s", "u"), "new container")  # 7 attempts, past max_retries=1
            self.assertEqual(len(server.requests), 7)
            self.assertEqual(c.failed_calls, 0)
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
