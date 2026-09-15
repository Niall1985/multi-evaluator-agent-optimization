import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.runner import CodeExecutionRunner

extract = CodeExecutionRunner.extract_code
SOLVE = "def solve(grid):\n    return [list(reversed(r)) for r in reversed(grid)]"


class TestCodeExtraction(unittest.TestCase):
    """Shapes of real gpt-oss-120b responses seen while running the ARC benchmark live."""

    def test_plain_python_fence(self):
        self.assertEqual(extract(f"Here you go:\n```python\n{SOLVE}\n```\nDone."), SOLVE)

    def test_math_fence_before_python_fence(self):
        # First fence is pseudo-code with a NON-BREAKING HYPHEN (U+2011): not Python, must be skipped.
        text = (
            "**Solution Explanation**\n\nFormally\n\n```\noutput[i][j] = grid[R‑1‑i][C‑1‑j]      for 0 ≤ i < R\n```\n\n"
            f"So we reverse rows and columns:\n\n```python\n{SOLVE}\n```\n"
        )
        self.assertEqual(extract(text), SOLVE)

    def test_math_fence_then_unfenced_code(self):
        text = (
            "The rule is a 180° rotation.\n\n```\noutput[i][j] = g[R‑1‑i][C‑1‑j]          (0‑based indices)\n```\n\n"
            f"Implementation:\n\n{SOLVE}\n"
        )
        self.assertEqual(extract(text), SOLVE)

    def test_unlabelled_fence_with_code_is_used(self):
        self.assertEqual(extract(f"```\n{SOLVE}\n```"), SOLVE)

    def test_prefers_block_that_defines_a_function(self):
        text = f"```python\nprint('hello')\n```\nand the solution\n```python\n{SOLVE}\n```"
        self.assertEqual(extract(text), SOLVE)

    def test_typographic_hyphen_inside_code_is_normalised(self):
        # U+2011 in an index expression: only fix punctuation when that is what breaks parsing
        text = "```python\ndef solve(grid):\n    n = len(grid)\n    return [grid[n‑1‑i] for i in range(n)]\n```"
        code = extract(text)
        self.assertIn("grid[n-1-i]", code)
        compile(code, "<t>", "exec")

    def test_bare_code_and_empty_response(self):
        self.assertEqual(extract(SOLVE), SOLVE)
        self.assertEqual(extract(""), "")
        self.assertEqual(extract("I cannot solve this."), "I cannot solve this.")  # caller reports the SyntaxError

    def test_canonical_task_mock_outputs_still_extract(self):
        from core.groq_client import GroqLLMClient
        from benchmarks.benchmark_tasks import BENCHMARK_TASKS
        client = GroqLLMClient(is_mock=True)
        for task in BENCHMARK_TASKS:
            code = extract(client.generate("Write code", f"Task: {task.id}\n{task.description}"))
            self.assertTrue(code.startswith("def solve"), task.id)

    def test_unusable_fence_falls_back_to_first_block(self):
        text = "```\nnot python at all ‑‑ ???\n```"
        self.assertEqual(extract(text), "not python at all ‑‑ ???")


if __name__ == "__main__":
    unittest.main()
