import ast
import time
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from benchmarks.benchmark_tasks import BenchmarkTask


@dataclass
class CaseExecutionResult:
    case_index: int
    case_type: str  # 'test' or 'edge'
    args: Tuple[Any, ...]
    expected: Any
    actual: Any = None
    passed: bool = False
    is_exception_expected: bool = False
    exception_raised: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionReport:
    passed_count: int = 0
    total_count: int = 0
    pass_rate: float = 0.0
    latency_ms: float = 0.0
    case_results: List[CaseExecutionResult] = field(default_factory=list)
    syntax_error: Optional[str] = None
    code_extracted: str = ""

    @property
    def total_test_cases(self) -> int:
        return sum(1 for c in self.case_results if c.case_type == "test")

    @property
    def passed_test_cases(self) -> int:
        return sum(1 for c in self.case_results if c.case_type == "test" and c.passed)

    @property
    def total_edge_cases(self) -> int:
        return sum(1 for c in self.case_results if c.case_type == "edge")

    @property
    def passed_edge_cases(self) -> int:
        return sum(1 for c in self.case_results if c.case_type == "edge" and c.passed)

    @property
    def errors(self) -> List[str]:
        errs = []
        if self.syntax_error:
            errs.append(self.syntax_error)
        for c in self.case_results:
            if not c.passed:
                errs.append(f"Case {c.case_type} #{c.case_index}: expected {c.expected}, got {c.actual}, err: {c.exception_raised or c.error}")
        return errs


class CodeExecutionRunner:
    """Executes a candidate solution against BenchmarkTask test cases and edge cases."""

    def __init__(self, timeout_sec: float = 2.0):
        self.timeout_sec = timeout_sec

    # Typographic characters LLMs emit in prose that also leak into code (non-breaking hyphen, dashes, smart quotes).
    _UNICODE_PUNCT = str.maketrans({"‑": "-", "‐": "-", "–": "-", "—": "-",
                                    "‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})

    @staticmethod
    def _parses(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @classmethod
    def extract_code(cls, text: str) -> str:
        """Extracts the Python program from an LLM response.

        Responses often contain several fenced blocks (a math/pseudo-code fence first, the real
        ```python block later) or unfenced code after prose, so "first fence wins" is not enough.
        Preference order: a block that parses AND defines a function (python-labelled first), then any
        block that parses, then the whole text if it parses, then the tail from the first `def`/`import`
        line. Typographic hyphens/quotes are normalised only when that is what stops a block parsing.
        """
        fences = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", text, re.DOTALL)
        candidates = [(lang.lower(), body.strip()) for lang, body in fences if body.strip()]
        # python-labelled blocks first, then unlabelled/other, keeping original order within each group
        candidates.sort(key=lambda c: 0 if c[0] in ("python", "py", "python3") else 1)

        def _usable(body: str) -> Optional[str]:
            for variant in (body, body.translate(cls._UNICODE_PUNCT)):
                if cls._parses(variant):
                    return variant
            return None

        parsed = [(lang, u) for lang, body in candidates if (u := _usable(body)) is not None]
        for _lang, code in parsed:
            if re.search(r"^\s*def\s+\w+", code, re.MULTILINE):
                return code
        if parsed:
            return parsed[0][1]

        whole = _usable(text.strip())
        if whole is not None:
            return whole
        # Unfenced code after prose: keep everything from the first code-looking line
        m = re.search(r"^(?:def|import|from|class)\s", text, re.MULTILINE)
        if m:
            tail = _usable(text[m.start():].strip())
            if tail is not None:
                return tail
        if candidates:  # fenced but nothing parses anywhere: first block, as before (caller reports SyntaxError)
            return candidates[0][1]
        return text

    @staticmethod
    def _are_values_equal(actual: Any, expected: Any) -> bool:
        """Checks equality with float tolerance if applicable."""
        # Grid-style outputs (ARC tasks) may come back as numpy arrays; `==` on arrays is
        # element-wise and ambiguous in a bool context, so normalise to nested lists first.
        if hasattr(actual, "tolist"):
            actual = actual.tolist()
        if hasattr(expected, "tolist"):
            expected = expected.tolist()
        if isinstance(actual, float) and isinstance(expected, float):
            return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5)
        return actual == expected

    @classmethod
    def run_task(
        cls,
        task: Optional[BenchmarkTask] = None,
        agent_output: Optional[str] = None,
        include_test_cases: bool = True,
        include_edge_cases: bool = True,
        code: Optional[str] = None,
        entry_point: Optional[str] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        edge_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionReport:
        """Executes candidate code against test_cases and/or edge_cases using normalized args."""
        raw_code = code if code is not None else (agent_output or "")
        code_str = cls.extract_code(raw_code)
        report = ExecutionReport(code_extracted=code_str)

        fn_entry = entry_point or (task.entry_point if task else "solve")
        t_cases = test_cases if test_cases is not None else (task.test_cases if task else [])
        e_cases = edge_cases if edge_cases is not None else (task.edge_cases if task else [])

        cases_to_run: List[Tuple[str, int, Dict[str, Any]]] = []
        if include_test_cases and t_cases:
            for idx, c in enumerate(t_cases):
                cases_to_run.append(("test", idx, c))
        if include_edge_cases and e_cases:
            for idx, c in enumerate(e_cases):
                cases_to_run.append(("edge", idx, c))

        report.total_count = len(cases_to_run)
        if report.total_count == 0:
            report.pass_rate = 1.0
            return report

        # Compile and execute code in isolated namespace
        namespace: Dict[str, Any] = {}
        try:
            exec(code_str, namespace)
        except Exception as e:
            report.syntax_error = f"{type(e).__name__}: {str(e)}"
            for case_type, case_idx, case_dict in cases_to_run:
                report.case_results.append(CaseExecutionResult(
                    case_index=case_idx,
                    case_type=case_type,
                    args=case_dict.get("args", ()),
                    expected=case_dict.get("expected"),
                    passed=False,
                    error=report.syntax_error,
                ))
            return report

        solve_fn = namespace.get(fn_entry) or namespace.get("solve")
        if not callable(solve_fn):
            report.syntax_error = f"Function '{fn_entry}' not found or not callable."
            for case_type, case_idx, case_dict in cases_to_run:
                report.case_results.append(CaseExecutionResult(
                    case_index=case_idx,
                    case_type=case_type,
                    args=case_dict.get("args", ()),
                    expected=case_dict.get("expected"),
                    passed=False,
                    error=report.syntax_error,
                ))
            return report

        start_time = time.perf_counter()
        passed_count = 0

        for case_type, case_idx, case_dict in cases_to_run:
            raw_args = case_dict.get("args", ())
            if not isinstance(raw_args, tuple):
                raw_args = (raw_args,)

            expected = case_dict.get("expected")
            expect_exception = case_dict.get("expect_exception", False)

            case_res = CaseExecutionResult(
                case_index=case_idx,
                case_type=case_type,
                args=raw_args,
                expected=expected,
                is_exception_expected=expect_exception,
            )

            try:
                actual = solve_fn(*raw_args)
                case_res.actual = actual

                if not expect_exception and cls._are_values_equal(actual, expected):
                    case_res.passed = True
                    passed_count += 1
                elif expect_exception:
                    case_res.passed = False  # Exception was expected but not raised
            except Exception as e:
                case_res.exception_raised = f"{type(e).__name__}: {str(e)}"
                if expect_exception:
                    case_res.passed = True
                    passed_count += 1
                else:
                    case_res.passed = False

            report.case_results.append(case_res)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        report.latency_ms = elapsed_ms
        report.passed_count = passed_count
        report.pass_rate = float(passed_count / report.total_count) if report.total_count > 0 else 0.0

        return report
