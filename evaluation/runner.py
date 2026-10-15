import os
import time
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from tasks.benchmark_tasks import BenchmarkTask

_CACHED_SENTI_TOOLKIT = None


class FallbackSentiToolKit:
    """Deterministic fallback sentiment classifier matching SentiToolKit interface."""
    def __call__(self, text: str) -> str:
        t = str(text).lower()
        if any(w in t for w in ["great", "excellent", "love", "good", "amazing", "smooth", "pleasant", "impressed", "wonderful", "flawless", "helpful"]):
            if "not good" in t or "not great" in t or "not impressive" in t or "disappoint" in t:
                return "Negative"
            return "Positive"
        if any(w in t for w in ["bad", "terrible", "awful", "horrible", "defect", "fail", "slow", "poor", "freeze", "crash", "worst", "unacceptable"]):
            if "not bad" in t:
                return "Positive"
            return "Negative"
        return "Neutral"


def get_cached_senti_toolkit():
    """Singleton cached instance of user's SentiToolKit to prevent re-initializing Keras graph."""
    global _CACHED_SENTI_TOOLKIT
    if _CACHED_SENTI_TOOLKIT is None:
        try:
            from SentiAnalyzer.Main_tensor_model import SentiToolKit
            _CACHED_SENTI_TOOLKIT = SentiToolKit()

            # Optimize __call__ with direct tensor inference (avoids slow progress bar & builds)
            def _fast_call(sentence):
                prepared = _CACHED_SENTI_TOOLKIT.prepare_text(sentence)
                probs = _CACHED_SENTI_TOOLKIT.model(prepared, training=False).numpy()[0]
                if probs[2] > 0.7:
                    return 'Positive'
                elif probs[1] > 0.4:
                    return 'Neutral'
                else:
                    return 'Negative'

            _CACHED_SENTI_TOOLKIT.__call__ = _fast_call

            try:
                import SentiAnalyzer.Main_tensor_model
                SentiAnalyzer.Main_tensor_model.SentiToolKit = lambda *a, **kw: _CACHED_SENTI_TOOLKIT
            except Exception:
                pass
        except BaseException:
            _CACHED_SENTI_TOOLKIT = FallbackSentiToolKit()
    return _CACHED_SENTI_TOOLKIT


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
    """Executes a candidate solution against BenchmarkTask test cases and edge cases across

    Coding and Sentiment Analysis (SentiToolKit) domains.
    """

    def __init__(self, timeout_sec: float = 2.0):
        self.timeout_sec = timeout_sec

    @staticmethod
    def extract_code(text: str, entry_point: str = "solve") -> str:
        """Extracts pure python code from LLM responses.

        Reasoning models routinely emit several fenced blocks and leave them
        untagged, with prose snippets ahead of the real answer. Taking the first
        fence then yields a fragment like ``nums[i] + nums[j] == target``, so
        prefer the block that actually defines the entry point.
        """
        tags = [
            (tag or "").strip().lower()
            for tag in re.findall(r"```([A-Za-z0-9_+-]*)", text)
        ]
        bodies = re.findall(r"```[A-Za-z0-9_+-]*[^\S\n]*\n?(.*?)```", text, re.DOTALL)
        tagged = list(zip(tags, bodies)) if len(tags) == len(bodies) else [("", b) for b in bodies]

        python_first = [b for tag, b in tagged if tag in ("python", "py")] + [
            b for tag, b in tagged if tag not in ("python", "py")
        ]

        def _pick(predicate):
            for body in python_first:
                if predicate(body):
                    return body.strip()
            return None

        # 1. A block defining the requested entry point.
        candidate = _pick(
            lambda b: re.search(
                r"^\s*(?:async\s+)?def\s+" + re.escape(entry_point) + r"\s*\(", b, re.MULTILINE
            )
        )
        # 2. Any block defining a function or class.
        candidate = candidate or _pick(
            lambda b: re.search(r"^\s*(?:async\s+)?(?:def|class)\s+\w+", b, re.MULTILINE)
        )
        # 3. Any non-empty block.
        candidate = candidate or _pick(lambda b: b.strip())
        if candidate:
            return candidate

        # 4. No usable fence: the response may already be bare code, and the
        #    caller still has _extract_direct_answer as a fallback.
        return text

    @classmethod
    def _are_values_equal(cls, actual: Any, expected: Any) -> bool:
        """Checks equality across numbers, floats, sentiment strings, and MCQ/FIB answers."""
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False

        # Direct equality
        if actual == expected:
            return True

        # Float comparison with tolerance
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-5)

        # Numeric string comparison (e.g. "89" == 89 or "89.0" == 89)
        if isinstance(expected, (int, float)) or (isinstance(expected, str) and expected.replace(".", "", 1).lstrip("-").isdigit()):
            try:
                exp_f = float(expected)
                act_str = str(actual).strip().strip("`*\"'").rstrip(".").split()[0]
                if math.isclose(float(act_str), exp_f, rel_tol=1e-5, abs_tol=1e-5):
                    return True
            except (ValueError, IndexError):
                pass

        # String normalization (case-insensitive, punctuation stripped)
        str_act = str(actual).strip().strip("`*\"'").strip("()").strip()
        str_exp = str(expected).strip().strip("`*\"'").strip("()").strip()

        if str_act.lower() == str_exp.lower():
            return True

        # Sentiment normalization ('positive', 'negative', 'neutral')
        if str_exp.lower() in ["positive", "negative", "neutral"]:
            clean_act = str_act.lower()
            if str_exp.lower() in clean_act:
                return True

        # MCQ Option normalization ('A', 'B', 'C', 'D')
        if str_exp.upper() in ["A", "B", "C", "D"]:
            match = re.search(r"\b([A-D])\b", str_act.upper())
            if match and match.group(1) == str_exp.upper():
                return True

        return False

    @classmethod
    def _extract_direct_answer(cls, text: str, expected: Any) -> Any:
        """Fallback extractor when agent outputs textual answer without def solve()."""
        clean_text = text.strip()

        # Sentiment Extraction
        if isinstance(expected, str) and expected.lower() in ["positive", "negative", "neutral"]:
            match = re.search(r"\b(positive|negative|neutral)\b", clean_text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()

        # MCQ Option Extraction ('A', 'B', 'C', 'D')
        if isinstance(expected, str) and expected.upper() in ["A", "B", "C", "D"]:
            match = re.search(r"(?:answer|option|choice)?\s*[:\-\)]?\s*\(?([A-D])\)?", clean_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # FIB / Numeric answer extraction
        if isinstance(expected, (int, float)):
            match = re.search(r"(?:answer|result|is|equals)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)", clean_text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                return int(val) if val.is_integer() else val

        return clean_text

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
        code_str = cls.extract_code(
            raw_code, entry_point=entry_point or (task.entry_point if task else "solve")
        )
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

        # Compile and execute code in isolated namespace with pre-warmed libraries
        namespace: Dict[str, Any] = {
            "math": math,
            "re": re,
            "time": time,
        }

        # Wire SentiToolKit into the execution namespace for sentiment tasks
        if getattr(task, "domain", "coding") == "sentiment":
            cached_senti = get_cached_senti_toolkit()
            if cached_senti is not None:
                namespace["senti_model"] = cached_senti
                namespace["SentiToolKit"] = lambda *a, **kw: cached_senti
                try:
                    import sys
                    if "SentiAnalyzer.Main_tensor_model" in sys.modules:
                        sys.modules["SentiAnalyzer.Main_tensor_model"].SentiToolKit = lambda *a, **kw: cached_senti
                except Exception:
                    pass

        code_compiled = False
        try:
            exec(code_str, namespace)
            code_compiled = True
        except Exception as e:
            report.syntax_error = f"{type(e).__name__}: {str(e)}"

        solve_fn = namespace.get(fn_entry) or namespace.get("solve") if code_compiled else None
        is_callable = callable(solve_fn)

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

            if is_callable:
                try:
                    actual = solve_fn(*raw_args)
                    case_res.actual = actual

                    if not expect_exception and cls._are_values_equal(actual, expected):
                        case_res.passed = True
                        passed_count += 1
                    elif expect_exception:
                        case_res.passed = False
                except Exception as e:
                    case_res.exception_raised = f"{type(e).__name__}: {str(e)}"
                    if expect_exception:
                        case_res.passed = True
                        passed_count += 1
                    else:
                        case_res.passed = False
            else:
                # Fallback: direct text answer extraction
                actual = cls._extract_direct_answer(raw_code, expected)
                case_res.actual = actual
                if cls._are_values_equal(actual, expected):
                    case_res.passed = True
                    passed_count += 1
                else:
                    case_res.error = report.syntax_error or f"Function '{fn_entry}' not callable and answer could not be extracted."

            report.case_results.append(case_res)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        report.latency_ms = elapsed_ms
        report.passed_count = passed_count
        report.pass_rate = float(passed_count / report.total_count) if report.total_count > 0 else 0.0

        return report
