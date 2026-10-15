"""
ARC-AGI scoring helpers, following the official rules (pixel-perfect match,
up to 2 attempts per test input => pass@2) and the metrics layout used by
OpenEvolve's arc_benchmark evaluator (`combined_score`, `runs_successfully`,
`<example>_pass_at_2`, `<example>_attempt_<n>`).

`score_arc_cases` / `score_arc_program` accept programs exposing
`transform_grid_attempt_1/2` (OpenEvolve seed style) or a lone `solve`, and back
the evaluators in `evaluators.py` (official metrics) and `partial_evaluators.py`.
"""

import numbers
from typing import Any, Dict, List, Optional, Sequence

from evaluation.runner import CodeExecutionRunner
from evaluation.sandbox import run_program
from benchmarks.benchmark_tasks import BenchmarkTask

Grid = List[List[int]]

ATTEMPT_FUNCTIONS = ("transform_grid_attempt_1", "transform_grid_attempt_2")


def _cell(v: Any) -> Optional[int]:
    """An integer cell value (numpy ints and integral floats like 1.0 included); None for anything else.
    `int(v)` alone accepted "3" and truncated 2.9 to 2, matches the official exact comparison would reject."""
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def to_grid(value: Any) -> Optional[Grid]:
    """Normalises numpy arrays / nested sequences to list[list[int]]; None if not grid-like."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    rows: Grid = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            return None
        cells = [_cell(v) for v in row]
        if any(c is None for c in cells):
            return None
        rows.append(cells)
    if len({len(r) for r in rows}) != 1:
        return None
    return rows


def grids_equal(pred: Any, truth: Grid) -> bool:
    g = to_grid(pred)
    return g is not None and g == to_grid(truth)


def pixel_accuracy(pred: Any, truth: Grid) -> float:
    """Fraction of correctly predicted cells; 0.0 on shape mismatch (secondary diagnostic only)."""
    g = to_grid(pred)
    t = to_grid(truth)
    if g is None or t is None or len(g) != len(t) or len(g[0]) != len(t[0]):
        return 0.0
    total = len(t) * len(t[0])
    correct = sum(1 for r in range(len(t)) for c in range(len(t[0])) if g[r][c] == t[r][c])
    return correct / total if total else 0.0


def pass_at_2_single(attempts: Sequence[Any], truth: Grid) -> int:
    """1 if any of (at most) two attempts is pixel-perfect, else 0."""
    return int(any(grids_equal(a, truth) for a in list(attempts)[:2]))


def pass_at_2_multi(attempts_per_example: Sequence[Sequence[Any]], truths: Sequence[Grid]) -> float:
    """Mean pass@2 over examples."""
    if not truths:
        return 0.0
    return sum(pass_at_2_single(a, t) for a, t in zip(attempts_per_example, truths)) / len(truths)


def score_arc_cases(
    cases: Sequence[Dict[str, Any]],
    agent_output: str,
    entry_point: str = "solve",
    prefix: str = "train_example",
) -> Dict[str, Any]:
    """Pass@2-scores `cases` ([{"args": (grid,), "expected": grid}, ...]) with OpenEvolve-style keys.

    Always includes `runs_successfully` and `combined_score` (mean pass@2); per example
    `{prefix}_{i}_pass_at_2`, `{prefix}_{i}_attempt_{j}` and `..._pixel_acc`. Raw attempt
    outputs are returned under `outputs` for partial-credit evaluators to reuse.

    The program runs in a child process (evaluation.sandbox) with a per-call time limit; it uses OpenEvolve's
    two attempt functions when present, else `entry_point`.
    """
    code = CodeExecutionRunner.extract_code(agent_output)
    run = run_program(code, [ATTEMPT_FUNCTIONS[:2], (entry_point,)], [tuple(case["args"]) for case in cases])
    if run.load_error:
        return {"runs_successfully": 0.0, "combined_score": 0.0, "error": run.load_error, "outputs": []}
    if not run.functions:
        err = f"No `{entry_point}` or {ATTEMPT_FUNCTIONS} found."
        return {"runs_successfully": 0.0, "combined_score": 0.0, "error": err, "outputs": []}

    metrics: Dict[str, Any] = {"runs_successfully": 1.0}
    per_example: List[int] = []
    errors: List[str] = []
    outputs: List[List[Any]] = []

    for i, case in enumerate(cases):
        truth = case["expected"]
        attempts: List[Any] = []
        for j, (status, value) in enumerate(run.results[i]):
            out = value if status == "ok" else None
            if status != "ok":
                errors.append(f"{prefix}_{i}_attempt_{j}: {value}")
            attempts.append(out)
            metrics[f"{prefix}_{i}_attempt_{j}"] = grids_equal(out, truth)
            metrics[f"{prefix}_{i}_attempt_{j}_pixel_acc"] = round(pixel_accuracy(out, truth), 4)
        p2 = pass_at_2_single(attempts, truth)
        metrics[f"{prefix}_{i}_pass_at_2"] = p2
        per_example.append(p2)
        outputs.append(attempts)

    metrics["combined_score"] = sum(per_example) / len(per_example) if per_example else 0.0
    if errors:
        metrics["errors"] = errors
    metrics["outputs"] = outputs
    return metrics


def score_arc_program(
    task: BenchmarkTask,
    agent_output: str,
    use_held_out: bool = False,
) -> Dict[str, Any]:
    """Runs a candidate program with pass@2 semantics and returns OpenEvolve-style metrics.

    `use_held_out=False` scores the demonstration (train) pairs; `True` scores the held-out (test) pairs.
    """
    cases = task.edge_cases if use_held_out else task.test_cases
    prefix = "test_example" if use_held_out else "train_example"
    metrics = score_arc_cases(cases, agent_output, task.entry_point, prefix)
    metrics.pop("outputs", None)
    return metrics
