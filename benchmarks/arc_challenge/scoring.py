"""
ARC-AGI scoring helpers, following the official rules (pixel-perfect match,
up to 2 attempts per test input => pass@2) and the metrics layout used by
OpenEvolve's arc_benchmark evaluator (`combined_score`, `runs_successfully`,
`<example>_pass_at_2`, `<example>_attempt_<n>`).

`score_arc_cases` / `score_arc_program` accept programs exposing
`transform_grid_attempt_1/2` (OpenEvolve seed style) or a lone `solve`, and back
the evaluators in `evaluators.py` (official metrics) and `partial_evaluators.py`.
"""

from typing import Any, Dict, List, Optional, Sequence

from evaluation.runner import CodeExecutionRunner
from benchmarks.benchmark_tasks import BenchmarkTask

Grid = List[List[int]]

ATTEMPT_FUNCTIONS = ("transform_grid_attempt_1", "transform_grid_attempt_2")


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
        try:
            rows.append([int(v) for v in row])
        except (TypeError, ValueError):
            return None
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


def _load_attempt_functions(agent_output: str, entry_point: str = "solve"):
    """Execs candidate code; returns (attempt_fns, error). Prefers OpenEvolve's two attempts, else `solve`."""
    code = CodeExecutionRunner.extract_code(agent_output)
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as e:  # noqa: BLE001 - candidate code is untrusted
        return [], f"{type(e).__name__}: {e}"
    fns = [namespace[n] for n in ATTEMPT_FUNCTIONS if callable(namespace.get(n))]
    if not fns and callable(namespace.get(entry_point)):
        fns = [namespace[entry_point]]
    if not fns:
        return [], f"No `{entry_point}` or {ATTEMPT_FUNCTIONS} found."
    return fns, None


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
    """
    fns, err = _load_attempt_functions(agent_output, entry_point)
    if err:
        return {"runs_successfully": 0.0, "combined_score": 0.0, "error": err, "outputs": []}

    metrics: Dict[str, Any] = {"runs_successfully": 1.0}
    per_example: List[int] = []
    errors: List[str] = []
    outputs: List[List[Any]] = []

    for i, case in enumerate(cases):
        grid_in = case["args"][0]
        truth = case["expected"]
        attempts: List[Any] = []
        for j, fn in enumerate(fns[:2]):
            try:
                out = fn(grid_in)
            except Exception as e:  # noqa: BLE001
                out = None
                errors.append(f"{prefix}_{i}_attempt_{j}: {type(e).__name__}: {e}")
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
