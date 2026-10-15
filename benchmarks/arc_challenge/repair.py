"""Program-archive hooks for ARC: where a program fails on the demonstration pairs, and the repair prompt.

Only the demonstration pairs (`task.test_cases`) feed prompts and selection. The held-out outputs never reach
a prompt or a selection decision, which is what lets the run evolve on the evaluation split itself; they are
read only by archive_pass_at_2, to score the two programs picked from the archive.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.benchmark_tasks import BenchmarkTask
from evaluation.runner import CodeExecutionRunner
from evaluation.sandbox import run_program

from .loader import render_grid
from .scoring import ATTEMPT_FUNCTIONS, pass_at_2_single, score_arc_cases, to_grid

# Cap on failing pairs shown per program: each expected/produced pair can be two 30x30 grids
MAX_FEEDBACK_PAIRS = 3
MAX_ERROR_CHARS = 300


@dataclass
class ProgramInspection:
    cell: str  # MAP-Elites cell: demonstration pass mask + count of right-shaped outputs, e.g. "101|s3"
    solved: bool  # passes every demonstration pair
    passed: int
    total: int
    feedback: str


def _shape(grid) -> str:
    g = to_grid(grid)
    return f"{len(g)}x{len(g[0])}" if g else "none"


def inspect_program(task: BenchmarkTask, solution: str) -> ProgramInspection:
    cases = task.test_cases
    metrics = score_arc_cases(cases, solution, task.entry_point, prefix="train_example")
    total = len(cases)
    if metrics.get("error") or not metrics.get("outputs"):
        error = str(metrics.get("error") or "program produced no output")[:MAX_ERROR_CHARS]
        return ProgramInspection(cell="x|s0", solved=False, passed=0, total=total,
                                 feedback=f"The program does not run: {error}")

    outputs = metrics["outputs"]
    mask, right_shape, lines = [], 0, []
    for i, (case, attempts) in enumerate(zip(cases, outputs)):
        ok = bool(metrics.get(f"train_example_{i}_pass_at_2"))
        mask.append("1" if ok else "0")
        produced = next((a for a in attempts if a is not None), None)
        if to_grid(produced) is not None and _shape(produced) == _shape(case["expected"]):
            right_shape += 1
        if ok or len(lines) >= MAX_FEEDBACK_PAIRS * 4:
            continue
        lines.append(f"Demonstration {i + 1}: expected ({_shape(case['expected'])}):")
        lines.append(render_grid(case["expected"]))
        if to_grid(produced) is None:
            errs = [e for e in metrics.get("errors", []) if e.startswith(f"train_example_{i}_")]
            lines.append("got: " + (errs[0].split(": ", 1)[-1][:MAX_ERROR_CHARS] if errs else "no valid grid"))
        else:
            lines.append(f"got ({_shape(produced)}):")
            lines.append(render_grid(to_grid(produced)))
    passed = mask.count("1")
    header = f"Passes {passed} of {total} demonstrations."
    return ProgramInspection(
        cell=f"{''.join(mask)}|s{right_shape}",
        solved=passed == total and total > 0,
        passed=passed,
        total=total,
        feedback="\n".join([header, *lines]),
    )


def build_repair_prompt(
    task: BenchmarkTask,
    parent_solution: Optional[str],
    parent_feedback: str = "",
    inspirations: Sequence[str] = (),
) -> str:
    """The user message for one call. The task comes first so repeated calls on a task share a prompt prefix
    (vLLM prefix caching); with no parent it is the plain task, i.e. a fresh attempt."""
    if parent_solution is None:
        return task.description
    parts: List[str] = [
        task.description,
        "",
        "Current program:",
        "```python",
        CodeExecutionRunner.extract_code(parent_solution).strip(),
        "```",
        "",
        parent_feedback,
    ]
    for j, other in enumerate(inspirations):
        parts += ["", f"Another program from the archive ({j + 1}), for ideas:", "```python",
                  CodeExecutionRunner.extract_code(other).strip(), "```"]
    parts += ["", "Fix the current program so it reproduces every demonstration output exactly. "
                  "Answer with the complete program as one ```python block defining solve(grid)."]
    return "\n".join(parts)


def held_out_outputs(task: BenchmarkTask, solution: str) -> Optional[tuple]:
    """The program's first valid output on each held-out input (no expected outputs are read), or None if it
    produces no grid for some input. Hashable, so two programs can be compared for distinct answers."""
    code = CodeExecutionRunner.extract_code(solution)
    run = run_program(code, [ATTEMPT_FUNCTIONS[:2], (task.entry_point,)], [tuple(c["args"]) for c in task.edge_cases])
    if run.load_error or not run.functions:
        return None
    outputs = []
    for results in run.results:
        grid = next((to_grid(v) for status, v in results if status == "ok" and to_grid(v) is not None), None)
        if grid is None:
            return None
        outputs.append(tuple(tuple(row) for row in grid))
    return tuple(outputs)


def archive_pass_at_2(task: BenchmarkTask, ranked_solutions: Sequence[Tuple[str, str]]) -> Dict[str, object]:
    """Official pass@2 with the two attempts taken from the archive.

    `ranked_solutions` is [(program_id, solution), ...], best first, ranked on the demonstration pairs only.
    Attempt 1 is the best program that produces a grid for every held-out input; attempt 2 the next one whose
    answers differ (a second identical answer wastes the attempt). The expected outputs are read only to
    score the two picks."""
    picks: List[Tuple[str, tuple]] = []
    for program_id, solution in ranked_solutions:
        outputs = held_out_outputs(task, solution)
        if outputs is None or any(outputs == seen for _, seen in picks):
            continue
        picks.append((program_id, outputs))
        if len(picks) == 2:
            break
    per_input = []
    for i, case in enumerate(task.edge_cases):
        attempts = [[list(row) for row in outputs[i]] for _, outputs in picks]
        per_input.append(pass_at_2_single(attempts, case["expected"]))
    score = sum(per_input) / len(per_input) if per_input else 0.0
    return {"score": score, "solved": bool(per_input) and score == 1.0,
            "attempt_program_ids": [pid for pid, _ in picks]}
