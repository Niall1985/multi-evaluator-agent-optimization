"""
Loads ARC-AGI tasks and adapts them to the project's BenchmarkTask structure.

ARC task JSON (fchollet/ARC-AGI, arcprize/ARC-AGI-2, Kaggle arc-prize-20xx):
    {
        "train": [{"input": Grid, "output": Grid}, ...],   # demonstration pairs
        "test":  [{"input": Grid, "output": Grid}, ...]    # held-out pairs
    }
A Grid is a rectangular list-of-lists of ints 0-9, from 1x1 up to 30x30.

Mapping onto BenchmarkTask:
    train pairs -> test_cases  (mu_1 Functional Correctness: fit the demonstrations)
    test  pairs -> edge_cases  (mu_5 Edge-Case Stress: generalise to the held-out input)
    entry_point -> solve(grid: list[list[int]]) -> list[list[int]]
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tasks.benchmark_tasks import BenchmarkTask

Grid = List[List[int]]

ARC_TASKS_DIR = Path(__file__).parent / "tasks"
ARC_ID_PREFIX = "arc_"

# Human-readable labels for the bundled tasks (ARC-AGI-1 public data).
ARC_TASK_LABELS: Dict[str, str] = {
    "00576224": "Tile 2x2 into 6x6 with alternating mirrored rows",
    "25ff71a9": "Shift pattern down by one row",
    "3c9b0459": "Rotate grid 180 degrees",
    "c8f0f002": "Recolour 7 to 5",
}


def render_grid(grid: Grid) -> str:
    """Renders a grid as compact rows of digits, e.g. '8 6\\n6 4'."""
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def _grid_shape(grid: Grid) -> str:
    return f"{len(grid)}x{len(grid[0]) if grid else 0}"


def _build_description(task_id: str, train: List[Dict[str, Grid]], test: List[Dict[str, Grid]]) -> str:
    # NOTE: wording deliberately avoids keywords the offline mock client matches on
    # ("target", "differ", "stock", "fib", ...) so ARC prompts route to the ARC mock branch.
    lines = [
        f"ARC-AGI task {task_id}. Write a Python function `solve(grid: list[list[int]]) -> list[list[int]]` "
        "that applies the single hidden transformation shown by the demonstration pairs below to any new input grid. "
        "Grids are lists of rows of ints 0-9 (0 = background). The returned grid must match the expected output "
        "exactly in shape and every cell. Return a plain list of lists (not a numpy array).",
        "",
    ]
    for i, pair in enumerate(train):
        lines.append(f"Demonstration {i + 1} input ({_grid_shape(pair['input'])}):")
        lines.append(render_grid(pair["input"]))
        lines.append(f"Demonstration {i + 1} output ({_grid_shape(pair['output'])}):")
        lines.append(render_grid(pair["output"]))
        lines.append("")
    for i, pair in enumerate(test):
        lines.append(f"Held-out input {i + 1} ({_grid_shape(pair['input'])}):")
        lines.append(render_grid(pair["input"]))
        lines.append("")
    return "\n".join(lines).rstrip()


def arc_task_to_benchmark(
    arc_id: str,
    data: Dict[str, Any],
    source: str = "arc-agi-1",
    test_solutions: Optional[List[Grid]] = None,
) -> BenchmarkTask:
    """Converts a raw ARC task dict into a BenchmarkTask.

    `test_solutions` supplies held-out outputs when they live in a separate
    solutions file (Kaggle layout). Test pairs without an output are skipped
    from edge_cases since they cannot be scored.
    """
    train = data.get("train", [])
    test = list(data.get("test", []))
    if test_solutions:
        test = [
            {**pair, "output": test_solutions[i]}
            for i, pair in enumerate(test)
            if i < len(test_solutions)
        ]

    test_cases = [{"args": (pair["input"],), "expected": pair["output"]} for pair in train]
    edge_cases = [
        {"args": (pair["input"],), "expected": pair["output"]}
        for pair in test
        if "output" in pair
    ]

    label = ARC_TASK_LABELS.get(arc_id, "Grid transformation")
    return BenchmarkTask(
        id=f"{ARC_ID_PREFIX}{arc_id}",
        name=f"ARC {arc_id}: {label}",
        category=f"ARC-AGI Abstract Reasoning ({source})",
        description=_build_description(arc_id, train, test),
        entry_point="solve",
        test_cases=test_cases,
        edge_cases=edge_cases,
        constraints=(
            "Output must be pixel-perfect: identical shape and cell values (ints 0-9) to the expected grid. "
            "Return list[list[int]]. Must generalise from the demonstration pairs to the held-out input; "
            "no hard-coding of the expected outputs. Pure Python / numpy only."
        ),
    )


def load_arc_task_file(path: os.PathLike, source: str = "arc-agi-1") -> BenchmarkTask:
    """Loads one `<task_id>.json` file (fchollet/ARC-AGI layout)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return arc_task_to_benchmark(path.stem, data, source=source)


def load_bundled_arc_tasks(tasks_dir: os.PathLike = ARC_TASKS_DIR) -> List[BenchmarkTask]:
    """Loads every task JSON shipped in benchmarks/arc_challenge/tasks/."""
    tasks_dir = Path(tasks_dir)
    return [load_arc_task_file(p) for p in sorted(tasks_dir.glob("*.json"))]


def load_kaggle_split(
    data_root: os.PathLike,
    split: str = "evaluation",
    limit: Optional[int] = None,
) -> List[BenchmarkTask]:
    """Loads a Kaggle arc-prize split (same layout OpenEvolve's example uses).

    Expects `arc-agi_{split}_challenges.json` and, optionally,
    `arc-agi_{split}_solutions.json` under `data_root`. Task order follows the
    challenges file, matching the numbering on https://arcprize.org/tasks/.
    """
    data_root = Path(data_root)
    with open(data_root / f"arc-agi_{split}_challenges.json", "r", encoding="utf-8") as f:
        challenges: Dict[str, Any] = json.load(f)

    solutions: Dict[str, List[Grid]] = {}
    sol_path = data_root / f"arc-agi_{split}_solutions.json"
    if sol_path.exists():
        with open(sol_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

    tasks = []
    for i, (arc_id, data) in enumerate(challenges.items()):
        if limit is not None and i >= limit:
            break
        tasks.append(arc_task_to_benchmark(arc_id, data, source=f"arc-prize {split}", test_solutions=solutions.get(arc_id)))
    return tasks


ARC_TASKS: List[BenchmarkTask] = load_bundled_arc_tasks()


def get_arc_task(task_id: str) -> Optional[BenchmarkTask]:
    """Looks up a bundled ARC task by BenchmarkTask id, raw ARC id, or display name."""
    for t in ARC_TASKS:
        if task_id in (t.id, t.name) or t.id == f"{ARC_ID_PREFIX}{task_id}":
            return t
    return None
