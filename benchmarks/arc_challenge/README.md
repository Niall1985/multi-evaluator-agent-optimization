# ARC-AGI Benchmark (`benchmarks/arc_challenge`)

Adapts tasks from the [ARC-AGI benchmark](https://arcprize.org/) ("Easy for Humans, Hard for AI") to this
project's `BenchmarkTask` structure so the multi-evaluator optimisation loop can evolve agents on abstract
grid-reasoning puzzles alongside the 7 canonical coding tasks.

## The task format

Every ARC task is a JSON dict with two fields (identical for ARC-AGI-1, ARC-AGI-2 and the Kaggle arc-prize datasets):

```json
{
  "train": [{"input": [[8, 6], [6, 4]], "output": [[8, 6, 8, 6, 8, 6], ...]}, ...],
  "test":  [{"input": [[3, 2], [7, 8]], "output": [[3, 2, 3, 2, 3, 2], ...]}]
}
```

A grid is a rectangular list-of-lists of ints `0-9` (0 = background), from 1x1 up to 30x30. `train`
holds the demonstration pairs (typically 3), `test` the held-out pair(s) (typically 1, sometimes 2).

**Scoring** (per ARC Prize rules): an output is correct only if it is pixel-perfect — same shape, every cell
equal. Two attempts are allowed per test input; a task is solved if either matches (**pass@2**). Pixel
accuracy is a secondary diagnostic only, never the leaderboard metric.

## Mapping onto `BenchmarkTask`

| ARC field | `BenchmarkTask` field | Evaluator that consumes it |
| :--- | :--- | :--- |
| `train` pairs | `test_cases` — `{"args": (input,), "expected": output}` | μ₁ Functional Correctness (fit the demonstrations) |
| `test` pairs | `edge_cases` — same shape | μ₅ Edge-Case Stress (generalise to the held-out input) |
| task id `00576224` | `id="arc_00576224"`, `name="ARC 00576224: …"` | task selector / CSV `task_id` |
| — | `entry_point="solve"`, `solve(grid) -> grid` | `CodeExecutionRunner` |
| rendered grids | `description` (prompt shown to the agent) | agent LLM, μ₄ judge |

Only the demonstration pairs are used for the agent's fitness signal (μ₁); held-out pairs are scored
separately (μ₅), which mirrors the OpenEvolve split of evolving on `train` and evaluating on `test`.
`CodeExecutionRunner` normalises numpy arrays via `.tolist()` before comparing, so `solve` may return either.

## Files

```
benchmarks/arc_challenge/
├── tasks/*.json      # bundled ARC-AGI-1 public tasks (raw, unmodified)
├── loader.py         # JSON -> BenchmarkTask; bundled-dir loader; Kaggle split loader
├── scoring.py        # pass@2 / pixel-accuracy helpers + OpenEvolve-style metrics dict
└── README.md
```

Bundled tasks (from `fchollet/ARC-AGI`, chosen for being small and easy to inspect):

| Task | Split | Rule |
| :--- | :--- | :--- |
| [`00576224`](https://arcprize.org/tasks/00576224/) | evaluation | tile 2x2 → 6x6, middle band mirrored (the example used in the OpenEvolve docs) |
| `25ff71a9` | training | shift pattern down one row |
| `3c9b0459` | training | rotate 180° |
| `c8f0f002` | training | recolour 7 → 5 |

## Usage

```python
from benchmarks.arc_challenge import ARC_TASKS, get_arc_task, load_kaggle_split, score_arc_program
from controller import EvolutionController

# Bundled tasks are also reachable through the normal task lookup and the Streamlit dropdown
ctrl = EvolutionController(selected_task_id="arc_00576224")
ctrl.run_generation()

# Full Kaggle dataset (https://www.kaggle.com/competitions/arc-prize-2025/data), same layout OpenEvolve uses
tasks = load_kaggle_split("data/arc-prize-2025", split="evaluation", limit=10)   # order == arcprize.org/tasks

# Optional pass@2 scoring for programs exposing transform_grid_attempt_1/2 (or a lone solve)
metrics = score_arc_program(get_arc_task("00576224"), program_text, use_held_out=True)
# -> {"runs_successfully": 1.0, "combined_score": 1.0, "test_example_0_pass_at_2": 1,
#     "test_example_0_attempt_0": True, "test_example_0_attempt_1": True, ...}
```

To add more tasks, drop `<task_id>.json` files into `tasks/` (e.g. from
`https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/{training|evaluation}/<id>.json`) and
optionally add a label in `loader.ARC_TASK_LABELS`.

## Offline / mock mode

`GroqLLMClient` in mock mode recognises the `ARC-AGI task <id>` prompt header and returns hand-written
solutions for the four bundled tasks (identity transform for any other id), consistent with how it mocks
the seven coding tasks. Real evaluation of unseen ARC tasks requires a live `GROQ_API_KEY`.

## Design notes / differences from OpenEvolve's example

- OpenEvolve evolves a program with **two** functions (`transform_grid_attempt_1/2`) and its evaluator computes
  pass@2 over them. This project's runner is single-entry-point (`solve`), so the default μ₁/μ₅ path is
  effectively pass@1 per pair; `scoring.score_arc_program` provides the two-attempt path when wanted.
- OpenEvolve reads the Kaggle `arc-agi_{split}_challenges.json` (+ `_solutions.json` for test outputs)
  via `DATA_ROOT` / `TASK_FILE` / `TASK_NUM` env vars; `load_kaggle_split` reads the same files, keeping task
  order so index `N` matches "Task N" on arcprize.org.
- Prompt wording avoids `target`, `differ`, `stock`, `fib`, `bracket`, `parenthes`: the mock client keyword-matches
  on those to pick canonical-task solutions (see `tests/test_arc_challenge.py`).

## Sources

- ARC Prize — task format, datasets, scoring: https://arcprize.org/guide/1
- ARC-AGI-1 data & README (JSON structure, 400 train / 400 eval): https://github.com/fchollet/ARC-AGI
- ARC-AGI-2 repo (same format): https://github.com/arcprize/ARC-AGI-2
- ARC-AGI-2 announcement (pass@2 rationale): https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025
- OpenEvolve ARC example (evaluator, seed program): https://github.com/codelion/openevolve/tree/main/examples/arc_benchmark
- Kaggle dataset: https://www.kaggle.com/competitions/arc-prize-2025/data
