"""ARC-AGI (Abstraction and Reasoning Corpus) benchmark adapted to BenchmarkTask."""

from .loader import (
    ARC_TASKS,
    ARC_TASKS_DIR,
    arc_task_to_benchmark,
    get_arc_task,
    load_arc_task_file,
    load_bundled_arc_tasks,
    load_kaggle_split,
    render_grid,
)
from .scoring import (
    grids_equal,
    pass_at_2_single,
    pass_at_2_multi,
    pixel_accuracy,
    score_arc_program,
)

__all__ = [
    "ARC_TASKS",
    "ARC_TASKS_DIR",
    "arc_task_to_benchmark",
    "get_arc_task",
    "load_arc_task_file",
    "load_bundled_arc_tasks",
    "load_kaggle_split",
    "render_grid",
    "grids_equal",
    "pass_at_2_single",
    "pass_at_2_multi",
    "pixel_accuracy",
    "score_arc_program",
]
