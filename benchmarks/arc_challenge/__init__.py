"""ARC-AGI (Abstraction and Reasoning Corpus) benchmark adapted to BenchmarkTask."""

from .loader import (
    ARC_DATA_ROOT,
    ARC_TASKS,
    DEFAULT_DATA_ROOT,
    arc_task_to_benchmark,
    available_splits,
    get_arc_task,
    load_arc_task_file,
    load_arc_tasks,
    load_kaggle_split,
    render_grid,
)
from .scoring import (
    grids_equal,
    pass_at_2_single,
    pass_at_2_multi,
    pixel_accuracy,
    score_arc_cases,
    score_arc_program,
)
from .evaluators import (
    REAL_ARC_EVALUATORS,
    ArcHeldOutPassAt2Evaluator,
    ArcPassAt2Evaluator,
    ArcRunsSuccessfullyEvaluator,
)
from .partial_evaluators import (
    AI_GENERATED_HEADER,
    PARTIAL_ARC_EVALUATORS,
    ArcColorPaletteEvaluator,
    ArcPixelAccuracyEvaluator,
    ArcShapeMatchEvaluator,
)
from .pool import build_arc_evaluator_pool
from .viz import (
    ARC_PALETTE,
    benchmark_summary_frame,
    pass_at_2_trajectory_figure,
    task_gallery_figure,
)

__all__ = [
    "ARC_DATA_ROOT",
    "ARC_TASKS",
    "DEFAULT_DATA_ROOT",
    "arc_task_to_benchmark",
    "available_splits",
    "get_arc_task",
    "load_arc_task_file",
    "load_arc_tasks",
    "load_kaggle_split",
    "render_grid",
    "grids_equal",
    "pass_at_2_single",
    "pass_at_2_multi",
    "pixel_accuracy",
    "score_arc_cases",
    "score_arc_program",
    "REAL_ARC_EVALUATORS",
    "ArcHeldOutPassAt2Evaluator",
    "ArcPassAt2Evaluator",
    "ArcRunsSuccessfullyEvaluator",
    "AI_GENERATED_HEADER",
    "PARTIAL_ARC_EVALUATORS",
    "ArcColorPaletteEvaluator",
    "ArcPixelAccuracyEvaluator",
    "ArcShapeMatchEvaluator",
    "build_arc_evaluator_pool",
    "ARC_PALETTE",
    "benchmark_summary_frame",
    "pass_at_2_trajectory_figure",
    "task_gallery_figure",
]
