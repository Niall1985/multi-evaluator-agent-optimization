"""Benchmark suites for Multi-Objective Agent Optimization.

- `benchmark_tasks`: the 7 canonical HumanEval / MBPP / algorithmic tasks (+ BenchmarkTask dataclass)
- `arc_challenge`:   ARC-AGI grid tasks, evaluators and visualisations
"""

from .benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask, get_benchmark_task, get_benchmark_tasks, get_random_task

__all__ = [
    "BENCHMARK_TASKS",
    "BenchmarkTask",
    "get_benchmark_task",
    "get_benchmark_tasks",
    "get_random_task",
]
