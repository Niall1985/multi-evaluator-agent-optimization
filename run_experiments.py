import argparse
import logging
import random
import time
import numpy as np
import pandas as pd
from typing import List, Optional

from controller import EvolutionController
from tasks.benchmark_tasks import BENCHMARK_TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ALL_CONDITIONS = [
    "single_metric",
    "static_cascade",
    "ucb1_bandit",
    "no_pruning",
    "full_adaptive",
]

ALL_TASKS = [t.id for t in BENCHMARK_TASKS]


def run_experiment_suite(
    conditions: List[str],
    tasks: List[str],
    num_runs: int = 5,
    num_generations: int = 30,
    api_key: Optional[str] = None,
    model: str = "openai/gpt-oss-120b",
    inter_call_delay: float = 0.0,
    is_mock: Optional[bool] = None,
    output_csv: str = "experiment_results.csv",
) -> pd.DataFrame:
    """Executes the full experiment suite across conditions, tasks, and independent seed runs."""
    all_frames = []
    total_experiments = len(conditions) * len(tasks) * num_runs
    current_experiment = 0

    logger.info(
        f"Starting experiment suite: {len(conditions)} conditions x {len(tasks)} tasks x {num_runs} runs "
        f"= {total_experiments} total runs ({num_generations} generations each)."
    )

    for c_idx, condition in enumerate(conditions, 1):
        for t_idx, task_id in enumerate(tasks, 1):
            for run_idx in range(num_runs):
                current_experiment += 1
                seed = 42 + run_idx * 1000 + c_idx * 100 + t_idx
                random.seed(seed)
                np.random.seed(seed)

                logger.info(
                    f"[{current_experiment}/{total_experiments}] Running condition='{condition}', "
                    f"task='{task_id}', run={run_idx + 1}/{num_runs}, seed={seed}"
                )

                controller = EvolutionController(
                    api_key=api_key,
                    model=model,
                    default_strategy=condition,
                    selected_task_id=task_id,
                    inter_call_delay=inter_call_delay,
                    is_mock=is_mock,
                )

                for gen in range(1, num_generations + 1):
                    controller.run_generation(strategy=condition, task_id=task_id)

                # Extract detailed dataframe
                df_run = controller.archive.to_detailed_dataframe()
                df_run["run_index"] = run_idx
                df_run["condition"] = condition
                all_frames.append(df_run)

                # Checkpoint save after each run
                combined_df = pd.concat(all_frames, ignore_index=True)
                combined_df.to_csv(output_csv, index=False)

    logger.info(f"Experiment suite complete. Saved {len(combined_df)} total records to '{output_csv}'.")
    return combined_df


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Objective Agent Optimization Experiment Harness")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent seed runs per condition/task (default: 5)")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations per run (default: 30)")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help=f"List of conditions to run (default: {ALL_CONDITIONS})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        choices=ALL_TASKS,
        help=f"List of task IDs to run (default: {ALL_TASKS})",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Groq API Key (optional, defaults to mock)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--delay", type=float, default=0.0, help="Inter-call delay in seconds (default: 0.0)")
    parser.add_argument("--output", type=str, default="experiment_results.csv", help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment_suite(
        conditions=args.conditions,
        tasks=args.tasks,
        num_runs=args.runs,
        num_generations=args.generations,
        api_key=args.api_key,
        model=args.model,
        inter_call_delay=args.delay,
        output_csv=args.output,
    )
