"""Headless run of the whole ARC-AGI benchmark against the live Groq API.

Does what "Run N Gen" does in the Streamlit UI with the ARC benchmark + ARC evaluator suite
selected, but prints everything so a run can be inspected: per-generation mutation, active
evaluator subset, fitness, dF, cost, averaged metrics and per-task pass@2, then the best
agent's program for every task.

    uv run python scripts/live_arc_run.py                 # 5 generations
    uv run python scripts/live_arc_run.py --generations 10 --strategy no_pruning --delay 2.0

Reads GROQ_API_KEY from .env (falls back to the deterministic mock if unset). The run's CSV is
written next to the normal one as `live_arc_results.csv` (git-ignored via *_results.csv).
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.arc_challenge import ARC_BENCHMARK_ID, build_arc_evaluator_pool  # noqa: E402
from controller import EvolutionController  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--strategy", default="full_adaptive",
                        choices=["full_adaptive", "no_pruning", "ucb1_bandit", "static_cascade", "single_metric"])
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between LLM calls (rate-limit pacing)")
    parser.add_argument("--archetype", default="General Balanced Assistant")
    parser.add_argument("--csv", default="live_arc_results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ctrl = EvolutionController(
        selected_task_id=ARC_BENCHMARK_ID,
        evaluator_pool_factory=build_arc_evaluator_pool,
        inter_call_delay=args.delay,
        default_strategy=args.strategy,
        initial_archetype=args.archetype,
    )
    ctrl.csv_filepath = args.csv
    print("mock mode:", ctrl.llm_client.is_mock, "| model:", ctrl.llm_client.model, flush=True)

    seed = ctrl.history_records[0]
    print(f"gen 0 seed  fitness={seed['fitness']:.4f} metrics={seed['metrics']}", flush=True)
    print("   per task pass@2 test:", {k: v.get("arc_pass_at_2_test") for k, v in seed["task_metrics"].items()}, flush=True)

    for _ in range(args.generations):
        r = ctrl.run_generation()
        print(
            f"gen {r['generation']} {r['mutation_type']:<22} active={len(r['active_evaluators'])} "
            f"fitness={r['fitness']:.4f} dF={r['delta_f']:+.4f} cost=${r['cost_spent']:.4f} metrics={r['metrics']}",
            flush=True,
        )
        print("   per task pass@2 train:", {k: v.get("arc_pass_at_2_train") for k, v in r["task_metrics"].items()}, flush=True)

    best = ctrl.archive.get_best_agent()
    print("\nBEST:", best.id, f"fitness={best.fitness:.4f}", best.metrics, flush=True)
    print("telemetry:", json.dumps(ctrl.get_telemetry_summary(), indent=1, default=str), flush=True)
    for tid, sol in best.task_solutions.items():
        print(f"\n----- {tid} -----\n{sol[:700]}", flush=True)


if __name__ == "__main__":
    main()
