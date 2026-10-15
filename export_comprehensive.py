"""Re-export the live ablation into the comprehensive multi-domain benchmark schema.

The experiment driver writes the archive's native per-generation schema. This
script widens it to the comprehensive layout, deriving each additional column
from recorded data rather than inventing it:

  * test-case counts come from mu_1, which is defined as the pass rate, so
    passed = round(rate * total) is exact;
  * per-evaluator costs come from the evaluator registry crossed with which
    evaluators actually ran that generation;
  * the fANOVA block is replayed with the same estimator and the same inputs
    the controller used (metric matrix against fitness, refitted each
    generation once four agents exist).

Usage:
    python export_comprehensive.py [--in results/ablation_live.csv]
                                   [--out results/comprehensive_live_benchmark_results.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd

from core.agent import AGENT_ARCHETYPES
from optimization.fanova import FunctionalANOVA
from tasks.benchmark_tasks import BENCHMARK_TASKS

METRICS = ["mu_1_correctness", "mu_2_latency", "mu_3_conciseness",
           "mu_4_reasoning", "mu_5_edge_cases", "mu_6_safety"]
# Matches evaluation/metrics.py registrations.
COSTS = {"mu_1_correctness": 0.001, "mu_2_latency": 0.000, "mu_3_conciseness": 0.000,
         "mu_4_reasoning": 0.015, "mu_5_edge_cases": 0.005, "mu_6_safety": 0.015}
DEEP = {"mu_4_reasoning", "mu_5_edge_cases", "mu_6_safety"}
SHORT = {"mu_1_correctness": "mu1_correctness", "mu_2_latency": "mu2_latency",
         "mu_3_conciseness": "mu3_conciseness", "mu_4_reasoning": "mu4_reasoning",
         "mu_5_edge_cases": "mu5_edge_cases", "mu_6_safety": "mu6_safety"}

COLUMNS = [
    "run_id", "agent_domain", "ablation_strategy", "seed_archetype", "bo_acquisition",
    "generation", "agent_id", "parent_id", "task_id", "task_name", "fitness",
    "testcases_passed", "testcases_total", "testcases_pass_rate_pct",
    "mu_1_functional_correctness", "pruning_status", "is_deep_tier_pruned",
    "active_evaluators_count", "active_evaluators_list", "adaptive_cost_spent",
    "total_cost_without_pruning", "net_savings_dollars", "net_savings_pct",
    "cost_mu1_correctness", "cost_mu2_latency", "cost_mu3_conciseness",
    "cost_mu4_reasoning", "cost_mu5_edge_cases", "cost_mu6_safety",
    "fanova_total_variance", "fanova_S1_correctness", "fanova_S2_latency",
    "fanova_S3_conciseness", "fanova_S4_reasoning", "fanova_S5_edge_cases",
    "fanova_S6_safety", "fanova_redundant_pruned_metrics",
    "mu_2_latency", "mu_3_conciseness", "mu_4_reasoning", "mu_5_edge_cases", "mu_6_safety",
    "weight_mu1_correctness", "weight_mu2_latency", "weight_mu3_conciseness",
    "weight_mu4_reasoning", "weight_mu5_edge_cases", "weight_mu6_safety",
    "delta_f", "mutation_type", "harness_temperature", "harness_top_p",
    "harness_max_retries", "cumulative_adaptive_cost", "cumulative_naive_cost",
    "cumulative_net_savings", "solution_preview", "system_prompt",
    # Provenance: absent from the original schema, retained because it is the
    # evidence that these rows came from real completions.
    "api_calls_live", "api_calls_mock", "model",
]

TASKS = {t.id: t for t in BENCHMARK_TASKS}
ARCHETYPES = {a["system_prompt"][:80]: n for n, a in AGENT_ARCHETYPES.items()} \
    if isinstance(AGENT_ARCHETYPES, dict) else {}


def _num(v, default=0.0) -> float:
    """float() that also absorbs NaN, which is truthy and so survives `or`."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(f) else f


def pruning_status(active: list, strategy: str) -> str:
    if strategy == "single_metric" or active == ["mu_1_correctness"]:
        return "Single Metric Only (Core)"
    if len(active) == len(METRICS):
        return "No Pruning (Full Ensemble)"
    if DEEP & set(active):
        return "Deep Tier Evaluated (Triggered by Preview Gate)"
    return "Deep Tier Pruned (Preview Gate)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=os.path.join("results", "ablation_live.csv"))
    ap.add_argument("--out", dest="dst",
                    default=os.path.join("results", "comprehensive_live_benchmark_results.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    out_rows = []

    cells = list(df.groupby(["condition", "task_id", "run_index"], sort=True))
    for run_no, ((cond, task_id, run_idx), cell) in enumerate(cells, start=1):
        cell = cell.sort_values("generation")
        task = TASKS.get(task_id)
        n_tests = len(task.test_cases) if task else 0
        domain = getattr(task, "domain", "coding") if task else "coding"

        seed_row = cell[cell.generation == 0]
        seed_prompt = str(seed_row.system_prompt.iloc[0])[:80] if len(seed_row) else ""
        archetype = ARCHETYPES.get(seed_prompt, "")

        # Replay fANOVA exactly as the controller did: metric matrix vs fitness,
        # refitted each generation once four scored agents exist.
        fan = FunctionalANOVA(evaluator_names=METRICS)
        X_hist, y_hist = [], []

        for _, r in cell.iterrows():
            present = [m for m in METRICS if pd.notna(r.get(m))]
            X_hist.append([_num(r.get(m)) for m in METRICS])
            y_hist.append(_num(r.get("fitness")))

            fa = {"total_variance": 0.0, "evaluator_importances": {}, "redundant_evaluators": []}
            if len(X_hist) >= 4:
                try:
                    fan.fit(np.array(X_hist), np.array(y_hist))
                    fa = fan.get_fanova_summary()
                except Exception:
                    pass

            rate = _num(r.get("mu_1_correctness"))
            adaptive = _num(r.get("cost_spent"))
            naive = _num(r.get("naive_cost_spent"))
            cum_a = _num(r.get("cumulative_adaptive_cost"))
            cum_n = _num(r.get("cumulative_naive_cost"))
            imp = fa.get("evaluator_importances", {})

            row = {
                "run_id": run_no,
                "agent_domain": domain,
                "ablation_strategy": cond,
                "seed_archetype": archetype,
                "bo_acquisition": "gp_ucb",
                "generation": int(r["generation"]),
                "agent_id": r.get("agent_id"),
                "parent_id": r.get("parent_id"),
                "task_id": task_id,
                "task_name": task.name if task else task_id,
                "fitness": r.get("fitness"),
                "testcases_passed": int(round(rate * n_tests)),
                "testcases_total": n_tests,
                "testcases_pass_rate_pct": round(rate * 100, 2),
                "mu_1_functional_correctness": r.get("mu_1_correctness"),
                "pruning_status": pruning_status(present, cond),
                "is_deep_tier_pruned": not bool(DEEP & set(present)),
                "active_evaluators_count": len(present),
                "active_evaluators_list": ", ".join(present),
                "adaptive_cost_spent": adaptive,
                "total_cost_without_pruning": naive,
                "net_savings_dollars": round(naive - adaptive, 6),
                "net_savings_pct": round((naive - adaptive) / naive * 100, 2) if naive else 0.0,
                "fanova_total_variance": round(_num(fa.get("total_variance")), 6),
                "fanova_redundant_pruned_metrics": fa.get("redundant_evaluators", []),
                "delta_f": r.get("delta_f"),
                "mutation_type": r.get("mutation_type"),
                "harness_temperature": r.get("temperature"),
                "harness_top_p": r.get("top_p"),
                "harness_max_retries": r.get("max_retries"),
                "cumulative_adaptive_cost": cum_a,
                "cumulative_naive_cost": cum_n,
                "cumulative_net_savings": round(cum_n - cum_a, 6),
                "solution_preview": r.get("solution_preview"),
                "system_prompt": r.get("system_prompt"),
                "api_calls_live": r.get("live_calls"),
                "api_calls_mock": r.get("mock_calls"),
                "model": r.get("model"),
            }
            for i, m in enumerate(METRICS, start=1):
                row[f"cost_{SHORT[m]}"] = COSTS[m] if m in present else 0.0
                row[f"fanova_S{i}_{SHORT[m].split('_', 1)[1]}"] = round(_num(imp.get(m)), 6)
                row[f"weight_{SHORT[m]}"] = r.get(f"weight_{m}")
                if m != "mu_1_correctness":
                    row[m] = r.get(m)
            out_rows.append(row)

    out = pd.DataFrame(out_rows)
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = None
    out = out[COLUMNS]
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    out.to_csv(args.dst, index=False)
    print(f"wrote {len(out)} rows x {len(out.columns)} columns -> {args.dst}")
    print(f"runs: {out.run_id.nunique()}   tasks: {out.task_id.nunique()}   "
          f"strategies: {out.ablation_strategy.nunique()}")
    # live/mock counts are per-cell, so dedupe before summing or every row re-counts them.
    per_cell = out.drop_duplicates(subset=["run_id"])
    print(f"provenance: live={int(per_cell.api_calls_live.sum())} "
          f"mock={int(per_cell.api_calls_mock.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
