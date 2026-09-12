import argparse
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import wilcoxon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_wilcoxon_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Computes paired Wilcoxon signed-rank tests for Condition 5 (full_adaptive)

    versus Conditions 1-4 (single_metric, static_cascade, ucb1_bandit, no_pruning)
    across all benchmark tasks.
    """
    results = []
    condition_5 = "full_adaptive"
    baselines = ["single_metric", "static_cascade", "ucb1_bandit", "no_pruning"]

    # Filter to final generation for each run or aggregate best fitness
    # Group by (task_id, condition, run_index) and get max fitness
    df_runs = df.groupby(["task_id", "condition", "run_index"]).agg({
        "fitness": "max",
        "cumulative_adaptive_cost": "max",
        "mu_1_correctness": "last",
    }).reset_index()

    tasks = sorted(df_runs["task_id"].unique())

    for task in tasks:
        df_task = df_runs[df_runs["task_id"] == task]
        c5_runs = df_task[df_task["condition"] == condition_5].sort_values("run_index")

        if len(c5_runs) == 0:
            continue

        c5_fitness = c5_runs["fitness"].values

        for base_cond in baselines:
            base_runs = df_task[df_task["condition"] == base_cond].sort_values("run_index")
            if len(base_runs) == 0:
                continue

            base_fitness = base_runs["fitness"].values
            n_pairs = min(len(c5_fitness), len(base_fitness))
            
            c5_vals = c5_fitness[:n_pairs]
            b_vals = base_fitness[:n_pairs]
            diffs = c5_vals - b_vals

            mean_c5 = float(np.mean(c5_vals))
            mean_base = float(np.mean(b_vals))
            mean_diff = float(np.mean(diffs))

            if np.all(diffs == 0):
                w_stat, p_two, p_one = 0.0, 1.0, 0.5
            else:
                try:
                    res_two = wilcoxon(c5_vals, b_vals, alternative="two-sided")
                    res_one = wilcoxon(c5_vals, b_vals, alternative="greater")
                    w_stat = float(res_two.statistic)
                    p_two = float(res_two.pvalue)
                    p_one = float(res_one.pvalue)
                except Exception as e:
                    logger.warning(f"Wilcoxon computation error for {task} ({condition_5} vs {base_cond}): {e}")
                    w_stat, p_two, p_one = float("nan"), float("nan"), float("nan")

            results.append({
                "task_id": task,
                "comparison": f"{condition_5} vs {base_cond}",
                "baseline_condition": base_cond,
                "n_runs": n_pairs,
                "mean_full_adaptive_fitness": round(mean_c5, 4),
                "mean_baseline_fitness": round(mean_base, 4),
                "mean_diff": round(mean_diff, 4),
                "wilcoxon_W": round(w_stat, 2) if not np.isnan(w_stat) else "N/A",
                "p_value_one_sided": round(p_one, 4) if not np.isnan(p_one) else "N/A",
                "p_value_two_sided": round(p_two, 4) if not np.isnan(p_two) else "N/A",
                "statistically_significant_0_05": (p_one < 0.05) if not np.isnan(p_one) else False,
            })

    return pd.DataFrame(results)


def compute_convergence_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Computes generations-to-convergence and cost-efficiency metrics across conditions."""
    records = []
    
    for (task, cond, run), group in df.groupby(["task_id", "condition", "run_index"]):
        group = group.sort_values("generation")
        fitnesses = group["fitness"].values
        generations = group["generation"].values
        max_f = np.max(fitnesses)
        threshold_95 = 0.95 * max_f if max_f > 0 else 0.95

        # Find first generation reaching >= 95% of peak fitness
        conv_gen = generations[-1]
        for g, f in zip(generations, fitnesses):
            if f >= threshold_95:
                conv_gen = g
                break

        final_cost = group["cumulative_adaptive_cost"].iloc[-1]
        final_naive = group["cumulative_naive_cost"].iloc[-1]
        cost_saved = max(0.0, final_naive - final_cost)
        pct_saved = (cost_saved / final_naive * 100.0) if final_naive > 0 else 0.0

        records.append({
            "task_id": task,
            "condition": cond,
            "run_index": run,
            "peak_fitness": round(max_f, 4),
            "convergence_generation": conv_gen,
            "cumulative_cost": round(final_cost, 5),
            "cumulative_naive_cost": round(final_naive, 5),
            "cost_saved_usd": round(cost_saved, 5),
            "cost_saved_pct": round(pct_saved, 2),
        })

    conv_df = pd.DataFrame(records)
    summary = conv_df.groupby(["task_id", "condition"]).agg({
        "peak_fitness": ["mean", "std"],
        "convergence_generation": ["mean", "std"],
        "cumulative_cost": "mean",
        "cost_saved_pct": "mean",
    }).reset_index()

    # Flatten column multi-index
    summary.columns = [
        "task_id", "condition",
        "mean_peak_fitness", "std_peak_fitness",
        "mean_conv_generation", "std_conv_generation",
        "mean_cost_spent", "mean_pct_saved"
    ]
    return summary.round(4)


def analyze_csv_results(csv_filepath: str, output_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads results CSV and performs statistical & convergence analysis."""
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Results CSV not found: {csv_filepath}")

    df = pd.read_csv(csv_filepath)
    logger.info(f"Loaded {len(df)} records from {csv_filepath}")

    # Ensure required columns exist
    if "condition" not in df.columns and "strategy" in df.columns:
        df["condition"] = df["strategy"]
    if "run_index" not in df.columns:
        df["run_index"] = 0

    wilcoxon_df = compute_wilcoxon_paired_tests(df)
    conv_df = compute_convergence_analysis(df)

    wilcoxon_csv = os.path.join(output_dir, "statistical_analysis_summary.csv")
    conv_csv = os.path.join(output_dir, "task_condition_metrics.csv")

    wilcoxon_df.to_csv(wilcoxon_csv, index=False)
    conv_df.to_csv(conv_csv, index=False)

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY (WILCOXON PAIRED SIGNED-RANK TESTS)")
    print("=" * 80)
    print(wilcoxon_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK CONDITION CONVERGENCE & EFFICIENCY METRICS")
    print("=" * 80)
    print(conv_df.to_string(index=False))

    return wilcoxon_df, conv_df


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Multi-Objective Agent Optimization Experiment Results")
    parser.add_argument("--csv", type=str, default="experiment_results.csv", help="Input experiment CSV path")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for summary CSVs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_csv_results(csv_filepath=args.csv, output_dir=args.output_dir)
