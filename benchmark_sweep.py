"""Multi-Domain Full Benchmark Sweep Harness.

Executes the complete experimental grid across Coding and Sentiment agents:
- Coding Agent: 5 Ablations x 6 Seed Archetypes x 2 BO Acquisitions = 60 Runs (180 Evolved Agents)
- Sentiment Agent: 5 Ablations x 4 Seed Archetypes x 2 BO Acquisitions = 40 Runs (120 Evolved Agents)
Total: 100 Runs x 3 Generations = 300 Evolved Agents Generated + 100 Seed Agents (Gen 0).

Outputs comprehensive evaluation results into:
1. `comprehensive_multi_domain_benchmark_results.csv` (300 Evolved Agents, Gen 1-3)
2. `seed_agents_benchmark_results.csv` (100 Seed Agents, Gen 0 Baseline)
3. `multi_domain_benchmark_results_all_sheets.xlsx` (Excel Workbook containing all sheets:
   - Sheet 1: `Evolved_Agents_Gen1_3`
   - Sheet 2: `Seed_Agents_Gen0`
   - Sheet 3: `Run_Comparison`)
"""

import os
import threading
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Callable

from controller import EvolutionController
from tasks.benchmark_tasks import BenchmarkTask


ABLATION_CONDITIONS = [
    "full_adaptive",
    "no_pruning",
    "static_cascade",
    "single_metric",
    "ucb1_bandit",
]

BO_ACQUISITIONS = [
    "gp_ucb",
    "expected_improvement",
]

CODING_ARCHETYPES = [
    "General Balanced Assistant",
    "Defensive Edge-Case Engineer",
    "Chain-of-Thought Analytic Agent",
    "Asymptotic Complexity Optimizer",
    "Minimalist Economy Synthesizer",
    "Safety & Constraint-Aligned Agent",
]

SENTIMENT_ARCHETYPES = [
    "SentiToolKit Neural Specialist",
    "Nuance & Aspect Sentiment Reasoner",
    "Sarcasm & Negation Analyst",
    "Fast Production Sentiment Classifier",
]

# Static unit evaluator costs
EVALUATOR_UNIT_COSTS = {
    "mu_1_correctness": 0.0010,
    "mu_2_latency": 0.0001,
    "mu_3_conciseness": 0.0001,
    "mu_4_reasoning": 0.0050,
    "mu_5_edge_cases": 0.0050,
    "mu_6_safety": 0.0050,
}

PRIORITIZED_COLUMNS = [
    # 1. Core Run Identification
    "run_id",
    "agent_domain",
    "ablation_strategy",
    "seed_archetype",
    "bo_acquisition",
    "generation",
    "agent_id",
    "parent_id",
    "task_id",
    "task_name",

    # 2. Key Performance & Verification Metrics (Most Important)
    "fitness",
    "testcases_passed",
    "testcases_total",
    "testcases_pass_rate_pct",
    "mu_1_functional_correctness",

    # 3. Pruning Status & Economic Efficiency
    "pruning_status",
    "is_deep_tier_pruned",
    "active_evaluators_count",
    "active_evaluators_list",
    "adaptive_cost_spent",
    "total_cost_without_pruning",
    "net_savings_dollars",
    "net_savings_pct",

    # 4. Cost of Every Evaluator for this Generation
    "cost_mu1_correctness",
    "cost_mu2_latency",
    "cost_mu3_conciseness",
    "cost_mu4_reasoning",
    "cost_mu5_edge_cases",
    "cost_mu6_safety",

    # 5. fANOVA (Functional ANOVA) Scores & Variance
    "fanova_total_variance",
    "fanova_S1_correctness",
    "fanova_S2_latency",
    "fanova_S3_conciseness",
    "fanova_S4_reasoning",
    "fanova_S5_edge_cases",
    "fanova_S6_safety",
    "fanova_redundant_pruned_metrics",

    # 6. Secondary Evaluator Dimension Scores
    "mu_2_latency",
    "mu_3_conciseness",
    "mu_4_reasoning",
    "mu_5_edge_cases",
    "mu_6_safety",

    # 7. Dynamic Weights on Simplex Δ⁵
    "weight_mu1_correctness",
    "weight_mu2_latency",
    "weight_mu3_conciseness",
    "weight_mu4_reasoning",
    "weight_mu5_edge_cases",
    "weight_mu6_safety",

    # 8. Optimization & Search Parameters
    "delta_f",
    "mutation_type",
    "harness_temperature",
    "harness_top_p",
    "harness_max_retries",
    "cumulative_adaptive_cost",
    "cumulative_naive_cost",
    "cumulative_net_savings",
    "solution_preview",
    "system_prompt",
]


class BenchmarkSweepManager:
    """Manages execution, telemetry, and multi-sheet persistence for 100 runs (300 Evolved + 100 Seed Agents)."""

    def __init__(
        self,
        output_csv: str = "comprehensive_multi_domain_benchmark_results.csv",
        seed_csv: str = "seed_agents_benchmark_results.csv",
        excel_path: str = "multi_domain_benchmark_results_all_sheets.xlsx",
    ):
        self.output_csv = output_csv
        self.seed_csv = seed_csv
        self.excel_path = excel_path
        self.lock = threading.Lock()
        self.is_running = False
        self.progress_pct = 0.0
        self.current_run_idx = 0
        self.total_runs = 100
        self.current_generation = 0
        self.total_agents = 300
        self.current_domain = "coding"
        self.current_ablation = "full_adaptive"
        self.current_archetype = "General Balanced Assistant"
        self.current_acq = "gp_ucb"
        self.records: List[Dict[str, Any]] = []
        self.seed_records: List[Dict[str, Any]] = []
        self.comparison_records: List[Dict[str, Any]] = []
        self.summary_stats: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

        # Load existing results if available
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                self.records = df.to_dict(orient="records")
                self.current_run_idx = len(set(df["run_id"])) if "run_id" in df.columns else len(df) // 3
                self.total_agents = len(df)
                self.progress_pct = 100.0 if len(df) >= 300 else round(len(df) / 300 * 100, 1)
            except Exception:
                pass

        if os.path.exists(self.seed_csv):
            try:
                df_seeds = pd.read_csv(self.seed_csv)
                self.seed_records = df_seeds.to_dict(orient="records")
            except Exception:
                pass

        self._compute_summary_stats()

    def start_sweep_async(self, is_mock: bool = True) -> bool:
        """Launches the sweep in a background thread."""
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.error_message = None
            self.progress_pct = 0.0
            self.current_run_idx = 0
            self.records = []
            self.seed_records = []
            self.comparison_records = []
            self._thread = threading.Thread(target=self._execute_sweep, args=(is_mock,), daemon=True)
            self._thread.start()
            return True

    def _execute_sweep(self, is_mock: bool = True):
        try:
            # Prewarm cached sentiment toolkit
            try:
                from evaluation.runner import get_cached_senti_toolkit
                get_cached_senti_toolkit()
            except Exception:
                pass

            # Build the 100 experiment configurations:
            run_configs = []
            run_counter = 1

            # Coding Domain (60 runs)
            for ablation in ABLATION_CONDITIONS:
                for archetype in CODING_ARCHETYPES:
                    for acq in BO_ACQUISITIONS:
                        run_configs.append({
                            "run_id": run_counter,
                            "domain": "coding",
                            "ablation": ablation,
                            "archetype": archetype,
                            "acq": acq,
                        })
                        run_counter += 1

            # Sentiment Domain (40 runs)
            for ablation in ABLATION_CONDITIONS:
                for archetype in SENTIMENT_ARCHETYPES:
                    for acq in BO_ACQUISITIONS:
                        run_configs.append({
                            "run_id": run_counter,
                            "domain": "sentiment",
                            "ablation": ablation,
                            "archetype": archetype,
                            "acq": acq,
                        })
                        run_counter += 1

            total_runs_count = len(run_configs)  # exactly 100
            all_evolved_records: List[Dict[str, Any]] = []
            all_seed_records: List[Dict[str, Any]] = []
            all_comparison_records: List[Dict[str, Any]] = []

            for run_idx, cfg in enumerate(run_configs, 1):
                self.current_run_idx = run_idx
                self.current_domain = cfg["domain"]
                self.current_ablation = cfg["ablation"]
                self.current_archetype = cfg["archetype"]
                self.current_acq = cfg["acq"]

                # Initialize controller with domain, seed archetype, and acquisition type
                controller = EvolutionController(
                    agent_domain=cfg["domain"],
                    initial_archetype=cfg["archetype"],
                    acquisition_type=cfg["acq"],
                    default_strategy=cfg["ablation"],
                    selected_task_id=None,  # "All tasks in domain (randomized)" suite
                    is_mock=is_mock,
                )

                # ==============================================================
                # 1. EXTRACT & RECORD GENERATION 0 SEED AGENT
                # ==============================================================
                seed_rec = controller.history_records[0]
                seed_agent = controller.archive.agents.get(seed_rec["child_id"])
                seed_suite = seed_rec.get("test_suite", {})
                seed_passed = seed_suite.get("passed_count", 0)
                seed_total = seed_suite.get("total_count", 0)
                seed_pass_rate = round((seed_passed / seed_total * 100.0), 2) if seed_total > 0 else 0.0
                seed_metrics = seed_rec.get("metrics", {})
                seed_weights = seed_rec.get("weights", {})
                seed_cost = seed_rec.get("cost_spent", 0.0162)

                seed_row = {
                    "run_id": cfg["run_id"],
                    "agent_domain": cfg["domain"],
                    "ablation_strategy": cfg["ablation"],
                    "seed_archetype": cfg["archetype"],
                    "bo_acquisition": cfg["acq"],
                    "generation": 0,
                    "agent_id": seed_rec["child_id"],
                    "parent_id": "Root",
                    "task_id": seed_rec.get("task_id", ""),
                    "task_name": seed_suite.get("task_name", seed_rec.get("task_id", "")),
                    "fitness": round(seed_rec.get("fitness", 0.0), 5),
                    "testcases_passed": seed_passed,
                    "testcases_total": seed_total,
                    "testcases_pass_rate_pct": seed_pass_rate,
                    "mu_1_functional_correctness": round(seed_metrics.get("mu_1_correctness", 0.0), 4),
                    "pruning_status": "Seed Baseline (Full Ensemble)",
                    "is_deep_tier_pruned": False,
                    "active_evaluators_count": len(seed_rec.get("active_evaluators", [])),
                    "active_evaluators_list": ", ".join(seed_rec.get("active_evaluators", [])),
                    "adaptive_cost_spent": round(seed_cost, 6),
                    "total_cost_without_pruning": round(seed_rec.get("naive_cost", 0.0162), 6),
                    "net_savings_dollars": 0.0,
                    "net_savings_pct": 0.0,
                    "cost_mu1_correctness": EVALUATOR_UNIT_COSTS["mu_1_correctness"],
                    "cost_mu2_latency": EVALUATOR_UNIT_COSTS["mu_2_latency"],
                    "cost_mu3_conciseness": EVALUATOR_UNIT_COSTS["mu_3_conciseness"],
                    "cost_mu4_reasoning": EVALUATOR_UNIT_COSTS["mu_4_reasoning"],
                    "cost_mu5_edge_cases": EVALUATOR_UNIT_COSTS["mu_5_edge_cases"],
                    "cost_mu6_safety": EVALUATOR_UNIT_COSTS["mu_6_safety"],
                    "fanova_total_variance": 0.0,
                    "fanova_S1_correctness": 0.1667,
                    "fanova_S2_latency": 0.1667,
                    "fanova_S3_conciseness": 0.1667,
                    "fanova_S4_reasoning": 0.1667,
                    "fanova_S5_edge_cases": 0.1667,
                    "fanova_S6_safety": 0.1667,
                    "fanova_redundant_pruned_metrics": "None (Seed Baseline)",
                    "mu_2_latency": round(seed_metrics.get("mu_2_latency", 0.0), 4),
                    "mu_3_conciseness": round(seed_metrics.get("mu_3_conciseness", 0.0), 4),
                    "mu_4_reasoning": round(seed_metrics.get("mu_4_reasoning", 0.0), 4),
                    "mu_5_edge_cases": round(seed_metrics.get("mu_5_edge_cases", 0.0), 4),
                    "mu_6_safety": round(seed_metrics.get("mu_6_safety", 0.0), 4),
                    "weight_mu1_correctness": round(seed_weights.get("mu_1_correctness", 0.167), 4),
                    "weight_mu2_latency": round(seed_weights.get("mu_2_latency", 0.167), 4),
                    "weight_mu3_conciseness": round(seed_weights.get("mu_3_conciseness", 0.167), 4),
                    "weight_mu4_reasoning": round(seed_weights.get("mu_4_reasoning", 0.167), 4),
                    "weight_mu5_edge_cases": round(seed_weights.get("mu_5_edge_cases", 0.167), 4),
                    "weight_mu6_safety": round(seed_weights.get("mu_6_safety", 0.167), 4),
                    "delta_f": 0.0,
                    "mutation_type": seed_rec.get("mutation_type", "seed"),
                    "harness_temperature": round(seed_agent.harness_knobs.temperature, 2) if seed_agent else 0.7,
                    "harness_top_p": round(seed_agent.harness_knobs.top_p, 2) if seed_agent else 0.95,
                    "harness_max_retries": seed_agent.harness_knobs.max_retries if seed_agent else 2,
                    "cumulative_adaptive_cost": round(seed_cost, 6),
                    "cumulative_naive_cost": round(seed_cost, 6),
                    "cumulative_net_savings": 0.0,
                    "solution_preview": seed_rec.get("solution", "")[:120].replace("\n", " "),
                    "system_prompt": seed_rec.get("child_prompt", "")[:200].replace("\n", " "),
                }
                all_seed_records.append(seed_row)

                # ==============================================================
                # 2. EXECUTE 3 EVOLVED GENERATIONS PER RUN
                # ==============================================================
                run_evolved_rows = []
                for gen in range(1, 4):
                    self.current_generation = (run_idx - 1) * 3 + gen
                    self.progress_pct = round((self.current_generation / 300.0) * 100.0, 1)

                    rec = controller.run_generation(strategy=cfg["ablation"])
                    fanova_summary = controller.get_fanova_summary()
                    active_evals = rec.get("active_evaluators", [])
                    test_suite = rec.get("test_suite", {})

                    passed_cases = test_suite.get("passed_count", 0)
                    total_cases = test_suite.get("total_count", 0)
                    pass_rate_pct = round((passed_cases / total_cases * 100.0), 2) if total_cases > 0 else 0.0

                    if cfg["ablation"] == "single_metric":
                        pruning_status = "Single Metric Only (Core)"
                        is_pruned = True
                    elif cfg["ablation"] == "no_pruning":
                        pruning_status = "No Pruning (Full Ensemble)"
                        is_pruned = False
                    elif len(active_evals) < 6:
                        pruning_status = "Pruned (Core Tier Only - Deep Tier Pruned)"
                        is_pruned = True
                    elif rec.get("soft_probe_triggered"):
                        pruning_status = "Deep Tier Evaluated (Epsilon-Probe Audit)"
                        is_pruned = False
                    else:
                        pruning_status = "Deep Tier Evaluated (Triggered by Preview Gate)"
                        is_pruned = False

                    spent_cost = rec.get("cost_spent", 0.0)
                    naive_cost = rec.get("naive_cost", 0.0162)
                    net_savings_dollars = round(max(0.0, naive_cost - spent_cost), 6)
                    net_savings_pct = round((net_savings_dollars / naive_cost * 100.0), 2) if naive_cost > 0 else 0.0

                    metrics = rec.get("metrics", {})
                    weights = rec.get("weights", {})
                    child_agent = controller.archive.agents.get(rec["child_id"])

                    cost_mu1 = EVALUATOR_UNIT_COSTS["mu_1_correctness"] if "mu_1_correctness" in active_evals else 0.0
                    cost_mu2 = EVALUATOR_UNIT_COSTS["mu_2_latency"] if "mu_2_latency" in active_evals else 0.0
                    cost_mu3 = EVALUATOR_UNIT_COSTS["mu_3_conciseness"] if "mu_3_conciseness" in active_evals else 0.0
                    cost_mu4 = EVALUATOR_UNIT_COSTS["mu_4_reasoning"] if "mu_4_reasoning" in active_evals else 0.0
                    cost_mu5 = EVALUATOR_UNIT_COSTS["mu_5_edge_cases"] if "mu_5_edge_cases" in active_evals else 0.0
                    cost_mu6 = EVALUATOR_UNIT_COSTS["mu_6_safety"] if "mu_6_safety" in active_evals else 0.0

                    fanova_importances = fanova_summary.get("evaluator_importances", {})
                    redundant_list = fanova_summary.get("redundant_evaluators", [])

                    row = {
                        "run_id": cfg["run_id"],
                        "agent_domain": cfg["domain"],
                        "ablation_strategy": cfg["ablation"],
                        "seed_archetype": cfg["archetype"],
                        "bo_acquisition": cfg["acq"],
                        "generation": gen,
                        "agent_id": rec["child_id"],
                        "parent_id": rec["parent_id"],
                        "task_id": rec.get("task_id", ""),
                        "task_name": test_suite.get("task_name", rec.get("task_id", "")),
                        "fitness": round(rec.get("fitness", 0.0), 5),
                        "testcases_passed": passed_cases,
                        "testcases_total": total_cases,
                        "testcases_pass_rate_pct": pass_rate_pct,
                        "mu_1_functional_correctness": round(metrics.get("mu_1_correctness", 0.0), 4),
                        "pruning_status": pruning_status,
                        "is_deep_tier_pruned": is_pruned,
                        "active_evaluators_count": len(active_evals),
                        "active_evaluators_list": ", ".join(active_evals),
                        "adaptive_cost_spent": round(spent_cost, 6),
                        "total_cost_without_pruning": round(naive_cost, 6),
                        "net_savings_dollars": net_savings_dollars,
                        "net_savings_pct": net_savings_pct,
                        "cost_mu1_correctness": cost_mu1,
                        "cost_mu2_latency": cost_mu2,
                        "cost_mu3_conciseness": cost_mu3,
                        "cost_mu4_reasoning": cost_mu4,
                        "cost_mu5_edge_cases": cost_mu5,
                        "cost_mu6_safety": cost_mu6,
                        "fanova_total_variance": round(fanova_summary.get("total_variance", 0.0), 6),
                        "fanova_S1_correctness": round(fanova_importances.get("mu_1_correctness", 0.0), 4),
                        "fanova_S2_latency": round(fanova_importances.get("mu_2_latency", 0.0), 4),
                        "fanova_S3_conciseness": round(fanova_importances.get("mu_3_conciseness", 0.0), 4),
                        "fanova_S4_reasoning": round(fanova_importances.get("mu_4_reasoning", 0.0), 4),
                        "fanova_S5_edge_cases": round(fanova_importances.get("mu_5_edge_cases", 0.0), 4),
                        "fanova_S6_safety": round(fanova_importances.get("mu_6_safety", 0.0), 4),
                        "fanova_redundant_pruned_metrics": ", ".join(redundant_list) if redundant_list else "None",
                        "mu_2_latency": round(metrics.get("mu_2_latency", 0.0), 4),
                        "mu_3_conciseness": round(metrics.get("mu_3_conciseness", 0.0), 4),
                        "mu_4_reasoning": round(metrics.get("mu_4_reasoning", 0.0), 4),
                        "mu_5_edge_cases": round(metrics.get("mu_5_edge_cases", 0.0), 4),
                        "mu_6_safety": round(metrics.get("mu_6_safety", 0.0), 4),
                        "weight_mu1_correctness": round(weights.get("mu_1_correctness", 0.0), 4),
                        "weight_mu2_latency": round(weights.get("mu_2_latency", 0.0), 4),
                        "weight_mu3_conciseness": round(weights.get("mu_3_conciseness", 0.0), 4),
                        "weight_mu4_reasoning": round(weights.get("mu_4_reasoning", 0.0), 4),
                        "weight_mu5_edge_cases": round(weights.get("mu_5_edge_cases", 0.0), 4),
                        "weight_mu6_safety": round(weights.get("mu_6_safety", 0.0), 4),
                        "delta_f": round(rec.get("delta_f", 0.0), 4),
                        "mutation_type": rec.get("mutation_type", ""),
                        "harness_temperature": round(child_agent.harness_knobs.temperature, 2) if child_agent else 0.7,
                        "harness_top_p": round(child_agent.harness_knobs.top_p, 2) if child_agent else 0.95,
                        "harness_max_retries": child_agent.harness_knobs.max_retries if child_agent else 2,
                        "cumulative_adaptive_cost": round(controller.cumulative_adaptive_cost, 6),
                        "cumulative_naive_cost": round(controller.cumulative_naive_cost, 6),
                        "cumulative_net_savings": round(max(0.0, controller.cumulative_naive_cost - controller.cumulative_adaptive_cost), 6),
                        "solution_preview": rec.get("solution", "")[:120].replace("\n", " "),
                        "system_prompt": rec.get("child_prompt", "")[:200].replace("\n", " "),
                    }
                    all_evolved_records.append(row)
                    run_evolved_rows.append(row)

                # ==============================================================
                # 3. RUN COMPARISON: SEED VS PEAK EVOLVED AGENT
                # ==============================================================
                best_evolved = max(run_evolved_rows, key=lambda x: x["fitness"]) if run_evolved_rows else seed_row
                fitness_gain = round(best_evolved["fitness"] - seed_row["fitness"], 5)
                pass_gain = round(best_evolved["testcases_pass_rate_pct"] - seed_row["testcases_pass_rate_pct"], 2)
                run_spent = sum(r["adaptive_cost_spent"] for r in run_evolved_rows)
                run_naive = sum(r["total_cost_without_pruning"] for r in run_evolved_rows)
                run_saved = round(max(0.0, run_naive - run_spent), 6)
                run_saved_pct = round((run_saved / run_naive * 100.0), 2) if run_naive > 0 else 0.0

                comp_row = {
                    "run_id": cfg["run_id"],
                    "agent_domain": cfg["domain"],
                    "ablation_strategy": cfg["ablation"],
                    "seed_archetype": cfg["archetype"],
                    "bo_acquisition": cfg["acq"],
                    "seed_agent_id": seed_row["agent_id"],
                    "seed_fitness": seed_row["fitness"],
                    "seed_pass_rate_pct": seed_row["testcases_pass_rate_pct"],
                    "best_evolved_agent_id": best_evolved["agent_id"],
                    "best_evolved_generation": best_evolved["generation"],
                    "best_evolved_fitness": best_evolved["fitness"],
                    "best_evolved_pass_rate_pct": best_evolved["testcases_pass_rate_pct"],
                    "fitness_improvement_delta": fitness_gain,
                    "pass_rate_gain_pct": pass_gain,
                    "adaptive_cost_spent": round(run_spent, 6),
                    "total_cost_without_pruning": round(run_naive, 6),
                    "net_savings_dollars": run_saved,
                    "net_savings_pct": run_saved_pct,
                }
                all_comparison_records.append(comp_row)

                # Periodic state update
                with self.lock:
                    self.records = list(all_evolved_records)
                    self.seed_records = list(all_seed_records)
                    self.comparison_records = list(all_comparison_records)
                    self._compute_summary_stats()

            # Final persistence to CSVs and multi-sheet Excel
            with self.lock:
                self.progress_pct = 100.0
                self.is_running = False
                self._compute_summary_stats()
                self._save_all_files(all_evolved_records, all_seed_records, all_comparison_records)

        except Exception as e:
            with self.lock:
                self.is_running = False
                self.error_message = str(e)

    def _save_all_files(
        self,
        evolved_records: List[Dict[str, Any]],
        seed_records: List[Dict[str, Any]],
        comparison_records: List[Dict[str, Any]],
    ):
        """Saves evolved CSV, seed CSV, and combined multi-sheet Excel workbook."""
        # 1. Evolved agents CSV (300 rows)
        df_evolved = pd.DataFrame(evolved_records)[PRIORITIZED_COLUMNS] if evolved_records else pd.DataFrame(columns=PRIORITIZED_COLUMNS)
        df_evolved.to_csv(self.output_csv, index=False)

        # 2. Seed agents CSV (100 rows)
        df_seeds = pd.DataFrame(seed_records)[PRIORITIZED_COLUMNS] if seed_records else pd.DataFrame(columns=PRIORITIZED_COLUMNS)
        df_seeds.to_csv(self.seed_csv, index=False)

        # 3. Multi-sheet Excel workbook
        try:
            df_comp = pd.DataFrame(comparison_records) if comparison_records else pd.DataFrame()
            with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                df_evolved.to_excel(writer, sheet_name="Evolved_Agents_Gen1_to_3", index=False)
                df_seeds.to_excel(writer, sheet_name="Seed_Agents_Gen0", index=False)
                if not df_comp.empty:
                    df_comp.to_excel(writer, sheet_name="Run_Comparison_Summary", index=False)
        except Exception as e:
            print(f"Warning: Excel export error: {e}")

    def _compute_summary_stats(self):
        if not self.records:
            self.summary_stats = {}
            return

        df = pd.DataFrame(self.records)
        df_seeds = pd.DataFrame(self.seed_records) if self.seed_records else None

        total_spent = float(df["adaptive_cost_spent"].sum()) if "adaptive_cost_spent" in df.columns else 0.0
        total_naive = float(df["total_cost_without_pruning"].sum()) if "total_cost_without_pruning" in df.columns else 0.0
        total_saved = max(0.0, total_naive - total_spent)
        pct_saved = round((total_saved / total_naive * 100.0), 2) if total_naive > 0 else 0.0

        mean_fit = round(float(df["fitness"].mean()), 4) if "fitness" in df.columns else 0.0
        mean_pass = round(float(df["testcases_pass_rate_pct"].mean()), 2) if "testcases_pass_rate_pct" in df.columns else 0.0
        mean_correctness = round(float(df["mu_1_functional_correctness"].mean()), 4) if "mu_1_functional_correctness" in df.columns else 0.0

        seed_mean_fit = round(float(df_seeds["fitness"].mean()), 4) if df_seeds is not None and "fitness" in df_seeds.columns else 0.0
        seed_mean_pass = round(float(df_seeds["testcases_pass_rate_pct"].mean()), 2) if df_seeds is not None and "testcases_pass_rate_pct" in df_seeds.columns else 0.0

        coding_count = int((df["agent_domain"] == "coding").sum()) if "agent_domain" in df.columns else 0
        sentiment_count = int((df["agent_domain"] == "sentiment").sum()) if "agent_domain" in df.columns else 0

        self.summary_stats = {
            "total_agents_generated": len(df),
            "total_seed_agents": len(df_seeds) if df_seeds is not None else 0,
            "total_runs_completed": self.current_run_idx,
            "coding_agents_count": coding_count,
            "sentiment_agents_count": sentiment_count,
            "mean_fitness": mean_fit,
            "seed_mean_fitness": seed_mean_fit,
            "fitness_gain": round(mean_fit - seed_mean_fit, 4),
            "mean_test_pass_rate": mean_pass,
            "seed_mean_pass_rate": seed_mean_pass,
            "pass_rate_gain": round(mean_pass - seed_mean_pass, 2),
            "mean_correctness": mean_correctness,
            "total_adaptive_cost": round(total_spent, 5),
            "total_naive_cost": round(total_naive, 5),
            "total_saved_dollars": round(total_saved, 5),
            "overall_pct_saved": pct_saved,
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current progress, state, and telemetry of the sweep."""
        with self.lock:
            recent = self.records[-10:] if len(self.records) > 10 else self.records
            return {
                "is_running": self.is_running,
                "progress_pct": self.progress_pct,
                "current_run_idx": self.current_run_idx,
                "total_runs": self.total_runs,
                "current_generation": self.current_generation,
                "total_agents": self.total_agents,
                "total_seed_agents": len(self.seed_records),
                "current_domain": self.current_domain,
                "current_ablation": self.current_ablation,
                "current_archetype": self.current_archetype,
                "current_acq": self.current_acq,
                "summary_stats": self.summary_stats,
                "recent_records": recent,
                "total_records_count": len(self.records),
                "total_seed_records_count": len(self.seed_records),
                "error_message": self.error_message,
                "csv_path": self.output_csv,
                "seed_csv_path": self.seed_csv,
                "excel_path": self.excel_path,
            }


# Singleton manager instance
SWEEP_MANAGER = BenchmarkSweepManager()
