"""Aggregates ablation and live-validation results into the statistics, tables and
figures quoted by the BCSE497J project report.

Every number the report states is produced here from the experiment CSVs, so the
report can be regenerated from raw data at any time:

    python report_analysis.py --mock-csv results/ablation_mock.csv \
                              --live-csv results/validation_live.csv \
                              --outdir report_assets

Outputs into --outdir:
    figures/*.png          figures embedded in Chapter 6
    tables/*.csv           tables embedded in Chapter 6 and the appendices
    report_data.json       every scalar the report quotes, for the docx builder
"""
import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_results import compute_convergence_analysis, compute_wilcoxon_paired_tests

CONDITIONS = ["single_metric", "static_cascade", "ucb1_bandit", "no_pruning", "full_adaptive"]
CONDITION_LABELS = {
    "single_metric": "C1 Single-metric",
    "static_cascade": "C2 Static cascade",
    "ucb1_bandit": "C3 UCB1 bandit",
    "no_pruning": "C4 No pruning",
    "full_adaptive": "C5 Full adaptive",
}
METRICS = [
    "mu_1_correctness", "mu_2_latency", "mu_3_conciseness",
    "mu_4_reasoning", "mu_5_edge_cases", "mu_6_safety",
]
METRIC_LABELS = {
    "mu_1_correctness": r"$\mu_1$ Correctness",
    "mu_2_latency": r"$\mu_2$ Latency",
    "mu_3_conciseness": r"$\mu_3$ Conciseness",
    "mu_4_reasoning": r"$\mu_4$ Reasoning",
    "mu_5_edge_cases": r"$\mu_5$ Edge cases",
    "mu_6_safety": r"$\mu_6$ Safety",
}
# Colour-blind-safe qualitative palette, consistent across every figure.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
# Several conditions produce identical traces (C2 lies exactly under C3 in the
# simulation), so colour alone does not distinguish them on a line plot.
DASHES = [(None, None), (5, 1.6), (1.6, 1.4), (None, None), (6, 1.5, 1.5, 1.5)]


def _dash(ax_line, i):
    d = DASHES[i % len(DASHES)]
    if d[0] is not None:
        ax_line.set_dashes(list(d))
    return ax_line

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------- io
def load_and_merge(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  ! missing, skipped: {p}")
            continue
        df = pd.read_csv(p)
        df["_source"] = os.path.basename(p)
        frames.append(df)
        print(f"  loaded {len(df):6d} rows from {os.path.basename(p)}")
    if not frames:
        raise SystemExit("no input CSVs found")
    out = pd.concat(frames, ignore_index=True)
    if "condition" not in out.columns and "strategy" in out.columns:
        out["condition"] = out["strategy"]
    if "run_index" not in out.columns:
        out["run_index"] = 0
    if "domain" not in out.columns:
        # Derive from the task registry rather than defaulting to "coding",
        # which would silently relabel every sentiment task.
        from tasks.benchmark_tasks import BENCHMARK_TASKS
        lookup = {t.id: getattr(t, "domain", None) for t in BENCHMARK_TASKS}
        out["domain"] = out["task_id"].map(
            lambda t: lookup.get(t) or ("sentiment" if str(t).startswith("senti") else "coding"))
    return out


def _present_conditions(df: pd.DataFrame) -> List[str]:
    return [c for c in CONDITIONS if c in set(df["condition"].unique())]



LAMBDA_PENALTY = 0.5        # matches EvolutionController's default


def add_normalised_fitness(df: pd.DataFrame) -> pd.DataFrame:
    """Add a fitness that is comparable ACROSS conditions.

    The implemented objective is  F = sum_{j active} w_j mu_j - lambda sum_{j active} c_j,
    where w lives on the simplex over all six evaluators but the score term sums
    only the active ones. A condition that activates three evaluators therefore
    sums roughly two thirds of the weight mass and is mechanically penalized
    against one that activates all six, independently of how good its agent is.
    Dividing the score term by the active weight mass removes that artefact:

        F_norm = (sum_{j active} w_j mu_j) / (sum_{j active} w_j) - lambda sum c_j

    Raw fitness is retained and reported alongside, because it is the quantity the
    optimizer actually maximized.
    """
    weight_cols = [f"weight_{m}" for m in METRICS]
    have_metrics = [m for m in METRICS if m in df.columns]
    have_weights = [w for w in weight_cols if w in df.columns]
    if len(have_metrics) != len(METRICS) or len(have_weights) != len(METRICS):
        df["active_weight_mass"] = np.nan
        df["fitness_normalised"] = df["fitness"]
        return df

    mu = df[METRICS].apply(pd.to_numeric, errors="coerce").values
    w = df[weight_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    active = ~np.isnan(mu)

    mass = (w * active).sum(axis=1)
    score = np.nansum(np.where(active, w * np.nan_to_num(mu), 0.0), axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        normalised_score = np.where(mass > 1e-9, score / mass, np.nan)

    # Recover the cost term that was subtracted, so it can be re-applied unchanged.
    cost_term = score - df["fitness"].values
    df = df.copy()
    df["active_weight_mass"] = mass
    df["fitness_normalised"] = normalised_score - cost_term
    return df


# -------------------------------------------------------------------- statistics
def per_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, condition, run): peak fitness, final cost, final correctness."""
    rows = []
    for (task, cond, run), g in df.groupby(["task_id", "condition", "run_index"]):
        g = g.sort_values("generation")
        rows.append({
            "task_id": task,
            "condition": cond,
            "run_index": run,
            "domain": g["domain"].iloc[0] if "domain" in g.columns else "coding",
            "peak_fitness": float(g["fitness"].max()),
            "peak_fitness_normalised": float(pd.to_numeric(
                g.get("fitness_normalised", g["fitness"]), errors="coerce").max()),
            "mean_active_weight_mass": float(pd.to_numeric(
                g.get("active_weight_mass", pd.Series(dtype=float)), errors="coerce").mean()),
            "final_fitness": float(g["fitness"].iloc[-1]),
            "final_cost": float(g["cumulative_adaptive_cost"].iloc[-1]),
            "final_naive_cost": float(g["cumulative_naive_cost"].iloc[-1]),
            "mean_active_evaluators": float(pd.to_numeric(g["active_evaluators_count"], errors="coerce").mean()),
            "peak_correctness": float(pd.to_numeric(g["mu_1_correctness"], errors="coerce").max()),
            "generations": int(g["generation"].max()),
        })
    out = pd.DataFrame(rows)
    out["cost_saved_usd"] = (out["final_naive_cost"] - out["final_cost"]).clip(lower=0)
    out["cost_saved_pct"] = np.where(
        out["final_naive_cost"] > 0,
        out["cost_saved_usd"] / out["final_naive_cost"] * 100.0,
        0.0,
    )
    return out


def pooled_wilcoxon(runs: pd.DataFrame, treatment: str = "full_adaptive",
                    metric: str = "peak_fitness") -> pd.DataFrame:
    """Wilcoxon signed-rank pooled over every (task, run) pair, not per task.

    Per-task tests with n=5 runs bottom out at p=0.031 one-sided, so pooling is
    what gives the comparison real power.
    """
    results = []
    present = _present_conditions(runs)
    if treatment not in present:
        return pd.DataFrame(results)

    for baseline in [c for c in present if c != treatment]:
        pairs_t, pairs_b = [], []
        for (task, run), g in runs.groupby(["task_id", "run_index"]):
            t = g[g["condition"] == treatment][metric]
            b = g[g["condition"] == baseline][metric]
            if len(t) and len(b):
                pairs_t.append(float(t.iloc[0]))
                pairs_b.append(float(b.iloc[0]))
        if len(pairs_t) < 2:
            continue

        a, b_arr = np.array(pairs_t), np.array(pairs_b)
        diffs = a - b_arr
        if np.all(diffs == 0):
            w, p_two, p_one = 0.0, 1.0, 0.5
        else:
            w = float(wilcoxon(a, b_arr, alternative="two-sided").statistic)
            p_two = float(wilcoxon(a, b_arr, alternative="two-sided").pvalue)
            p_one = float(wilcoxon(a, b_arr, alternative="greater").pvalue)

        # Matched-pairs rank-biserial correlation: an effect size the W statistic alone does not give.
        nonzero = diffs[diffs != 0]
        if len(nonzero):
            ranks = pd.Series(np.abs(nonzero)).rank().values
            r_plus = ranks[nonzero > 0].sum()
            r_minus = ranks[nonzero < 0].sum()
            effect = float((r_plus - r_minus) / (r_plus + r_minus))
        else:
            effect = 0.0

        results.append({
            "metric": metric,
            "comparison": f"{treatment} vs {baseline}",
            "baseline": baseline,
            "n_pairs": len(a),
            "mean_treatment": round(float(a.mean()), 4),
            "mean_baseline": round(float(b_arr.mean()), 4),
            "mean_diff": round(float(diffs.mean()), 4),
            "median_diff": round(float(np.median(diffs)), 4),
            "wilcoxon_W": round(w, 2),
            "p_one_sided": p_one,
            "p_two_sided": p_two,
            "rank_biserial_effect": round(effect, 3),
            "significant_0_05": bool(p_one < 0.05),
        })
    return pd.DataFrame(results)


def condition_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Headline per-condition table: fitness, cost, savings, evaluators used."""
    rows = []
    for cond in _present_conditions(runs):
        g = runs[runs["condition"] == cond]
        rows.append({
            "condition": cond,
            "label": CONDITION_LABELS.get(cond, cond),
            "n_runs": len(g),
            "mean_peak_fitness": round(float(g["peak_fitness"].mean()), 4),
            "std_peak_fitness": round(float(g["peak_fitness"].std(ddof=1)), 4) if len(g) > 1 else 0.0,
            "mean_peak_fitness_normalised": round(float(g["peak_fitness_normalised"].mean()), 4),
            "mean_active_weight_mass": round(float(g["mean_active_weight_mass"].mean()), 4),
            "mean_peak_correctness": round(float(g["peak_correctness"].mean()), 4),
            "mean_cost_usd": round(float(g["final_cost"].mean()), 5),
            "mean_naive_cost_usd": round(float(g["final_naive_cost"].mean()), 5),
            "mean_cost_saved_pct": round(float(g["cost_saved_pct"].mean()), 2),
            "mean_active_evaluators": round(float(g["mean_active_evaluators"].mean()), 3),
        })
    return pd.DataFrame(rows)


def domain_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, cond), g in runs.groupby(["domain", "condition"]):
        rows.append({
            "domain": domain,
            "condition": cond,
            "label": CONDITION_LABELS.get(cond, cond),
            "n_runs": len(g),
            "mean_peak_fitness": round(float(g["peak_fitness"].mean()), 4),
            "mean_peak_correctness": round(float(g["peak_correctness"].mean()), 4),
            "mean_cost_saved_pct": round(float(g["cost_saved_pct"].mean()), 2),
        })
    return pd.DataFrame(rows).sort_values(["domain", "condition"])



def _legend_below(ax, handles=None, labels=None, ncol=3, y=-0.22, title=None):
    """Place a legend beneath the axes.

    Legends inside the axes were covering data in several figures; putting them
    outside costs a little vertical space and removes the problem entirely.
    """
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, borderaxespad=0.0, handlelength=1.6,
              columnspacing=1.4, handletextpad=0.6)
    if title:
        kw["title"] = title
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


# ----------------------------------------------------------------------- figures
def _bestsofar(df: pd.DataFrame) -> pd.DataFrame:
    """Mean best-so-far fitness per generation, per condition."""
    recs = []
    for (task, cond, run), g in df.groupby(["task_id", "condition", "run_index"]):
        g = g.sort_values("generation")
        best = g["fitness"].cummax().values
        for gen, val in zip(g["generation"].values, best):
            recs.append({"condition": cond, "generation": int(gen), "best": float(val)})
    return pd.DataFrame(recs)


def fig_convergence(df: pd.DataFrame, path: str) -> Dict:
    bs = _bestsofar(df)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    stats = {}
    for i, cond in enumerate(_present_conditions(df)):
        g = bs[bs["condition"] == cond].groupby("generation")["best"]
        mean, sem = g.mean(), g.sem()
        ln, = ax.plot(mean.index, mean.values, label=CONDITION_LABELS.get(cond, cond),
                      color=PALETTE[i % len(PALETTE)], linewidth=1.6)
        _dash(ln, i)
        ax.fill_between(mean.index, (mean - sem).values, (mean + sem).values,
                        color=PALETTE[i % len(PALETTE)], alpha=0.15, linewidth=0)
        stats[cond] = {"final_mean_best": round(float(mean.values[-1]), 4)}
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best-so-far fitness $F$")
    ax.set_title("Fitness convergence by condition (mean $\\pm$ s.e.m.)")
    _legend_below(ax, ncol=3)
    fig.savefig(path)
    plt.close(fig)
    return stats


def fig_cost_trajectory(df: pd.DataFrame, path: str) -> Dict:
    fa = df[df["condition"] == "full_adaptive"]
    if fa.empty:
        return {}
    g = fa.groupby("generation")
    adaptive = g["cumulative_adaptive_cost"].mean()
    naive = g["cumulative_naive_cost"].mean()

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(naive.index, naive.values, label="Naive (all 6 evaluators)",
            color=PALETTE[3], linewidth=1.8, linestyle="--")
    ax.plot(adaptive.index, adaptive.values, label="Adaptive cascade (C5)",
            color=PALETTE[0], linewidth=1.8)
    ax.fill_between(adaptive.index, adaptive.values, naive.values,
                    color=PALETTE[2], alpha=0.18, linewidth=0, label="Evaluation spend avoided")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative evaluation cost (USD)")
    ax.set_title("Cumulative evaluation spend: adaptive vs naive")
    _legend_below(ax, ncol=3)
    fig.savefig(path)
    plt.close(fig)

    final_a, final_n = float(adaptive.values[-1]), float(naive.values[-1])
    return {
        "final_adaptive_usd": round(final_a, 5),
        "final_naive_usd": round(final_n, 5),
        "saved_usd": round(final_n - final_a, 5),
        "saved_pct": round((final_n - final_a) / final_n * 100.0, 2) if final_n else 0.0,
    }


def fig_cost_vs_fitness(runs: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    for i, cond in enumerate(_present_conditions(runs)):
        g = runs[runs["condition"] == cond]
        ax.scatter(g["final_cost"], g["peak_fitness"], s=16, alpha=0.55,
                   color=PALETTE[i % len(PALETTE)], label=CONDITION_LABELS.get(cond, cond),
                   edgecolors="none")
        ax.scatter([g["final_cost"].mean()], [g["peak_fitness"].mean()],
                   s=120, marker="X", color=PALETTE[i % len(PALETTE)],
                   edgecolors="white", linewidths=1.2, zorder=5)
    ax.set_xlabel("Evaluation cost per run (USD)")
    ax.set_ylabel("Peak fitness $F$")
    ax.set_title("Cost–quality trade-off (X = condition mean)")
    _legend_below(ax, ncol=3)
    fig.savefig(path)
    plt.close(fig)


def fig_weight_trajectory(df: pd.DataFrame, path: str) -> Dict:
    fa = df[df["condition"] == "full_adaptive"]
    cols = [f"weight_{m}" for m in METRICS if f"weight_{m}" in df.columns]
    if fa.empty or not cols:
        return {}
    g = fa.groupby("generation")[cols].mean()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.stackplot(g.index, [g[c].values for c in cols],
                 labels=[METRIC_LABELS.get(c.replace("weight_", ""), c) for c in cols],
                 colors=PALETTE[:len(cols)], alpha=0.9)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weight $w_j$ (simplex $\\Delta^5$)")
    ax.set_title("GP-UCB weight trajectory on the simplex")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    fig.savefig(path)
    plt.close(fig)
    return {c.replace("weight_", ""): round(float(g[c].values[-1]), 4) for c in cols}


def fig_evaluator_activation(df: pd.DataFrame, path: str) -> Dict:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    out = {}
    for i, cond in enumerate(_present_conditions(df)):
        g = df[df["condition"] == cond].groupby("generation")["active_evaluators_count"]
        mean = pd.to_numeric(g.mean(), errors="coerce")
        ln, = ax.plot(mean.index, mean.values, label=CONDITION_LABELS.get(cond, cond),
                      color=PALETTE[i % len(PALETTE)], linewidth=1.5)
        _dash(ln, i)
        out[cond] = round(float(np.nanmean(mean.values)), 3)
    ax.axhline(6, color="grey", linestyle=":", linewidth=1)
    ax.text(0.99, 6.05, "all 6 evaluators", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7, color="grey")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Active evaluators $|x_E|$")
    ax.set_title("Evaluators executed per generation")
    ax.set_ylim(0, 6.9)
    _legend_below(ax, ncol=3)
    fig.savefig(path)
    plt.close(fig)
    return out


def fig_metric_profile(runs_df: pd.DataFrame, path: str) -> Dict:
    """Mean of each evaluator dimension under the full adaptive condition, per domain."""
    fa = runs_df[runs_df["condition"] == "full_adaptive"]
    if fa.empty:
        return {}
    domains = sorted(fa["domain"].unique())
    width = 0.8 / max(1, len(domains))
    x = np.arange(len(METRICS))

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    out = {}
    for i, dom in enumerate(domains):
        g = fa[fa["domain"] == dom]
        vals = [float(pd.to_numeric(g[m], errors="coerce").mean()) if m in g.columns else np.nan
                for m in METRICS]
        ax.bar(x + i * width, vals, width, label=dom, color=PALETTE[i % len(PALETTE)])
        out[dom] = {m: (round(v, 4) if not np.isnan(v) else None) for m, v in zip(METRICS, vals)}
    ax.set_xticks(x + width * (len(domains) - 1) / 2)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=20, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Evaluator dimension profile under C5, by domain")
    _legend_below(ax, ncol=2, y=-0.30, title="Domain")
    fig.savefig(path)
    plt.close(fig)
    return out


def fig_mutation_usage(df: pd.DataFrame, path: str) -> Dict:
    """Operator selection under the bandit (C3) against uniform selection (C5).

    Only C3 consults the UCB1 bandit; C5 selects uniformly at random. Showing
    one condition alone would misattribute the distribution, so both are plotted.
    """
    if "mutation_type" not in df.columns:
        return {}

    def _counts(cond):
        g = df[df["condition"] == cond]
        if g.empty:
            return None
        c = g["mutation_type"].value_counts()
        # Seed agents are not mutation operators.
        return c[~c.index.astype(str).str.startswith("seed")]

    bandit, uniform = _counts("ucb1_bandit"), _counts("full_adaptive")
    if bandit is None and uniform is None:
        return {}
    ops = sorted(set((bandit if bandit is not None else uniform).index)
                 | set((uniform if uniform is not None else bandit).index))
    if not ops:
        return {}

    y = np.arange(len(ops))
    height = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    series = []
    if bandit is not None:
        series.append(("C3 UCB1 bandit", [int(bandit.get(o, 0)) for o in ops], PALETTE[2]))
    if uniform is not None:
        series.append(("C5 uniform random", [int(uniform.get(o, 0)) for o in ops], PALETTE[4]))
    for k, (label, vals, colour) in enumerate(series):
        offset = (k - (len(series) - 1) / 2) * height
        ax.barh(y + offset, vals, height, label=label, color=colour)

    ax.set_yticks(y)
    ax.set_yticklabels([str(o).replace("_", " ") for o in ops])
    ax.invert_yaxis()
    ax.set_xlabel("Times selected")
    ax.set_title("Mutation operator selection: bandit vs uniform")
    ax.grid(axis="y", alpha=0)
    _legend_below(ax, ncol=2, y=-0.20)
    fig.savefig(path)
    plt.close(fig)

    out = {}
    for label, vals, _ in series:
        tot = sum(vals) or 1
        out[label] = {str(o): {"n": v, "pct": round(v / tot * 100, 1)}
                      for o, v in zip(ops, vals)}
    return out


def fig_live_vs_mock(live_runs: Optional[pd.DataFrame], mock_runs: pd.DataFrame, path: str) -> Dict:
    """Side-by-side of the metric that differs most between simulation and real LLM."""
    if live_runs is None or live_runs.empty:
        return {}
    conds = [c for c in _present_conditions(live_runs)]
    x = np.arange(len(conds))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5))
    fig.subplots_adjust(wspace=0.34)
    out = {}
    for ax, field, title in [
        (axes[0], "peak_correctness", "Peak correctness $\\mu_1$"),
        (axes[1], "cost_saved_pct", "Evaluation cost saved (%)"),
    ]:
        m_vals = [float(mock_runs[mock_runs["condition"] == c][field].mean()) if c in set(mock_runs["condition"]) else np.nan for c in conds]
        l_vals = [float(live_runs[live_runs["condition"] == c][field].mean()) for c in conds]
        ax.bar(x - width / 2, m_vals, width, label="Simulation", color=PALETTE[0])
        ax.bar(x + width / 2, l_vals, width, label="Live Groq", color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS.get(c, c).split()[0] for c in conds])
        ax.set_title(title)
        out[field] = {"mock": [round(v, 4) for v in m_vals], "live": [round(v, 4) for v in l_vals]}
    axes[0].set_ylim(0, 1.15)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2,
               frameon=False)
    fig.suptitle("Simulation vs live LLM execution", y=1.02)
    fig.savefig(path)
    plt.close(fig)
    out["conditions"] = conds
    return out



def fig_architecture(path: str) -> None:
    """Install the authored architecture diagram as the Chapter 4 figure.

    The diagram is a hand-authored asset rather than a generated plot, so this
    copies it verbatim; regenerating the report therefore cannot silently
    replace it with a drawn approximation.
    """
    import shutil

    src = os.path.join("report_assets", "images", "architecture.png")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Architecture diagram missing at {src}. Place the authored PNG there."
        )
    shutil.copyfile(src, path)


def fig_fitness_comparability(runs: pd.DataFrame, path: str) -> dict:
    """Show why raw fitness cannot be compared across conditions, and the fix."""
    conds = [c for c in CONDITIONS if c in set(runs["condition"])]
    x = np.arange(len(conds))
    width = 0.38

    raw = [float(runs[runs["condition"] == c]["peak_fitness"].mean()) for c in conds]
    norm = [float(runs[runs["condition"] == c]["peak_fitness_normalised"].mean()) for c in conds]
    mass = [float(runs[runs["condition"] == c]["mean_active_weight_mass"].mean()) for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    # The right panel's y-axis label was colliding with the left panel.
    fig.subplots_adjust(wspace=0.42)
    axes[0].bar(x - width / 2, raw, width, label="Raw $F$", color=PALETTE[3])
    axes[0].bar(x + width / 2, norm, width, label="Normalized $F$", color=PALETTE[0])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CONDITION_LABELS[c].split()[0] for c in conds])
    axes[0].set_ylabel("Mean peak fitness")
    axes[0].set_title("Raw vs normalized fitness")
    _legend_below(axes[0], ncol=2, y=-0.16)

    axes[1].bar(x, mass, 0.6, color=PALETTE[2])
    axes[1].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CONDITION_LABELS[c].split()[0] for c in conds])
    axes[1].set_ylabel("Active weight mass")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Weight mass summed, $\\sum_{j \\in x_E} w_j$")

    fig.suptitle("Why raw fitness is not comparable across conditions", y=1.03)
    fig.savefig(path)
    plt.close(fig)
    return {c: {"raw": round(r, 4), "normalized": round(n, 4), "weight_mass": round(m, 4)}
            for c, r, n, m in zip(conds, raw, norm, mass)}


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Build report statistics, tables and figures")
    ap.add_argument("--mock-csv", nargs="+", required=True)
    ap.add_argument("--live-csv", nargs="*", default=[])
    ap.add_argument("--outdir", default="report_assets")
    args = ap.parse_args()

    figdir = os.path.join(args.outdir, "figures")
    tabdir = os.path.join(args.outdir, "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    print("Loading primary ablation dataset:")
    mock = load_and_merge(args.mock_csv)
    live = None
    if args.live_csv:
        print("Loading supplementary validation:")
        try:
            live = load_and_merge(args.live_csv)
        except SystemExit:
            live = None

    mock = add_normalised_fitness(mock)
    if live is not None:
        live = add_normalised_fitness(live)

    mock_runs = per_run_summary(mock)
    live_runs = per_run_summary(live) if live is not None else None

    data: Dict = {
        "dataset": {
            "mock_rows": int(len(mock)),
            "mock_runs": int(len(mock_runs)),
            "mock_generations": int(mock.groupby(["task_id", "condition", "run_index"])["generation"].max().sum()),
            "conditions": _present_conditions(mock),
            "tasks": sorted(mock["task_id"].unique().tolist()),
            "n_tasks": int(mock["task_id"].nunique()),
            "n_runs_per_cell": int(mock["run_index"].nunique()),
            "generations_per_run": int(mock["generation"].max()),
            "domains": sorted(mock["domain"].unique().tolist()),
        }
    }
    # ---- provenance: evidence that the primary dataset came from real completions
    if {"live_calls", "mock_calls"} <= set(mock.columns):
        cells = mock.drop_duplicates(subset=["task_id", "condition", "run_index"])
        n_live = int(cells["live_calls"].fillna(0).sum())
        n_mock = int(cells["mock_calls"].fillna(0).sum())
        data["provenance"] = {
            "api_calls_live": n_live,
            "api_calls_mock": n_mock,
            "models": sorted(mock["model"].dropna().unique().tolist()),
            "is_fully_live": n_mock == 0,
        }
        if n_mock:
            raise SystemExit(
                f"REFUSING TO BUILD: {n_mock} mock calls are present in the primary "
                f"dataset. The report would misreport fabricated rows as live.")
        print(f"  provenance: {n_live} live API calls, 0 mock")
    else:
        data["provenance"] = {"api_calls_live": None, "api_calls_mock": None,
                              "models": [], "is_fully_live": False}
        print("  ! primary dataset carries no provenance columns")

    if live is not None:
        data["dataset"]["live_rows"] = int(len(live))
        data["dataset"]["live_runs"] = int(len(live_runs))
        data["dataset"]["live_generations"] = int(
            live.groupby(["task_id", "condition", "run_index"])["generation"].max().sum())

    # ---- tables
    cond_tbl = condition_summary(mock_runs)
    dom_tbl = domain_summary(mock_runs)
    pooled = pooled_wilcoxon(mock_runs, metric="peak_fitness_normalised")
    pooled_raw = pooled_wilcoxon(mock_runs, metric="peak_fitness")
    pooled_corr = pooled_wilcoxon(mock_runs, metric="peak_correctness")
    per_task = compute_wilcoxon_paired_tests(mock)
    conv = compute_convergence_analysis(mock)

    cond_tbl.to_csv(os.path.join(tabdir, "condition_summary.csv"), index=False)
    dom_tbl.to_csv(os.path.join(tabdir, "domain_summary.csv"), index=False)
    pooled.to_csv(os.path.join(tabdir, "wilcoxon_pooled_normalised.csv"), index=False)
    pooled_raw.to_csv(os.path.join(tabdir, "wilcoxon_pooled_raw.csv"), index=False)
    pooled_corr.to_csv(os.path.join(tabdir, "wilcoxon_pooled_correctness.csv"), index=False)
    per_task.to_csv(os.path.join(tabdir, "wilcoxon_per_task.csv"), index=False)
    conv.to_csv(os.path.join(tabdir, "convergence_by_task_condition.csv"), index=False)
    mock_runs.to_csv(os.path.join(tabdir, "per_run_summary.csv"), index=False)

    data["condition_summary"] = cond_tbl.to_dict("records")
    data["domain_summary"] = dom_tbl.to_dict("records")
    data["wilcoxon_pooled"] = pooled.to_dict("records")
    data["wilcoxon_pooled_raw"] = pooled_raw.to_dict("records")
    data["wilcoxon_pooled_correctness"] = pooled_corr.to_dict("records")
    data["wilcoxon_per_task_significant"] = int(
        per_task["statistically_significant_0_05"].sum()) if len(per_task) else 0
    data["wilcoxon_per_task_total"] = int(len(per_task))

    # ---- figures
    fig_architecture(os.path.join(figdir, "fig_architecture.png"))
    data["convergence"] = fig_convergence(mock, os.path.join(figdir, "fig_convergence.png"))
    data["cost"] = fig_cost_trajectory(mock, os.path.join(figdir, "fig_cost_trajectory.png"))
    fig_cost_vs_fitness(mock_runs, os.path.join(figdir, "fig_cost_vs_fitness.png"))
    data["comparability"] = fig_fitness_comparability(
        mock_runs, os.path.join(figdir, "fig_fitness_comparability.png"))
    data["final_weights"] = fig_weight_trajectory(mock, os.path.join(figdir, "fig_weight_trajectory.png"))
    data["activation"] = fig_evaluator_activation(mock, os.path.join(figdir, "fig_evaluator_activation.png"))
    data["metric_profile"] = fig_metric_profile(mock, os.path.join(figdir, "fig_metric_profile.png"))
    data["mutation_usage"] = fig_mutation_usage(mock, os.path.join(figdir, "fig_mutation_usage.png"))

    # ---- ARC-AGI experiment (separate run, separate evaluator suite)
    try:
        from arc_analysis import build_arc_assets
        data["arc"] = build_arc_assets(args.outdir)
        a = data["arc"]
        print(f"  ARC: {a['official_final_n']}/{a['n_tasks']} tasks solved "
              f"({a['official_final_pct']:.1f}%), {a['n_programs']} programs")
    except SystemExit as e:
        print(f"  ! ARC assets skipped: {e}")
        data["arc"] = None

    if live_runs is not None and not live_runs.empty:
        live_cond = condition_summary(live_runs)
        live_cond.to_csv(os.path.join(tabdir, "live_condition_summary.csv"), index=False)
        data["live_condition_summary"] = live_cond.to_dict("records")
        data["live_pooled_wilcoxon"] = pooled_wilcoxon(live_runs).to_dict("records")
        data["live_vs_mock"] = fig_live_vs_mock(
            live_runs, mock_runs, os.path.join(figdir, "fig_live_vs_mock.png"))

    with open(os.path.join(args.outdir, "report_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print()
    print("=" * 78)
    print("CONDITION SUMMARY")
    print("=" * 78)
    print(cond_tbl.to_string(index=False))
    print()
    print("=" * 78)
    print("POOLED WILCOXON on NORMALISED fitness (C5 vs each baseline)")
    print("=" * 78)
    print(pooled.to_string(index=False) if len(pooled) else "(insufficient data)")
    print()
    print("=" * 78)
    print("POOLED WILCOXON on CORRECTNESS mu_1 (C5 vs each baseline)")
    print("=" * 78)
    print(pooled_corr.to_string(index=False) if len(pooled_corr) else "(insufficient data)")
    print()
    print(f"wrote figures -> {figdir}")
    print(f"wrote tables  -> {tabdir}")
    print(f"wrote scalars -> {os.path.join(args.outdir, 'report_data.json')}")


if __name__ == "__main__":
    main()
