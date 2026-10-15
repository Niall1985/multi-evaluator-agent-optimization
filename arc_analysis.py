"""ARC-AGI experiment statistics, tables and figures for Chapter 6 of the report.

Consumes the artefacts written by scripts/live_arc_run.py on the
arc-agi-integration branch:

    results/arc/live_arc_agent_results.csv     one row per child agent
    results/arc/live_arc_program_results.csv   one row per program written
    results/arc/live_arc_heldout_results.csv   final held-out pass@2 per task
    results/arc/live_arc_archive.log           per-generation run log

The official ARC score is strict: a task counts only when every held-out input
matches exactly. The mean of per-task `held_out_pass_at_2` gives partial credit
on the four two-input tasks and therefore reads higher; both are reported, with
the strict figure as the headline.
"""
import os
import re
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from report_analysis import PALETTE, _dash, _legend_below

ARC_DIR = os.path.join("results", "arc")
# In-loop evaluator pool. arc_pass_at_2_test is deliberately absent: the official
# held-out score is computed out-of-band so it can never feed selection.
ARC_METRICS = ["arc_runs_successfully", "arc_pass_at_2_train", "arc_pixel_accuracy",
               "arc_shape_match", "arc_color_palette"]
ARC_LABELS = {
    "arc_runs_successfully": "runs successfully",
    "arc_pass_at_2_train": "pass@2 (demonstrations)",
    "arc_pixel_accuracy": "pixel accuracy *",
    "arc_shape_match": "shape match *",
    "arc_color_palette": "colour palette *",
}
N_TASKS = 120


def load_arc(d: str = ARC_DIR) -> Dict[str, object]:
    """Loads the three CSVs and the per-generation trajectory parsed from the log."""
    out: Dict[str, object] = {}
    for key, name in [("agents", "live_arc_agent_results.csv"),
                      ("programs", "live_arc_program_results.csv"),
                      ("heldout", "live_arc_heldout_results.csv")]:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise SystemExit(f"ARC results missing: {p}")
        out[key] = pd.read_csv(p)
    out["trajectory"] = parse_log(os.path.join(d, "live_arc_archive.log"))
    return out


def parse_log(path: str) -> pd.DataFrame:
    """Per-generation held-out score. Only the log carries the trajectory: the
    held-out CSV is rewritten each generation and so holds the final one only."""
    if not os.path.exists(path):
        return pd.DataFrame()
    txt = open(path, encoding="utf-8", errors="replace").read()
    rows = []
    for gen, m in enumerate(re.finditer(
            r"held-out pass@2 \(archive picks\): ([\d.]+)/(\d+) = ([\d.]+) solved=\[(.*?)\]", txt)):
        total, denom, frac, solved = m.groups()
        ids = [s.strip().strip("'\"") for s in solved.split(",") if s.strip()]
        rows.append({"generation": gen, "mean_pass_at_2": float(frac),
                     "tasks_solved": len(ids), "n_tasks": int(denom),
                     "official_pct": len(ids) / int(denom) * 100,
                     "solved_ids": ids})
    return pd.DataFrame(rows)


def arc_summary(data: Dict[str, object]) -> Dict:
    """Every ARC scalar the report quotes."""
    ag, pr, ho = data["agents"], data["programs"], data["trajectory"]
    hd = data["heldout"]
    final = ho.iloc[-1] if len(ho) else None
    first = ho.iloc[0] if len(ho) else None

    demos = ag.groupby("generation")["solved_tasks"].max().dropna()
    # solved_tasks is recorded on the seed row only; the cumulative count is
    # recoverable from the per-generation new_solves column.
    seed_solved = float(demos.iloc[0]) if len(demos) else 0.0
    new_solves = ag[ag.generation > 0].groupby("generation")["new_solves"].sum()
    cumulative = (seed_solved + new_solves.cumsum()).round().astype(int)

    return {
        "n_tasks": int(final["n_tasks"]) if final is not None else N_TASKS,
        "generations": int(ag.generation.max()),
        "model": "Qwen3.8-27B",
        "provider": "self-hosted vLLM (Modal)",
        "islands": int(ag[ag.generation > 0]["island"].nunique()) if "island" in ag else 0,
        "children_per_generation": int(ag[ag.generation > 0].groupby("generation").size().mode()[0]),
        "tasks_per_child": 30,
        # official = strict ARC rule, every held-out input of the task must match
        "official_start_pct": float(first["official_pct"]) if first is not None else 0.0,
        "official_final_pct": float(final["official_pct"]) if final is not None else 0.0,
        "official_start_n": int(first["tasks_solved"]) if first is not None else 0,
        "official_final_n": int(final["tasks_solved"]) if final is not None else 0,
        # mean of per-task scores: partial credit on two-input tasks, reads higher
        "mean_start_pct": float(first["mean_pass_at_2"]) * 100 if first is not None else 0.0,
        "mean_final_pct": float(final["mean_pass_at_2"]) * 100 if final is not None else 0.0,
        "demos_start": int(seed_solved),
        "demos_final": int(cumulative.iloc[-1]) if len(cumulative) else int(seed_solved),
        "n_programs": int(len(pr)),
        "n_elites": int(pr.is_elite.sum()),
        "n_cells": int(pr.cell.nunique()),
        "programs_solved": int(pr.solved.sum()),
        "prompt_tokens": int(pr.prompt_tokens.fillna(0).sum()),
        "completion_tokens": int(pr.completion_tokens.fillna(0).sum()),
        "reasoning_tokens": int(pr.reasoning_tokens.fillna(0).sum()),
        "truncated_pct": float((pr.finish_reason == "length").mean() * 100),
        "repairs": int(ag.repairs.fillna(0).sum()),
        "solved_ids": list(final["solved_ids"]) if final is not None else [],
        "partial_tasks": int(((hd.held_out_pass_at_2 > 0) & (hd.held_out_pass_at_2 < 1)).sum()),
        "generalisation_gap": (int(cumulative.iloc[-1]) if len(cumulative) else 0)
                              - (int(final["tasks_solved"]) if final is not None else 0),
    }


def fig_arc_progress(data: Dict[str, object], path: str) -> Dict:
    """Held-out score and demonstration fits against generation."""
    ho, ag = data["trajectory"], data["agents"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax1.plot(ho.generation, ho.official_pct, marker="o", ms=3.5, lw=1.6,
             color=PALETTE[0], label="official (all inputs match)")
    l, = ax1.plot(ho.generation, ho.mean_pass_at_2 * 100, marker="s", ms=3.0, lw=1.3,
                  color=PALETTE[1], label="mean per held-out input")
    _dash(l, 1)
    ax1.set_xlabel("generation")
    ax1.set_ylabel(f"held-out pass@2 (% of {int(ho.n_tasks.iloc[0])} tasks)")
    ax1.set_title("ARC-AGI held-out score")
    ax1.set_ylim(bottom=0)
    _legend_below(ax1, ncol=1, y=-0.30)

    seed = float(ag.groupby("generation")["solved_tasks"].max().dropna().iloc[0])
    cum = (seed + ag[ag.generation > 0].groupby("generation")["new_solves"].sum().cumsum())
    gens = [0] + list(cum.index)
    vals = [seed] + list(cum.values)
    ax2.step(gens, vals, where="post", color=PALETTE[2], lw=1.6)
    ax2.fill_between(gens, vals, step="post", alpha=0.18, color=PALETTE[2])
    ax2.set_xlabel("generation")
    ax2.set_ylabel("tasks")
    ax2.set_title("Tasks fitting their demonstrations")
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"final_official": float(ho.official_pct.iloc[-1]),
            "final_demos": int(vals[-1])}


def fig_arc_metrics(data: Dict[str, object], path: str) -> Dict:
    """The five in-loop evaluators against generation; partial-credit ones dashed."""
    ag = data["agents"]
    g = ag.groupby("generation")[ARC_METRICS].mean()
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    for i, m in enumerate(ARC_METRICS):
        line, = ax.plot(g.index, g[m], marker="o", ms=2.6, lw=1.4,
                        color=PALETTE[i % len(PALETTE)], label=ARC_LABELS[m])
        if m in ("arc_pixel_accuracy", "arc_shape_match", "arc_color_palette"):
            _dash(line, i)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean score over children")
    ax.set_title("ARC evaluator scores (* = AI-generated partial credit, not ARC metrics)")
    ax.set_ylim(0, 1.0)
    _legend_below(ax, ncol=3, y=-0.30)
    fig.savefig(path)
    plt.close(fig)
    return {m: round(float(g[m].iloc[-1]), 4) for m in ARC_METRICS}


def fig_arc_weights(data: Dict[str, object], path: str) -> Dict:
    """GP-UCB weight on each ARC evaluator, generation by generation."""
    ag = data["agents"]
    cols = [f"weight_{m}" for m in ARC_METRICS]
    g = ag.groupby("generation")[cols].mean()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.stackplot(g.index, *[g[c] for c in cols],
                 labels=[ARC_LABELS[m] for m in ARC_METRICS],
                 colors=[PALETTE[i % len(PALETTE)] for i in range(len(cols))], alpha=0.85)
    ax.set_xlabel("generation")
    ax.set_ylabel("weight $w_j$")
    ax.set_title("GP-UCB weight vector on the ARC evaluator simplex")
    ax.set_ylim(0, 1)
    ax.set_xlim(g.index.min(), g.index.max())
    ax.grid(False)
    _legend_below(ax, ncol=3, y=-0.30)
    fig.savefig(path)
    plt.close(fig)
    return {"final": {m: round(float(g[f"weight_{m}"].iloc[-1]), 3) for m in ARC_METRICS}}


def fig_arc_archive(data: Dict[str, object], path: str) -> Dict:
    """Archive growth and how the programs distribute over MAP-Elites cells."""
    pr = data["programs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    by_gen = pr.groupby("generation").size().cumsum()
    el = pr[pr.is_elite == 1].groupby("generation").size().reindex(by_gen.index, fill_value=0).cumsum()
    ax1.plot(by_gen.index, by_gen.values, lw=1.6, color=PALETTE[0], label="programs written")
    l, = ax1.plot(el.index, el.values, lw=1.4, color=PALETTE[3], label="cell-holding elites")
    _dash(l, 1)
    ax1.set_xlabel("generation")
    ax1.set_ylabel("cumulative count")
    ax1.set_title("Program archive growth")
    _legend_below(ax1, ncol=1, y=-0.30)

    occ = pr.groupby("cell").size().sort_values(ascending=False).head(12)
    ax2.barh(range(len(occ)), occ.values, color=PALETTE[4], alpha=0.85)
    ax2.set_yticks(range(len(occ)))
    ax2.set_yticklabels([str(c)[:18] for c in occ.index], fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("programs")
    ax2.set_title("Most occupied MAP-Elites cells")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"programs": int(len(pr)), "elites": int(pr.is_elite.sum()),
            "cells": int(pr.cell.nunique())}


def arc_generation_table(data: Dict[str, object]) -> pd.DataFrame:
    """Per-generation summary for the Chapter 6 table."""
    ag, ho = data["agents"], data["trajectory"]
    g = ag.groupby("generation").agg(
        children=("agent_id", "size"),
        tasks_attempted=("tasks_attempted", "sum"),
        repairs=("repairs", "sum"),
        new_solves=("new_solves", "sum"),
        mean_fitness=("fitness", "mean"),
    )
    g = g.join(ho.set_index("generation")[["tasks_solved", "official_pct"]], how="left")
    g["repairs"] = g["repairs"].fillna(0).astype(int)
    g["new_solves"] = g["new_solves"].fillna(0).astype(int)
    return g.reset_index().round(4)


def build_arc_assets(outdir: str = "report_assets") -> Dict:
    """Writes every ARC figure and table; returns the scalars for report_data.json."""
    figdir, tabdir = os.path.join(outdir, "figures"), os.path.join(outdir, "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    data = load_arc()
    out = arc_summary(data)
    out["progress"] = fig_arc_progress(data, os.path.join(figdir, "fig_arc_progress.png"))
    out["metrics"] = fig_arc_metrics(data, os.path.join(figdir, "fig_arc_metrics.png"))
    out["weights"] = fig_arc_weights(data, os.path.join(figdir, "fig_arc_weights.png"))
    out["archive"] = fig_arc_archive(data, os.path.join(figdir, "fig_arc_archive.png"))

    tbl = arc_generation_table(data)
    tbl.to_csv(os.path.join(tabdir, "arc_generation_summary.csv"), index=False)
    out["generation_table"] = tbl.to_dict("records")

    hd = data["heldout"].sort_values("held_out_pass_at_2", ascending=False)
    hd.to_csv(os.path.join(tabdir, "arc_heldout_by_task.csv"), index=False)
    out["solved_table"] = hd[hd.held_out_pass_at_2 > 0].to_dict("records")
    return out


if __name__ == "__main__":
    import json
    s = build_arc_assets()
    print(json.dumps({k: v for k, v in s.items()
                      if not isinstance(v, (list, dict))}, indent=2))
