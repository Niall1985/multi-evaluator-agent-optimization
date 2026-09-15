"""
Plotly visualisations for ARC-AGI benchmark results (no Streamlit dependency so
they can be unit-tested and reused from notebooks / analyze_results.py).

- `task_gallery_figure`: one row per demonstration / held-out pair, columns
  Input | Expected | Predicted, drawn with the official ARC colour palette.
- `pass_at_2_trajectory_figure`: real (train / test pass@2) and partial-credit
  scores per generation.
- `benchmark_summary_frame`: best-so-far ARC scores per task from the archive.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from benchmarks.benchmarks_tasks import BenchmarkTask

from .scoring import grids_equal, pixel_accuracy, score_arc_cases, to_grid

# Official ARC task-viewer palette (arcprize.org / fchollet/ARC-AGI apps/css), index == colour value.
ARC_PALETTE: Tuple[str, ...] = (
    "#000000",  # 0 black (background)
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 sky blue
    "#870C25",  # 9 maroon
)

# Discrete colourscale: value v is drawn with ARC_PALETTE[v] when zmin=-0.5, zmax=9.5.
ARC_COLORSCALE: List[List[Any]] = []
for _i, _hex in enumerate(ARC_PALETTE):
    ARC_COLORSCALE.append([_i / 10, _hex])
    ARC_COLORSCALE.append([(_i + 1) / 10, _hex])

MISSING_GRID = [[-1]]  # sentinel drawn as an empty cell when a program returned nothing usable

# Metric names the ARC pool registers (kept here so the app can stay generic)
REAL_METRICS = ("arc_runs_successfully", "arc_pass_at_2_train", "arc_pass_at_2_test")
PARTIAL_METRICS = ("arc_pixel_accuracy", "arc_shape_match", "arc_color_palette")


def grid_trace(grid: Optional[Sequence[Sequence[int]]], showscale: bool = False) -> go.Heatmap:
    """Heatmap trace for one grid; rows are drawn top-down like the ARC viewer."""
    g = to_grid(grid) if grid is not None else None
    z = g if g is not None else MISSING_GRID
    return go.Heatmap(
        z=z[::-1],  # plotly's y axis points up; reverse so row 0 is on top
        zmin=-0.5,
        zmax=9.5,
        colorscale=ARC_COLORSCALE,
        showscale=showscale,
        xgap=1,
        ygap=1,
        hovertemplate="row %{customdata}<br>col %{x}<br>colour %{z}<extra></extra>",
        customdata=[[len(z) - 1 - r] * len(z[0]) for r in range(len(z))],
    )


def _pair_title(kind: str, idx: int, predicted: Any, expected: Sequence[Sequence[int]]) -> str:
    if predicted is None or to_grid(predicted) is None:
        return f"{kind} {idx + 1} — Predicted ✗ (no grid)"
    ok = grids_equal(predicted, expected)
    acc = pixel_accuracy(predicted, expected)
    mark = "✓ exact" if ok else f"✗ {acc:.0%} px"
    return f"{kind} {idx + 1} — Predicted {mark}"


def task_gallery_figure(
    task: BenchmarkTask,
    agent_output: str,
    include_held_out: bool = True,
    cell_px: int = 22,
) -> go.Figure:
    """Input | Expected | Predicted for every pair of `task`, running `agent_output` to get predictions.

    Predicted uses attempt 1 (or `solve`); when the program exposes a second attempt and only
    that one is correct, the second attempt is shown instead so the pass@2 credit is visible.
    """
    sections: List[Tuple[str, List[Dict[str, Any]], str]] = [("Demonstration", task.test_cases, "train_example")]
    if include_held_out and task.edge_cases:
        sections.append(("Held-out", task.edge_cases, "test_example"))

    rows: List[Tuple[str, int, Any, Any, Any]] = []  # kind, idx, input, expected, predicted
    for kind, cases, prefix in sections:
        metrics = score_arc_cases(cases, agent_output, task.entry_point, prefix=prefix)
        outputs = metrics.get("outputs", []) or [[None] for _ in cases]
        for i, (case, attempts) in enumerate(zip(cases, outputs)):
            expected = case["expected"]
            shown = attempts[0] if attempts else None
            if len(attempts) > 1 and not grids_equal(shown, expected) and grids_equal(attempts[1], expected):
                shown = attempts[1]
            rows.append((kind, i, case["args"][0], expected, shown))

    n = max(len(rows), 1)
    titles: List[str] = []
    for kind, i, _in, expected, predicted in rows:
        titles += [f"{kind} {i + 1} — Input", f"{kind} {i + 1} — Expected", _pair_title(kind, i, predicted, expected)]
    if not rows:
        titles = ["Input", "Expected", "Predicted"]

    fig = make_subplots(rows=n, cols=3, subplot_titles=titles, horizontal_spacing=0.06, vertical_spacing=0.6 / n)
    max_dim = 1
    for r, (kind, i, grid_in, expected, predicted) in enumerate(rows, start=1):
        for c, g in enumerate((grid_in, expected, predicted), start=1):
            fig.add_trace(grid_trace(g), row=r, col=c)
            gg = to_grid(g) if g is not None else None
            if gg:
                max_dim = max(max_dim, len(gg), len(gg[0]))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain")
    for ann in fig.layout.annotations:
        ann.font = dict(size=11)
        if "✓" in ann.text:
            ann.font.color = "#10B981"
        elif "✗" in ann.text:
            ann.font.color = "#DC2626"
    fig.update_layout(
        height=max(260, n * max(160, min(max_dim, 30) * cell_px // 2 + 90)),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pass_at_2_trajectory_figure(records: Iterable[Dict[str, Any]]) -> go.Figure:
    """Per-generation ARC scores: real metrics as solid lines, partial-credit ones dashed."""
    recs = [r for r in records if isinstance(r.get("metrics"), dict)]
    fig = go.Figure()
    styles = {
        "arc_pass_at_2_train": ("Pass@2 — demonstrations (real)", "#2563EB", "solid"),
        "arc_pass_at_2_test": ("Pass@2 — held-out (real, official)", "#DC2626", "solid"),
        "arc_runs_successfully": ("Runs successfully (real)", "#64748B", "dot"),
        "arc_pixel_accuracy": ("Pixel accuracy (partial, AI-gen)", "#10B981", "dash"),
        "arc_shape_match": ("Shape match (partial, AI-gen)", "#F59E0B", "dash"),
        "arc_color_palette": ("Colour palette (partial, AI-gen)", "#8B5CF6", "dash"),
    }
    for key, (label, colour, dash) in styles.items():
        pts = [(r["generation"], r["metrics"][key]) for r in recs if key in r["metrics"]]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=label,
            line=dict(color=colour, width=2, dash=dash), marker=dict(size=7),
        ))
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Score", yaxis=dict(range=[-0.05, 1.05]),
        height=340, margin=dict(l=30, r=30, t=30, b=30), legend=dict(orientation="h", y=-0.25),
    )
    return fig


def benchmark_summary_frame(agents: Iterable[Any]) -> pd.DataFrame:
    """Best-so-far ARC metrics per task across archive agents (task ids starting with `arc_`)."""
    rows: Dict[str, Dict[str, Any]] = {}
    for a in agents:
        tid = getattr(a, "task_id", None)
        if not tid or not str(tid).startswith("arc_") or not getattr(a, "metrics", None):
            continue
        row = rows.setdefault(tid, {"Task": tid, "Candidates": 0, "Best Fitness": float("-inf"), "Best Agent": ""})
        row["Candidates"] += 1
        if a.fitness > row["Best Fitness"]:
            row["Best Fitness"], row["Best Agent"] = a.fitness, a.id
        for key in REAL_METRICS + PARTIAL_METRICS:
            if key in a.metrics:
                row[key] = max(row.get(key, 0.0), a.metrics[key])
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    df["Best Fitness"] = df["Best Fitness"].round(4)
    if "arc_pass_at_2_test" in df.columns:
        df["Solved (official)"] = df["arc_pass_at_2_test"] >= 1.0
    return df.sort_values("Task").reset_index(drop=True)
