import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from controller import EvolutionController
from core.agent import AGENT_ARCHETYPES
from benchmarks.benchmarks_tasks import BENCHMARK_TASKS
from benchmarks.arc_challenge import (
    ARC_TASKS,
    benchmark_summary_frame,
    build_arc_evaluator_pool,
    get_arc_task,
    pass_at_2_trajectory_figure,
    task_gallery_figure,
)

# ---------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------
def _load_env_file(filepath: str = ".env"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Objective Agent Optimization (BCSE497J)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-core {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-deep {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .panelist-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
EVALUATOR_SUITES = {
    "Canonical (6 mu evaluators)": None,
    "ARC-AGI (3 real + 3 partial)": build_arc_evaluator_pool,
}


def _build_controller(suite_name: str, **overrides) -> EvolutionController:
    kwargs = dict(
        api_key=os.getenv("GROQ_API_KEY", ""),
        model="openai/gpt-oss-120b",
        lambda_penalty=0.5,
        default_strategy="adaptive",
        initial_archetype="General Balanced Assistant",
        inter_call_delay=1.5,
        evaluator_pool_factory=EVALUATOR_SUITES[suite_name],
    )
    kwargs.update(overrides)
    return EvolutionController(**kwargs)


if "evaluator_suite" not in st.session_state:
    st.session_state.evaluator_suite = "Canonical (6 mu evaluators)"
if "controller" not in st.session_state:
    st.session_state.controller = _build_controller(st.session_state.evaluator_suite)
if "generation_count" not in st.session_state:
    st.session_state.generation_count = 0


ctrl: EvolutionController = st.session_state.controller

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Optimization Configuration")
    
    current_key = ctrl.api_key or os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key (Auto-loaded from .env)",
        value=current_key,
        type="password",
        help="Reads GROQ_API_KEY from .env file or system environment. If blank, deterministic simulation mock is used.",
    )
    if api_key_input != ctrl.api_key:
        ctrl.api_key = api_key_input
        ctrl.llm_client.api_key = api_key_input
        ctrl.llm_client.is_mock = not bool(api_key_input.strip())

    model_options = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "Custom Model...",
    ]
    model_selection = st.selectbox("Groq LLM Model", options=model_options, index=0)
    if model_selection == "Custom Model...":
        custom_model = st.text_input("Enter Custom Groq Model Identifier", value=ctrl.model)
        selected_model = custom_model.strip()
    else:
        selected_model = model_selection

    if selected_model != ctrl.model:
        ctrl.model = selected_model
        ctrl.llm_client.model = selected_model

    delay_slider = st.slider(
        "Inter-Call Delay (Pacing for TPM Rate Limits)",
        min_value=0.0,
        max_value=5.0,
        value=float(ctrl.inter_call_delay),
        step=0.5,
        help="Adds delay between consecutive LLM calls to prevent Groq Tokens-Per-Minute (TPM) rate limit throttling.",
    )
    ctrl.inter_call_delay = delay_slider
    ctrl.llm_client.inter_call_delay = delay_slider

    st.markdown("---")
    st.markdown("### Selectable Agent Archetypes")
    
    archetype_choices = list(AGENT_ARCHETYPES.keys())
    selected_archetype = st.selectbox(
        "Initial Seed Agent Persona",
        options=archetype_choices,
        index=archetype_choices.index(ctrl.initial_archetype) if ctrl.initial_archetype in archetype_choices else 0,
        help="Choose among 6 distinct specialized agent archetypes to initialize and evolve.",
    )
    st.caption(AGENT_ARCHETYPES[selected_archetype]["description"])

    st.markdown("---")
    st.markdown("### Selectable Benchmark Task")

    task_choices = ["All Benchmark Tasks (Suite / Random)"] + [t.name for t in BENCHMARK_TASKS] + [t.name for t in ARC_TASKS]
    current_task_idx = 0
    if ctrl.selected_task_id:
        for idx, t_name in enumerate(task_choices):
            if t_name == ctrl.selected_task_id:
                current_task_idx = idx
                break

    selected_task_name = st.selectbox(
        "Benchmark Problem / Task Input",
        options=task_choices,
        index=current_task_idx,
        help="Select a specific coding/reasoning problem (e.g. Stock Exchange, Fibonacci, or an ARC-AGI grid puzzle) or test across the full suite.",
    )
    ctrl.selected_task_id = None if selected_task_name == "All Benchmark Tasks (Suite / Random)" else selected_task_name
    
    if selected_task_name != "All Benchmark Tasks (Suite / Random)":
        matched_task = next((t for t in BENCHMARK_TASKS + ARC_TASKS if t.name == selected_task_name), None)
        if matched_task:
            st.caption(f"**Category:** {matched_task.category}\n\n**Goal:** {matched_task.description[:120]}...")

    st.markdown("---")
    st.markdown("### Search & Penalty Knobs")

    strategy_mode = st.selectbox(
        "Evolution Strategy Condition",
        options=["full_adaptive", "no_pruning", "ucb1_bandit", "static_cascade", "single_metric"],
        format_func=lambda x: {
            "full_adaptive": "Condition 5: Full Joint Adaptive (Gaps 1, 2, 3)",
            "no_pruning": "Condition 4: Adaptive Weights, No Pruning",
            "ucb1_bandit": "Condition 3: UCB1 Bandit Mutations, Uniform Weights",
            "static_cascade": "Condition 2: Static Cascade (Uniform Weights)",
            "single_metric": "Condition 1: Single Metric (mu_1 Correctness Only)",
        }.get(x, x),
        index=0 if ctrl.default_strategy in ["full_adaptive", "adaptive"] else (
            ["full_adaptive", "no_pruning", "ucb1_bandit", "static_cascade", "single_metric"].index(ctrl.default_strategy)
            if ctrl.default_strategy in ["full_adaptive", "no_pruning", "ucb1_bandit", "static_cascade", "single_metric"]
            else 0
        ),
    )
    ctrl.default_strategy = strategy_mode

    lambda_penalty = st.slider(
        "Cost Penalty Lambda (lambda)",
        min_value=0.0,
        max_value=2.0,
        value=float(ctrl.lambda_penalty),
        step=0.05,
        help="Higher lambda heavily penalizes invoking expensive Deep Tier evaluators.",
    )
    ctrl.lambda_penalty = lambda_penalty

    st.markdown("---")
    st.markdown("### Evaluator Suite Tiers")
    suite_names = list(EVALUATOR_SUITES)
    chosen_suite = st.radio(
        "Evaluator Suite",
        options=suite_names,
        index=suite_names.index(st.session_state.evaluator_suite),
        help="Canonical: the 6 mu evaluators for coding tasks. ARC-AGI: official pass@2 metrics plus "
             "AI-generated partial-credit heuristics for grid tasks (see benchmarks/arc_challenge).",
    )
    if chosen_suite != st.session_state.evaluator_suite:
        # Metric set changes -> GP, sampler and archive must be rebuilt; start a fresh controller.
        st.session_state.evaluator_suite = chosen_suite
        st.session_state.controller = _build_controller(
            chosen_suite,
            lambda_penalty=ctrl.lambda_penalty,
            default_strategy=ctrl.default_strategy,
            initial_archetype=ctrl.initial_archetype,
            selected_task_id=ctrl.selected_task_id,
        )
        st.session_state.generation_count = 0
        st.rerun()

    def _tier_caption(tier: str) -> str:
        return "\n".join(
            f"• {ev.name} (${ev.cost:.3f})" for ev in ctrl.evaluator_pool.evaluators.values() if ev.tier == tier
        ) or "• (none)"

    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.markdown("<span class='badge-core'>Subset 1: Core Tier</span>", unsafe_allow_html=True)
        st.caption(_tier_caption("core"))
    with c_sub2:
        st.markdown("<span class='badge-deep'>Subset 2: Deep Tier</span>", unsafe_allow_html=True)
        st.caption(_tier_caption("deep"))

    st.markdown("---")
    st.markdown("### Optimization Actions")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("Run 1 Gen", use_container_width=True, type="primary"):
            with st.spinner("Executing 1 Generation..."):
                ctrl.run_generation(strategy=strategy_mode)
                st.session_state.generation_count = ctrl.current_generation
                st.rerun()

    with col_a2:
        if st.button("Run 5 Gen", use_container_width=True):
            with st.spinner("Executing 5 Generations with rate-limit pacing..."):
                ctrl.run_n_generations(5, strategy=strategy_mode)
                st.session_state.generation_count = ctrl.current_generation
                st.rerun()

    if st.button("Reset with Selected Archetype & Task", use_container_width=True):
        ctrl.reset(initial_archetype=selected_archetype, selected_task_id=ctrl.selected_task_id)
        st.session_state.generation_count = 0
        st.rerun()


# ---------------------------------------------------------
# Main Header & Team Info
# ---------------------------------------------------------
st.markdown("<div class='main-header'>Multi-Objective Agent Optimization Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>"
    "<b>Course:</b> BCSE497J – Project 1 | "
    "<b>Team:</b> Subhrojyoti Sen (23BCE1259), Kumar Shreyash (23BCE1882), Niall Francis Ajeet Dcunha (23BCE1985) | "
    "<b>Guide:</b> Dr. Sreeja P S"
    "</div>",
    unsafe_allow_html=True,
)

# Telemetry Summary KPIs
telemetry = ctrl.get_telemetry_summary()
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Generations Run", f"{telemetry['total_generations']}")
with kpi2:
    st.metric("Best Fitness", f"{telemetry['best_fitness']:.4f}")
with kpi3:
    st.metric("Total Spent", f"${telemetry['cumulative_cost_spent']:.5f}")
with kpi4:
    st.metric("Naive Cost", f"${telemetry['cumulative_naive_cost']:.5f}")
with kpi5:
    st.metric("Cost Savings", f"${telemetry['cost_saved_usd']:.5f}", f"{telemetry['cost_saved_pct']:.1f}%")

st.markdown("---")

# ---------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------
tab_panelist, tab_inspector, tab_arc, tab1, tab2, tab3, tab4 = st.tabs([
    "PRESENTATION: Graphs & Results Table",
    "GENERATION INSPECTOR: Prompts & Solutions",
    "ARC BENCHMARK: Grid Results",
    "Tab 1: Live Evolution & Pareto Radar",
    "Tab 2: Gaussian Process & Adaptive Weights (Gap 2)",
    "Tab 3: Cost Awareness & Evaluator Pruning (Gaps 1 & 3)",
    "Tab 4: Population Archive & Data Export (CSV)",
])


# ---------------------------------------------------------
# TAB PANELIST: Graphs & Results Table (All-in-One Presentation)
# ---------------------------------------------------------
with tab_panelist:
    st.markdown("### Executive Summary & Presentation Dashboard")
    st.caption("Consolidated analytical results, Pareto trade-offs, Bayesian acquisition curves, evaluator pruning matrix, and complete tabular performance records.")

    # Top Row Graphs: Radar Profile & Pareto Frontier
    p_row1_c1, p_row1_c2 = st.columns(2)
    
    best_agent = ctrl.archive.get_best_agent()
    all_agents = ctrl.archive.get_all_agents()
    seed_agent = all_agents[0] if all_agents else None

    with p_row1_c1:
        st.markdown("#### Figure 1: Multi-Objective Performance Radar (Best Agent vs Seed Baseline)")
        if best_agent and best_agent.metrics:
            categories = list(best_agent.metrics.keys())
            best_vals = [best_agent.metrics[k] for k in categories]
            
            fig_rad = go.Figure()
            # Best agent trace
            r_best = best_vals + [best_vals[0]]
            t_cat = categories + [categories[0]]
            fig_rad.add_trace(go.Scatterpolar(
                r=r_best,
                theta=t_cat,
                fill='toself',
                name=f"Evolved Best ({best_agent.id})",
                line_color='#2563EB',
                fillcolor='rgba(37, 99, 235, 0.25)',
            ))

            # Seed agent trace
            if seed_agent and seed_agent.metrics:
                seed_vals = [seed_agent.metrics.get(k, 0.0) for k in categories]
                r_seed = seed_vals + [seed_vals[0]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=r_seed,
                    theta=t_cat,
                    fill='toself',
                    name=f"Initial Seed ({seed_agent.id})",
                    line_color='#9CA3AF',
                    fillcolor='rgba(156, 163, 175, 0.15)',
                    line=dict(dash='dot'),
                ))

            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
                showlegend=True,
                height=360,
                margin=dict(l=30, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_rad, use_container_width=True)
        else:
            st.info("Execute generations to populate Figure 1.")

    with p_row1_c2:
        st.markdown("#### Figure 2: Non-Dominated Pareto Frontier (Accuracy vs Latency vs Cost)")
        pareto_agents = ctrl.archive.get_pareto_front()
        pareto_ids = {a.id for a in pareto_agents}

        # Axes depend on the active evaluator suite: (x = accuracy-like, y = secondary, hover = deep metric)
        if "mu_1_correctness" in ctrl.metric_names:
            pareto_axes = [("Correctness (mu_1)", "mu_1_correctness"), ("Latency Score (mu_2)", "mu_2_latency"), ("Reasoning (mu_4)", "mu_4_reasoning")]
        else:
            pareto_axes = [("Pass@2 Train (real)", "arc_pass_at_2_train"), ("Pixel Accuracy (partial)", "arc_pixel_accuracy"), ("Pass@2 Held-out (real)", "arc_pass_at_2_test")]
        (x_label, x_key), (y_label, y_key), (h_label, h_key) = pareto_axes

        if all_agents:
            scatter_data = []
            for a in all_agents:
                scatter_data.append({
                    "Agent ID": a.id,
                    "Generation": a.generation,
                    x_label: a.metrics.get(x_key, 0.0),
                    y_label: a.metrics.get(y_key, 0.0),
                    h_label: a.metrics.get(h_key, 0.0),
                    "Cost ($)": a.cost_spent,
                    "Fitness": a.fitness,
                    "Pareto Status": "Pareto Optimal (Non-Dominated)" if a.id in pareto_ids else "Dominated Candidate",
                })
            df_scatter = pd.DataFrame(scatter_data)
            fig_p_scat = px.scatter(
                df_scatter,
                x=x_label,
                y=y_label,
                size="Fitness",
                color="Pareto Status",
                hover_data=["Agent ID", "Generation", "Cost ($)", h_label],
                color_discrete_map={"Pareto Optimal (Non-Dominated)": "#DC2626", "Dominated Candidate": "#3B82F6"},
            )
            fig_p_scat.update_layout(height=360, margin=dict(l=30, r=30, t=30, b=30))
            st.plotly_chart(fig_p_scat, use_container_width=True)
        else:
            st.info("Execute generations to populate Figure 2.")

    # Middle Row Graphs: Cost Savings & Evaluator Pruning Heatmap
    p_row2_c1, p_row2_c2 = st.columns(2)
    with p_row2_c1:
        st.markdown("#### Figure 3: Cumulative Cost Savings (Adaptive Pruned vs Naive Full Suite)")
        df_rec = pd.DataFrame(ctrl.archive.generation_records)
        if not df_rec.empty and "cumulative_adaptive_cost" in df_rec.columns:
            fig_p_cost = go.Figure()
            fig_p_cost.add_trace(go.Scatter(
                x=df_rec["generation"],
                y=df_rec["cumulative_naive_cost"],
                mode='lines+markers',
                name='Naive Full Suite (All 6 Evaluators)',
                line=dict(color='#DC2626', width=2, dash='dash'),
            ))
            fig_p_cost.add_trace(go.Scatter(
                x=df_rec["generation"],
                y=df_rec["cumulative_adaptive_cost"],
                mode='lines+markers',
                name='Our Selective Adaptive Suite (Gaps 1 & 3)',
                line=dict(color='#10B981', width=3),
            ))
            fig_p_cost.update_layout(
                xaxis_title="Generation",
                yaxis_title="Cumulative Cost (USD $)",
                height=340,
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_p_cost, use_container_width=True)
        else:
            st.info("Execute generations to populate Figure 3.")

    with p_row2_c2:
        st.markdown("#### Figure 4: Selective Evaluator Activation & Pruning Matrix")
        if all_agents:
            heatmap_data = []
            gen_labels = []
            for a in all_agents:
                gen_labels.append(f"G{a.generation} ({a.id[:7]})")
                row_flags = [1 if m in a.active_evaluators else 0 for m in ctrl.metric_names]
                heatmap_data.append(row_flags)

            fig_p_heat = go.Figure(data=go.Heatmap(
                z=np.array(heatmap_data).T,
                x=gen_labels,
                y=ctrl.metric_names,
                colorscale=[[0, '#F1F5F9'], [1, '#10B981']],
                showscale=False,
            ))
            fig_p_heat.update_layout(
                xaxis_title="Evaluated Candidates per Generation",
                yaxis_title="Evaluator Dimension",
                height=340,
                margin=dict(l=30, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_p_heat, use_container_width=True)
        else:
            st.info("Execute generations to populate Figure 4.")

    st.markdown("---")
    st.markdown("---")
    st.markdown("#### Comprehensive Experimental Results Table")
    df_panelist_results = ctrl.archive.to_detailed_dataframe()
    if not df_panelist_results.empty:
        # Display with download button
        col_down1, col_down2 = st.columns([3, 1])
        with col_down1:
            st.caption("Complete table of parameters, harness knobs (temperature, retries, top_p), multi-metric evaluation scores, scalarized fitness, and costs.")
        with col_down2:
            st.download_button(
                label="Download Results CSV",
                data=df_panelist_results.to_csv(index=False).encode('utf-8'),
                file_name="multi_objective_agent_optimization_results.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
        st.dataframe(df_panelist_results, use_container_width=True)

        with st.expander("Inspect Prompts and Solutions for Each Generation (Live Panelist View)", expanded=False):
            agents_all = ctrl.archive.get_all_agents()
            for ag in agents_all:
                p_ag = ctrl.archive.agents.get(ag.parent_id) if ag.parent_id else None
                st.markdown(f"##### Generation {ag.generation}: Agent `{ag.id}` (Parent: `{ag.parent_id or 'Root'}` | Fitness: `{ag.fitness:.4f}` | Cost: `${ag.cost_spent:.5f}`)")
                c_p, c_c, c_s = st.columns(3)
                with c_p:
                    st.markdown(f"**Parent Prompt (`{p_ag.id if p_ag else 'Root'}`)**")
                    st.code(p_ag.system_prompt if p_ag else "Initial Seed Archetype (No Parent)", language="text")
                with c_c:
                    st.markdown(f"**Mutated Child Prompt (`{ag.id}`)**")
                    st.caption(f"Mutation: `{ag.mutation_type}`")
                    st.code(ag.system_prompt, language="text")
                with c_s:
                    st.markdown(f"**Generated Solution Code (`{ag.task_id or 'Benchmark Task'}`)**")
                    st.code(ag.last_solution if ag.last_solution else "# Solution executed during evaluation", language="python")
                st.divider()
    else:
        st.info("No experimental records available. Run generations to populate.")


# ---------------------------------------------------------
# TAB INSPECTOR: Prompts, Lineage & Code Solutions
# ---------------------------------------------------------
with tab_inspector:
    st.subheader("Generation-by-Generation Agent Lineage, Prompts & Solutions Explorer")
    st.caption("Compare parent system prompts, mutated child system prompts, and synthesized Python code solutions across every evolutionary generation.")

    all_agents_list = ctrl.archive.get_all_agents()
    if not all_agents_list:
        st.info("No generations have been executed yet. Click 'Run 1 Gen' or 'Run 5 Gen' in the sidebar.")
    else:
        # Generation Selector
        gen_options = [
            f"Generation {a.generation}: {a.id} (Parent: {a.parent_id or 'Root'}) — Fitness: {a.fitness:.4f}"
            for a in all_agents_list
        ]
        selected_gen_idx = st.selectbox(
            "Select Generation / Agent Candidate to Inspect",
            options=range(len(gen_options)),
            format_func=lambda i: gen_options[i],
            index=len(gen_options) - 1,
        )
        selected_agent = all_agents_list[selected_gen_idx]
        parent_agent = ctrl.archive.agents.get(selected_agent.parent_id) if selected_agent.parent_id else None

        # Summary KPIs for selected generation
        st.markdown(f"#### Selected Candidate: `{selected_agent.id}` (Generation {selected_agent.generation})")
        kpi_g1, kpi_g2, kpi_g3, kpi_g4 = st.columns(4)
        with kpi_g1:
            st.metric("Fitness Score F", f"{selected_agent.fitness:.4f}")
        with kpi_g2:
            st.metric("Cost Spent", f"${selected_agent.cost_spent:.5f}")
        with kpi_g3:
            st.metric("Active Evaluators", f"{len(selected_agent.active_evaluators)} / {len(ctrl.metric_names)}")
        with kpi_g4:
            st.metric("Mutation Strategy", f"{selected_agent.mutation_type}")

        # 3-Column Comparative View
        col_view1, col_view2, col_view3 = st.columns([1, 1, 1])

        with col_view1:
            st.markdown("##### 1. Parent Agent System Prompt")
            if parent_agent:
                st.caption(f"**Parent ID:** `{parent_agent.id}` (Gen {parent_agent.generation})")
                st.caption(f"**Parent Fitness:** {parent_agent.fitness:.4f} | **Cost:** ${parent_agent.cost_spent:.5f}")
                st.caption(f"**Parent Knobs:** Temp={parent_agent.harness_knobs.temperature:.2f}, Retries={parent_agent.harness_knobs.max_retries}, Top_p={parent_agent.harness_knobs.top_p:.2f}")
                st.code(parent_agent.system_prompt, language="text")
            else:
                st.caption("**Root Baseline Agent (No Parent)**")
                st.info("This is the Generation 0 seed agent initialized from the selected archetype.")
                st.code(selected_agent.system_prompt, language="text")

        with col_view2:
            st.markdown("##### 2. Mutated Child System Prompt")
            st.caption(f"**Child ID:** `{selected_agent.id}` (Gen {selected_agent.generation})")
            st.caption(f"**Mutation Operator:** `{selected_agent.mutation_type}`")
            st.caption(f"**Child Knobs:** Temp={selected_agent.harness_knobs.temperature:.2f}, Retries={selected_agent.harness_knobs.max_retries}, Top_p={selected_agent.harness_knobs.top_p:.2f}")
            st.code(selected_agent.system_prompt, language="text")

        with col_view3:
            st.markdown("##### 3. Synthesized Python Solution Code")
            st.caption(f"**Target Task:** `{selected_agent.task_id or 'Benchmark Task'}`")
            if selected_agent.metrics:
                m_str = " | ".join([f"{k}: {v:.2f}" for k, v in selected_agent.metrics.items()])
                st.caption(f"**Scores:** {m_str}")
            if selected_agent.last_solution:
                st.code(selected_agent.last_solution, language="python")
            else:
                st.info("No solution cached for this agent.")

        # Master Table of all Generations
        st.markdown("---")
        st.markdown("#### Master Evolution History: Prompts & Solutions")
        table_rows = []
        for a in all_agents_list:
            p_a = ctrl.archive.agents.get(a.parent_id) if a.parent_id else None
            table_rows.append({
                "Gen": a.generation,
                "Agent ID": a.id,
                "Parent ID": a.parent_id or "Root",
                "Mutation": a.mutation_type,
                "Fitness": round(a.fitness, 4),
                "Cost ($)": round(a.cost_spent, 5),
                "Task": a.task_id or "Benchmark",
                "Parent Prompt Preview": (p_a.system_prompt[:50] + "...") if p_a else "Root (Initial Seed)",
                "Child Prompt Preview": (a.system_prompt[:50] + "..."),
                "Solution Preview": (a.last_solution[:60] + "...") if a.last_solution else "N/A",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


# ---------------------------------------------------------
# TAB ARC: ARC-AGI Benchmark Grid Results
# ---------------------------------------------------------
with tab_arc:
    st.subheader("ARC-AGI Benchmark: Input / Expected / Predicted Grids")
    st.caption(
        "Renders the candidate program's outputs next to the ground truth using the official ARC colour palette. "
        "Solid-line metrics are the real ARC scores (pixel-perfect pass@2); dashed ones are AI-generated partial-credit "
        "heuristics from `benchmarks/arc_challenge/partial_evaluators.py`."
    )

    arc_agents = [a for a in ctrl.archive.get_all_agents() if a.task_id and str(a.task_id).startswith("arc_")]
    if not arc_agents:
        st.info(
            "No ARC candidates yet. In the sidebar pick an `ARC …` task (and optionally the ARC-AGI evaluator suite), "
            "then run generations."
        )
    else:
        # Candidate selector, defaulting to the best ARC agent
        best_arc = max(arc_agents, key=lambda a: a.fitness)
        arc_options = [
            f"Gen {a.generation}: {a.id} — {a.task_id} — Fitness {a.fitness:.4f}" for a in arc_agents
        ]
        sel_idx = st.selectbox(
            "Candidate to visualise",
            options=range(len(arc_options)),
            format_func=lambda i: arc_options[i],
            index=arc_agents.index(best_arc),
        )
        arc_agent = arc_agents[sel_idx]
        arc_task = get_arc_task(arc_agent.task_id)

        if arc_task is None:
            st.warning(f"Task `{arc_agent.task_id}` is not loaded (check ARC_DATA_ROOT / ARC_TASK_FILE).")
        elif not arc_agent.last_solution:
            st.info("No solution cached for this candidate.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            m = arc_agent.metrics or {}
            with k1:
                st.metric("Pass@2 — demonstrations", f"{m.get('arc_pass_at_2_train', m.get('mu_1_correctness', 0.0)):.2f}")
            with k2:
                st.metric("Pass@2 — held-out (official)", f"{m.get('arc_pass_at_2_test', m.get('mu_5_edge_cases', 0.0)):.2f}")
            with k3:
                st.metric("Pixel accuracy (partial)", f"{m.get('arc_pixel_accuracy', float('nan')):.2f}")
            with k4:
                st.metric("Fitness F", f"{arc_agent.fitness:.4f}")

            st.markdown(f"##### {arc_task.name}")
            st.caption(f"{len(arc_task.test_cases)} demonstration pair(s), {len(arc_task.edge_cases)} held-out pair(s). "
                       "Predicted shows attempt 1 (or `solve`); attempt 2 is shown when only it is correct.")
            st.plotly_chart(task_gallery_figure(arc_task, arc_agent.last_solution), use_container_width=True)

            with st.expander("Candidate program", expanded=False):
                st.code(arc_agent.last_solution, language="python")

        st.markdown("---")
        c_traj, c_sum = st.columns([3, 2])
        with c_traj:
            st.markdown("#### ARC Scores over Generations")
            traj_records = [r for r in ctrl.history_records if str(r.get("task_id", "")).startswith("arc_")]
            fig_arc_traj = pass_at_2_trajectory_figure(traj_records)
            if fig_arc_traj.data:
                st.plotly_chart(fig_arc_traj, use_container_width=True)
            else:
                st.info("ARC metrics appear here when the ARC-AGI evaluator suite is active.")
        with c_sum:
            st.markdown("#### Best-so-far per ARC Task")
            df_arc = benchmark_summary_frame(ctrl.archive.get_all_agents())
            if df_arc.empty:
                st.info("No ARC metrics recorded yet.")
            else:
                st.dataframe(df_arc, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 1: Live Evolution & Pareto Radar
# ---------------------------------------------------------
with tab1:
    st.subheader("Pareto Optimization & Multi-Objective Profile")
    col_t1_left, col_t1_right = st.columns([1, 1])

    with col_t1_left:
        st.markdown("#### Best Agent Multi-Objective Radar Profile")
        if best_agent and best_agent.metrics:
            categories = list(best_agent.metrics.keys())
            values = [best_agent.metrics[k] for k in categories]
            categories.append(categories[0])
            values.append(values[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"Best ({best_agent.id})",
                line_color='#2563EB',
                fillcolor='rgba(37, 99, 235, 0.25)',
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
                showlegend=True,
                height=380,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Run at least 1 generation to populate radar profile.")

    with col_t1_right:
        st.markdown("#### Fitness Trajectory over Generations")
        records = ctrl.archive.generation_records
        if records:
            df_records = pd.DataFrame(records)
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=df_records["generation"],
                y=df_records["fitness"],
                mode='lines+markers',
                name='Agent Fitness',
                line=dict(color='#10B981', width=2),
                marker=dict(size=7),
            ))
            df_records["cum_best"] = df_records["fitness"].cummax()
            fig_traj.add_trace(go.Scatter(
                x=df_records["generation"],
                y=df_records["cum_best"],
                mode='lines',
                name='Incumbent Best',
                line=dict(color='#2563EB', width=2, dash='dash'),
            ))
            fig_traj.update_layout(
                xaxis_title="Generation",
                yaxis_title="Scalarized Fitness F(Phi, x_E, w)",
                height=380,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("No generation records yet.")


# ---------------------------------------------------------
# TAB 2: Gaussian Process & Adaptive Weights (Gap 2)
# ---------------------------------------------------------
with tab2:
    st.subheader("Gap 2: Bayesian Gaussian Process Weight Optimization & Expected Improvement")
    st.markdown(
        "The continuous weight vector w = [w_1, ..., w_k] with sum(w_j) = 1 is actively updated via "
        "Gaussian Process Regression with a Matern kernel. The acquisition function **Expected Improvement (EI)** "
        "proposes optimal weights that maximize relative improvement Delta_F."
    )

    c_gp_ctrl, c_gp_view = st.columns([1, 2])
    with c_gp_ctrl:
        st.markdown("#### 1D Slice Metric Target")
        target_metric = st.selectbox(
            "Select Metric to Inspect 1D GP Slice",
            options=ctrl.metric_names,
            index=0,
        )
        target_idx = ctrl.metric_names.index(target_metric)
        st.markdown(f"**Selected Dimension:** `{target_metric}` (index {target_idx})")
        st.caption("A 1D slice varies the weight of the selected metric from 0 to 1 while distributing the remainder uniformly.")

    with c_gp_view:
        w_grid, mu, sigma, ei, sampled_x, sampled_y = ctrl.bayesian_optimizer.get_1d_gp_slice(target_idx=target_idx)

        fig_gp = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                f"Gaussian Process Mean & 95% Confidence Interval for w[{target_metric}]",
                "Expected Improvement (EI) Acquisition Function",
            ),
        )

        upper_bound = mu + 1.96 * sigma
        lower_bound = mu - 1.96 * sigma

        fig_gp.add_trace(
            go.Scatter(
                x=np.concatenate([w_grid, w_grid[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='95% Confidence Interval (+-1.96 sigma)',
            ),
            row=1, col=1,
        )

        fig_gp.add_trace(
            go.Scatter(
                x=w_grid,
                y=mu,
                mode='lines',
                line=dict(color='#2563EB', width=2.5),
                name='GP Mean mu(w)',
            ),
            row=1, col=1,
        )

        if len(sampled_x) > 0:
            fig_gp.add_trace(
                go.Scatter(
                    x=sampled_x,
                    y=sampled_y,
                    mode='markers',
                    marker=dict(color='#DC2626', size=8, symbol='circle'),
                    name='Observed Trials (w_i, Delta_F_i)',
                ),
                row=1, col=1,
            )

        fig_gp.add_trace(
            go.Scatter(
                x=w_grid,
                y=ei,
                mode='lines',
                line=dict(color='#D97706', width=2),
                fill='tozeroy',
                fillcolor='rgba(217, 119, 6, 0.2)',
                name='Expected Improvement EI(w)',
            ),
            row=2, col=1,
        )

        fig_gp.update_xaxes(title_text=f"Weight w_{target_idx} [{target_metric}]", row=2, col=1)
        fig_gp.update_yaxes(title_text="Predicted Delta_F", row=1, col=1)
        fig_gp.update_yaxes(title_text="EI Value", row=2, col=1)
        fig_gp.update_layout(height=480, margin=dict(l=40, r=40, t=40, b=30))
        st.plotly_chart(fig_gp, use_container_width=True)

    st.markdown("#### Dynamic Weight Evolution over Generations")
    df_weights = ctrl.bayesian_optimizer.get_weights_history_df()
    if not df_weights.empty:
        fig_area = go.Figure()
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#6366F1']
        for i, col_name in enumerate(ctrl.metric_names):
            if col_name in df_weights.columns:
                fig_area.add_trace(go.Scatter(
                    x=df_weights["Generation"],
                    y=df_weights[col_name],
                    mode='lines',
                    stackgroup='one',
                    name=col_name,
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                ))
        fig_area.update_layout(
            xaxis_title="Generation",
            yaxis_title="Weight Proportion (Sum = 1.0)",
            height=340,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("Run generations to view dynamic weight adaptation trajectories.")


# ---------------------------------------------------------
# TAB 3: Cost Awareness & Evaluator Pruning (Gaps 1 & 3)
# ---------------------------------------------------------
with tab3:
    st.subheader("Gaps 1 & 3: Cost-Penalized Search & Evaluator Pruning Telemetry")
    
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown("#### Cumulative Cost Comparison")
        df_rec = pd.DataFrame(ctrl.archive.generation_records)
        if not df_rec.empty and "cumulative_adaptive_cost" in df_rec.columns:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=df_rec["generation"],
                y=df_rec["cumulative_naive_cost"],
                mode='lines+markers',
                name='Naive Full Evaluation Suite (All 6 Evaluators)',
                line=dict(color='#DC2626', width=2, dash='dash'),
            ))
            fig_cost.add_trace(go.Scatter(
                x=df_rec["generation"],
                y=df_rec["cumulative_adaptive_cost"],
                mode='lines+markers',
                name='Our Selective Adaptive Suite (Gaps 1 & 3)',
                line=dict(color='#10B981', width=3),
            ))
            fig_cost.update_layout(
                xaxis_title="Generation",
                yaxis_title="Cumulative Cost (USD $)",
                height=360,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("No cost history available yet.")

    with col_c2:
        st.markdown("#### Selective Evaluator Activation Heatmap")
        agents = ctrl.archive.get_all_agents()
        if agents:
            heatmap_data = []
            gen_labels = []
            for a in agents:
                gen_labels.append(f"Gen {a.generation} ({a.id[:8]})")
                row_flags = [1 if m in a.active_evaluators else 0 for m in ctrl.metric_names]
                heatmap_data.append(row_flags)

            fig_heat = go.Figure(data=go.Heatmap(
                z=np.array(heatmap_data).T,
                x=gen_labels,
                y=ctrl.metric_names,
                colorscale=[[0, '#F1F5F9'], [1, '#10B981']],
                showscale=False,
            ))
            fig_heat.update_layout(
                xaxis_title="Agent Candidates per Generation",
                yaxis_title="Evaluator Suite",
                height=360,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No evaluator activation data yet.")


# ---------------------------------------------------------
# TAB 4: Population Archive & Data Export (CSV)
# ---------------------------------------------------------
with tab4:
    st.subheader("Population Archive & Complete CSV Results")

    col_exp1, col_exp2 = st.columns([3, 1])
    with col_exp1:
        st.markdown("#### Searchable Population Archive")
    with col_exp2:
        df_detailed = ctrl.archive.to_detailed_dataframe()
        csv_data = df_detailed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Results (CSV)",
            data=csv_data,
            file_name="optimization_results.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

    st.caption("All parameters, harness knobs (theta_H), active evaluators, weights, costs, and prompts are automatically persisted to optimization_results.csv.")

    if not df_detailed.empty:
        st.dataframe(df_detailed, use_container_width=True)
    else:
        st.info("Archive is empty. Run generations to evolve agents.")

    st.markdown("---")
    st.markdown("#### Prompt & Harness Knob Diff Inspector")

    agents_dict = ctrl.archive.agents
    child_candidates = [a_id for a_id in agents_dict.keys() if agents_dict[a_id].parent_id is not None]
    if child_candidates:
        selected_agent_id = st.selectbox(
            "Select Mutated Child Agent to Inspect vs its Parent",
            options=child_candidates,
            index=len(child_candidates) - 1,
        )
        child_agent = agents_dict[selected_agent_id]
        parent_agent = agents_dict.get(child_agent.parent_id)

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(f"**Parent Agent (`{parent_agent.id if parent_agent else 'Root'}`)**")
            if parent_agent:
                st.code(parent_agent.system_prompt, language="text")
                st.caption(f"Knobs: Temp={parent_agent.harness_knobs.temperature:.2f}, Retries={parent_agent.harness_knobs.max_retries}, Top_p={parent_agent.harness_knobs.top_p:.2f}")
                st.caption(f"Fitness: {parent_agent.fitness:.4f} | Cost: ${parent_agent.cost_spent:.5f}")
        with col_p2:
            st.markdown(f"**Mutated Child Agent (`{child_agent.id}`)**")
            st.caption(f"Mutation: `{child_agent.mutation_type}`")
            st.code(child_agent.system_prompt, language="text")
            st.caption(f"Knobs: Temp={child_agent.harness_knobs.temperature:.2f}, Retries={child_agent.harness_knobs.max_retries}, Top_p={child_agent.harness_knobs.top_p:.2f}")
            st.caption(f"Fitness: {child_agent.fitness:.4f} | Cost: ${child_agent.cost_spent:.5f}")
        with col_p3:
            st.markdown(f"**Generated Solution Code (`{child_agent.task_id or 'Benchmark Task'}`)**")
            if child_agent.metrics:
                m_str = " | ".join([f"{k}: {v:.2f}" for k, v in child_agent.metrics.items()])
                st.caption(f"Scores: {m_str}")
            st.code(child_agent.last_solution if child_agent.last_solution else "# Solution executed during evaluation", language="python")
    else:
        st.info("Evolve at least 1 child generation to inspect parent-child prompt diffs and generated solutions.")
