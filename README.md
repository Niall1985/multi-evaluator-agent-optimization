# Multi-Objective Agent Optimization Through Adaptive Evaluation

> **Course / Capstone Project**: BCSE497J  
> **Core Framework**: Adaptive Multi-Evaluator Co-Optimization, Functional ANOVA (fANOVA), Dimension Archive, and GP-UCB Joint Search.

---

## 🚀 Executive Summary & Addressed Research Gaps

Existing prompt and agent optimization frameworks (PromptBreeder, DSPy, AlphaEvolve, GEPA) typically optimize code generation agents against fixed, single-metric fitness functions or static, hand-weighted benchmarks. This introduces severe evaluation cost bottlenecks, evaluator redundancy, silent regressions, and heuristic tuning.

This framework introduces a **joint discrete-continuous optimization system** solving all **6 fundamental research gaps** identified in multi-metric agentic literature:

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │                    JOINT SEARCH SPACE (S)                   │
                               │  • Discrete: Parent π (Pareto-UCB), Mutation Arms (UCB1)   │
                               │  • Discrete Subset: Active Evaluators x_E ∈ {0, 1}⁶         │
                               │  • Continuous: Simplex Weights w ∈ Δ⁵ (GP-UCB / EI)         │
                               │  • Continuous Harness Knobs: θ_H (Temperature, Top-p)      │
                               └─────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                                 [ Agent Execution on Task ]
                                                              │
                                                              ▼
                               ┌─────────────────────────────────────────────────────────────┐
                               │             GAP 1: 2-TIER CASCADE & PRUNING (x_E)           │
                               │  • Tier 1 (Core): [μ₁, μ₂, μ₃] ($0.0012)                    │
                               │  • Preview Score Gate: avg(μ₁, μ₂, μ₃) ≥ 0.65?              │
                               │  • Tier 2 (Deep): [μ₄, μ₅, μ₆] ($0.0150) — triggered or     │
                               │    pruned with Gap 5 ε-probing audit (p = 0.30)             │
                               └─────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                               ┌─────────────────────────────────────────────────────────────┐
                               │        GAP 3: COST-PENALIZED SCALARIZED FITNESS             │
                               │  F(Φ, x_E, w) = ∑_{j ∈ x_E} w_j · μ_j - λ ∑_{j ∈ x_E} c_j  │
                               │  ΔF = F(Φ_child, x_E, w) - F(Φ_parent, x_E, w)              │
                               └─────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                               ┌─────────────────────────────────────────────────────────────┐
                               │        GAP 2 & 6: BAYESIAN GP-UCB ON SIMPLEX Δ⁵             │
                               │  Fit GP: w ↦ ΔF (Matérn 5/2 Kernel)                         │
                               │  Acquisition: α_GP-UCB(w) = μ(w) + β_t · σ(w)               │
                               │  Sublinear Regret Bound: O(√(T γ_T))                        │
                               └─────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                               ┌─────────────────────────────────────────────────────────────┐
                               │        GAP 4 & 5: fANOVA & DIMENSION ELITE ARCHIVE          │
                               │  • Gap 4 fANOVA: Sobol S_j = V_j / V_total (Prune S_j < 5%) │
                               │  • Gap 5 Archive: Best-ever agent per dimension μ_j         │
                               │  • Silent Regression Monitor: Flag drops Δ > 0.25           │
                               └─────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                               ┌─────────────────────────────────────────────────────────────┐
                               │            REACT STUDIO UI & TEST SUITE INSPECTOR           │
                               │  • 8 Dedicated Tabbed Views (Overview, Gaps 1-6, Tests)     │
                               │  • Test Suite Inspector: Input/Output & PASS/FAIL Badges    │
                               │  • Live 1D GP Posterior Slices & Simplex Evolution Charts   │
                               └─────────────────────────────────────────────────────────────┘
```

---

## 🔬 The 6 Research Gaps Addressed

### 1. Gap 1: Fixed Evaluation Strategy & 2-Tier Cascade Pruning ($x_E \in \{0, 1\}^6$)
* **Problem**: Conventional systems evaluate every candidate against the entire evaluator ensemble, wasting costly LLM judge tokens on non-viable solutions.
* **Our Solution**: Implements an active evaluator selection vector $x_E$. Fast deterministic metrics (Core Tier) run first; expensive LLM judges (Deep Tier) are triggered only if the preview score meets the threshold ($\tau = 0.65$) or via $\epsilon$-probing.

### 2. Gap 2: Static Hand-Weighted Multi-Metric Setups ($w \in \Delta^5$)
* **Problem**: Multi-metric setups manually hardcode weights $w$, preventing dynamic adaptation as the agent improves.
* **Our Solution**: Models the metric weight vector $w = [w_1, \dots, w_6]$ on the continuous probability simplex $\Delta^5$ ($\sum w_j = 1, w_j \ge 0$) via **Gaussian Process Regression** with a Matérn 5/2 kernel, proposing optimal continuous weights dynamically.

### 3. Gap 3: No Cost-Awareness in Evaluation ($F = \sum w_j \mu_j - \lambda \sum c_j$)
* **Problem**: Evaluator compute and API dollar costs are treated as free.
* **Our Solution**: Directly embeds unit evaluator costs ($c_1, \dots, c_6$) into the optimization objective with penalty coefficient $\lambda = 0.5$, reducing evaluation expenditure by $>60\%$ while maintaining high benchmark pass rates.

### 4. Gap 4: No Redundancy Detection (Functional ANOVA / Sobol Indices)
* **Problem**: Ensembles run uninformative or redundant evaluators indefinitely without detecting which metrics provide true optimization signal.
* **Our Solution**: Implements **Functional ANOVA (fANOVA)** variance decomposition over the surrogate model:
  $$V(f) = \sum_{j=1}^K V_j + \sum_{j < k} V_{jk} + \dots$$
  Computes the **Sobol First-Order Sensitivity Index** $S_j = \frac{V_j}{V(f)}$. Evaluators with $S_j < 0.05$ (explaining $<5\%$ of variance) are identified as redundant and pruned automatically.

### 5. Gap 5: Risk of Silent Regression & Per-Evaluator Dimension Archive
* **Problem**: When an evaluator is pruned or de-emphasized, agents can exploit loopholes and quietly regress on unwatched dimensions.
* **Our Solution**:
  1. **Per-Evaluator Dimension Archive (MAP-Elites view)**: Tracks and preserves the historical peak agent for every individual dimension $\mu_1, \dots, \mu_6$.
  2. **$\epsilon$-Probing Audit Mechanism**: Overrides pruning decisions with probability $\epsilon = 0.30$ to periodically audit dropped dimensions.
  3. **Silent Regression Detection**: Compares candidate scores against historical elites and triggers real-time regression alerts if any dimension drops by $\Delta > 0.25$.

### 6. Gap 6: Joint Discrete + Continuous Search with Theoretical Grounding
* **Problem**: Frameworks optimize discrete prompt texts and continuous hyperparameters with separate, disconnected heuristics lacking formal convergence guarantees.
* **Our Solution**: Unified joint space $\mathcal{S} = \mathcal{X}_{\text{prompt}} \times \{0, 1\}^6 \times \Delta^5 \times \Theta_H$.
  * **Discrete Variables**: Parent selection via Pareto tournament + Mutation operators via **UCB1 Bandit** with sublinear regret $\mathcal{O}(\sqrt{K T \ln T})$.
  * **Continuous Variables**: Simplex weights $w$ and harness knobs $\theta_H$ via **GP-UCB** with acquisition $\alpha(w) = \mu(w) + \beta_t \cdot \sigma(w)$, providing a proven **sublinear cumulative regret bound** $\mathcal{O}(\sqrt{T \gamma_T})$.

---

## 🤖 The 2 Specialized Agent Domains & Benchmark Suites

The framework supports multi-objective evolution and benchmarking across **2 distinct agent domains**:

### 1. Python Coding Agent (`coding`)
* **Core Modality**: Code synthesis, algorithmic problem-solving, dynamic programming, and edge-case handling.
* **7 Canonical Benchmark Tasks**:
  1. `he_000_has_close_elements`: Float proximity checks with precision tolerance (HumanEval/0).
  2. `he_003_below_zero`: Prefix scan bank account balance underflow (HumanEval/3).
  3. `he_055_fib`: Dynamic programming Fibonacci sequence computation (HumanEval/55).
  4. `mbpp_006_differ_bits`: Bitwise XOR power-of-two differ-by-one bit check (MBPP/6).
  5. `mbpp_274_stock_exchange`: Maximum single-transaction stock profit (MBPP/274).
  6. `task_valid_parentheses`: Bracket matching and stack validation.
  7. `task_two_sum`: Optimal hash-map two-sum index lookup.
* **6 Specialized Archetypes**: General Balanced Assistant, Defensive Edge-Case Engineer, Chain-of-Thought Analytic Agent, Asymptotic Complexity Optimizer, Minimalist Economy Synthesizer, Safety & Constraint-Aligned Agent.

---

### 2. Sentiment Analysis Agent (`sentiment`)
* **Core Modality**: Tensor-based NLP sentiment analysis powered by the user's proprietary neural package:
  ```python
  from SentiAnalyzer.Main_tensor_model import SentiToolKit
  ```
* **Neural Architecture**: Employs pre-trained Keras sequential neural embeddings (`SentiToolKit.keras`) and pickle tokenizer (`tokenizer.pkl`), generating probability distributions across `[Negative, Neutral, Positive]`.
* **5 Comprehensive Benchmark Suites**:
  1. `senti_001_product_reviews`: E-commerce hardware and consumer electronics feedback with subtle satisfaction indicators.
  2. `senti_002_movie_critiques`: Nuanced cinematic reviews evaluating acting, pacing, and artistic direction.
  3. `senti_003_contrastive_clauses`: Complex compound reviews with contrastive conjunctions (*"great screen, but the battery dies in 2 hours"*), evaluating dominant clause polarity extraction.
  4. `senti_004_negation_sarcasm`: Double negations (*"not bad at all"*, *"hardly a disappointment"*) and ironic compliments.
  5. `senti_005_aspect_service`: Multi-aspect hospitality, food quality, and customer service reviews.
* **4 Specialized Archetypes**: SentiToolKit Neural Specialist, Nuance & Aspect Sentiment Reasoner, Sarcasm & Negation Analyst, Fast Production Sentiment Classifier.

---

## 📊 Concrete Evaluator Suite Matrix

| Metric | Dimension | Tier | Type | Unit Cost ($c_j$) |
| :--- | :--- | :--- | :--- | :--- |
| **$\mu_1$** | Functional Correctness | **Core** | Deterministic Unit Test Sandbox | $\$0.0010$ |
| **$\mu_2$** | Execution Latency | **Core** | High-Precision Timer ($ms$) | $\$0.0001$ |
| **$\mu_3$** | Token Conciseness | **Core** | Byte / Token Length Analyzer | $\$0.0001$ |
| **$\mu_4$** | LLM Reasoning Quality | **Deep** | Groq LLM-as-a-Judge (GPT-OSS) | $\$0.0050$ |
| **$\mu_5$** | Edge-Case Stress Test | **Deep** | Sandbox Boundary Assertion Suite | $\$0.0050$ |
| **$\mu_6$** | Safety & Anti-Hallucination | **Deep** | Groq LLM-as-a-Judge (GPT-OSS) | $\$0.0050$ |

---

## 🖥️ Modern React Studio UI Architecture

The React frontend (`frontend/`) features an uncluttered, responsive **Tabbed Navigation Bar** organizing the studio into 9 dedicated views:

```
[ Overview ] [ Batch Benchmark (100 Runs) ] [ Gap 1: Pruning ] [ Gap 2: Bayesian w ] [ Gap 3: Cost Fitness ] [ Gap 4: fANOVA ] [ Gap 5: Archive ] [ Gap 6: Joint Search ] [ Test Suite Execution ]
```

### 1. `Overview Tab`
* Global KPI telemetry cards (Best Fitness, Adaptive Cost Spent, Naive Baseline Cost, Net Dollar & % Savings, Pareto Size).
* Interactive 2D **Pareto Frontier Scatter Plot** (Fitness vs. Evaluator Cost in \$).
* Evolved Active System Prompt card with live metric badges.

### 2. `Batch Benchmark Studio (100 Runs / 300 Agents)` ⭐ NEW
* **One-Click Execution**: "Run Everything (100 Runs)" button triggering the complete multi-domain experimental matrix:
  * **Coding Agent**: 5 Ablations × 6 Seed Archetypes × 2 BO Acquisitions = 60 Runs (180 Agents)
  * **Sentiment Agent**: 5 Ablations × 4 Seed Archetypes × 2 BO Acquisitions = 40 Runs (120 Agents)
  * **Total**: 100 Runs × 3 Generations = **300 Agents Generated**
* **Live Progress Bar**: Real-time run counter (`Run X/100`), agent generation progress (`Gen Y/300`), domain indicator, and percentage completion.
* **Instant CSV Export**: Direct one-click download of `comprehensive_multi_domain_benchmark_results.csv` with columns arranged by metric importance.
* **Interactive Results Table**: Filter by Domain (`Coding` / `Sentiment`), Ablation Condition (all 5), BO Acquisition (`GP-UCB` / `EI`), and search by Task or Agent ID.

### 3. `Gap 1: Pruning Tab`
* Interactive 2-Tier Cascade architecture flow diagram.
* Preview Score Gate trigger visualization ($\tau = 0.65$).
* Evaluator Activation Vector Table: generation-by-generation audit of active vs. skipped metrics ($x_E \in \{0, 1\}^6$) and per-step dollar savings.

### 4. `Gap 2: Bayesian w Tab`
* Live **1D Gaussian Process Posterior Slice**: Interactive chart showing posterior mean $\mu(w)$, $\pm 1.96\sigma$ uncertainty envelope, and Expected Improvement (EI) curve for any selected metric weight.
* Probability Simplex ($\Delta^5$) active distribution grid.

### 5. `Gap 3: Cost Fitness Tab`
* Cost-penalized fitness equation breakdown ($F = \sum w_j \mu_j - \lambda \sum c_j$).
* Cumulative Cost Trajectory chart: Adaptive System (\$ spent) vs. Naive Baseline (\$ spent).
* Evaluator Unit Costing Matrix ($c_1 \dots c_6$).

### 6. `Gap 4: fANOVA Tab`
* Interactive **Sobol First-Order Sensitivity Index ($S_j$) Bar Chart**.
* Total Surrogate Variance $V(f)$ counter.
* Redundant Evaluators Badge list & automatic pruning status.

### 7. `Gap 5: Archive Tab`
* **6 Dimension Elite Cards** (one per evaluator) displaying peak historical score, generation found, mutation type, and elite prompt.
* Live **Silent Regression Alert Monitor** with regression magnitude and affected metric warnings.
* $\epsilon$-Probing audit log tracker.

### 8. `Gap 6: Joint Search Tab`
* Discrete **UCB1 Mutation Bandit Arm Distribution** bar chart.
* Continuous Simplex Weight Trajectory ($w_1 \dots w_6$) stacked area chart across all generations.
* Continuous harness knob evolution (temperature, top-p, retries).

### 9. `Test Suite Execution Tab`
* **Generation Pill Selector**: Inspect candidate execution for any generation (Gen 0, Gen 1, Gen 2...).
* **Pass Rate Status**: Overall test pass rate percentage badge.
* **Generated Code / Model Invocation Sandbox**: Full, syntax-highlighted code output (Python algorithms for Coding agent; neural model invocation for Sentiment agent) with a one-click copy button.
* **Multi-Domain Assertion & Case Results Table**:
  * **Coding Domain**: Function call arguments (`args`), expected return value, actual evaluated result, execution status (**`✓ PASS`** vs **`✗ FAIL`**), and exception/traceback on failure.
  * **Sentiment Analysis Domain**: Review text input, target sentiment polarity (`Positive`, `Neutral`, `Negative`), neural output prediction, confidence probability distribution, and classification status (**`✓ PASS`** vs **`✗ FAIL`**).

---

## 📁 Repository Structure

```
multi-evaluator-agent-optimization/
├── core/
│   ├── __init__.py
│   ├── agent.py               # Agent dataclass & harness knobs θ_H
│   ├── archive.py             # Population archive, Pareto front & Gap 5 Dimension Elites
│   ├── groq_client.py         # Groq API client with fallback & offline mock simulation
│   └── mutator.py             # Prompt & knob mutation operators
├── evaluation/
│   ├── __init__.py
│   ├── base.py                # BaseEvaluator abstract class & EvaluatorResult
│   ├── pool.py                # EvaluatorPool: runs active subset x_E
│   ├── metrics.py             # 6 concrete evaluators (Core & Deep tiers) with case results
│   └── runner.py              # Isolated CodeExecutionRunner & syntax validator
├── optimization/
│   ├── __init__.py
│   ├── bayesian_weights.py    # GP-UCB & Expected Improvement over probability simplex Δ⁵
│   ├── joint_sampler.py       # Orchestrator for joint space (π, x_E, w, θ_H) & fANOVA pruning
│   ├── scoring.py             # Cost-penalized scalarization: ∑(w·μ) - λ∑c
│   ├── bandit.py              # UCB1 multi-armed bandit for mutation operators
│   └── fanova.py              # Gap 4: Functional ANOVA Sobol variance decomposition
├── tasks/
│   ├── __init__.py
│   └── benchmark_tasks.py     # Benchmark suites: 7 Coding (HumanEval/MBPP) + 5 Sentiment Analysis
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # 17 comprehensive unit & integration tests covering Gaps 1-6 & domains
├── frontend/                  # Modern React + Vite + Tailwind UI Studio
│   ├── src/
│   │   ├── components/
│   │   │   ├── tabs/          # Dedicated Tab Views for Overview, Gaps 1-6, and Test Suite
│   │   │   ├── Navbar.jsx
│   │   │   ├── TelemetryCards.jsx
│   │   │   ├── ControlPanel.jsx
│   │   │   └── NavigationTabs.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── controller.py              # Evolutionary optimization loop orchestrator
├── server.py                  # FastAPI REST API backend serving React frontend
├── app.py                     # Streamlit alternative Web Application
├── run_experiments.py         # Automated ablation benchmark runner (5 conditions)
├── analyze_results.py         # Wilcoxon signed-rank test & statistical significance analysis
├── requirements.txt           # Python dependencies
├── optimization_results.csv   # Auto-saved run records (32-column schema)
└── README.md                  # System documentation
```

---

## ⚡ Quickstart & Execution Guide

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/your-username/multi-evaluator-agent-optimization.git
cd multi-evaluator-agent-optimization

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Set your Groq API key in `.env` (optional — offline mock simulation runs automatically if omitted):
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

### 3. Launching the Applications

#### Option A: Full-Stack React Studio (Recommended)
1. **Start the FastAPI Backend:**
   ```powershell
   python server.py
   ```
2. **Launch the React Frontend in Dev Mode (in a second terminal):**
   ```powershell
   cd frontend
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.  
   *(The production bundle is also served directly by `server.py` at `http://localhost:8000`).*

#### Option B: Launching Streamlit Web App
```powershell
streamlit run app.py
```

#### Option C: Running Benchmark Experiments & Statistical Analysis
```powershell
# Run the 5 ablation conditions across benchmark tasks
python run_experiments.py --runs 5 --generations 10

# Perform Wilcoxon signed-rank test and convergence analysis
python analyze_results.py
```

---

## 🧪 Unit & Integration Test Suite

All 18 unit and integration tests verify the end-to-end functionality of all 6 research gaps and multi-domain benchmarking:
```powershell
python -m unittest tests.test_pipeline
```
```
----------------------------------------------------------------------
Ran 18 tests

OK
```
Covering:
- **Gap 1**: 2-Tier cascade evaluation, preview score gating, and active subset filtering ($x_E \in \{0, 1\}^6$).
- **Gap 2 & 6**: Bayesian GP-UCB continuous weight optimization on probability simplex $\Delta^5$ with Matérn 5/2 kernel & Expected Improvement.
- **Gap 3**: Unit evaluator cost accounting and scalarized cost-penalized fitness ($F = \sum w_j \mu_j - \lambda \sum c_j$).
- **Gap 4**: Functional ANOVA (fANOVA) surrogate variance decomposition and Sobol index calculation ($S_j = V_j / V_{\text{total}}$).
- **Gap 5**: Per-evaluator Dimension Elite archive, $\epsilon$-probing audit mechanisms, and silent regression detection ($\Delta > 0.25$).
- **Agent Domains**: Python code synthesis sandbox execution and neural NLP `SentiToolKit` (`SentiAnalyzer`) integration with fallback handling.
- **Batch Benchmark Harness**: Complete 100-run / 300-agent experimental sweep verification and prioritized schema validation.

---

## 📄 Comprehensive Benchmark Schema (Prioritized Column Order)

The full experimental matrix (60 Coding runs + 40 Sentiment runs = 100 runs × 3 generations = 300 agents) exports directly to `comprehensive_multi_domain_benchmark_results.csv` with columns strictly ordered by metric importance:

1. **Core Identification**: `run_id`, `agent_domain`, `ablation_strategy`, `seed_archetype`, `bo_acquisition`, `generation`, `agent_id`, `parent_id`, `task_id`, `task_name`
2. **Primary Performance & Verification (Rank 1)**: `fitness`, `testcases_passed`, `testcases_total`, `testcases_pass_rate_pct`, `mu_1_functional_correctness`
3. **Pruning Status & Economics (Rank 2)**: `pruning_status`, `is_deep_tier_pruned`, `active_evaluators_count`, `active_evaluators_list`, `adaptive_cost_spent`, `total_cost_without_pruning`, `net_savings_dollars`, `net_savings_pct`
4. **Evaluator Unit Costs for Generation (Rank 3)**: `cost_mu1_correctness`, `cost_mu2_latency`, `cost_mu3_conciseness`, `cost_mu4_reasoning`, `cost_mu5_edge_cases`, `cost_mu6_safety`
5. **fANOVA Sensitivity Scores (Rank 4)**: `fanova_total_variance`, `fanova_S1_correctness`, `fanova_S2_latency`, `fanova_S3_conciseness`, `fanova_S4_reasoning`, `fanova_S5_edge_cases`, `fanova_S6_safety`, `fanova_redundant_pruned_metrics`
6. **Secondary Evaluator Dimensions (Rank 5)**: `mu_2_latency`, `mu_3_conciseness`, `mu_4_reasoning`, `mu_5_edge_cases`, `mu_6_safety`
7. **Dynamic Simplex Weights (Rank 6)**: `weight_mu1_correctness`, `weight_mu2_latency`, `weight_mu3_conciseness`, `weight_mu4_reasoning`, `weight_mu5_edge_cases`, `weight_mu6_safety`
8. **Optimization Knobs & Metadata (Rank 7)**: `delta_f`, `mutation_type`, `harness_temperature`, `harness_top_p`, `harness_max_retries`, `cumulative_adaptive_cost`, `cumulative_naive_cost`, `cumulative_net_savings`, `solution_preview`, `system_prompt`