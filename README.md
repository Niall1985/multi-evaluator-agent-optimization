# Multi-Objective Agent Optimization through Evaluation

## Project Overview & Addressed Gaps

This project implements a complete, from-scratch framework for **Multi-Objective Agent Optimization** across 3 foundational research gaps:

```
                              +-------------------------------------------------------------+
                              |                    JOINT SEARCH SPACE (V)                   |
                              |  - Parent Agent (pi) & Groq LLM Prompt Mutation             |
                              |  - Active Evaluator Subset: x_E subset of {E_Core, E_Deep}  |
                              |  - Continuous Weights: w in R^k (via Gaussian Process + EI) |
                              |  - Harness Knobs: theta_H (temperature, max_retries)        |
                              +-------------------------------------------------------------+
                                                             |
                                                             v
                                                 [ Agent Execution on Task ]
                                                             |
                                                             v
                              +-------------------------------------------------------------+
                              |                  SELECTIVE EVALUATION SUITE                 |
                              |  Subset 1 (Core Tier):   [mu_1, mu_2, mu_3]  (Low cost)     |
                              |  Subset 2 (Deep Tier):   [mu_4, mu_5, mu_6]  (Higher cost)  |
                              |  Runs ONLY evaluators in active subset x_E                  |
                              +-------------------------------------------------------------+
                                                             |
                                                             v
                              +-------------------------------------------------------------+
                              |                 COST-PENALIZED SCALARIZATION                |
                              |  Fitness = sum_{j in x_E} (w_j * mu_j) - lambda * sum(c_j)  |
                              |  Delta_F = Fitness_child - Fitness_parent                   |
                              +-------------------------------------------------------------+
                                                             |
                                                             v
                              +-------------------------------------------------------------+
                              |                 BAYESIAN GAUSSIAN PROCESS                   |
                              |  Fit GP: w -> Delta_F with RBF/Matern Kernel                |
                              |  Acquisition: Propose next w via Expected Improvement (EI)  |
                              +-------------------------------------------------------------+
                                                             |
                                                             v
                              +-------------------------------------------------------------+
                              |                      STREAMLIT UI                           |
                              |  - Live Generation Tracker & Trade-off Radar                |
                              |  - Interactive 1D/2D GP Mean + Uncertainty & EI Curves      |
                              |  - Cost Savings vs Naive Full Evaluation Benchmark          |
                              +-------------------------------------------------------------+
```

### 1. Gap 1: Selective Evaluator Search ($x_E \subseteq \mathcal{E}$)
Rather than running an unchangeable static suite for every candidate, the optimization loop dynamically selects an active evaluator subset $x_E$. Fast and cheap unit tests (Core Tier) filter broken candidates, while promising candidates trigger deeper semantic evaluations (Deep Tier).

### 2. Gap 2: Bayesian Weight Optimization ($w \in \mathbb{R}^k$)
The metric weight vector $w = [w_1, \dots, w_k]$ with $\sum w_j = 1$ is modeled as a continuous search parameter updated dynamically each generation using **Gaussian Process Regression** with a Matérn-5/2 kernel:
$$\hat{\mu}(w), \hat{\sigma}^2(w) = \text{GP}(\{(w^{(i)}, \Delta F^{(i)})\}_{i=1}^t)$$
Optimal weights are proposed by maximizing the **Expected Improvement (EI)** acquisition function:
$$\text{EI}(w) = (\hat{\mu}(w) - f^* - \xi)\Phi(Z) + \hat{\sigma}(w)\phi(Z), \quad Z = \frac{\hat{\mu}(w) - f^* - \xi}{\hat{\sigma}(w)}$$

### 3. Gap 3: Cost-Penalized Fitness Function
Evaluator compute and API costs $c_j$ are explicitly penalized in the composite fitness score:
$$\text{Fitness } F(\Phi, x_E, w) = \sum_{j \in x_E} (w_j \mu_j) - \lambda \sum_{j \in x_E} c_j$$
$$\text{Relative Improvement } \Delta F = F(\Phi_{\text{child}}, x_E, w) - F(\Phi_{\text{parent}}, x_E, w)$$

---

## Evaluator Suite (6 Concrete Evaluators)

| Evaluator | Tier | Description | Cost ($c_j$) |
| :--- | :--- | :--- | :--- |
| $\mu_1$ Functional Correctness | **Subset 1 (Core)** | Deterministic pass rate on benchmark unit test assertions | $\$0.001$ |
| $\mu_2$ Execution Latency | **Subset 1 (Core)** | Normalized runtime and speed of execution | $\$0.000$ |
| $\mu_3$ Token Conciseness | **Subset 1 (Core)** | Output brevity and absence of conversational filler | $\$0.000$ |
| $\mu_4$ LLM Reasoning Quality | **Subset 2 (Deep)** | Step-by-step logic clarity evaluated via Groq LLM-as-a-judge | $\$0.015$ |
| $\mu_5$ Edge-Case Stress Test | **Subset 2 (Deep)** | Multi-case boundary conditions and exception handling | $\$0.005$ |
| $\mu_6$ LLM Safety & Anti-Hallucination | **Subset 2 (Deep)** | Factuality, API validity, and constraint conformance | $\$0.015$ |

---

## Repository Structure

```
multi-evaluator-agent-optimization/
├── core/
│   ├── __init__.py
│   ├── agent.py               # Agent dataclass (prompts, harness knobs theta_H)
│   ├── archive.py             # Population archive & lineage tracking
│   ├── groq_client.py         # Groq API client with fallback chain & mock simulation
│   └── mutator.py             # Prompt & knob mutator
├── evaluation/
│   ├── __init__.py
│   ├── base.py                # BaseEvaluator abstract class
│   ├── pool.py                # EvaluatorPool: runs active subset x_E
│   └── metrics.py             # 6 concrete evaluators (Subsets 1 & 2)
├── optimization/
│   ├── __init__.py
│   ├── bayesian_weights.py    # Gaussian Process + Expected Improvement for weights w
│   ├── joint_sampler.py       # Samples joint search space V = (pi, x_E, w, theta_H)
│   └── scoring.py             # Cost-penalized scalarization: sum(w*mu) - lambda*sum(c)
├── tasks/
│   ├── __init__.py
│   └── benchmark_tasks.py     # Python algorithmic & reasoning benchmark suite
├── controller.py              # Evolutionary optimization loop orchestrator
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Python dependencies
├── optimization_results.csv   # Auto-saved run records and evaluation telemetry
└── README.md                  # Project documentation
```

---

## Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Groq API Key
The system automatically reads your API key from `.env` or system environment variables:
```bash
# In .env file
GROQ_API_KEY=your_groq_api_key_here
```
If no key is provided, the framework operates in offline mock simulation mode.

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```

---

## Automatic CSV Export
All run parameters, harness knobs ($\theta_H$), active evaluator subsets, weights, individual metric scores, and cost spent are automatically saved to `optimization_results.csv` on disk and can be directly downloaded from the Streamlit UI (Tab 4).