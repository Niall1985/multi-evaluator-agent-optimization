import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from controller import EvolutionController
from tasks.benchmark_tasks import BENCHMARK_TASKS
from core.agent import AGENT_ARCHETYPES

load_dotenv()

app = FastAPI(title="Multi-Objective Agent Optimization API", version="2.0.0")

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API key from .env / environment. Blank key => deterministic offline mock mode.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_IS_MOCK = not bool(GROQ_API_KEY)

# Global controller instance
controller = EvolutionController(
    api_key=GROQ_API_KEY,
    is_mock=DEFAULT_IS_MOCK,
    model="openai/gpt-oss-120b",
)


class StepRequest(BaseModel):
    n_steps: int = 1
    strategy: Optional[str] = "full_adaptive"
    task_id: Optional[str] = None


class ResetRequest(BaseModel):
    agent_domain: Optional[str] = "coding"
    archetype: Optional[str] = None
    task_id: Optional[str] = None
    lambda_penalty: Optional[float] = 0.5
    acquisition_type: Optional[str] = "gp_ucb"
    is_mock: Optional[bool] = None  # None => live if GROQ_API_KEY is set, else mock


@app.get("/api/status")
def get_status():
    """Returns real-time KPI metrics and overall optimization telemetry."""
    telemetry = controller.get_telemetry_summary()
    best_agent = controller.archive.get_best_agent()
    return {
        "telemetry": telemetry,
        "is_mock": controller.llm_client.is_mock,
        "model": controller.llm_client.model,
        "current_generation": controller.current_generation,
        "agent_domain": controller.agent_domain,
        "selected_task_id": controller.selected_task_id,
        "initial_archetype": controller.initial_archetype,
        "strategy": controller.default_strategy,
        "lambda_penalty": controller.lambda_penalty,
        "acquisition_type": controller.acquisition_type,
        "best_agent": {
            "id": best_agent.id if best_agent else None,
            "fitness": round(best_agent.fitness, 4) if best_agent else 0.0,
            "system_prompt": best_agent.system_prompt if best_agent else "",
            "metrics": best_agent.metrics if best_agent else {},
            "mutation_type": best_agent.mutation_type if best_agent else "",
            "domain": best_agent.domain if best_agent else "coding",
        } if best_agent else None,
    }


@app.get("/api/fanova")
def get_fanova():
    """Gap 4: Returns Functional ANOVA variance decomposition and evaluator importances."""
    return controller.get_fanova_summary()


@app.get("/api/gp_slice")
def get_gp_slice(target_idx: int = 0):
    """Gap 2: Returns 1D slice of GP posterior mean, uncertainty bands, and acquisition curve."""
    safe_idx = max(0, min(target_idx, len(controller.metric_names) - 1))
    w_grid, mu, sigma, ei, sampled_x, sampled_y = controller.bayesian_optimizer.get_1d_gp_slice(
        target_idx=safe_idx, resolution=60
    )
    curve_data = [
        {
            "w": round(float(w_grid[i]), 3),
            "mu": round(float(mu[i]), 4),
            "upper": round(float(mu[i] + 1.96 * sigma[i]), 4),
            "lower": round(float(mu[i] - 1.96 * sigma[i]), 4),
            "ei": round(float(ei[i]), 4),
        }
        for i in range(len(w_grid))
    ]
    samples_data = [
        {"x": round(float(sampled_x[i]), 3), "y": round(float(sampled_y[i]), 4)}
        for i in range(len(sampled_x))
    ]
    return {
        "metric_name": controller.metric_names[safe_idx],
        "target_idx": safe_idx,
        "curve": curve_data,
        "samples": samples_data,
        "all_metrics": controller.metric_names,
    }


@app.get("/api/dimension_elites")
def get_dimension_elites():
    """Gap 5: Returns the per-evaluator archive view and regression alert status."""
    elites = controller.archive.get_dimension_elites()
    # Check silent regressions for the latest agent
    latest_agent = controller.archive.get_all_agents()[-1] if controller.archive.agents else None
    regression_info = controller.archive.detect_silent_regression(latest_agent) if latest_agent else {"has_regression": False}
    return {
        "dimension_elites": elites,
        "latest_agent_id": latest_agent.id if latest_agent else None,
        "silent_regression": regression_info,
    }


@app.get("/api/history")
def get_history():
    """Returns chronological generation records."""
    return {
        "records": controller.history_records,
        "total": len(controller.history_records),
    }


@app.get("/api/pareto")
def get_pareto():
    """Returns non-dominated Pareto frontier candidates."""
    pareto_agents = controller.archive.get_pareto_front()
    return [
        {
            "id": a.id,
            "generation": a.generation,
            "fitness": round(a.fitness, 4),
            "cost_spent": round(a.cost_spent, 5),
            "metrics": {k: round(v, 3) for k, v in a.metrics.items()},
            "mutation_type": a.mutation_type,
            "prompt_preview": a.system_prompt[:120] + "...",
            "harness_knobs": {
                "temperature": a.harness_knobs.temperature,
                "top_p": a.harness_knobs.top_p,
                "max_retries": a.harness_knobs.max_retries,
            },
        }
        for a in pareto_agents
    ]


@app.get("/api/metadata")
def get_metadata():
    """Returns metadata for selectors (domains, tasks, archetypes, strategies)."""
    return {
        "agent_domains": [
            {
                "id": "coding",
                "name": "Coding Agent",
                "description": "Python synthesis, algorithmic problem solving & edge case compliance",
            },
            {
                "id": "sentiment",
                "name": "Sentiment Analysis Agent",
                "description": "Tensor-based NLP sentiment analysis powered by SentiToolKit",
            },
        ],
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "category": t.category,
                "description": t.description,
                "domain": getattr(t, "domain", "coding"),
                "task_type": getattr(t, "task_type", "code"),
                "difficulty": getattr(t, "difficulty", "Medium"),
            }
            for t in BENCHMARK_TASKS
        ],
        "archetypes": list(AGENT_ARCHETYPES.keys()),
        "archetypes_by_domain": {
            "coding": [k for k, v in AGENT_ARCHETYPES.items() if v.get("domain", "coding") == "coding"],
            "sentiment": [k for k, v in AGENT_ARCHETYPES.items() if v.get("domain") == "sentiment"],
        },
        "strategies": [
            {"id": "full_adaptive", "name": "Full Adaptive (Joint BO + fANOVA + Epsilon-Probe)"},
            {"id": "no_pruning", "name": "Ablation 1: No Pruning (Always All 6 Evaluators)"},
            {"id": "static_cascade", "name": "Ablation 2: Static Cascade (Fixed Hand-Weights)"},
            {"id": "single_metric", "name": "Ablation 3: Single Metric (Execution Correctness Only)"},
            {"id": "ucb1_bandit", "name": "Ablation 4: UCB1 Bandit Only (Discrete Arm Search)"},
        ],
        "acquisition_types": [
            {"id": "gp_ucb", "name": "GP-UCB (Proven Sublinear Regret Bound)"},
            {"id": "expected_improvement", "name": "Expected Improvement (EI)"},
        ],
    }


@app.post("/api/step")
def run_step(req: StepRequest):
    """Executes 1 or more generation cycles."""
    new_records = []
    for _ in range(req.n_steps):
        rec = controller.run_generation(strategy=req.strategy, task_id=req.task_id)
        new_records.append(rec)
    return {
        "success": True,
        "completed_steps": len(new_records),
        "latest_generation": controller.current_generation,
        "records": new_records,
    }


@app.post("/api/reset")
def reset_controller(req: ResetRequest):
    """Resets the optimization population and state with domain support."""
    global controller
    from core.agent import get_default_archetype
    domain = req.agent_domain or "coding"
    default_arch = get_default_archetype(domain)
    controller = EvolutionController(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        is_mock=req.is_mock if req.is_mock is not None else DEFAULT_IS_MOCK,
        agent_domain=domain,
        initial_archetype=req.archetype or default_arch,
        selected_task_id=req.task_id,
        lambda_penalty=req.lambda_penalty if req.lambda_penalty is not None else 0.5,
        acquisition_type=req.acquisition_type or "gp_ucb",
    )
    return {
        "success": True,
        "message": f"Controller reset successfully for domain: {domain}",
        "agent_domain": domain,
        "initial_archetype": controller.initial_archetype,
        "selected_task_id": controller.selected_task_id,
    }


# Mount built frontend static files if present
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
