import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class HarnessKnobs:
    """Execution harness hyperparameters for an agent (theta_H)."""
    temperature: float = 0.7
    max_retries: int = 2
    top_p: float = 0.95
    max_tokens: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": round(self.temperature, 3),
            "max_retries": self.max_retries,
            "top_p": round(self.top_p, 3),
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessKnobs":
        return cls(
            temperature=data.get("temperature", 0.7),
            max_retries=data.get("max_retries", 2),
            top_p=data.get("top_p", 0.95),
            max_tokens=data.get("max_tokens", 1024),
        )


AGENT_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "General Balanced Assistant": {
        "id_suffix": "general",
        "description": "Balanced baseline assistant prioritizing clarity, correctness, and moderate conciseness.",
        "system_prompt": (
            "You are an expert, precise, and robust AI coding assistant. "
            "Solve the given problem clearly, accurately, and efficiently with minimal fluff."
        ),
        "harness_knobs": HarnessKnobs(temperature=0.7, max_retries=2, top_p=0.95, max_tokens=1024),
    },
    "Algorithmic Specialist Agent": {
        "id_suffix": "algorithmic",
        "description": "Specialized in optimal asymptotic time complexity O(N)/O(log N) and space complexity O(1).",
        "system_prompt": (
            "You are an algorithmic optimization specialist. Prioritize optimal time and memory complexity. "
            "Design the most computationally efficient algorithm using optimal dynamic programming, hashing, "
            "or two-pointer techniques. Output only verified, high-performance code."
        ),
        "harness_knobs": HarnessKnobs(temperature=0.5, max_retries=2, top_p=0.90, max_tokens=1024),
    },
    "Defensive & Fault-Tolerant Agent": {
        "id_suffix": "defensive",
        "description": "Specialized in corner-case validation, exception trapping, and boundary condition resilience.",
        "system_prompt": (
            "You are a defensive, fault-tolerant AI software engineer. Before implementing, rigorously identify "
            "and trap all corner cases (empty inputs, zero values, negative indices, boundary overflows). "
            "Wrap risky operations in explicit exception handling and parameter validations."
        ),
        "harness_knobs": HarnessKnobs(temperature=0.4, max_retries=3, top_p=0.90, max_tokens=1024),
    },
    "Chain-of-Thought Reasoning Agent": {
        "id_suffix": "cot",
        "description": "Mandates explicit step-by-step invariant reasoning and logical decomposition before code synthesis.",
        "system_prompt": (
            "You are an analytical AI reasoning specialist. For every problem: "
            "1. Outline a concise, 3-step chain of thought establishing input invariants and algorithm logic. "
            "2. Immediately synthesize the complete, verified Python implementation with no redundant conversational text."
        ),
        "harness_knobs": HarnessKnobs(temperature=0.6, max_retries=2, top_p=0.95, max_tokens=1024),
    },
    "Minimalist Economy Synthesizer": {
        "id_suffix": "minimalist",
        "description": "Ultra-concise agent minimizing token footprint and maximizing response speed with zero commentary.",
        "system_prompt": (
            "You are a hyper-concise AI code synthesizer. Deliver pure, production-grade Python implementations. "
            "Omit all conversational introductions, greetings, markdown explanations, and comments. Return only optimal code."
        ),
        "harness_knobs": HarnessKnobs(temperature=0.3, max_retries=1, top_p=0.85, max_tokens=512),
    },
    "Safety & Constraint-Aligned Agent": {
        "id_suffix": "safety",
        "description": "Strict compliance with safety constraints, type signatures, and anti-hallucination guarantees.",
        "system_prompt": (
            "You are a security and constraint-focused AI assistant. Strictly enforce input types, bounds, "
            "and safety requirements. Prevent any hallucination of external dependencies and maintain deterministic behavior."
        ),
        "harness_knobs": HarnessKnobs(temperature=0.3, max_retries=2, top_p=0.90, max_tokens=1024),
    },
}


@dataclass
class Agent:
    """Agent candidate representation in the joint search space."""
    id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    parent_id: Optional[str] = None
    generation: int = 0
    system_prompt: str = (
        "You are an expert, precise, and robust AI coding assistant. "
        "Solve the given problem clearly, accurately, and efficiently with minimal fluff."
    )
    user_template: str = "{task_description}"
    harness_knobs: HarnessKnobs = field(default_factory=HarnessKnobs)
    
    # Evaluation telemetry
    metrics: Dict[str, float] = field(default_factory=dict)  # mu_1 ... mu_6
    fitness: float = 0.0
    cost_spent: float = 0.0
    active_evaluators: List[str] = field(default_factory=list)
    mutation_type: str = "initial"
    mutation_description: str = "Seed baseline agent"
    lineage_path: List[str] = field(default_factory=list)
    last_solution: str = ""
    task_id: Optional[str] = None
    # Multi-task benchmarks (e.g. the whole ARC set): per-task outputs and scores behind the averaged `metrics`
    task_solutions: Dict[str, str] = field(default_factory=dict)
    task_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.lineage_path:
            self.lineage_path = [self.id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "harness_knobs": self.harness_knobs.to_dict(),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "fitness": round(self.fitness, 4),
            "cost_spent": round(self.cost_spent, 6),
            "active_evaluators": self.active_evaluators,
            "mutation_type": self.mutation_type,
            "mutation_description": self.mutation_description,
            "lineage_path": self.lineage_path,
            "last_solution": self.last_solution,
            "task_id": self.task_id,
            "task_solutions": self.task_solutions,
            "task_metrics": self.task_metrics,
            "created_at": self.created_at,
        }

    def clone(self, generation: int, mutation_type: str = "mutation", description: str = "") -> "Agent":
        new_id = f"agent_g{generation}_{uuid.uuid4().hex[:6]}"
        child = Agent(
            id=new_id,
            parent_id=self.id,
            generation=generation,
            system_prompt=self.system_prompt,
            user_template=self.user_template,
            harness_knobs=HarnessKnobs(
                temperature=self.harness_knobs.temperature,
                max_retries=self.harness_knobs.max_retries,
                top_p=self.harness_knobs.top_p,
                max_tokens=self.harness_knobs.max_tokens,
            ),
            metrics=self.metrics.copy(),
            fitness=self.fitness,
            cost_spent=0.0,
            mutation_type=mutation_type,
            mutation_description=description,
            lineage_path=self.lineage_path + [new_id],
            last_solution="",
            task_id=self.task_id,
        )
        return child
