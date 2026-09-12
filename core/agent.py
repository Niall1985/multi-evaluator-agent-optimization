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
        )
        return child
