from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from core.agent import Agent


@dataclass
class EvaluatorResult:
    """Result of an evaluation execution."""
    evaluator_name: str
    score: float  # 0.0 to 1.0
    cost: float   # Cost in USD
    execution_time_ms: float
    details: Optional[Dict[str, Any]] = None


class BaseEvaluator(ABC):
    """Abstract base class for all metric evaluators."""

    def __init__(self, name: str, cost: float, tier: str = "core"):
        self.name = name
        self.cost = cost
        self.tier = tier  # 'core' (Subset 1) or 'deep' (Subset 2)

    @abstractmethod
    def evaluate(
        self,
        agent: Agent,
        task: Dict[str, Any],
        agent_output: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorResult:
        """Evaluates agent output and returns an EvaluatorResult."""
        pass
