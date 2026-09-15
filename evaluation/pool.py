from typing import Dict, List, Any, Tuple
from core.agent import Agent
from core.groq_client import GroqLLMClient
from .base import BaseEvaluator, EvaluatorResult
from .metrics import (
    FunctionalCorrectnessEvaluator,
    ExecutionLatencyEvaluator,
    TokenConcisenessEvaluator,
    LLMReasoningQualityEvaluator,
    EdgeCaseStressEvaluator,
    LLMSafetyHallucinationEvaluator,
)


class EvaluatorPool:
    """Orchestrates evaluator registration and runs selective subsets x_E on candidate agents."""

    def __init__(self, llm_client: GroqLLMClient, register_defaults: bool = True):
        self.llm_client = llm_client
        self.evaluators: Dict[str, BaseEvaluator] = {}
        if register_defaults:
            self._register_default_evaluators()

    def _register_default_evaluators(self):
        # Subset 1: Fast Operational / Core Tier
        self.register(FunctionalCorrectnessEvaluator())
        self.register(ExecutionLatencyEvaluator())
        self.register(TokenConcisenessEvaluator())

        # Subset 2: Deep Semantic & Robustness Tier
        self.register(LLMReasoningQualityEvaluator(self.llm_client))
        self.register(EdgeCaseStressEvaluator())
        self.register(LLMSafetyHallucinationEvaluator(self.llm_client))

    def register(self, evaluator: BaseEvaluator):
        self.evaluators[evaluator.name] = evaluator

    def get_evaluator_names(self) -> List[str]:
        return list(self.evaluators.keys())

    def get_evaluators_by_tier(self, tier: str) -> List[str]:
        return [name for name, ev in self.evaluators.items() if ev.tier == tier]

    def get_full_eval_cost(self) -> float:
        """Returns the total cost if all registered evaluators were executed."""
        return sum(ev.cost for ev in self.evaluators.values())

    def evaluate_subset(
        self,
        agent: Agent,
        task: Dict[str, Any],
        agent_output: str,
        active_subset: List[str],
        execution_context: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, float], List[float], Dict[str, EvaluatorResult]]:
        """Runs ONLY the evaluators in active_subset x_E.
        
        Returns:
            scores_dict: {evaluator_name: score}
            costs_list: [cost_j for active evaluators]
            detailed_results: {evaluator_name: EvaluatorResult}
        """
        scores: Dict[str, float] = {}
        costs: List[float] = []
        details: Dict[str, EvaluatorResult] = {}

        for name in active_subset:
            if name in self.evaluators:
                evaluator = self.evaluators[name]
                res = evaluator.evaluate(
                    agent=agent,
                    task=task,
                    agent_output=agent_output,
                    execution_context=execution_context,
                )
                scores[name] = res.score
                costs.append(res.cost)
                details[name] = res

        return scores, costs, details
