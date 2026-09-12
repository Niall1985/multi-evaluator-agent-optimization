import random
from typing import List, Dict, Tuple, Optional
from core.agent import Agent
from core.archive import PopulationArchive
from evaluation.pool import EvaluatorPool
from .bayesian_weights import BayesianWeightOptimizer


class JointSearchSampler:
    """Gap 1 & Joint Space Orchestrator: Samples (parent pi, active subset x_E, weights w, knobs theta_H)."""

    def __init__(
        self,
        evaluator_pool: EvaluatorPool,
        bayesian_optimizer: BayesianWeightOptimizer,
        deep_tier_threshold: float = 0.65,
        exploration_prob: float = 0.30,
    ):
        self.evaluator_pool = evaluator_pool
        self.bayesian_optimizer = bayesian_optimizer
        self.deep_tier_threshold = deep_tier_threshold
        self.exploration_prob = exploration_prob

    def sample_parent(self, archive: PopulationArchive) -> Agent:
        """Samples a parent agent using Pareto-preference or fitness-proportional tournament."""
        all_agents = archive.get_all_agents()
        if not all_agents:
            # Create default seed agent
            return Agent()

        pareto_front = archive.get_pareto_front()
        if pareto_front and random.random() < 0.75:
            # Select from Pareto-optimal candidates
            return random.choice(pareto_front)

        # Tournament selection of size 3
        k = min(3, len(all_agents))
        candidates = random.sample(all_agents, k)
        return max(candidates, key=lambda a: a.fitness)

    def sample_weights(self) -> Dict[str, float]:
        """Proposes continuous weights w via Bayesian Gaussian Process + Expected Improvement."""
        return self.bayesian_optimizer.propose_next_weights()

    def determine_active_evaluator_subset(
        self,
        strategy: str = "adaptive",
        core_score_preview: Optional[float] = None,
    ) -> List[str]:
        """Determines active evaluator subset x_E.
        
        - 'baseline': Always runs all 6 evaluators (Core + Deep).
        - 'adaptive': Runs Core first; triggers Deep tier only if candidate shows promise
          (core_score >= threshold) or via epsilon-exploration.
        """
        core_evals = self.evaluator_pool.get_evaluators_by_tier("core")
        deep_evals = self.evaluator_pool.get_evaluators_by_tier("deep")

        if strategy == "baseline":
            return core_evals + deep_evals

        # Adaptive Strategy:
        if core_score_preview is None:
            # Initial stage of generation: run Core tier first
            return core_evals
        
        # Deep trigger evaluation
        should_trigger_deep = (
            core_score_preview >= self.deep_tier_threshold
            or random.random() < self.exploration_prob
        )

        if should_trigger_deep:
            return core_evals + deep_evals
        else:
            return core_evals
