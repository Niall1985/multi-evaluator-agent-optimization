import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from core.agent import Agent, HarnessKnobs
from core.archive import PopulationArchive
from evaluation.pool import EvaluatorPool
from .bayesian_weights import BayesianWeightOptimizer
from .fanova import FunctionalANOVA


class JointSearchSampler:
    """Gap 1, 4, 5, 6: Joint Discrete + Continuous Search Space Orchestrator.
    
    Manages:
    - Discrete variables: Parent candidate selection (Pareto-UCB tournament),
      Mutation operator selection (UCB1 bandit), Evaluator subset x_E (fANOVA + 2-Tier + epsilon-probe).
    - Continuous variables: Evaluator weights w on probability simplex Delta^(K-1) (GP-UCB with sublinear regret),
      Harness parameters theta_H (continuous temperature, top_p).
    """

    def __init__(
        self,
        evaluator_pool: EvaluatorPool,
        bayesian_optimizer: BayesianWeightOptimizer,
        fanova: Optional[FunctionalANOVA] = None,
        deep_tier_threshold: float = 0.65,
        exploration_prob: float = 0.30,
        enable_fanova_pruning: bool = True,
    ):
        self.evaluator_pool = evaluator_pool
        self.bayesian_optimizer = bayesian_optimizer
        self.fanova = fanova or FunctionalANOVA(evaluator_names=evaluator_pool.get_evaluator_names())
        self.deep_tier_threshold = deep_tier_threshold
        self.exploration_prob = exploration_prob  # Gap 5: Soft-pruning epsilon probing probability
        self.enable_fanova_pruning = enable_fanova_pruning

    def sample_parent(self, archive: PopulationArchive) -> Agent:
        """Gap 6: Samples a parent agent using Pareto-preference and fitness tournament."""
        all_agents = archive.get_all_agents()
        if not all_agents:
            return Agent()

        pareto_front = archive.get_pareto_front()
        if pareto_front and random.random() < 0.75:
            # High-fitness Pareto exploration
            return random.choice(pareto_front)

        # Tournament selection of size 3
        k = min(3, len(all_agents))
        candidates = random.sample(all_agents, k)
        return max(candidates, key=lambda a: a.fitness)

    def sample_weights(self, acquisition: Optional[str] = None) -> Dict[str, float]:
        """Gap 2 & 6: Proposes continuous weights w via GP-UCB (or EI) on the simplex Delta^(K-1)."""
        return self.bayesian_optimizer.propose_next_weights(acquisition=acquisition)

    def sample_harness_knobs(self, parent_knobs: HarnessKnobs) -> HarnessKnobs:
        """Gap 6: Continuous search over agent harness knobs theta_H."""
        # Perturb temperature with continuous Gaussian step, bounded [0.1, 1.0]
        new_temp = float(np.clip(parent_knobs.temperature + np.random.normal(0, 0.08), 0.1, 1.0))
        # Perturb top_p bounded [0.6, 1.0]
        new_top_p = float(np.clip(parent_knobs.top_p + np.random.normal(0, 0.05), 0.6, 1.0))
        # Discrete retries with small transition probability
        new_retries = parent_knobs.max_retries
        if random.random() < 0.20:
            new_retries = int(np.clip(parent_knobs.max_retries + random.choice([-1, 1]), 1, 4))

        return HarnessKnobs(
            temperature=round(new_temp, 3),
            max_retries=new_retries,
            top_p=round(new_top_p, 3),
            max_tokens=parent_knobs.max_tokens,
        )

    def determine_active_evaluator_subset(
        self,
        strategy: str = "adaptive",
        core_score_preview: Optional[float] = None,
        generation: int = 1,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Gap 1, 4, 5: Determines active evaluator subset x_E.
        
        Incorporates:
        - 2-Tier Cascade (Core vs Deep)
        - Gap 4: Functional ANOVA (fANOVA) redundancy detection (pruning uninformative evaluators)
        - Gap 5: Soft-pruning with epsilon-probing (probing dropped dimensions with probability epsilon)
        """
        core_evals = self.evaluator_pool.get_evaluators_by_tier("core")
        deep_evals = self.evaluator_pool.get_evaluators_by_tier("deep")
        all_evals = core_evals + deep_evals

        metadata: Dict[str, Any] = {
            "tier_triggered": "all" if strategy == "baseline" else "core",
            "fanova_pruned": [],
            "soft_probe_triggered": False,
        }

        if strategy == "baseline":
            return all_evals, metadata

        # Stage 1: Core Tier execution preview
        if core_score_preview is None:
            return core_evals, metadata

        # Stage 2: Deep Trigger Decision
        is_promising = core_score_preview >= self.deep_tier_threshold
        is_soft_probe = random.random() < self.exploration_prob

        if not is_promising and not is_soft_probe:
            # Hard prune Deep Tier due to poor Core preview
            metadata["tier_triggered"] = "core_only_pruned"
            return core_evals, metadata

        if not is_promising and is_soft_probe:
            metadata["soft_probe_triggered"] = True
            metadata["tier_triggered"] = "soft_probe_audit"

        # Stage 3: Gap 4 fANOVA Redundancy Pruning
        active_subset = list(all_evals)
        if self.enable_fanova_pruning and generation >= 4:
            redundant_evals = self.fanova.identify_redundant_evaluators(threshold=0.04)
            if redundant_evals:
                # Soft-prune: Keep informative, but allow epsilon probing on redundant ones
                kept = []
                for e in active_subset:
                    if e in redundant_evals:
                        # With probability epsilon, probe it anyway to protect against silent regression
                        if random.random() < self.exploration_prob:
                            kept.append(e)
                            metadata["soft_probe_triggered"] = True
                        else:
                            metadata["fanova_pruned"].append(e)
                    else:
                        kept.append(e)
                # Ensure at least core metrics remain
                active_subset = kept if kept else core_evals

        return active_subset, metadata
