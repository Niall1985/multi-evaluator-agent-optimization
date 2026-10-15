import random
from typing import List, Dict, Tuple, Optional, Any
import optuna
from optuna.distributions import CategoricalDistribution
from core.agent import Agent, HarnessKnobs
from core.archive import PopulationArchive, ProgramRecord, weighted_score
from evaluation.pool import EvaluatorPool
from .bayesian_weights import BayesianWeightOptimizer
from .fanova import FunctionalANOVA
from .tpe import categorical_tpe_choice


class JointSearchSampler:
    """Gap 1, 4, 5, 6: Joint Discrete + Continuous Search Space Orchestrator.

    Manages:
    - Discrete variables: parent pi and mutation idea (tree-structured Parzen estimator, spec §3.1: Optuna's
      TPESampler for the fixed idea menu, optimization.tpe for the changing set of candidate parents),
      evaluator subset x_E (2-tier cascade + leave-one-out soft pruning + epsilon-probe).
    - Continuous variables: evaluator weights w on the simplex Delta^(K-1) and harness parameters theta_H
      (temperature, top_p), proposed jointly by one GP (spec §3.2).
    """

    def __init__(
        self,
        evaluator_pool: EvaluatorPool,
        bayesian_optimizer: BayesianWeightOptimizer,
        fanova: Optional[FunctionalANOVA] = None,
        deep_tier_threshold: float = 0.65,
        exploration_prob: float = 0.30,
        enable_fanova_pruning: bool = True,
        prune_patience: int = 3,
        tpe_startup_trials: int = 5,
    ):
        self.evaluator_pool = evaluator_pool
        self.bayesian_optimizer = bayesian_optimizer
        self.fanova = fanova or FunctionalANOVA(evaluator_names=evaluator_pool.get_evaluator_names())
        self.deep_tier_threshold = deep_tier_threshold
        self.exploration_prob = exploration_prob  # Gap 5: Soft-pruning epsilon probing probability
        self.enable_fanova_pruning = enable_fanova_pruning
        self.prune_patience = prune_patience  # spec §3.3: W consecutive windows below tau before pruning
        self.tpe_startup_trials = tpe_startup_trials
        # (parent_id, idea, delta_f) per generation: the observations the parent's density-ratio estimator splits
        self.trials: List[Tuple[str, str, float]] = []
        # The idea menu is fixed, so it is Optuna's TPESampler over one categorical, driven by ask-and-tell:
        # asked in sample_idea, told Delta_F in record_trial. Created on first use, when the menu is known.
        self.idea_study: Optional[optuna.Study] = None
        # Open asks by key: one per child, since program-archive mode asks for several children per generation
        self._pending_idea_trials: Dict[str, optuna.trial.Trial] = {}

    # ------------------------------------------------------------------ discrete: parent and idea

    def record_trial(self, parent_id: str, idea: str, delta_f: float, key: str = "default") -> None:
        self.trials.append((parent_id, idea, float(delta_f)))
        pending = self._pending_idea_trials.pop(key, None)
        if pending is not None:
            self.idea_study.tell(pending, float(delta_f))

    def restore_trial(self, parent_id: str, idea: str, delta_f: float, menu: List[str]) -> None:
        """Replays a finished generation's (parent, idea, Delta_F) into the trial list and the idea study."""
        self.trials.append((parent_id, idea, float(delta_f)))
        if idea not in menu:
            return
        self._ensure_idea_study()
        self.idea_study.add_trial(optuna.trial.create_trial(
            params={"idea": idea},
            distributions={"idea": CategoricalDistribution(list(menu))},
            value=float(delta_f),
        ))

    def _ensure_idea_study(self) -> None:
        if self.idea_study is None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.idea_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(n_startup_trials=self.tpe_startup_trials),
            )

    def _tpe_ready(self) -> bool:
        return len(self.trials) >= self.tpe_startup_trials

    def sample_parent(self, archive: PopulationArchive, use_tpe: bool = False) -> Agent:
        """Samples a parent. With `use_tpe` (and enough trials) the Pareto front is ranked by the density
        ratio p(pi | G) / p(pi | B) of past parents; otherwise Pareto-preference plus fitness tournament."""
        all_agents = archive.get_all_agents()
        if not all_agents:
            return Agent()

        pareto_front = archive.get_pareto_front()
        if use_tpe and pareto_front and self._tpe_ready():
            by_id = {a.id: a for a in pareto_front}
            chosen = categorical_tpe_choice(
                candidates=list(by_id),
                observed=[t[0] for t in self.trials],
                scores=[t[2] for t in self.trials],
            )
            return by_id[chosen]

        if pareto_front and random.random() < 0.75:
            # High-fitness Pareto exploration
            return random.choice(pareto_front)

        # Tournament selection of size 3
        k = min(3, len(all_agents))
        candidates = random.sample(all_agents, k)
        return max(candidates, key=lambda a: a.fitness)

    def sample_idea(
        self, strategies: List[Tuple[str, str]], use_tpe: bool = False, key: str = "default"
    ) -> Tuple[str, str]:
        """Samples a mutation idea from the menu. With `use_tpe` it is Optuna's TPESampler (random for its
        first `tpe_startup_trials` trials); the trial stays open until record_trial tells it Delta_F."""
        if not use_tpe:
            return random.choice(strategies)
        by_name = dict(strategies)
        self._ensure_idea_study()
        stale = self._pending_idea_trials.pop(key, None)
        if stale is not None:
            # The previous generation died between ask and tell: close its trial rather than leave it running
            self.idea_study.tell(stale, state=optuna.trial.TrialState.FAIL)
        trial = self.idea_study.ask({"idea": CategoricalDistribution(list(by_name))})
        self._pending_idea_trials[key] = trial
        chosen = trial.params["idea"]
        return chosen, by_name[chosen]

    # ------------------------------------------------------------------ program-archive mode

    # Floor on a prior weight, so a candidate with a zero archived score stays reachable
    # (OpenEvolve's fitness-weighted island sampling uses the same 0.001 floor).
    PRIOR_FLOOR = 0.001

    def sample_weighted_parent(self, candidates: List[Agent]) -> Agent:
        """TPE over `candidates` with their archived fitness as the prior p_i (tpe.categorical_tpe_choice):
        before any trial this is fitness-proportional selection; past (parent, Delta_F) trials then move it."""
        by_id = {a.id: a for a in candidates}
        chosen = categorical_tpe_choice(
            candidates=list(by_id),
            observed=[t[0] for t in self.trials],
            scores=[t[2] for t in self.trials],
            prior=[max(a.fitness, 0.0) + self.PRIOR_FLOOR for a in by_id.values()],
        )
        return by_id[chosen]

    def sample_program_parent(
        self, archive: PopulationArchive, task_id: str, weights: Dict[str, float]
    ) -> Optional[ProgramRecord]:
        """Which archived program of `task_id` to repair, or None for a fresh attempt.

        TPE over the task's MAP-Elites elites plus the FRESH choice, observed on that task's own
        (parent, Delta_F) trials. Prior: a program's weighted score; FRESH gets the mean of those, so it is as
        likely as an average elite until repairs start paying off (or failing)."""
        elites = archive.task_elites(task_id)
        if not elites:
            return None
        prior = [weighted_score(p.scores, weights) + self.PRIOR_FLOOR for p in elites]
        candidates = [p.id for p in elites] + [archive.FRESH]
        prior.append(sum(prior) / len(prior))
        trials = archive.task_trials.get(task_id, [])
        chosen = categorical_tpe_choice(
            candidates=candidates,
            observed=[t[0] for t in trials],
            scores=[t[1] for t in trials],
            prior=prior,
        )
        return None if chosen == archive.FRESH else archive.programs[chosen]

    # ------------------------------------------------------------------ continuous: w and theta_H

    def sample_weights(self, acquisition: Optional[str] = None) -> Dict[str, float]:
        """Gap 2 & 6: Proposes continuous weights w via GP-UCB (or EI) on the simplex Delta^(K-1)."""
        return self.bayesian_optimizer.propose_next_weights(acquisition=acquisition)

    def sample_continuous(
        self, parent_knobs: HarnessKnobs, acquisition: Optional[str] = None
    ) -> Tuple[Dict[str, float], HarnessKnobs]:
        """Proposes (w, theta_H) from the joint GP. Knobs the GP does not model are inherited from the parent."""
        weights, knob_values = self.bayesian_optimizer.propose_next_config(acquisition=acquisition)
        knobs = HarnessKnobs(
            temperature=round(knob_values.get("temperature", parent_knobs.temperature), 3),
            max_retries=parent_knobs.max_retries,
            top_p=round(knob_values.get("top_p", parent_knobs.top_p), 3),
            max_tokens=parent_knobs.max_tokens,
        )
        return weights, knobs

    # ------------------------------------------------------------------ discrete: evaluator subset x_E

    def _soft_prune(self, names: List[str], metadata: Dict[str, Any]) -> List[str]:
        """Gap 4/5: drops evaluators fANOVA has held below tau for `prune_patience` windows, except that each
        is still probed with probability epsilon so a wrong prune costs an occasional audit, not coverage."""
        pruned = set(self.fanova.get_pruned_evaluators(self.prune_patience))
        kept = []
        for e in names:
            if e not in pruned:
                kept.append(e)
            elif random.random() < self.exploration_prob:
                kept.append(e)
                metadata["soft_probe_triggered"] = True
            else:
                metadata["fanova_pruned"].append(e)
        return kept

    def select_core_subset(self, enable_pruning: bool) -> Tuple[List[str], Dict[str, Any]]:
        """Stage 1 of the cascade: the core evaluators that run before the deep-tier decision."""
        metadata: Dict[str, Any] = {"tier_triggered": "core", "fanova_pruned": [], "soft_probe_triggered": False}
        core_evals = self.evaluator_pool.get_evaluators_by_tier("core")
        if enable_pruning and self.enable_fanova_pruning:
            kept = self._soft_prune(core_evals, metadata)
            # Never let pruning empty the preview: the deep-tier gate needs at least one core score
            core_evals = kept or core_evals[:1]
        return core_evals, metadata

    def select_deep_subset(
        self, core_score_preview: float, metadata: Dict[str, Any], enable_pruning: bool
    ) -> List[str]:
        """Stage 2: the deep evaluators to run on top of the core preview (empty when the gate stays shut)."""
        deep_evals = self.evaluator_pool.get_evaluators_by_tier("deep")
        if not deep_evals:
            return []

        is_promising = core_score_preview >= self.deep_tier_threshold
        is_soft_probe = random.random() < self.exploration_prob

        if not is_promising and not is_soft_probe:
            # Hard prune Deep Tier due to poor Core preview
            metadata["tier_triggered"] = "core_only_pruned"
            return []

        if not is_promising:
            metadata["soft_probe_triggered"] = True
            metadata["tier_triggered"] = "soft_probe_audit"

        if enable_pruning and self.enable_fanova_pruning:
            return self._soft_prune(deep_evals, metadata)
        return deep_evals
