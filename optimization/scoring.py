from typing import Dict, List, Tuple


def calculate_cost_penalized_fitness(
    evaluator_scores: Dict[str, float],      # {name: mu_j}
    weights: Dict[str, float],               # {name: w_j}
    active_evaluator_costs: List[float],     # [c_j]
    lambda_penalty: float = 0.5,
) -> Tuple[float, float]:
    """Gap 3: Calculates cost-penalized scalarized fitness score and total active evaluation cost.
    
    Formula:
        Fitness = sum_{j in x_E} (w_j * mu_j) - lambda * sum_{j in x_E} (c_j)
    """
    weighted_score = sum(weights.get(k, 0.0) * score for k, score in evaluator_scores.items())
    total_eval_cost = sum(active_evaluator_costs)
    scalarized_fitness = weighted_score - (lambda_penalty * total_eval_cost)
    return float(scalarized_fitness), float(total_eval_cost)


def calculate_relative_improvement(
    child_fitness: float,
    parent_fitness: float,
) -> float:
    """Computes relative improvement Delta_F for Bayesian Gaussian Process feedback:
    
    Delta_F = Fitness_child - Fitness_parent
    """
    return float(child_fitness - parent_fitness)


def calculate_delta_f(
    child_scores: Dict[str, float],
    parent_metrics: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Spec §3.2 GP regression target, both sides scalarized under the *current* w:

        Delta_F = sum_{j in x_E} w_j mu_j(child) - sum_{j in x_E} w_j mu_j(parent)

    No cost term (the cost penalty is a property of x_E, not of the weights the GP is learning), and the
    parent is re-scalarized from its stored metrics rather than reusing `parent.fitness`, which was computed
    under the parent's own weights and subset. Only evaluators observed on both sides are compared, so a
    parent that was evaluated on a smaller subset does not bias the difference.
    """
    shared = [j for j in child_scores if j in parent_metrics]
    return float(sum(weights.get(j, 0.0) * (child_scores[j] - parent_metrics[j]) for j in shared))
