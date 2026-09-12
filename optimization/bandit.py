import math
import random
from typing import List, Dict, Tuple, Optional


class UCB1MutationBandit:
    """UCB1 Multi-Armed Bandit for selecting prompt mutation operators in Condition 3."""

    def __init__(self, arms: List[Tuple[str, str]], c: float = 1.414):
        self.arms = arms  # List of (strategy_name, strategy_goal)
        self.arm_names = [a[0] for a in arms]
        self.c = c
        self.counts: Dict[str, int] = {name: 0 for name in self.arm_names}
        self.rewards: Dict[str, float] = {name: 0.0 for name in self.arm_names}
        self.total_pulls = 0

    def select_arm(self) -> Tuple[str, str]:
        """Selects an arm using UCB1 formula. Unpulled arms are selected first."""
        # Check for unpulled arms
        for arm in self.arms:
            name = arm[0]
            if self.counts[name] == 0:
                return arm

        # Compute UCB1 values for all arms
        best_arm = self.arms[0]
        max_ucb = -float('inf')

        for arm in self.arms:
            name = arm[0]
            n_a = self.counts[name]
            mean_reward = self.rewards[name] / n_a
            bonus = self.c * math.sqrt(math.log(self.total_pulls) / n_a)
            ucb_score = mean_reward + bonus

            if ucb_score > max_ucb:
                max_ucb = ucb_score
                best_arm = arm

        return best_arm

    def update(self, arm_name: str, reward: float):
        """Updates the empirical reward and pull count for the selected arm."""
        if arm_name in self.counts:
            self.counts[arm_name] += 1
            self.rewards[arm_name] += reward
            self.total_pulls += 1
