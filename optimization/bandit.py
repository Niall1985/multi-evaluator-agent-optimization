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


class UCB1TaskSelector:
    """UCB1 over benchmark tasks (program-archive mode): which tasks get a call this generation.

    Index of task t: mean reward + c * sqrt(ln N / n_t), with c = sqrt(2) as in Auer, Cesa-Bianchi & Fischer
    (2002) UCB1 and UCB1MutationBandit. UCB1 starts by playing every arm once; here that first pull is the
    seed generation, and its reward is the task's archived weighted score, so a task whose seed program is
    already close starts with a higher estimate. Later rewards are the gain in the task's best archived score.
    A solved task is retired and never selected again.
    """

    def __init__(self, c: float = 1.414):
        self.c = c
        self.counts: Dict[str, int] = {}
        self.rewards: Dict[str, float] = {}
        self.retired: set = set()
        self.total_pulls = 0

    def update(self, task_id: str, reward: float) -> None:
        self.counts[task_id] = self.counts.get(task_id, 0) + 1
        self.rewards[task_id] = self.rewards.get(task_id, 0.0) + float(reward)
        self.total_pulls += 1

    def retire(self, task_id: str) -> None:
        self.retired.add(task_id)

    def index(self, task_id: str) -> float:
        n = self.counts.get(task_id, 0)
        if n == 0:
            return float("inf")
        return self.rewards[task_id] / n + self.c * math.sqrt(math.log(max(1, self.total_pulls)) / n)

    def select(self, task_ids: List[str], k: int) -> List[str]:
        """The k active tasks with the highest index (unpulled first), ties broken at random."""
        active = [t for t in task_ids if t not in self.retired]
        random.shuffle(active)
        return sorted(active, key=self.index, reverse=True)[:max(0, k)]

    def state(self) -> Dict[str, object]:
        return {"c": self.c, "counts": self.counts, "rewards": self.rewards,
                "retired": sorted(self.retired), "total_pulls": self.total_pulls}

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "UCB1TaskSelector":
        sel = cls(c=float(state["c"]))
        sel.counts = dict(state["counts"])
        sel.rewards = dict(state["rewards"])
        sel.retired = set(state["retired"])
        sel.total_pulls = int(state["total_pulls"])
        return sel
