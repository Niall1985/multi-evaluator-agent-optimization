import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BenchmarkTask:
    id: str
    name: str
    category: str
    description: str
    entry_point: str
    test_cases: List[Dict[str, Any]]
    edge_cases: List[Dict[str, Any]]
    constraints: str
    prompt_template: str = "{task_description}"


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        id="task_fibonacci",
        name="Efficient N-th Fibonacci",
        category="Algorithms & Dynamic Programming",
        description=(
            "Write a Python function `solve(n: int) -> int` that computes the n-th Fibonacci number in O(N) time "
            "and O(1) space. Assume 0-indexed where solve(0)=0, solve(1)=1, solve(2)=1, solve(3)=2. "
            "For negative n, it MUST raise ValueError."
        ),
        entry_point="solve",
        test_cases=[
            {"input": 0, "expected": 0},
            {"input": 1, "expected": 1},
            {"input": 6, "expected": 8},
            {"input": 10, "expected": 55},
            {"input": 15, "expected": 610},
        ],
        edge_cases=[
            {"input": 0, "expected": 0},
            {"input": 1, "expected": 1},
            {"input": -5, "expected": None, "expect_exception": True},
            {"input": -1, "expected": None, "expect_exception": True},
        ],
        constraints="Must execute in under 1ms, raise ValueError on negative inputs, use no 3rd-party libraries.",
    ),
    BenchmarkTask(
        id="task_stock_exchange",
        name="Best Time to Buy & Sell Stock (Stock Exchange)",
        category="Financial Algorithms & Greedy",
        description=(
            "Write a Python function `solve(prices: list) -> int` that calculates the maximum profit achievable "
            "from buying and selling a stock at most once. You cannot sell before buying. Return 0 if no profit can be made "
            "or if the list has fewer than 2 prices. Must run in O(N) time and O(1) space."
        ),
        entry_point="solve",
        test_cases=[
            {"input": [[7, 1, 5, 3, 6, 4]], "expected": 5},
            {"input": [[7, 6, 4, 3, 1]], "expected": 0},
            {"input": [[2, 4, 1]], "expected": 2},
            {"input": [[1, 2, 3, 4, 5]], "expected": 4},
        ],
        edge_cases=[
            {"input": [[]], "expected": 0},
            {"input": [[5]], "expected": 0},
            {"input": [[3, 3, 3]], "expected": 0},
        ],
        constraints="O(N) single-pass scan, O(1) auxiliary space, handles empty and decreasing lists.",
    ),
    BenchmarkTask(
        id="task_merge_intervals",
        name="Merge Overlapping Intervals",
        category="Intervals & Scheduling",
        description=(
            "Write a Python function `solve(intervals: list) -> list` that merges all overlapping intervals. "
            "Each interval is a list of two ints [start, end]. Return the list of non-overlapping intervals sorted by start time."
        ),
        entry_point="solve",
        test_cases=[
            {"input": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expected": [[1, 6], [8, 10], [15, 18]]},
            {"input": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
            {"input": [[[1, 4], [2, 3]]], "expected": [[1, 4]]},
        ],
        edge_cases=[
            {"input": [[]], "expected": []},
            {"input": [[[1, 5]]], "expected": [[1, 5]]},
            {"input": [[[2, 3], [1, 4]]], "expected": [[1, 4]]},
        ],
        constraints="Handles unsorted inputs, single-element list, empty list.",
    ),
    BenchmarkTask(
        id="task_palindrome",
        name="Valid Alphanumeric Palindrome",
        category="String Manipulation",
        description=(
            "Write a Python function `solve(s: str) -> bool` that returns True if the given string is a palindrome "
            "considering only alphanumeric characters and ignoring cases. Return True for empty strings."
        ),
        entry_point="solve",
        test_cases=[
            {"input": "A man, a plan, a canal: Panama", "expected": True},
            {"input": "race a car", "expected": False},
            {"input": " ", "expected": True},
            {"input": "Madam, I'm Adam", "expected": True},
            {"input": "0P", "expected": False},
        ],
        edge_cases=[
            {"input": "", "expected": True},
            {"input": "!!!", "expected": True},
            {"input": "a", "expected": True},
            {"input": "ab", "expected": False},
        ],
        constraints="O(N) time complexity, case-insensitive, ignores all punctuation and whitespace.",
    ),
    BenchmarkTask(
        id="task_valid_parentheses",
        name="Balanced Bracket Sequence",
        category="Data Structures & Stack",
        description=(
            "Write a Python function `solve(s: str) -> bool` determining if the input string containing brackets "
            "'()[]{}' is valid. An input is valid if brackets are closed in the correct order."
        ),
        entry_point="solve",
        test_cases=[
            {"input": "()", "expected": True},
            {"input": "()[]{}", "expected": True},
            {"input": "(]", "expected": False},
            {"input": "([)]", "expected": False},
            {"input": "{[]}", "expected": True},
        ],
        edge_cases=[
            {"input": "", "expected": True},
            {"input": "[", "expected": False},
            {"input": "]", "expected": False},
            {"input": "(((((", "expected": False},
        ],
        constraints="O(N) time and memory, strictly validates standard bracket pairing.",
    ),
    BenchmarkTask(
        id="task_two_sum",
        name="Two Sum Target Index Lookup",
        category="Array Lookup & Hash Map",
        description=(
            "Write a Python function `solve(nums: list, target: int) -> list` that finds two indices in `nums` "
            "such that nums[i] + nums[j] == target (i != j). Return the sorted list [i, j]. If no solution exists, return []. "
            "Must run in O(N) time using a hash map."
        ),
        entry_point="solve",
        test_cases=[
            {"input": ([2, 7, 11, 15], 9), "expected": [0, 1]},
            {"input": ([3, 2, 4], 6), "expected": [1, 2]},
            {"input": ([3, 3], 6), "expected": [0, 1]},
        ],
        edge_cases=[
            {"input": ([], 0), "expected": []},
            {"input": ([1], 2), "expected": []},
            {"input": ([1, 2, 3], 10), "expected": []},
        ],
        constraints="O(N) time complexity using hash map lookup, returns 0-indexed list [i, j].",
    ),
]


def get_benchmark_task(task_id: str) -> Optional[BenchmarkTask]:
    for t in BENCHMARK_TASKS:
        if t.id == task_id or t.name == task_id:
            return t
    return None


def get_random_task() -> BenchmarkTask:
    return random.choice(BENCHMARK_TASKS)
