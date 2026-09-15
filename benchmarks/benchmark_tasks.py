import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


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


# Exactly 7 Canonical Benchmark Tasks (HumanEval / MBPP / Algorithmic)
BENCHMARK_TASKS: List[BenchmarkTask] = [
    # 1. HumanEval/0
    BenchmarkTask(
        id="he_000_has_close_elements",
        name="Has Close Elements (HumanEval/0)",
        category="Numeric Proximity & Sorting",
        description=(
            "Write a Python function `solve(numbers: list, threshold: float) -> bool` that checks if in given "
            "list of numbers, are any two numbers closer to each other than given threshold. "
            "Return True if any pair has absolute difference strictly less than threshold, else False."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ([1.0, 2.0, 3.9, 4.0, 5.0], 0.3), "expected": True},
            {"args": ([1.0, 2.0, 3.9, 4.0, 5.0], 0.05), "expected": False},
            {"args": ([1.0, 2.0, 5.9, 4.0, 5.0], 0.95), "expected": True},
            {"args": ([1.0, 2.0, 5.9, 4.0, 5.0], 0.8), "expected": False},
            {"args": ([1.0, 2.0, 3.0, 4.0, 5.0], 2.0), "expected": True},
        ],
        edge_cases=[
            {"args": ([], 0.5), "expected": False},
            {"args": ([1.0], 0.5), "expected": False},
            {"args": ([1.1, 1.1], 0.01), "expected": True},
        ],
        constraints="Handles empty lists, float comparisons with precision tolerance, threshold >= 0.",
    ),

    # 2. HumanEval/3
    BenchmarkTask(
        id="he_003_below_zero",
        name="Balance Below Zero (HumanEval/3)",
        category="Prefix Scan & Financial State",
        description=(
            "Write a Python function `solve(operations: list) -> bool` that takes a list of deposit and withdrawal "
            "operations on a bank account that starts with zero balance. Return True if at any point the balance "
            "falls strictly below zero, else return False."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ([1, 2, 3],), "expected": False},
            {"args": ([1, 2, -4, 5],), "expected": True},
            {"args": ([1, -1, 2, -2, 5, -5, 4, -4],), "expected": False},
            {"args": ([1, -1, 2, -2, 5, -5, 4, -5],), "expected": True},
            {"args": ([1, -2, 2, -2, 5, -5, 4, -4],), "expected": True},
        ],
        edge_cases=[
            {"args": ([],), "expected": False},
            {"args": ([-1],), "expected": True},
            {"args": ([0, 0, 0],), "expected": False},
        ],
        constraints="O(N) single-pass cumulative sum, handles zero transitions and empty operation list.",
    ),

    # 3. HumanEval/55
    BenchmarkTask(
        id="he_055_fib",
        name="N-th Fibonacci (HumanEval/55)",
        category="Dynamic Programming & Math",
        description=(
            "Write a Python function `solve(n: int) -> int` that computes the n-th Fibonacci number in O(N) time "
            "and O(1) space. Assume 0-indexed where solve(0)=0, solve(1)=1, solve(2)=1, solve(3)=2, solve(10)=55. "
            "For negative n, it MUST raise ValueError."
        ),
        entry_point="solve",
        test_cases=[
            {"args": (0,), "expected": 0},
            {"args": (1,), "expected": 1},
            {"args": (6,), "expected": 8},
            {"args": (10,), "expected": 55},
            {"args": (15,), "expected": 610},
        ],
        edge_cases=[
            {"args": (0,), "expected": 0},
            {"args": (1,), "expected": 1},
            {"args": (-5,), "expected": None, "expect_exception": True},
            {"args": (-1,), "expected": None, "expect_exception": True},
        ],
        constraints="Must execute in under 1ms, raise ValueError on negative inputs, use no 3rd-party libraries.",
    ),

    # 4. MBPP/6
    BenchmarkTask(
        id="mbpp_006_differ_bits",
        name="Differ at One Bit Position (MBPP/6)",
        category="Bit Manipulation",
        description=(
            "Write a Python function `solve(a: int, b: int) -> bool` to check whether two given non-negative integers "
            "differ at exactly one bit position in their binary representation. (i.e., XOR of a and b is a power of 2)."
        ),
        entry_point="solve",
        test_cases=[
            {"args": (13, 9), "expected": True},     # 1101 vs 1001 -> diff at bit 2 (xor=4=2^2)
            {"args": (15, 8), "expected": False},    # 1111 vs 1000 -> diff at 3 bits
            {"args": (2, 4), "expected": False},     # 0010 vs 0100 -> diff at 2 bits
            {"args": (1, 3), "expected": True},      # 0001 vs 0011 -> diff at bit 1 (xor=2=2^1)
            {"args": (0, 1), "expected": True},      # 0000 vs 0001 -> diff at bit 0 (xor=1=2^0)
        ],
        edge_cases=[
            {"args": (0, 0), "expected": False},    # 0 bits difference
            {"args": (7, 7), "expected": False},    # 0 bits difference
            {"args": (16, 0), "expected": True},    # power of 2 difference
        ],
        constraints="O(1) bitwise operations (xor > 0 and (xor & (xor - 1)) == 0).",
    ),

    # 5. MBPP/274 (LeetCode 121)
    BenchmarkTask(
        id="mbpp_274_stock_exchange",
        name="Best Time to Buy & Sell Stock (MBPP/274)",
        category="Financial Algorithms & Greedy",
        description=(
            "Write a Python function `solve(prices: list) -> int` that calculates the maximum profit achievable "
            "from buying and selling a stock at most once. You cannot sell before buying. Return 0 if no profit can be made "
            "or if the list has fewer than 2 prices. Must run in O(N) time and O(1) space."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ([7, 1, 5, 3, 6, 4],), "expected": 5},
            {"args": ([7, 6, 4, 3, 1],), "expected": 0},
            {"args": ([2, 4, 1],), "expected": 2},
            {"args": ([1, 2, 3, 4, 5],), "expected": 4},
        ],
        edge_cases=[
            {"args": ([],), "expected": 0},
            {"args": ([5],), "expected": 0},
            {"args": ([3, 3, 3],), "expected": 0},
        ],
        constraints="O(N) single-pass scan, O(1) auxiliary space, handles empty and decreasing lists.",
    ),

    # 6. LeetCode 20
    BenchmarkTask(
        id="task_valid_parentheses",
        name="Balanced Bracket Sequence (LeetCode 20)",
        category="Data Structures & Stack",
        description=(
            "Write a Python function `solve(s: str) -> bool` determining if the input string containing brackets "
            "'()[]{}' is valid. An input is valid if brackets are closed in the correct order."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ("()",), "expected": True},
            {"args": ("()[]{}",), "expected": True},
            {"args": ("(]",), "expected": False},
            {"args": ("([)]",), "expected": False},
            {"args": ("{[]}",), "expected": True},
        ],
        edge_cases=[
            {"args": ("",), "expected": True},
            {"args": ("[",), "expected": False},
            {"args": ("]",), "expected": False},
            {"args": ("(((((",), "expected": False},
        ],
        constraints="O(N) time and memory, strictly validates standard bracket pairing.",
    ),

    # 7. MBPP/77 (LeetCode 1)
    BenchmarkTask(
        id="task_two_sum",
        name="Two Sum Target Index Lookup (MBPP/77)",
        category="Array Lookup & Hash Map",
        description=(
            "Write a Python function `solve(nums: list, target: int) -> list` that finds two indices in `nums` "
            "such that nums[i] + nums[j] == target (i != j). Return the sorted list [i, j]. If no solution exists, return []. "
            "Must run in O(N) time using a hash map."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ([2, 7, 11, 15], 9), "expected": [0, 1]},
            {"args": ([3, 2, 4], 6), "expected": [1, 2]},
            {"args": ([3, 3], 6), "expected": [0, 1]},
        ],
        edge_cases=[
            {"args": ([], 0), "expected": []},
            {"args": ([1], 2), "expected": []},
            {"args": ([1, 2, 3], 10), "expected": []},
        ],
        constraints="O(N) time complexity using hash map lookup, returns 0-indexed list [i, j].",
    ),
]


def get_benchmark_task(task_id: str) -> Optional[BenchmarkTask]:
    for t in BENCHMARK_TASKS:
        if t.id == task_id or t.name == task_id:
            return t
    # Fall back to the ARC suite. Imported lazily: arc_challenge imports BenchmarkTask from this module.
    from benchmarks.arc_challenge import get_arc_task
    return get_arc_task(task_id)


def get_benchmark_tasks(task_id: str) -> List[BenchmarkTask]:
    """Resolves a selector to the list of tasks it stands for: a single task, or a whole benchmark
    (e.g. `arc_benchmark` -> every loaded ARC task). Empty list if unknown."""
    from benchmarks.arc_challenge import ARC_TASKS, is_arc_benchmark
    if is_arc_benchmark(task_id):
        return list(ARC_TASKS)
    single = get_benchmark_task(task_id)
    return [single] if single else []


def get_random_task() -> BenchmarkTask:
    return random.choice(BENCHMARK_TASKS)
