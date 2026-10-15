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
    domain: str = "coding"         # 'coding' | 'sentiment'
    task_type: str = "code"        # 'code' | 'sentiment' | 'mcq' | 'fib'


# =====================================================================
# DOMAIN 1: PYTHON CODING & ALGORITHMIC BENCHMARKS (7 Canonical Tasks)
# =====================================================================
CODING_TASKS: List[BenchmarkTask] = [
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
        domain="coding",
        task_type="code",
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
        domain="coding",
        task_type="code",
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
        domain="coding",
        task_type="code",
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
            {"args": (13, 9), "expected": True},
            {"args": (15, 8), "expected": False},
            {"args": (2, 4), "expected": False},
            {"args": (1, 3), "expected": True},
            {"args": (0, 1), "expected": True},
        ],
        edge_cases=[
            {"args": (0, 0), "expected": False},
            {"args": (7, 7), "expected": False},
            {"args": (16, 0), "expected": True},
        ],
        constraints="O(1) bitwise operations (xor > 0 and (xor & (xor - 1)) == 0).",
        domain="coding",
        task_type="code",
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
        domain="coding",
        task_type="code",
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
        domain="coding",
        task_type="code",
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
        domain="coding",
        task_type="code",
    ),
]


# =====================================================================
# DOMAIN 2: SENTIMENT ANALYSIS BENCHMARKS (SentiToolKit Integration)
# =====================================================================
SENTIMENT_TASKS: List[BenchmarkTask] = [
    # 1. Product & Gadget Reviews
    BenchmarkTask(
        id="senti_001_product_reviews",
        name="Consumer Electronics Product Reviews",
        category="E-Commerce & Gadget Reviews",
        description=(
            "Write a Python function `solve(review: str) -> str` using SentiToolKit from SentiAnalyzer "
            "to classify consumer product reviews into 'Positive', 'Negative', or 'Neutral'. "
            "Example: 'This product is absolutely amazing and fantastic!' -> 'Positive'."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ("This product is absolutely amazing and fantastic!",), "expected": "Positive"},
            {"args": ("The battery dies in twenty minutes and it overheats dangerously.",), "expected": "Negative"},
            {"args": ("The package arrived on Thursday in a brown cardboard box.",), "expected": "Neutral"},
            {"args": ("Superb build quality, lightning fast performance, highly recommended!",), "expected": "Positive"},
            {"args": ("Complete waste of money. Defective on arrival and customer support refused refund.",), "expected": "Negative"},
        ],
        edge_cases=[
            {"args": ("Average device, does the job but nothing extraordinary.",), "expected": "Neutral"},
            {"args": ("Unbelievably good value for the price!",), "expected": "Positive"},
            {"args": ("Terrible, broke on the first day.",), "expected": "Negative"},
        ],
        constraints="Must output exactly 'Positive', 'Negative', or 'Neutral'. Leverage SentiToolKit.",
        domain="sentiment",
        task_type="sentiment",
    ),

    # 2. Movie & Entertainment Critiques
    BenchmarkTask(
        id="senti_002_movie_critiques",
        name="Film & Entertainment Critiques",
        category="Media & Narrative Reviews",
        description=(
            "Write a Python function `solve(review: str) -> str` that evaluates cinematic critiques "
            "using SentiAnalyzer SentiToolKit, classifying emotional polarity into 'Positive', 'Negative', or 'Neutral'."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ("A breathtaking cinematic masterpiece with Oscar-worthy acting and sublime directing.",), "expected": "Positive"},
            {"args": ("Boring, predictable script with dreadful pacing and lifeless dialogue.",), "expected": "Negative"},
            {"args": ("The documentary runs for 94 minutes and was filmed in Berlin.",), "expected": "Neutral"},
            {"args": ("A stunning triumph of storytelling that kept me on the edge of my seat!",), "expected": "Positive"},
            {"args": ("An utter disaster from start to finish. Avoid at all costs.",), "expected": "Negative"},
        ],
        edge_cases=[
            {"args": ("It was an okay film, neither particularly good nor notably bad.",), "expected": "Neutral"},
            {"args": ("Mesmerizing visuals and emotional resonance throughout.",), "expected": "Positive"},
        ],
        constraints="Handles rich narrative prose, metaphors, and sentiment polarity via SentiToolKit.",
        domain="sentiment",
        task_type="sentiment",
    ),

    # 3. Contrastive Clauses & Conjunctions
    BenchmarkTask(
        id="senti_003_contrastive_clauses",
        name="Contrastive Clauses & Conjunctions",
        category="Linguistic Nuance & Transitions",
        description=(
            "Write a Python function `solve(review: str) -> str` that resolves contrastive conjunctions "
            "('but', 'however', 'although', 'despite') to correctly predict dominant sentiment polarity."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ("The design was sleek and modern, but the device stopped working after one week.",), "expected": "Negative"},
            {"args": ("I expected this software to be complicated, however it turned out to be remarkably intuitive and delightful.",), "expected": "Positive"},
            {"args": ("The delivery was slightly delayed, but the food was hot, fresh, and exceptionally delicious.",), "expected": "Positive"},
            {"args": ("Although the staff were friendly, the room was dirty and the air conditioner was broken.",), "expected": "Negative"},
        ],
        edge_cases=[
            {"args": ("The book has 300 pages and contains several chapters.",), "expected": "Neutral"},
            {"args": ("Despite minor flaws, this is the best purchase I have made all year.",), "expected": "Positive"},
        ],
        constraints="Properly weights primary clause polarity over concessive secondary clauses.",
        domain="sentiment",
        task_type="sentiment",
    ),

    # 4. Negation & Subtleties
    BenchmarkTask(
        id="senti_004_negation_sarcasm",
        name="Double Negation & Sentiment Subtleties",
        category="Negation Handling",
        description=(
            "Write a Python function `solve(review: str) -> str` to analyze reviews with subtle negation "
            "('not bad', 'hardly disappointing', 'failed to impress') using SentiAnalyzer SentiToolKit."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ("This restaurant is not bad at all; in fact, the pasta was quite delightful.",), "expected": "Positive"},
            {"args": ("The keynote speaker failed to impress and offered no fresh insights whatsoever.",), "expected": "Negative"},
            {"args": ("The update was released on Monday at 3 PM UTC.",), "expected": "Neutral"},
            {"args": ("I cannot recommend this enough—truly outstanding service!",), "expected": "Positive"},
            {"args": ("Nothing worked as advertised. A total letdown.",), "expected": "Negative"},
        ],
        edge_cases=[
            {"args": ("The flight departed at noon as scheduled.",), "expected": "Neutral"},
            {"args": ("Could not be more pleased with the outcome!",), "expected": "Positive"},
        ],
        constraints="Handles double negations and inverted polarities without hallucinating false negatives.",
        domain="sentiment",
        task_type="sentiment",
    ),

    # 5. Customer Service & Aspect Feedback
    BenchmarkTask(
        id="senti_005_aspect_service",
        name="Customer Experience & Hospitality Aspect",
        category="Service & Hospitality",
        description=(
            "Write a Python function `solve(review: str) -> str` evaluating hospitality and customer service "
            "feedback into 'Positive', 'Negative', or 'Neutral' with SentiToolKit integration."
        ),
        entry_point="solve",
        test_cases=[
            {"args": ("The concierge went above and beyond to make our wedding anniversary unforgettable!",), "expected": "Positive"},
            {"args": ("Rude representative hung up on me after keeping me on hold for 45 minutes.",), "expected": "Negative"},
            {"args": ("The hotel check-in time is 3:00 PM and check-out is 11:00 AM.",), "expected": "Neutral"},
            {"args": ("Impeccable cleanliness, warm hospitality, and divine breakfast buffet.",), "expected": "Positive"},
        ],
        edge_cases=[
            {"args": ("The receipt was sent to my email inbox.",), "expected": "Neutral"},
            {"args": ("Appalling hygiene in the bathroom, stains on the sheets.",), "expected": "Negative"},
        ],
        constraints="Extracts overall customer experience sentiment with high fidelity.",
        domain="sentiment",
        task_type="sentiment",
    ),
]


# Unified list containing all 12 benchmark tasks across the 2 supported domains (Coding and Sentiment)
BENCHMARK_TASKS: List[BenchmarkTask] = CODING_TASKS + SENTIMENT_TASKS


def get_tasks_by_domain(domain: str) -> List[BenchmarkTask]:
    """Returns tasks filtered by domain ('coding', 'sentiment')."""
    if domain == "sentiment":
        return list(SENTIMENT_TASKS)
    elif domain == "coding":
        return list(CODING_TASKS)
    return list(BENCHMARK_TASKS)


def get_benchmark_task(task_id: str) -> Optional[BenchmarkTask]:
    """Retrieves a task by ID or exact name across all domains."""
    for t in BENCHMARK_TASKS:
        if t.id == task_id or t.name == task_id:
            return t
    return None


def get_random_task(domain: Optional[str] = None) -> BenchmarkTask:
    """Returns a random task, optionally filtered by domain."""
    pool = get_tasks_by_domain(domain) if domain else BENCHMARK_TASKS
    return random.choice(pool)