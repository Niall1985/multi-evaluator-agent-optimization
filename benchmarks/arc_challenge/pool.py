"""ARC-specific EvaluatorPool: 3 real (official-metric) evaluators + 3 AI-generated partial-credit ones."""

from typing import Optional

from core.groq_client import GroqLLMClient
from evaluation.metrics import LLMReasoningQualityEvaluator, LLMSafetyHallucinationEvaluator
from evaluation.pool import EvaluatorPool

from .evaluators import REAL_ARC_EVALUATORS
from .partial_evaluators import PARTIAL_ARC_EVALUATORS


def build_arc_evaluator_pool(
    llm_client: GroqLLMClient,
    include_partial: bool = True,
    include_llm_judges: bool = False,
) -> EvaluatorPool:
    """Builds the ARC evaluator suite.

    Core tier: arc_runs_successfully, arc_pass_at_2_train (+ arc_pixel_accuracy, arc_shape_match).
    Deep tier: arc_pass_at_2_test (+ arc_color_palette; + mu_4/mu_6 LLM judges when requested).
    Pass as `EvolutionController(evaluator_pool_factory=build_arc_evaluator_pool)`.
    """
    pool = EvaluatorPool(llm_client=llm_client, register_defaults=False)
    for cls in REAL_ARC_EVALUATORS:
        pool.register(cls())
    if include_partial:
        for cls in PARTIAL_ARC_EVALUATORS:
            pool.register(cls())
    if include_llm_judges:
        pool.register(LLMReasoningQualityEvaluator(llm_client))
        pool.register(LLMSafetyHallucinationEvaluator(llm_client))
    return pool
