import random
import numpy as np
from typing import Tuple
from .agent import Agent, HarnessKnobs
from .groq_client import GroqLLMClient

MUTATION_STRATEGIES = [
    ("edge_case_awareness", "Instruct the agent to systematically identify and test corner cases before returning."),
    ("chain_of_thought", "Instruct the agent to use concise step-by-step reasoning before synthesizing code."),
    ("conciseness_economy", "Instruct the agent to omit conversational greetings, fluff, and unnecessary markdown formatting."),
    ("safety_adherence", "Instruct the agent to strictly follow parameter types, bounds, and security constraints."),
    ("algorithmic_efficiency", "Instruct the agent to prioritize lowest time and space asymptotic complexity."),
    ("robust_retry", "Instruct the agent to write defensive try-except blocks and explicit input validation."),
]


class PromptMutator:
    """Generates prompt mutations and harness knob perturbations (theta_H) for candidate agents."""

    def __init__(self, llm_client: GroqLLMClient):
        self.llm_client = llm_client

    def mutate(self, parent: Agent, generation: int) -> Tuple[Agent, str, str]:
        """Creates a mutated child agent from a parent candidate."""
        strategy_name, strategy_goal = random.choice(MUTATION_STRATEGIES)
        
        # 1. Mutate prompt via Groq LLM (or mock)
        system_instruction = (
            "You are an expert prompt optimization engineer. Your goal is to improve the provided "
            "agent system prompt to achieve higher functional correctness, faster execution, and conciseness."
        )
        user_instruction = (
            f"Parent System Prompt:\n\"\"\"\n{parent.system_prompt}\n\"\"\"\n\n"
            f"Strategy Name: {strategy_name}\n"
            f"Optimization Goal: {strategy_goal}\n\n"
            "Task: Rewrite and upgrade the system prompt. Make it direct, authoritative, and structured. "
            "Output ONLY the improved system prompt text without commentary or quotes."
        )

        mutated_prompt = self.llm_client.generate(
            system_prompt=system_instruction,
            user_prompt=user_instruction,
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
        ).strip()

        # Clean any surrounding markdown fences if returned
        if mutated_prompt.startswith("```") and mutated_prompt.endswith("```"):
            lines = mutated_prompt.split("\n")
            mutated_prompt = "\n".join(lines[1:-1]).strip()

        # 2. Perturb Harness Knobs (theta_H)
        parent_knobs = parent.harness_knobs
        new_temp = float(np.clip(parent_knobs.temperature + np.random.normal(0.0, 0.08), 0.1, 1.2))
        new_top_p = float(np.clip(parent_knobs.top_p + np.random.normal(0.0, 0.03), 0.7, 1.0))
        new_retries = int(np.clip(parent_knobs.max_retries + random.choice([-1, 0, 1]), 1, 5))

        child_knobs = HarnessKnobs(
            temperature=new_temp,
            max_retries=new_retries,
            top_p=new_top_p,
            max_tokens=parent_knobs.max_tokens,
        )

        child = parent.clone(
            generation=generation,
            mutation_type=strategy_name,
            description=f"Mutation [{strategy_name}]: {strategy_goal}",
        )
        child.system_prompt = mutated_prompt
        child.harness_knobs = child_knobs

        return child, strategy_name, strategy_goal
