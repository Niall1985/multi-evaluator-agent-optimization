import random
import numpy as np
from typing import Tuple, Optional, List
from .agent import Agent, HarnessKnobs
from .llm import LLMClient

MUTATION_STRATEGIES = [
    ("edge_case_awareness", "Instruct the agent to systematically identify and test corner cases before returning."),
    ("chain_of_thought", "Instruct the agent to use concise step-by-step reasoning before synthesizing code."),
    ("conciseness_economy", "Instruct the agent to omit conversational greetings, fluff, and unnecessary markdown formatting."),
    ("safety_adherence", "Instruct the agent to strictly follow parameter types, bounds, and security constraints."),
    ("algorithmic_efficiency", "Instruct the agent to prioritize lowest time and space asymptotic complexity."),
    ("robust_retry", "Instruct the agent to write defensive try-except blocks and explicit input validation."),
    ("reduce_overthinking", "Instruct the agent to stop overthinking: when its reasoning starts looping with phrases like "
     "'hmm', 'wait, but', 'actually', 'let me recheck' or 'let me reconsider', it must stop the loop, keep the "
     "best-supported hypothesis and continue from there, and it must not re-verify a step it already checked."),
    ("context_economy", "Instruct the agent to use what is already in its context: do not re-read or re-quote files, "
     "examples or earlier output unless they were updated since it last read them."),
    ("task_decomposition", "Instruct the agent to decompose the problem into small sub-tasks, solve and check them one "
     "by one, and build the final solution from one small helper function per sub-task."),
    ("hypothesis_first", "Instruct the agent to state one candidate rule in a single line, check it against every "
     "example once, and only revise it when an example contradicts it."),
]


class PromptMutator:
    """Generates prompt mutations and harness knob perturbations (theta_H) for candidate agents."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def get_available_strategies(self) -> List[Tuple[str, str]]:
        return list(MUTATION_STRATEGIES)

    def mutate(
        self,
        parent: Agent,
        generation: int,
        strategy_tuple: Optional[Tuple[str, str]] = None,
    ) -> Tuple[Agent, str, str]:
        """Creates a mutated child agent from a parent candidate."""
        if strategy_tuple:
            strategy_name, strategy_goal = strategy_tuple
        else:
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
            "Keep every output-format requirement of the parent prompt (for example, answering with one code "
            "block). Keep it short: at most about 150 words. "
            "Format: the first line is one role sentence; then one '- ' bulleted list with one rule per line. "
            "Merge the new instruction into that list; do not add headings, sections or paragraphs. "
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
