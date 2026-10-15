"""Sends one ARC task to the OpenAI-compatible server exactly as the evolution run does, and shows where the tokens went.

The request matches OpenAICompatLLMClient.generate (system prompt, task description, temperature, top_p, max_tokens
floor, chat_template_kwargs, thinking_token_budget). Unlike generate(), it prints finish_reason, token usage and the
reasoning text, which the run discards, then scores the answer on the demonstration and held-out pairs.

    source <env with OPENAI_BASE_URL / OPENAI_API_KEY>
    ARC_DATA_ROOT=data/arc-prize-2025 ARC_TASK_FILE=evaluation uv run python scripts/smoke_arc_prompt.py \
        --results runs/<run>/live_arc_qwen38_results.csv --generation 3 --task arc_1ae2feb7
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from benchmarks.arc_challenge.loader import load_arc_tasks
from benchmarks.arc_challenge.scoring import score_arc_cases
from core.openai_compat_client import OpenAICompatLLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="run archive CSV (one row per candidate)")
    parser.add_argument("--generation", type=int, required=True, help="candidate whose prompt and knobs to use")
    parser.add_argument("--task", required=True, help="ARC task id, e.g. arc_1ae2feb7")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    args = parser.parse_args()

    row = pd.read_csv(args.results).set_index("generation").loc[args.generation]
    task = next(t for t in load_arc_tasks() if t.id == args.task)
    client = OpenAICompatLLMClient(
        base_url=os.environ["OPENAI_BASE_URL"], model=args.model, api_key=os.environ.get("OPENAI_API_KEY"),
        min_max_tokens=args.max_tokens, enable_thinking=True, thinking_budget=args.thinking_budget,
    )
    print(f"gen {args.generation} {row.mutation_type} | T={row.temperature:.2f} top_p={row.top_p:.2f} "
          f"max_tokens={max(int(row.max_tokens), args.max_tokens)} | task {task.id}")
    print("=== SYSTEM PROMPT ===\n" + row.system_prompt)
    print(f"=== USER PROMPT ({len(task.description)} chars, first 600) ===\n" + task.description[:600])

    start = time.monotonic()
    response = client.client.chat.completions.create(
        model=client.model,
        messages=[{"role": "system", "content": row.system_prompt}, {"role": "user", "content": task.description}],
        temperature=float(row.temperature),
        top_p=float(row.top_p),
        max_tokens=max(int(row.max_tokens), args.max_tokens),
        extra_body=client._extra_body(),
    )
    elapsed = time.monotonic() - start
    choice = response.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
    content = message.content or ""
    usage = response.usage
    print(f"\n=== RESPONSE: finish_reason={choice.finish_reason} | prompt={usage.prompt_tokens} "
          f"completion={usage.completion_tokens} tokens | {elapsed:.0f}s ===")
    print(f"=== REASONING ({len(reasoning)} chars, last 1500) ===\n" + reasoning[-1500:])
    print(f"=== CONTENT ({len(content)} chars) ===\n" + content)

    for label, cases, prefix in (("demonstration", task.test_cases, "train_example"),
                                 ("held-out", task.edge_cases, "test_example")):
        result = score_arc_cases(cases, content, prefix=prefix)
        print(f"\n{label}: runs_successfully={result.get('runs_successfully')} "
              f"pass@2={result.get('combined_score')} error={result.get('error', '')}")
        for key, value in result.items():
            if key.endswith("_pixel_acc"):
                print(f"   {key} = {value}")


if __name__ == "__main__":
    main()
