"""Live ablation driver with checkpointing and graceful key-exhaustion handling.

Differs from run_experiments.py in three ways that matter for a trustworthy run:

  * strict mode is forced on, so a failed API call aborts the cell instead of
    silently recording mock output as if it were real;
  * the model is pinned (no silent fallback to a different model, which would
    mix two models' behaviour into one ablation);
  * progress is checkpointed per cell, so an exhausted key costs at most the
    cell in flight. Paste a new key into .env and re-run to resume.

Cell order is run-major: run 0 covers every task x condition before run 1
starts, so an interrupted sweep still leaves a balanced design behind.
"""
import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import core.groq_client as gc
from controller import EvolutionController
from tasks.benchmark_tasks import BENCHMARK_TASKS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

CONDITIONS = ["single_metric", "static_cascade", "ucb1_bandit", "no_pruning", "full_adaptive"]
ALL_TASKS = [t.id for t in BENCHMARK_TASKS]

CKPT = os.path.join("results", "live_checkpoint.json")
OUT = os.path.join("results", "ablation_live.csv")

# Groq returns 429 both for a spent daily allowance and for a burst over the
# per-minute token ceiling. The message text does not reliably separate them, so
# ask the API: the response headers carry the remaining daily request count
# whether the call succeeded or was rejected.
QUOTA_FLOOR = 5         # treat this few remaining requests as "key is finished"
COOLDOWN = 65.0         # the per-minute token bucket needs a full minute to refill
MAX_CELL_RETRIES = 4


# The daily *token* allowance is the real ceiling and it appears in no response
# header - only in the text of the 429 it eventually raises. The daily request
# header stays healthy while it is exhausted, so it cannot be used to detect it.
TPD_MARKERS = ("tokens per day", "tpd")


def probe_quota(key: str, model: str):
    """Return (remaining_requests, tpd_exhausted) for this key.

    remaining_requests is None when it cannot be read. tpd_exhausted is True
    when the probe came back as a tokens-per-day rejection.
    """
    from groq import Groq
    try:
        r = Groq(api_key=key).chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": "."}], max_tokens=1)
        return int(r.headers.get("x-ratelimit-remaining-requests", -1)), False
    except Exception as e:
        low = str(e).lower()
        if any(m in low for m in TPD_MARKERS):
            return None, True
        hdrs = getattr(getattr(e, "response", None), "headers", None)
        if hdrs and "x-ratelimit-remaining-requests" in hdrs:
            try:
                return int(hdrs["x-ratelimit-remaining-requests"]), False
            except (TypeError, ValueError):
                return None, False
        return None, False


def load_ckpt() -> set:
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as f:
            return {tuple(c) for c in json.load(f).get("completed", [])}
    return set()


def save_ckpt(done: set) -> None:
    os.makedirs("results", exist_ok=True)
    with open(CKPT, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(list(c) for c in done)}, f, indent=1)


def append_rows(df: pd.DataFrame) -> None:
    os.makedirs("results", exist_ok=True)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="repetitions per task x condition")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    ap.add_argument("--delay", type=float, default=6.0,
                    help="seconds between calls; 6s keeps ~646-token calls under 8000 TPM")
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    args = ap.parse_args()

    # Must be set before any controller is built: the seed agent is evaluated
    # inside EvolutionController.__init__, so a late strict flag would leave the
    # seed's calls free to fall back to mock.
    os.environ["GROQ_STRICT"] = "1"

    load_dotenv(override=True)
    # A single account's daily token allowance is far smaller than the grid, so
    # accept a pool: GROQ_API_KEYS as a comma-separated list, else GROQ_API_KEY.
    # Keys must belong to different Groq accounts to add capacity, because the
    # allowance is metered per organization rather than per key.
    raw = os.getenv("GROQ_API_KEYS", "") or os.getenv("GROQ_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        logger.error("No GROQ_API_KEY / GROQ_API_KEYS in .env")
        return 2
    key_idx = 0
    key = keys[key_idx]
    logger.info(f"Key pool: {len(keys)} key(s)")

    # Pin the model: fallback would silently swap models mid-ablation.
    gc.FALLBACK_MODELS = [args.model]

    done = load_ckpt()
    cells = [(r, t, c) for r in range(args.runs) for t in args.tasks for c in CONDITIONS]
    todo = [c for c in cells if tuple(c) not in done]

    logger.info(f"Grid: {len(cells)} cells ({args.runs} runs x {len(args.tasks)} tasks x "
                f"{len(CONDITIONS)} conditions), {args.generations} generations each")
    logger.info(f"Already complete: {len(done)}   Remaining: {len(todo)}")
    if not todo:
        logger.info("Nothing left to do - the grid is complete.")
        return 0

    live_total = 0
    t_start = time.time()

    for idx, (run_idx, task_id, cond) in enumerate(todo, 1):
        seed = 42 + run_idx * 1000 + CONDITIONS.index(cond) * 100 + args.tasks.index(task_id)
        random.seed(seed)
        np.random.seed(seed)

        logger.info(f"[{idx}/{len(todo)}] run={run_idx} task={task_id} cond={cond}")
        ctrl = None
        attempt = 0
        while True:
            try:
                ctrl = EvolutionController(
                    api_key=key, model=args.model, default_strategy=cond,
                    selected_task_id=task_id, inter_call_delay=args.delay, is_mock=False,
                )
                ctrl.llm_client.strict = True
                for _ in range(args.generations):
                    ctrl.run_generation(strategy=cond, task_id=task_id)
                break

            except RuntimeError as e:
                rem, tpd_done = probe_quota(key, args.model)
                spent = tpd_done or any(m in str(e).lower() for m in TPD_MARKERS)                     or (rem is not None and rem <= QUOTA_FLOOR)
                if spent:
                    if key_idx + 1 < len(keys):
                        key_idx += 1
                        key = keys[key_idx]
                        logger.warning(f"Key {key_idx} exhausted. Rotating to key "
                                       f"{key_idx + 1}/{len(keys)} and retrying this cell.")
                        continue        # rotation is not a failed attempt
                    save_ckpt(done)
                    el = (time.time() - t_start) / 60
                    logger.warning("=" * 72)
                    logger.warning(f"ALL {len(keys)} KEY(S) EXHAUSTED (daily token allowance).")
                    logger.warning(f"Completed {len(done)}/{len(cells)} cells "
                                   f"({live_total} live calls, {el:.0f} min this session).")
                    logger.warning("Add more keys to GROQ_API_KEYS in .env (comma-separated, "
                                   "one per Groq account) and re-run to resume.")
                    logger.warning("=" * 72)
                    return 3
                attempt += 1
                if attempt > MAX_CELL_RETRIES:
                    logger.error(f"Cell failed after {MAX_CELL_RETRIES} retries "
                                 f"({rem} requests left): {str(e)[:200]}")
                    save_ckpt(done)
                    return 1

                logger.warning(f"Transient limit ({rem} daily requests left). "
                               f"Cooling down {COOLDOWN:.0f}s, then retrying cell "
                               f"(attempt {attempt + 1}/{MAX_CELL_RETRIES + 1}).")
                time.sleep(COOLDOWN)

        df = ctrl.archive.to_detailed_dataframe()
        df["run_index"] = run_idx
        df["condition"] = cond
        # Provenance: proves every row came from a real completion.
        df["live_calls"] = ctrl.llm_client.live_calls
        df["mock_calls"] = ctrl.llm_client.mock_calls
        df["model"] = args.model
        append_rows(df)

        if ctrl.llm_client.mock_calls:
            logger.error(f"ABORT: {ctrl.llm_client.mock_calls} mock calls leaked into a live cell")
            save_ckpt(done)
            return 1

        live_total += ctrl.llm_client.live_calls
        done.add((run_idx, task_id, cond))
        save_ckpt(done)

        el = max(time.time() - t_start, 1e-9)
        rate = live_total / el * 3600
        remain = len(cells) - len(done)
        logger.info(f"    +{ctrl.llm_client.live_calls} live calls "
                    f"(total {live_total}, {rate:.0f}/hr, {remain} cells left)")

    logger.info(f"GRID COMPLETE: {len(done)} cells, {live_total} live calls -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
