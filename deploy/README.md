# Running the ARC optimizer on a self-hosted model

How to run `scripts/live_arc_run.py`, the MESA evolution loop on the ARC-AGI benchmark, against an
OpenAI-compatible model server instead of Groq: Qwen3.8-27B served by vLLM on Modal (`deploy/modal_vllm.py`),
or any local server such as llama.cpp.

## 1. Prerequisites

```bash
uv sync                      # project deps, including the openai SDK
uv tool install modal        # or: pip install modal
modal setup                  # logs the CLI in to your Modal workspace
```

ARC data: download the [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025/data) files from
Kaggle into `data/arc-prize-2025/` (git-ignored). The loader expects the Kaggle layout:

```
data/arc-prize-2025/arc-agi_evaluation_challenges.json   # 120 tasks, 172 held-out test inputs
data/arc-prize-2025/arc-agi_evaluation_solutions.json
data/arc-prize-2025/arc-agi_training_*.json              # optional
```

`ARC_DATA_ROOT` points the loader at that folder, `ARC_TASK_FILE=evaluation` restricts it to one split, and
`ARC_TASK_LIMIT=N` caps the number of tasks (useful for a cheap first run).

## 2. Deploy the model server (Modal)

```bash
modal deploy deploy/modal_vllm.py              # default: 4-bit AWQ on 1x A100-40GB
QUANT=fp8 modal deploy deploy/modal_vllm.py    # FP8 on 1x A100-80GB (twice the KV-cache room)
```

The deploy prints the endpoint URL, `https://<workspace>--qwen38-27b-vllm-server.us-west.modal.direct`.
The server needs Modal proxy auth (requests without a token get 401). Create a token once and keep it
out of git:

```bash
modal workspace proxy-tokens create --name mesa-arc --json
# -> {"Authorization": "Bearer wk-....ws-....", ...}
```

Put the URL and token in a private env file (not `.env`: setting `OPENAI_BASE_URL` there switches every
entry point - the Streamlit app, `run_experiments.py` - to this provider):

```bash
# ~/modal_run.env  (chmod 600)
export OPENAI_BASE_URL=https://<workspace>--qwen38-27b-vllm-server.us-west.modal.direct/v1
export OPENAI_API_KEY=wk-<id>.ws-<secret>
```

### Wake the server before a run

The server scales to zero after 4 idle minutes, and while no container is up Modal answers **503**. A cold
start takes 2-3 minutes with cached weights (5-10 minutes the first time, which downloads ~21 GB of weights). The
client's retries give up after about a minute, so wait for health first:

```bash
source ~/modal_run.env
until [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $OPENAI_API_KEY" \
        ${OPENAI_BASE_URL%/v1}/health)" = 200 ]; do sleep 10; done; echo ready
```

## 3. Run

```bash
source ~/modal_run.env
ARC_DATA_ROOT=data/arc-prize-2025 ARC_TASK_FILE=evaluation PYTHONUNBUFFERED=1 \
uv run python scripts/live_arc_run.py \
    --model Qwen/Qwen3.8-27B \
    --archetype "ARC Code-First Solver" \
    --thinking on --thinking-budget 4096 --max-tokens 10000 \
    --concurrency 22 \
    --rpm 0 --tpm 0 --delay 0 \
    --generations 10 \
    --csv live_arc_qwen38_results.csv --task-csv live_arc_qwen38_task_results.csv \
    > live_arc_qwen38.log 2>&1 &
```

For a first try add `ARC_TASK_LIMIT=16 ... --generations 1`: it exercises the whole path in a few minutes.

### Program-archive mode (`--mode archive`)

Evolves on the evaluation split itself. Programs are selected on the demonstration pairs only, and the
held-out outputs are used only to score the final picks. The archive keeps every program written for each
task (one MAP-Elites grid per task). Each generation:

- **Children:** one child agent per island. TPE picks its parent agent, using archived fitness as the
  prior, and Optuna TPE picks the mutation idea.
- **Tasks:** UCB1 picks the unsolved tasks, and each child gets `--tasks-per-child` of them.
- **Calls:** one call per task. For each one, TPE either picks an archived program to repair (the prompt
  shows its failing demonstrations plus other archived programs) or starts the task fresh.
- **Retiring:** a task is dropped once a program passes all its demonstrations.
- **Reported score:** held-out pass@2, where the two attempts are the two best archived programs that give
  different answers.

```bash
uv run python scripts/live_arc_run.py --mode archive \
    --model Qwen/Qwen3.8-27B --archetype "ARC Code-First Solver" \
    --thinking on --thinking-budget 4096 --max-tokens 10000 --concurrency 22 \
    --repair-thinking-budget 4096 --repair-reasoning-effort low \
    --rpm 0 --tpm 0 --delay 0 --generations 20 \
    --islands 4 --tasks-per-child 30 --max-temperature 0.7 --token-cap 25000000 \
    --state live_arc_state.json --program-csv live_arc_program_results.csv \
    --csv live_arc_agent_results.csv --task-csv live_arc_heldout_results.csv > live_arc_archive.log 2>&1 &
```

- **Run size:** the seed generation makes one call per task. After that, each generation makes
  islands × tasks-per-child calls plus one mutation call per island.
- **Checkpoint:** `--state` is rewritten after every generation. Add `--resume` to continue from it.
- **Program CSV:** one row per program, with its cell, scores, tokens and `finish_reason`.
- **Repair calls think less:** a repair prompt already shows the failing demonstrations, so repair calls get
  Qwen3.8's `reasoning_effort: low` (sent in `chat_template_kwargs`) with `--repair-thinking-budget 4096`; both
  are the defaults. Fresh attempts keep `--thinking-budget` and the model's default effort (xhigh). A 2048
  budget was tried first: 18 of 58 repairs then kept reasoning inside code comments, hit `--max-tokens`, and
  mostly never closed their code block.

### Why these flags

| Flag | Why |
|---|---|
| `--archetype "ARC Code-First Solver"` | Seed prompt that asks for one ```` ```python ```` block and keeps reasoning in comments. With the default seed, Qwen writes pages of analysis and hits the token cap before any code: 11% of programs ran, against 89% code-first. |
| `--thinking on --thinking-budget 4096` | A private reasoning pass. vLLM ends the thinking block after 4096 tokens (`thinking_token_budget`, enabled by `--reasoning-config` in the deploy) by inserting Qwen's official early-stop text, "Considering the limited time by the user, I have to give the solution based on the thinking directly now." followed by `</think>`. `--thinking off` turns it off; `server` sends nothing. |
| `--max-tokens 10000` | Floor on every call's `max_tokens` (the default for this provider): 4096 thinking + about 6k for the code. The first Qwen3.8 run used 1024 thinking tokens and solved 0-2 of 120 tasks per generation. 3000 was too small for Qwen on ARC; 6000 was used for the measured run below. |
| `--concurrency 22` | Tasks in flight at once (the default for this provider, and the server's `--max-num-seqs`). 32 filled the 16-bit KV cache and requests timed out (see troubleshooting); the FP8 KV cache (~385k tokens) has twice the room. Watch the KV-use line in the server log on the first generation. |
| `--rpm 0 --tpm 0 --delay 0` | The defaults are sized for Groq's free tier and would serialize a parallel run. |
| `--provider openai` / `--base-url` | Not needed when `OPENAI_BASE_URL` is set; the provider is picked from it. |

Other runner flags (`--strategy`, `--deep-threshold`, `--lambda-penalty`, ...) are described by
`uv run python scripts/live_arc_run.py --help`.

### Local server instead of Modal (llama.cpp, Ollama, vLLM on your own GPU)

```bash
llama-server -m model.gguf --parallel 4          # prints host:port
uv run python scripts/live_arc_run.py --base-url http://127.0.0.1:<port>/v1 --model local \
    --concurrency 4 --rpm 0 --tpm 0 --delay 0
```

Match `--concurrency` to the server's parallel slots. `--thinking-budget` is vLLM-only; other servers
ignore it.

## 4. Output and monitoring

| File | Content |
|---|---|
| `live_arc_qwen38.log` | One `gen N ...` line per generation (fitness, dF, per-metric means), retry warnings, and at the end the best agent and its programs. |
| `live_arc_qwen38_results.csv` | Archive export, one row per candidate (git-ignored). |
| `live_arc_qwen38_task_results.csv` | One row per (generation, task) with every metric and a `solved` flag (git-ignored). |

- **The log stays empty for the first 10-15 minutes**: the seed generation runs inside the controller's
  constructor, before the first print. That is normal.
- Server side: `modal app logs qwen38-27b-vllm | grep loggers.py` shows throughput, running/waiting
  requests and KV-cache use every 10 s.
- Client side, if it looks stuck: `uvx py-spy dump --pid <pid>` shows whether threads wait on HTTP (server busy)
  or sit in candidate code.

`arc_pass_at_2_test` is the official score (fraction of the 120 tasks solved). The 2025 evaluation split is
ARC-AGI-2, so expect low single digits. `arc_pixel_accuracy`, `arc_shape_match` and `arc_color_palette` are the
partial-credit signals that move first.

## 5. Time and cost (measured 2026-10-12 with Qwen3.6-27B AWQ, A100-40GB, 32-way, 6000 tokens)

| | |
|---|---|
| Throughput | 750-900 output tokens/s across 25-28 concurrent requests |
| Seed generation, code-first, 120 tasks | ~15 min |
| Later generations | 15-30+ min: mutated prompts can grow the answers again |
| Cost | about $2.42/h (A100-40GB $2.10 + 4 CPU + 16 GiB RAM, Modal list prices); billed only while a container runs, plus 4 idle minutes |

A full 10-generation run is a few hours. Use `ARC_TASK_LIMIT` to size it, or `QUANT=fp8` on an A100-80GB for
more KV-cache room (about $2.81/h).

## 6. Stopping

Kill the client by its PID (`pgrep -f scripts/live_arc_run.py`). Do not use a broad `pkill -f` pattern, which can
also match the shell that runs it. The Modal container stops on its own 4 minutes after the last request.
`modal app stop qwen38-27b-vllm` stops it at once (it then needs a new `modal deploy`).

To continue a stopped run, rerun the same command with `--resume` and `--generations` set to the number of
generations still wanted. It rebuilds the archive and search state from `--csv` / `--task-csv` and skips the seed.
A generation that was cut off is not in the CSVs and runs again. Copy the two CSVs first: the run overwrites them.

Revoke the proxy token when you are done: `modal workspace proxy-tokens list` for its ID, then
`modal workspace proxy-tokens delete <TOKEN_ID>`.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Every call fails at once with 503 | No container is up (cold start). Wake the server first (section 2). |
| A burst of `InternalServerError (attempt 1)` right after `modal deploy` | The old container drains while the new one starts. The retries succeed; you can ignore it. |
| `APITimeoutError` / `APIConnectionError` at attempt 2+ | Requests waited more than the client's 600 s timeout. The KV cache was full (98%) at 32-way concurrency with 6000-token answers, so requests queued. The defaults are now 22 with an FP8 KV cache; if it recurs, lower `--concurrency` further, lower `--max-tokens`, or deploy `QUANT=fp8` on 80 GB. |
| Bursts of `InternalServerError` mid-run, a second container in `modal container list`, then `503 no upstreams available` | Modal autoscaling. With `target_concurrency=16`, the 17th request scaled up a second A100, and the scale-down after it aborted in-flight requests. The deploy now runs one container (no `target_concurrency`) and drains for up to 10 minutes. |
| 500s then `503 no upstreams available` right as a new generation starts | The container scaled down while the client was scoring the previous generation (~2.5 min with no requests). The scale-down window is now 4 minutes; it was 150 s. Continue the stopped run with `--resume` (section 6). |
| `arc_runs_successfully` near 0 | Answers truncated before the code (`finish_reason=length`). Use the code-first archetype, a thinking budget, and enough `--max-tokens`. |
| Run hangs forever, no new log lines | Was: a generated `solve()` with an infinite loop. Candidate code now runs in a subprocess with a 5 s per-call limit (`evaluation/sandbox.py`), so this should not recur. If it does, `py-spy dump` shows where. |
| `We were unable to determine the version of Python` on deploy | The image needs `add_python="3.12"`, which `deploy/modal_vllm.py` already sets. |

## 8. Tests

```bash
uv run pytest   # offline: stub HTTP server + mock LLM, no model calls; ARC tests use tests/fixtures/arc_sample
```
