"""Qwen3.8-27B on 1x A100 (or smaller) with vLLM, no speculative decoding.

Adapted from the SGLang/H100 endpoint (Qwen/Qwen3.6-27B-FP8 + DFlash draft):
- engine: vLLM, public image vllm/vllm-openai (no patched nightly, no autoinference-utils)
- no --speculative-config, so no draft model or MTP head
- text-only (--language-model-only): the vision encoder is skipped, saving VRAM

Two weight variants, chosen with QUANT at deploy time:
    fp8            Qwen/Qwen3.8-27B-FP8          ~31 GB on disk  -> A100-80GB
    awq (default)  cyankiwi/Qwen3.8-27B-AWQ-INT4 ~21 GB on disk  -> A100-40GB (L40S also works)

Qwen3.8-27B has the same architecture as Qwen3.6-27B (Qwen3_5ForConditionalGeneration), and the vLLM recipe
uses the same parsers, so only the weights changed.

A100 has no FP8 tensor cores; vLLM runs FP8 checkpoints there as weight-only W8A16 with
Marlin kernels (picked automatically). This saves memory, but there is no FP8 compute speedup.

    modal deploy deploy/modal_vllm.py              # awq on A100-40GB
    QUANT=fp8 modal deploy deploy/modal_vllm.py    # fp8 on A100-80GB

Then point the optimizer at it (Modal proxy auth: `modal workspace proxy-tokens create`). The full run guide,
with the recommended runner flags, is deploy/README.md:
    OPENAI_BASE_URL=https://<workspace>--qwen38-27b-vllm-server.us-west.modal.direct/v1
    OPENAI_API_KEY=wk-<id>.ws-<secret>
    uv run python scripts/live_arc_run.py --model Qwen/Qwen3.8-27B --archetype "ARC Code-First Solver" \
        --thinking on --thinking-budget 4096 --max-tokens 10000 --concurrency 22 --rpm 0 --tpm 0 --delay 0

Sources: https://recipes.vllm.ai/Qwen/Qwen3.8-27B, https://huggingface.co/Qwen/Qwen3.8-27B,
https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html (thinking-budget early stop),
https://docs.vllm.ai/en/latest/features/reasoning_outputs/ (--reasoning-config),
https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/,
https://modal.com/docs/guide/servers
"""
import json
import os

import modal

MINUTES = 60
DEFAULT_PORT = 8000

QUANT = os.environ.get("QUANT", "awq")
VARIANTS = {
    "fp8": {
        "repo": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "gpu": "A100-80GB",
        "max_model_len": 65536,
    },
    "awq": {
        "repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4",  # compressed-tensors W4A16, group size 32
        "revision": "6e134bae811fb5adac50ee042ae5f029ac6779aa",
        "gpu": "A100-40GB",
        # Largest ARC prompts (30x30 grids) are ~15k tokens, plus up to 12k of answer
        "max_model_len": 49152,
    },
}
if QUANT not in VARIANTS:
    raise ValueError(f"QUANT must be one of {sorted(VARIANTS)}, got {QUANT!r}")
VARIANT = VARIANTS[QUANT]

# One served name for both variants, so clients do not change when QUANT changes
SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B"
ROUTING_REGION = "us-west"
REQUIRE_AUTHENTICATION = True
VLLM_IMAGE_TAG = "vllm/vllm-openai:v0.31.0"

N_GPUS = 1
GPU = f"{VARIANT['gpu']}:{N_GPUS}"
CPU = 4
MEMORY_MB = 16384

# Idle time before the container stops. Must outlast the client work between generations (scoring 120
# programs in the sandbox, the GP fit: ~2.5 min): at 150 s the container drained right as gen 2 started.
SCALEDOWN_WINDOW = 4 * MINUTES
# 32 filled the 16-bit KV cache with 6k-token answers and queued requests past the client timeout. The FP8 KV
# cache (~385k tokens on A100-40GB) has twice the room, so 22 fits with headroom for a few 30x30 prompts.
TARGET_INPUTS = 22
STARTUP_TIMEOUT = 30 * MINUTES

serving_image = (
    # add_python: Modal cannot detect the image's own venv Python; `vllm` keeps using /opt/venv via its shebang
    modal.Image.from_registry(VLLM_IMAGE_TAG, add_python="3.12")
    .entrypoint([])  # the image's entrypoint is `vllm serve`; Modal runs its own
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
            "QUANT": QUANT,  # the container re-imports this module, so it must see the same variant
        }
    )
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

SERVER_ARGS = {
    "--revision": VARIANT["revision"],
    "--served-model-name": SERVED_MODEL_NAME,
    "--host": "0.0.0.0",
    "--port": str(DEFAULT_PORT),
    "--tensor-parallel-size": str(N_GPUS),
    "--language-model-only": "",
    "--max-model-len": str(VARIANT["max_model_len"]),
    "--gpu-memory-utilization": "0.90",
    # 8-bit KV cache, about twice the cached tokens. On A100 (sm80) FlashAttention 2 has no FP8 KV, so vLLM picks
    # FlashInfer or Triton attention: https://docs.vllm.ai/en/latest/design/attention_backends/
    "--kv-cache-dtype": "fp8",
    "--max-num-seqs": str(TARGET_INPUTS),
    "--max-num-batched-tokens": "16384",  # chunked-prefill budget: a 15k-token 30x30 ARC prompt prefills in ~1 step
    "--max-cudagraph-capture-size": str(TARGET_INPUTS),  # recipe: lower this on Mamba cache errors
    "--reasoning-parser": "qwen3",
    # Lets requests cap thinking with `thinking_token_budget`; at the cap the server emits this end string. It is
    # Qwen's own early-stopping text (qwen.readthedocs.io quickstart, "thinking budget"), closed with the
    # template's `</think>\n\n`, so the model sees the same hand-off it was trained to summarize from.
    "--reasoning-config": json.dumps(
        {
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "\n\nConsidering the limited time by the user, I have to give the solution based on "
            "the thinking directly now.\n</think>\n\n",
        }
    ),
    "--enable-auto-tool-choice": "",
    "--tool-call-parser": "qwen3_coder",
    "--trust-remote-code": "",
    "--uvicorn-log-level": "warning",
}

WARMUP_PAYLOAD = {
    "model": SERVED_MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with JSON facts about Tokyo."}],
    "max_tokens": 64,
    "temperature": 0,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "city_facts",
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "population": {"type": "integer"},
                },
                "required": ["city", "population"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
}


def _wait_healthy(process, timeout: float, poll: float = 5.0) -> None:
    import time
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEFAULT_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(poll)
    raise TimeoutError(f"vllm serve not healthy after {timeout:.0f}s")


def _warmup(successful_requests: int = 2, request_timeout: float = 60.0) -> None:
    import urllib.request

    body = json.dumps(WARMUP_PAYLOAD).encode()
    for _ in range(successful_requests):
        req = urllib.request.Request(
            f"http://127.0.0.1:{DEFAULT_PORT}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=request_timeout) as r:
            r.read()


app = modal.App(name="qwen38-27b-vllm")


@app.server(
    image=serving_image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    min_containers=0,
    scaledown_window=SCALEDOWN_WINDOW,
    port=DEFAULT_PORT,
    routing_region=ROUTING_REGION,
    unauthenticated=not REQUIRE_AUTHENTICATION,
    # A drained container (redeploy) finishes its in-flight requests; ARC answers take minutes
    exit_grace_period=10 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    # No target_concurrency: one container. With target_concurrency=16, the 17th request (the mutation call)
    # scaled up a second A100, and the scale-down that followed aborted the other container's in-flight requests.
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
class Server:
    @modal.enter()
    def startup(self):
        import subprocess

        cmd = ["vllm", "serve", VARIANT["repo"]]
        for flag, value in SERVER_ARGS.items():
            cmd += [flag, value] if value else [flag]
        print(*cmd)
        self.process = subprocess.Popen(cmd)
        _wait_healthy(self.process, timeout=STARTUP_TIMEOUT)
        _warmup()
        print(f"{SERVED_MODEL_NAME} [{QUANT}] ({GPU}) vllm deployment is ready.")

    @modal.exit()
    def stop(self):
        if hasattr(self, "process"):
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except Exception:
                self.process.kill()
