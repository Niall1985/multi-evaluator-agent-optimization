"""Runs untrusted candidate code in a child process with a per-call time limit.

Candidate programs used to be exec'd and called in the evaluator's own process, so one `solve()` with an infinite
loop hung the whole run (threads cannot be interrupted). Here a child process (evaluation/sandbox_worker.py) execs
the code and runs every call; the parent waits at most `timeout` seconds per message and kills the child when a
call overruns, then restarts it for the remaining calls, so one bad input costs one TimeoutError, not the rest.

The child is a fresh `python -I` running a stdlib-only script, not a multiprocessing child: `fork` is unsafe in a
parent with threads (the controller evaluates tasks on a thread pool), and spawn/forkserver children re-run the
parent's __main__ (the whole runner, or the Streamlit app). Candidate stdout/stderr are discarded.
"""
import os
import subprocess
import sys
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, List, Optional, Sequence, Tuple

CALL_TIMEOUT_SEC = 5.0  # ARC solutions run in milliseconds; this only catches loops and runaway searches
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_worker.py")


@dataclass
class ProgramRun:
    """`load_error`: the code failed to exec (or timed out doing so). `functions`: the names that were called.
    `results[i][j]`: ("ok", value) or ("error", "Type: message") for input i and function j. `elapsed_ms`: time
    spent inside the calls themselves, excluding process start-up."""

    load_error: Optional[str] = None
    functions: List[str] = field(default_factory=list)
    results: List[List[Tuple[str, Any]]] = field(default_factory=list)
    elapsed_ms: float = 0.0


def _start_worker(job: tuple) -> Tuple[subprocess.Popen, Connection]:
    job_r, job_w = os.pipe()
    res_r, res_w = os.pipe()
    proc = subprocess.Popen(
        [sys.executable, "-I", _WORKER, str(job_r), str(res_w)],
        pass_fds=(job_r, res_w),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(job_r)
    os.close(res_w)
    with Connection(job_w, readable=False) as job_conn:
        job_conn.send(job)
    return proc, Connection(res_r, writable=False)


def run_program(
    code: str,
    name_groups: Sequence[Sequence[str]],
    inputs: Sequence[tuple],
    timeout: float = CALL_TIMEOUT_SEC,
) -> ProgramRun:
    """Execs `code` in a child process and calls each selected function on each argument tuple in `inputs`."""
    run = ProgramRun()
    flat: List[Tuple[str, Any]] = []
    total: Optional[int] = None
    groups = [tuple(g) for g in name_groups]
    while total is None or len(flat) < total:
        proc, conn = _start_worker((code, groups, [tuple(a) for a in inputs], len(flat)))
        try:
            # Start-up (interpreter, plus whatever the candidate imports) gets the same allowance as one call
            if not conn.poll(timeout + 5.0):
                run.load_error = f"TimeoutError: loading the program took over {timeout:g}s"
                return run
            kind, payload = conn.recv()
            if kind == "load_error":
                run.load_error = payload
                return run
            run.functions = payload
            total = len(inputs) * len(payload)
            while len(flat) < total:
                if not conn.poll(timeout):
                    flat.append(("error", f"TimeoutError: call exceeded {timeout:g}s"))
                    run.elapsed_ms += timeout * 1000.0
                    break  # kill the child; the outer loop restarts it at the next call
                status, value, elapsed = conn.recv()
                flat.append((status, value))
                run.elapsed_ms += elapsed * 1000.0
        except (EOFError, OSError):  # the child died (memory cap, os._exit, segfault)
            if total is None:
                run.load_error = "ChildProcessError: the candidate process died while loading"
                return run
            flat.append(("error", "ChildProcessError: the candidate process died mid-call"))
        finally:
            conn.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    width = len(run.functions)
    run.results = [flat[i * width:(i + 1) * width] for i in range(len(inputs))] if width else []
    return run
