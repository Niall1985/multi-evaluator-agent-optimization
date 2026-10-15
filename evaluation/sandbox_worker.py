"""Child side of evaluation/sandbox.py. Run as a script (`python -I sandbox_worker.py <job_fd> <result_fd>`), it
imports only the standard library, so start-up does not pull in the project or re-run the parent's __main__.

Reads one job (code, name_groups, inputs, start) and sends ("load_error", msg) or ("loaded", names), then one
(status, value, seconds) per call, function-major within each input, starting at flat call index `start`.
"""
import sys
import time
from multiprocessing.connection import Connection

MEMORY_LIMIT_BYTES = 2 * 1024 ** 3  # address-space cap so a runaway allocation fails here, not on the host


def main() -> None:
    job, out = Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False)
    code, name_groups, inputs, start = job.recv()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass

    namespace: dict = {"__name__": "__candidate__"}
    try:
        exec(code, namespace)
    except BaseException as e:  # noqa: BLE001 - candidate code is untrusted
        out.send(("load_error", f"{type(e).__name__}: {e}"))
        return
    # First group with any callable wins, e.g. [(attempt_1, attempt_2), (solve,)]
    names: list = []
    for group in name_groups:
        names = [n for n in group if callable(namespace.get(n))]
        if names:
            break
    out.send(("loaded", names))

    fns = [namespace[n] for n in names]
    for k in range(start, len(inputs) * len(fns)):
        args = inputs[k // len(fns)]
        t0 = time.perf_counter()
        try:
            msg = ("ok", fns[k % len(fns)](*args))
        except BaseException as e:  # noqa: BLE001
            msg = ("error", f"{type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        try:
            out.send((*msg, elapsed))
        except Exception as e:  # noqa: BLE001 - e.g. an unpicklable return value
            out.send(("error", f"{type(e).__name__}: result could not be sent back: {e}", elapsed))


if __name__ == "__main__":
    main()
