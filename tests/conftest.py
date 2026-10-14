"""Pins the ARC loader to the 4-task fixture before any test module imports it.

benchmarks/arc_challenge/loader.py reads ARC_DATA_ROOT / ARC_TASK_FILE / ARC_TASK_LIMIT once at import, so this has
to run first; it also keeps a shell export (e.g. the full arc-prize-2025 set for a live run) out of the tests.
"""
import os
from pathlib import Path

os.environ["ARC_DATA_ROOT"] = str(Path(__file__).parent / "fixtures" / "arc_sample")
os.environ.pop("ARC_TASK_FILE", None)
os.environ.pop("ARC_TASK_LIMIT", None)
