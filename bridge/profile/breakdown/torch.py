"""Kernels and steps from a PyTorch profiler trace (launch.py PROFILE=1)."""

import gzip
import json


def load(path):
    """Kernels as (start us, duration us, name, operation) and steps as (start, end). The trace links each kernel to
    the operation that launched it by "External id"."""
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        events = json.load(f)["traceEvents"]
    op_by_id = {
        e["args"]["External id"]: e["name"]
        for e in events
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }
    kernels = [
        (e["ts"], e["dur"], e["name"], op_by_id.get(e.get("args", {}).get("External id"), "?"))
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) > 0
    ]
    steps = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith("ProfilerStep#")
    )
    return kernels, steps
