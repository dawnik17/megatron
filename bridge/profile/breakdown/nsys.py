"""Kernels and steps from an Nsight Systems SQLite export (launch.py PROFILE=nsys), for rank 0's GPU."""

import bisect
import sqlite3
from collections import defaultdict


def load(path, n_steps=2):
    """Kernels as (start us, duration us, name, operation) and steps as (start, end). A kernel's operation is the
    innermost NVTX range (emit_nvtx) open on the thread that launched it. nsys marks no steps: its capture, launch.py's
    steps 30-31, is split into n_steps equal parts."""
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    tables = {t for (t,) in db.execute("SELECT name FROM sqlite_master")}
    device = db.execute("SELECT min(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]

    # a launch's thread id is its process id plus the thread in the low 24 bits; correlation ids are per process
    launches = {
        (tid >> 24, corr): (tid, start)
        for tid, start, corr in db.execute(
            "SELECT globalTid, start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    # emit_nvtx's ranges per thread, named like "aten::mm, seq = 0, op_id = 2"
    ranges = defaultdict(list)
    for tid, start, end, text in db.execute(
        "SELECT globalTid, start, end, text FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND text IS NOT NULL ORDER BY start"
    ):
        ranges[tid].append((start, end, text.split(",")[0]))
    range_starts = {tid: [r[0] for r in rs] for tid, rs in ranges.items()}

    def op(pid, corr):
        tid, t = launches.get((pid >> 24, corr), (None, None))
        rs = ranges.get(tid, [])
        i = bisect.bisect_right(range_starts.get(tid, []), t) - 1 if rs else -1
        while i >= 0 and rs[i][1] < t:  # the innermost range still open at the launch
            i -= 1
        return rs[i][2] if i >= 0 else "?"

    kernels = [
        (start / 1e3, (end - start) / 1e3, names[name], op(pid, corr))
        for start, end, name, pid, corr in db.execute(
            "SELECT start, end, demangledName, globalPid, correlationId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = ?",
            (device,),
        )
    ]
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "Memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "Memset"),
    ):
        if table in tables:
            rows = db.execute(f"SELECT start, end FROM {table} WHERE deviceId = ?", (device,))
            kernels += [(s / 1e3, (e - s) / 1e3, label, "?") for s, e in rows]

    first, last = min(k[0] for k in kernels), max(k[0] + k[1] for k in kernels)
    width = (last - first) / n_steps
    return kernels, [(first + i * width, first + (i + 1) * width) for i in range(n_steps)]
