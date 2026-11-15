"""Run exactly one scenario of the opt3s model in this process, then exit.

The opt3s counterpart of the legacy ``run_one.py``; reads the job as JSON on
stdin::

    {"scenario": {...}, "store_root": "results", "save_solution": true}

Run from the repository root as ``python -m depot.run_one``.
"""

from __future__ import annotations

import json
import sys

from benchmark import RunMetrics
from results import ResultStore
from scenario import Scenario

from .runner import main


def run(job: dict) -> int:
    scenario = Scenario.from_dict(job["scenario"])
    store = ResultStore(job.get("store_root", "results"))
    metrics = RunMetrics(trace_python_memory=bool(job.get("trace_python_memory", False)))
    result = main(
        scenario,
        metrics=metrics,
        verbose=bool(job.get("verbose", True)),
        exact_nonzeros_before_solve=bool(job.get("exact_nonzeros_before_solve", False)),
    )
    run_dir = store.save(result, save_solution=bool(job.get("save_solution", True)))
    print(f"  saved to {run_dir}")
    return 0


if __name__ == "__main__":
    payload = sys.stdin.read()
    if not payload.strip():
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    sys.exit(run(json.loads(payload)))
