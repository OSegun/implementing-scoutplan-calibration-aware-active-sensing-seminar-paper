"""
Resumable experiment driver.

The sandbox used to produce these results terminates long-running background
processes, so the sweep is expressed as an explicit job list that is worked
through incrementally.  Each invocation processes jobs until a wall-clock
budget is reached, appends completed rows to the CSV, and exits.  Re-running
resumes exactly where it stopped, so the full sweep is reproducible from a
cold start with `while python3 run_jobs.py; do :; done`.
"""
from __future__ import annotations
import csv, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as X

RESULTS = X.RESULTS
OUT = os.path.join(RESULTS, "pilot_results.csv")
CLUSTER = os.path.join(RESULTS, "cluster_sweep.csv")
STATE = os.path.join(RESULTS, "_done.json")

BUDGET_SEC = float(os.environ.get("JOB_BUDGET", "36"))

AGENTS = ["Random", "Lawnmower", "GreedyEntropy", "REINFORCE"]


def job_list():
    jobs = []
    for T in X.TEMPERATURES:
        for a in AGENTS:
            for s in X.SEEDS:
                jobs.append(("cal", a, s, T, 1.6))
    for sig in [1.0, 1.6, 2.5, 4.0, 7.0]:
        for a in ["Lawnmower", "GreedyEntropy", "REINFORCE"]:
            for s in X.SEEDS:
                jobs.append(("clu", a, s, 1.00, sig))
    return jobs


def load_done():
    if os.path.exists(STATE) and os.path.getsize(STATE) > 0:
        try:
            with open(STATE) as f:
                return set(tuple(x) for x in json.load(f))
        except json.JSONDecodeError:
            pass
    return set()


def save_done(done):
    with open(STATE, "w") as f:
        json.dump([list(d) for d in done], f)


def append(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _run_one(j):
    """Module-level worker so it is picklable by multiprocessing."""
    kind, agent, seed, T, sig = j
    row = X.run_agent(agent, seed, T, sigma=sig)
    row["kind"] = kind
    return j, row


def main():
    t0 = time.time()
    done = load_done()
    jobs = job_list()
    todo = [j for j in jobs if tuple(map(str, j)) not in done]
    if not todo:
        print("ALL_DONE", len(jobs))
        return 1

    # Two worker processes: the sandbox exposes two cores, and each job is
    # independent, so the sweep parallelises trivially across (agent, seed).
    import multiprocessing as mp

    def _work(j):
        kind, agent, seed, T, sig = j
        row = X.run_agent(agent, seed, T, sigma=sig)
        row["kind"] = kind
        return j, row

    n = 0
    with mp.Pool(2) as pool:
        it = pool.imap_unordered(_run_one, todo, chunksize=1)
        for j, row in it:
            append(OUT if row["kind"] == "cal" else CLUSTER, row)
            done.add(tuple(map(str, j)))
            save_done(done)      # checkpoint after every job, so a kill is safe
            n += 1
            if time.time() - t0 > BUDGET_SEC:
                pool.terminate()
                break

    print(f"completed {n} jobs this pass; {len(jobs) - len(done)} remaining "
          f"({100 * len(done) / len(jobs):.1f}% done)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
