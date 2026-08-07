"""
scoutplan CLI
=============
Single entry point for every stage of the pilot study, so the project is
runnable as `python run.py <stage>` instead of remembering five separate
commands. See README.md section 8 for the full walkthrough of what each
stage does and why it exists.

Stages:
  smoke     ~12s sanity check: field, calibration instrument, one rollout.
            Asserts accuracy invariance across temperature.
  sweep     Full 195-job resumable sweep via run_jobs.py. Safe to Ctrl-C and
            re-run; exits 1 and prints ALL_DONE once nothing is left to do.
  figures   Regenerate Figures 5-8 and results/summary.json from the CSVs.
  all       sweep, then figures.

This repository produces experiments, results and result figures. It does not
build documents and it does not author diagrams: both are presentation concerns
with their own dependencies, and keeping them out is what preserves the
NumPy-only claim in README.md.

Figures 1-4 and drawio/ are the system diagrams. They describe the design rather
than the data, so they are authored once outside the repo and shipped here for
reuse — see README.md, "Design".
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def smoke_test():
    import numpy as np
    from field import DiseaseField
    from perception import measure_calibration, CalibratedClassifier
    from env import ScoutEnv
    from agents import LawnmowerAgent

    fld = DiseaseField(size=12, n_parents=3, offspring_mean=10, sigma=1.6,
                        rng=np.random.default_rng(0))
    print("prevalence", round(fld.prevalence, 4), "| Morisita", round(fld.morisita_index(), 2))

    accs = set()
    for T in (0.3, 1.0, 4.0):
        r = measure_calibration(0.816, T, n=8000, seed=1)
        accs.add(round(r["accuracy"], 4))
        print(f"T={T:<4} acc={r['accuracy']:.4f}  ECE={r['ece']:.4f}")
    assert len(accs) == 1, f"instrument broken: accuracy moved with T -> {accs}"

    rng = np.random.default_rng(0)
    env = ScoutEnv(fld, CalibratedClassifier(0.816, 1.0, rng), budget=190,
                   prior=0.15, detect_threshold=0.75, rng=rng)
    agent = LawnmowerAgent()
    obs = env.reset()
    agent.reset(env)
    done = False
    while not done:
        obs, r, done, info = env.step(agent.act(obs, env))
    print("lawnmower:", {k: round(v, 3) for k, v in info.items()})
    print("smoke test OK")


def run_sweep():
    """Repeatedly invoke run_jobs.py until it reports ALL_DONE."""
    budget = os.environ.get("JOB_BUDGET", "60")
    env = dict(os.environ, JOB_BUDGET=budget)
    while True:
        result = subprocess.run([sys.executable, "run_jobs.py"], cwd=HERE, env=env)
        if result.returncode != 0:
            break


def run_figures():
    """Figures 5-8, the result plots. Reads results/*.csv, writes summary.json."""
    subprocess.run([sys.executable, "make_result_figures.py"], cwd=HERE, check=True)


def main():
    p = argparse.ArgumentParser(description="ScoutPlan pilot study entry point")
    p.add_argument("stage", choices=["smoke", "sweep", "figures", "all"])
    args = p.parse_args()

    if args.stage == "smoke":
        smoke_test()
    elif args.stage == "sweep":
        run_sweep()
    elif args.stage == "figures":
        run_figures()
    elif args.stage == "all":
        run_sweep()
        run_figures()


if __name__ == "__main__":
    main()
