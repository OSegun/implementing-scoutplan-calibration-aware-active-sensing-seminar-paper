"""
scoutplan.experiment
====================
Pilot experiment runner.

Produces three result files in ./results:
  calibration_table.csv   accuracy and ECE measured at each temperature
  pilot_results.csv       per-(temperature, agent, seed) episode metrics
  cluster_sweep.csv       per-(sigma, agent, seed) episode metrics

Statistics follow the guidance of Colas et al. (2018, 2019): a fixed seed
budget declared in advance, and interval estimates via bootstrap rather than
a bare mean.  The interquartile mean is reported alongside the mean because it
is robust to the outlier runs that dominate small-seed reinforcement learning
comparisons.
"""

from __future__ import annotations
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from field import DiseaseField
from perception import CalibratedClassifier, measure_calibration, expected_calibration_error
from env import ScoutEnv
from agents import RandomAgent, LawnmowerAgent, GreedyEntropyAgent, ReinforceAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

TEMPERATURES = [0.30, 0.50, 1.00, 2.00, 3.00, 4.00]
SEEDS = [0, 1, 2, 3, 4]
BASE_ACC = 0.816            # field-condition generalisation, Ahmad et al. (2023)
GRID = 12
BUDGET = 190.0
PRIOR = 0.15
TAU = 0.75
TRAIN_EPISODES = 400
EVAL_EPISODES = 20


# --------------------------------------------------------------------------

def iqm(x):
    """Interquartile mean."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 4:
        return float(x.mean())
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(x[lo:hi].mean())


def bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = np.array([iqm(x[i]) for i in idx])
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def make_env(seed, temperature, sigma=1.6, acc=BASE_ACC):
    rng = np.random.default_rng(seed)
    fld = DiseaseField(size=GRID, n_parents=3, offspring_mean=10,
                       sigma=sigma, rng=rng)
    if fld.n_diseased < 5:                       # guarantee a non-trivial task
        fld = DiseaseField(size=GRID, n_parents=4, offspring_mean=14,
                           sigma=sigma, rng=rng)
    clf = CalibratedClassifier(accuracy=acc, temperature=temperature, rng=rng)
    return ScoutEnv(fld, clf, budget=BUDGET, prior=PRIOR,
                    detect_threshold=TAU, rng=rng), fld


def rollout(env, agent, learn=False):
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset(env)
    traj, done = [], False
    while not done:
        a = agent.act(obs, env)
        nobs, r, done, info = env.step(a)
        if learn:
            o, h, p, act = agent._cache
            traj.append((o, h, p, act, r))
        obs = nobs
    return info, traj, env.obs_log


def run_agent(agent_name, seed, temperature, sigma=1.6):
    env, fld = make_env(seed, temperature, sigma)
    rng = np.random.default_rng(seed + 9973)

    if agent_name == "Random":
        agent = RandomAgent(rng)
    elif agent_name == "Lawnmower":
        agent = LawnmowerAgent(rng)
    elif agent_name == "GreedyEntropy":
        agent = GreedyEntropyAgent(rng)
    elif agent_name == "REINFORCE":
        agent = ReinforceAgent(env.obs_dim, env.n_actions, rng=rng)
        for _ in range(TRAIN_EPISODES):
            _, traj, _ = rollout(env, agent, learn=True)
            if traj:
                agent.update(traj)
        agent.eval_mode()
    else:
        raise ValueError(agent_name)

    recalls, dpj, cov, ttfd, eces, precs, fas = [], [], [], [], [], [], []
    for _ in range(EVAL_EPISODES):
        info, _, log = rollout(env, agent, learn=False)
        recalls.append(info["recall"])
        dpj.append(info["detections_per_joule"])
        cov.append(info["coverage"])
        precs.append(info["precision"])
        fas.append(info["false_alarms"])
        ttfd.append(info["time_to_first_detection"])
        if log:
            conf = np.array([c for c, _ in log])
            corr = np.array([k for _, k in log])
            eces.append(expected_calibration_error(conf, corr))

    return {
        "agent": agent_name,
        "seed": seed,
        "temperature": temperature,
        "sigma": sigma,
        "prevalence": fld.prevalence,
        "morisita": fld.morisita_index(),
        "recall": float(np.mean(recalls)),
        "det_per_joule": float(np.mean(dpj)),
        "precision": float(np.mean(precs)),
        "false_alarms": float(np.mean(fas)),
        "time_to_first_detection": float(np.mean(ttfd)),
        "coverage": float(np.mean(cov)),
        "episode_ece": float(np.mean(eces)) if eces else float("nan"),
    }


# --------------------------------------------------------------------------

def main():
    t0 = time.time()

    # --- 1. calibration table -------------------------------------------
    print("[1/3] measuring calibration table", flush=True)
    cal_rows = [measure_calibration(BASE_ACC, T, n=40000, seed=17)
                for T in TEMPERATURES]
    with open(os.path.join(RESULTS, "calibration_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    for r in cal_rows:
        print(f"    T={r['temperature']:.2f}  acc={r['accuracy']:.4f}  "
              f"ECE={r['ece']:.4f}  meanconf={r['mean_confidence']:.4f}", flush=True)

    # --- 2. calibration sweep (RQ2) --------------------------------------
    print("[2/3] calibration sweep", flush=True)
    rows = []
    agents = ["Random", "Lawnmower", "GreedyEntropy", "REINFORCE"]
    for T in TEMPERATURES:
        for a in agents:
            for s in SEEDS:
                rows.append(run_agent(a, s, T))
        done = [r for r in rows if r["temperature"] == T]
        for a in agents:
            v = [r["det_per_joule"] for r in done if r["agent"] == a]
            print(f"    T={T:.2f} {a:<14} dpj_iqm={iqm(v):.4f}", flush=True)
    with open(os.path.join(RESULTS, "pilot_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 3. cluster-strength sweep (RQ1) ---------------------------------
    print("[3/3] cluster-strength sweep", flush=True)
    crows = []
    for sig in [1.0, 1.6, 2.5, 4.0, 7.0]:
        for a in ["Lawnmower", "GreedyEntropy", "REINFORCE"]:
            for s in SEEDS:
                crows.append(run_agent(a, s, 1.00, sigma=sig))
        done = [r for r in crows if r["sigma"] == sig]
        for a in ["Lawnmower", "GreedyEntropy", "REINFORCE"]:
            v = [r["det_per_joule"] for r in done if r["agent"] == a]
            print(f"    sigma={sig:.1f} {a:<14} dpj_iqm={iqm(v):.4f}", flush=True)
    with open(os.path.join(RESULTS, "cluster_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0].keys()))
        w.writeheader()
        w.writerows(crows)

    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
