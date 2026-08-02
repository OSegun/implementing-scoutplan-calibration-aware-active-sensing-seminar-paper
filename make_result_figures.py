"""Result figures (5-8) built from the measured CSVs in ./results."""
from __future__ import annotations
import csv, os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import iqm, bootstrap_ci

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(HERE, "results")

GREEN = "#2C5F2D"; LEAF = "#5A8F4A"; WARM = "#B0764A"; GREY = "#6B7A6B"
BLUE = "#3C6E8F"; RED = "#A33B3B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

def load(p):
    with open(os.path.join(RES, p)) as f:
        return list(csv.DictReader(f))

cal = load("pilot_results.csv")
clu = load("cluster_sweep.csv")
ctab = load("calibration_table.csv")

Ts = sorted({float(r["temperature"]) for r in cal})
ECE = {float(r["temperature"]): float(r["ece"]) for r in ctab}
ACC = {float(r["temperature"]): float(r["accuracy"]) for r in ctab}
CONF = {float(r["temperature"]): float(r["mean_confidence"]) for r in ctab}
AGENTS = ["Random", "Lawnmower", "GreedyEntropy", "REINFORCE"]
COL = {"Random": GREY, "Lawnmower": WARM, "GreedyEntropy": BLUE, "REINFORCE": GREEN}


def sel(rows, agent, key, **eq):
    out = []
    for r in rows:
        if r["agent"] != agent:
            continue
        if all(abs(float(r[k]) - v) < 1e-9 for k, v in eq.items()):
            out.append(float(r[key]))
    return out


# ---------------------------------------------------- Figure 5
def figure5():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    ax = axes[0]
    ax.plot(Ts, [ACC[t] for t in Ts], "o-", color=GREEN, lw=2, label="Accuracy")
    ax.plot(Ts, [CONF[t] for t in Ts], "s--", color=WARM, lw=2, label="Mean confidence")
    ax.set_xlabel("Temperature $T$"); ax.set_ylabel("Proportion")
    ax.set_ylim(0.55, 1.0); ax.grid(alpha=.25)
    ax.set_title("Accuracy is invariant to $T$; confidence is not", fontsize=10, color=GREEN)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(Ts, [ECE[t] for t in Ts], "o-", color=RED, lw=2)
    ax.axvline(1.0, color=GREY, ls=":", lw=1)
    ax.annotate("calibrated\n$T=1$", xy=(1.0, ECE[1.0]), xytext=(1.35, 0.055),
                fontsize=8, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.annotate("overconfident", xy=(0.4, 0.115), fontsize=8, color=GREY)
    ax.annotate("underconfident", xy=(2.6, 0.13), fontsize=8, color=GREY)
    ax.set_xlabel("Temperature $T$"); ax.set_ylabel("Expected Calibration Error")
    ax.grid(alpha=.25)
    ax.set_title("ECE is minimised at $T=1$ and rises both ways", fontsize=10, color=GREEN)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure5_Calibration_Instrument.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("fig5")


# ---------------------------------------------------- Figure 6
def figure6():
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for a in AGENTS:
        m, lo, hi = [], [], []
        for t in Ts:
            v = sel(cal, a, "det_per_joule", temperature=t)
            m.append(iqm(v)); l, h = bootstrap_ci(v, seed=3); lo.append(l); hi.append(h)
        ax.plot(Ts, m, "o-", color=COL[a], lw=2, label=a, ms=5)
        ax.fill_between(Ts, lo, hi, color=COL[a], alpha=0.13, linewidth=0)
    ax.axvline(1.0, color=GREY, ls=":", lw=1)
    ax.text(1.03, 0.088, "perfect calibration", fontsize=8, color=GREY, rotation=90, va="top")
    ax.set_xlabel("Temperature $T$  (accuracy held constant at 0.816)")
    ax.set_ylabel("Detections per joule")
    ax.set_title("Planner performance under varying classifier calibration\n"
                 "IQM over 5 seeds, shaded band = 95% stratified bootstrap CI",
                 fontsize=10, color=GREEN)
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure6_Calibration_Sweep.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("fig6")


# ---------------------------------------------------- Figure 7
def figure7():
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for a in ["Lawnmower", "REINFORCE"]:
        rec = [np.mean(sel(cal, a, "recall", temperature=t)) for t in Ts]
        pre = [np.mean(sel(cal, a, "precision", temperature=t)) for t in Ts]
        ax.plot(Ts, rec, "o-", color=COL[a], lw=2, label=f"{a} — recall", ms=5)
        ax.plot(Ts, pre, "s--", color=COL[a], lw=1.6, alpha=0.75,
                label=f"{a} — precision", ms=4)
    ax.axvline(1.0, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Proportion")
    ax.set_title("The recall–precision trade-off created by miscalibration\n"
                 "Overconfidence buys recall and costs precision; "
                 "underconfidence loses both", fontsize=10, color=GREEN)
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure7_Recall_Precision.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("fig7")


# ---------------------------------------------------- Figure 8
def figure8():
    sigs = sorted({float(r["sigma"]) for r in clu})
    mor = [np.mean([float(r["morisita"]) for r in clu if abs(float(r["sigma"]) - s) < 1e-9])
           for s in sigs]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for a in ["Lawnmower", "GreedyEntropy", "REINFORCE"]:
        m = [iqm(sel(clu, a, "det_per_joule", sigma=s)) for s in sigs]
        ax.plot(sigs, m, "o-", color=COL[a], lw=2, label=a, ms=5)
    ax.set_xlabel("Cluster dispersal scale $\\sigma$   "
                  "(small = tight foci, large = near-random)")
    ax.set_ylabel("Detections per joule")
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(sigs, mor, "^:", color=RED, lw=1.4, ms=5, label="Morisita index")
    ax2.axhline(1.0, color=RED, ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylabel("Morisita index of dispersion", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.set_title("Effect of spatial aggregation on scouting efficiency\n"
                 "Morisita > 1 indicates aggregation; all planners peak at "
                 "intermediate clustering", fontsize=10, color=GREEN)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure8_Cluster_Sweep.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("fig8")


# ---------------------------------------------------- summary numbers
def summary():
    out = {}
    for a in AGENTS:
        out[a] = {}
        for t in Ts:
            v = sel(cal, a, "det_per_joule", temperature=t)
            lo, hi = bootstrap_ci(v, seed=3)
            out[a][t] = dict(
                dpj_iqm=iqm(v), ci=[lo, hi],
                recall=float(np.mean(sel(cal, a, "recall", temperature=t))),
                precision=float(np.mean(sel(cal, a, "precision", temperature=t))),
                coverage=float(np.mean(sel(cal, a, "coverage", temperature=t))),
                ttfd=float(np.mean(sel(cal, a, "time_to_first_detection", temperature=t))),
                fa=float(np.mean(sel(cal, a, "false_alarms", temperature=t))),
            )
    out["_calibration"] = {str(t): dict(ece=ECE[t], acc=ACC[t], conf=CONF[t]) for t in Ts}
    sigs = sorted({float(r["sigma"]) for r in clu})
    out["_cluster"] = {
        a: {str(s): dict(
            dpj=iqm(sel(clu, a, "det_per_joule", sigma=s)),
            recall=float(np.mean(sel(clu, a, "recall", sigma=s))),
            morisita=float(np.mean([float(r["morisita"]) for r in clu
                                    if abs(float(r["sigma"]) - s) < 1e-9])))
            for s in sigs}
        for a in ["Lawnmower", "GreedyEntropy", "REINFORCE"]}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("_")}, indent=1)[:1200])


if __name__ == "__main__":
    figure5(); figure6(); figure7(); figure8(); summary()
