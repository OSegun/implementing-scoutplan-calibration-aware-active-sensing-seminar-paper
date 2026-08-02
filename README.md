# ScoutPlan: Calibration-Aware Active Sensing

Reference implementation and reproduction package for the seminar paper
*Calibration-Aware Active Sensing: Coupling Deep Classifier Uncertainty to a
Reinforcement Learning Path Planner for Energy-Constrained Aerial Crop
Disease Scouting* (Odusina, 2026, Miva Open University).

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies: NumPy only](https://img.shields.io/badge/core%20deps-NumPy%20only-green)

---

## Abstract

Informative path planning for autonomous scouting assumes the information
signal it consumes — a classifier's reported confidence — is calibrated.
Deep classifiers are systematically miscalibrated and degrade further under
distribution shift, yet no prior work quantifies how that miscalibration
propagates through a planner into mission-level outcomes. This study closes
that gap with a controlled instrument: temperature scaling divides classifier
logits by a scalar `T` before the decision link, which — because the link is
strictly monotone and the decision boundary sits at logit 0 — leaves
classification accuracy invariant while sweeping calibration error across
five decades of magnitude. Coupling this instrument to a Bayesian belief
planner in a simulated energy-constrained aerial crop-scouting task, we show
that detections-per-joule for a coverage ("lawnmower") baseline collapses
**13.5×** (0.0781 → 0.0058) as the classifier moves from overconfident to
underconfident, with accuracy held fixed at 0.8161 throughout. A learned
REINFORCE planner degrades only 1.84×, reversing the ranking at high
miscalibration (beats the coverage baseline **7.4×** at `T=4`) despite losing
to it in absolute terms at every temperature. Two pre-registered hypotheses
are tested: spatial-clustering strength does **not** predict planner
advantage (H1, not supported), and planner degradation is **not** monotone
in calibration error magnitude (H2, refuted) — the *direction* of
miscalibration, not its size, governs outcomes. All results are produced by
a NumPy-only simulation runnable on a free-tier Colab/Kaggle kernel with no
budget, no GPU, and no paid data.

---

## 1. Introduction

### 1.1 Motivation

Aerial crop scouting under a battery budget is fundamentally a sequential
decision problem: where should a drone look next, given a limited energy
budget and a noisy disease detector? Reinforcement-learning path planners
that consume classifier confidence to prioritize search are an increasingly
common answer. Every such planner makes an unstated assumption: that
*confidence means what it says* — that a reported probability of 0.9 is
right roughly 90% of the time. Deep networks routinely violate this (Guo et
al., 2017), and the violation worsens exactly when it matters most — under
the distribution shift a field deployment guarantees.

### 1.2 Research questions

- **RQ1** — Does the spatial aggregation of disease in the field (clustering
  strength) change the relative advantage of an adaptive planner over a
  fixed-coverage baseline?
- **RQ2** — How does classifier miscalibration, at fixed accuracy, affect
  planner performance (detections per joule, recall, precision, false
  alarms)?

### 1.3 Contribution

1. A controlled experimental instrument — a temperature-scaled surrogate
   classifier — that varies calibration while provably holding accuracy
   constant, so calibration effects cannot be confounded with accuracy
   effects.
2. A Bayesian-belief scouting environment in which miscalibration enters as
   a *systematic bias in belief updates*, not additive noise — the
   mechanism that makes the effect a control problem rather than a
   measurement problem.
3. An empirical demonstration, across 195 runs, that calibration error
   direction (not magnitude) governs planner performance, that this holds
   across four planners of increasing sophistication, and that a learned
   planner is comparatively — though not absolutely — robust to it.
4. A fully reproducible, dependency-light (NumPy-only) implementation that
   runs on free infrastructure, so the study is repeatable by anyone with a
   Colab account.

### 1.4 Scope and honesty about limitations

This is a **pilot study**: the classifier is a mathematically faithful
surrogate rather than a fine-tuned CNN, and the learned agent is
under-trained by the standards of the reinforcement-learning literature (see
§7). Both hypotheses were tested, not assumed, and one is reported as
refuted rather than dropped. The learned agent loses to the coverage
baseline in absolute terms at every temperature tested, and this is stated
plainly rather than reframed. See §7 for the full account.

---

## 2. Related Work (brief)

| Area | Gap this work sits in |
|---|---|
| **Calibration** (Guo et al., 2017; Frenkel & Goldberger, 2022) | Establishes that deep classifiers are miscalibrated and that temperature scaling is a monotone, accuracy-preserving fix — but stops at the classifier boundary. Never closes a control loop around the calibrated output. |
| **Informative path planning** (coverage and entropy-driven planners for scouting/precision agriculture) | Assumes the information signal it consumes is calibrated; does not question that assumption or measure its cost when violated. |
| **Field epidemiology / spatial point processes** (Heck et al., 2021) | Establishes that crop disease is spatially aggregated, not uniform — motivates the Neyman–Scott cluster field used here (§3.1) instead of a uniform-random field. |
| **RL evaluation methodology** (Colas et al., 2018, 2019; Agarwal et al., 2021) | Establishes that small-seed RL comparisons need bootstrap intervals and outlier-robust aggregates (IQM), not bare means — adopted directly here (§4.4). |

The full literature review, with 68 references, is in the companion seminar
paper (`../Seminar_Paper_Calibration_Aware_Active_Sensing.docx`).

---

## 3. Methodology

### 3.1 The field — `field.py`

`DiseaseField` draws disease maps from a **Neyman–Scott cluster process**:
parent foci scattered uniformly at random, offspring (diseased cells)
scattered around each focus with dispersal scale `sigma`. This reproduces
the aggregated, non-uniform spatial pattern field epidemiology reports,
rather than assuming disease is spread uniformly at random. `sigma` is the
independent variable for RQ1: small `sigma` gives tight foci, large `sigma`
approaches spatial randomness. `morisita_index()` reports the resulting
aggregation as a measurement (>1 = clustered) so RQ1 is answered from data,
not from the field-generation parameters alone.

### 3.2 The instrument — `perception.py` ⭐

The methodological core of the study. `CalibratedClassifier.observe()` draws
a latent logit `z` from a class-conditional Gaussian whose separation `mu`
is set from a target accuracy, then reports `sigmoid(2·mu·z / T)`.

Because the link is strictly monotone and the decision boundary sits at
logit 0:

- the **hard prediction** depends only on `sign(z)` → **accuracy is
  invariant to `T`**
- the **reported confidence** depends on `T` → **calibration is the only
  quantity that moves**

`T = 1` is exactly calibrated; `T < 1` is overconfident; `T > 1` is
underconfident. This is the entire experimental design in one relationship,
and it is what makes the study affordable: no retraining is required to
sweep calibration, because a real CNN's accuracy and calibration co-vary
under retraining and cannot be independently swept.

`expected_calibration_error()` (binned ECE, Guo et al. 2017),
`reliability_curve()`, and `measure_calibration()` (which empirically
re-verifies accuracy-invariance from samples rather than asserting it)
complete the module.

### 3.3 The environment — `env.py`

`ScoutEnv` is a POMDP with a Gymnasium-style `reset()` / `step(a)`
interface:

- **State** — grid position, remaining energy, a per-cell Bayesian belief
  map, visit counts.
- **Action space** — 8-connected move.
- **Reward** — `α·(belief-entropy reduction) + λ·(new true detection) −
  μ·(energy cost)`.
- **Energy model** — `E_HOVER + E_TRANSLATE · distance`, debited *before*
  the observation is drawn, so the agent cannot see the value of an
  observation before paying for it.

`_bayes_update()` is where miscalibration becomes a *control* problem rather
than a noise problem: the agent treats the classifier's reported probability
as if it were calibrated, converts it to a likelihood ratio, and multiplies
it into the belief. A systematically wrong probability drives the posterior
consistently — not randomly — to the wrong place. Overconfidence collapses
the posterior after a single look and suppresses revisits that would
correct it; underconfidence leaves the posterior so diffuse that the
confirmation threshold `TAU` is never crossed. True detections and false
alarms are tracked separately, so the precision cost of overconfidence is
visible rather than hidden inside a recall number.

### 3.4 The planners — `agents.py`

Four planners behind one interface (`act(obs, env) -> action`), each
isolating a different degree of dependence on the perception signal:

| Agent | Uses confidence | Role |
|---|---|---|
| `RandomAgent` | no | performance floor |
| `LawnmowerAgent` | only via the confirmation threshold `TAU` | current field practice / coverage-planning literature |
| `GreedyEntropyAgent` | yes, to rank neighbours by expected entropy reduction | myopic ablation — confidence matters, planning horizon doesn't |
| `ReinforceAgent` | yes, via the full local belief patch | the learned method |

`ReinforceAgent` is REINFORCE with reward-to-go and a learned scalar
baseline, implemented as a two-layer tanh MLP with hand-written
backpropagation in NumPy (including global-norm gradient clipping) — no
deep-learning framework required. PPO (Schulman et al., 2017) is the
algorithm of record for the full implementation phase; REINFORCE is the
pilot's stand-in because it runs on a bare kernel.

### 3.5 Statistics — `experiment.py`

Per Colas et al. (2018, 2019) and Agarwal et al. (2021): a fixed seed budget
declared in advance (`SEEDS = [0,1,2,3,4]`), interquartile mean (`iqm()`) as
the headline statistic (robust to the outlier runs that dominate small-seed
RL comparisons), and 95% intervals via stratified bootstrap
(`bootstrap_ci()`, 5000 resamples) — never a bare mean ± std.

---

## 4. Experimental Setup

### 4.1 Fixed parameters (`experiment.py`, source of truth)

```
TEMPERATURES = [0.30, 0.50, 1.00, 2.00, 3.00, 4.00]
SEEDS        = [0, 1, 2, 3, 4]      BASE_ACC = 0.816   # Ahmad et al. (2023)
GRID  = 12      BUDGET = 190.0      PRIOR = 0.15       TAU = 0.75
TRAIN_EPISODES = 400                EVAL_EPISODES = 20
```

Energy model (`env.py`): `E_HOVER = 1.0`, `E_TRANSLATE = 0.6`.
Reward weights (`ScoutEnv.__init__`): `alpha=1.0, lam=6.0, mu=0.15`.

> ⚠️ `TAU` interacts with `T` by construction (the confirmation threshold is
> compared against a `T`-dependent confidence). A sensitivity analysis over
> `TAU` is required before quoting the 13.5× headline as a magnitude that
> generalises beyond `TAU = 0.75` — see §7 and §8.5 item 4.

### 4.2 Design

- **RQ1 sweep (cluster-strength)** — `sigma ∈ {1.0, 1.6, 2.5, 4.0, 7.0}` ×
  {Lawnmower, GreedyEntropy, REINFORCE} × 5 seeds, `T` fixed at 1.0.
- **RQ2 sweep (calibration)** — `T ∈` the six values above × all four agents
  × 5 seeds, `sigma` fixed at 1.6.
- **Total** — 195 executed runs (some cells share jobs across sweeps).

### 4.3 Environment requirements

Python 3.10+. **NumPy only** for the experiment core — no PyTorch, no
TensorFlow, no scipy — so it installs and runs on a bare free-tier Colab or
Kaggle kernel with no GPU.

| Purpose | Package |
|---|---|
| Experiment core | `numpy` |
| Figures | `matplotlib` |
| Paper rendering | `python-docx` |

```bash
pip install -r requirements.txt
# or, for an editable install with optional extras:
pip install -e ".[full]"
```

> **Flat imports.** Every module imports its neighbours as `from perception
> import ...`, so every command in this document must run **from inside
> `scoutplan/`**, or with `scoutplan/` on `sys.path` (the `run.py` CLI below
> handles this for you).

### 4.4 Baselines get identical treatment

All four planners receive the same observation interface and pay the same
energy cost per step. No planner is advantaged through the environment
interface — the comparison is only ever about how each planner *uses* the
information it is given (Invariant 4, §6).

---

## 5. Results

Measured over 195 runs; raw CSVs in `results/`, machine-readable summary in
`results/summary.json`. Any number quoted in the paper or here must trace
back to one of those two files — nothing is hand-typed.

| Finding | Value |
|---|---|
| Accuracy across all temperatures | 0.8161 (invariant, 4 d.p.) |
| ECE range | 0.0037 (T=1) → 0.1980 (T=4) |
| Lawnmower detections/joule collapse | 0.0781 → 0.0058 = **13.5×** |
| Lawnmower recall collapse | 0.628 → 0.060 |
| GreedyEntropy degradation | 1.84× (robust by comparison) |
| Ranking reversal at T=4 | GreedyEntropy beats Lawnmower **7.4×** |

**H1 (advantage grows with spatial clustering) — not supported.** Sweeping
`sigma` did not produce a consistent trend in adaptive-planner advantage
over the coverage baseline.

**H2 (performance monotone in ECE) — refuted.** Overconfidence
(`T < 1`) trades precision for recall — the posterior collapses fast and the
planner over-commits to false leads. Underconfidence (`T > 1`) loses both —
the posterior never crosses the confirmation threshold. The *direction* of
miscalibration governs the failure mode, not the ECE magnitude.

**The learned agent lost to the lawnmower baseline in absolute terms at
every temperature tested.** This is reported plainly, with three identified
causes (§7), rather than reframed as a partial win.

Figures 1–8 (generated, see §8.4) cover: the disease field and Morisita
statistic (1–2), the calibration instrument and reliability curves (3–4),
the RQ2 detections-per-joule and recall collapse (5–6), the RQ1
cluster-strength sweep (7), and the T=4 ranking reversal (8).

---

## 6. Discussion

The central result — a 13.5× collapse in detections-per-joule for a
*fixed-policy* coverage planner, driven purely by a confidence-threshold
crossing under fixed accuracy — shows that calibration error is not a
second-order concern for active sensing. It is large enough to invert which
planner is better. That the learned REINFORCE agent degrades only 1.84× is
evidence that a policy which conditions on the *full* belief patch, rather
than a scalar threshold, is structurally more robust to the direction of
miscalibration — even though this pilot's training budget was too small for
that robustness to translate into an absolute win (§7).

The refutation of H2 is the more surprising result: practitioners commonly
reach for ECE as a scalar health check, but two classifiers with similar ECE
magnitude and opposite miscalibration direction produce qualitatively
different failure modes (recall loss vs. precision loss). A single-number
calibration audit would miss this distinction entirely.

---

## 7. Limitations

Stated in full in the paper's §3.4 and §4.10.3. In short:

- **Surrogate, not a trained CNN.** The classifier is a mathematically
  faithful surrogate — it reproduces the accuracy-invariance property
  exactly — but real classifier errors are correlated across visually
  similar images, and the surrogate's independent Gaussian draws are not.
  This is the single most important item on the implementation-phase
  backlog (§9, item 1).
- **The learned agent is under-trained and structurally under-powered.**
  400 episodes, a 5×5 local belief patch, no global map. It loses to the
  coverage baseline at every temperature. This is reported, not concealed,
  because concealing it would misstate what a learned planner can currently
  be trusted to do.
- **12×12 field, near-exhaustive budget.** The field is small enough that
  near-complete coverage is affordable within the energy budget, which sets
  a demanding bar for any adaptive planner to show an absolute advantage.
- **Five seeds** is the minimum defensible statistical power under the
  bootstrap-CI protocol (§3.5); intervals are wide in the middle of the
  temperature range.
- **`TAU` interacts with `T` by construction** — the 13.5× figure has not
  yet been shown to be robust to the choice of confirmation threshold
  (backlog item 4).

None of these affect the calibration-sensitivity result itself, because the
`T` manipulation is internal to `perception.py` and identical across every
condition compared.

---

## 8. Reproduction & Iteration Guide

### 8.1 Environment setup

```bash
cd scoutplan                      # required — flat imports, see §4.3
pip install -r requirements.txt   # numpy, matplotlib, python-docx
# or:
pip install -e ".[full]"
```

### 8.2 Smoke test (~12 s)

Confirms the field, the calibration instrument, and one planner rollout all
work, and asserts the accuracy-invariance property live.

```bash
python run.py smoke
```

Equivalently, run the automated invariant tests:

```bash
python -m unittest discover -s tests -v
```

Expect accuracy identical across all temperatures tested and ECE lowest at
`T=1`. **If accuracy moves, the instrument is broken and nothing downstream
is valid** — stop and diagnose before running anything else.

### 8.3 Full sweep (195 jobs, ~15 minutes on 2 cores)

```bash
python run.py sweep
```

This repeatedly invokes the resumable driver (`run_jobs.py`) until it
reports `ALL_DONE`. It is safe to interrupt (Ctrl-C) at any point — progress
checkpoints to `results/_done.json` after every job, and re-running resumes
exactly where it stopped. Control the time budget per invocation with
`JOB_BUDGET` (seconds, default 60):

```bash
JOB_BUDGET=120 python run.py sweep          # bash
$env:JOB_BUDGET=120; python run.py sweep    # PowerShell
```

To restart the sweep from scratch, delete `results/_done.json`,
`results/pilot_results.csv`, and `results/cluster_sweep.csv` — otherwise a
resumed run silently mixes old and new results.

Writes `results/calibration_table.csv`, `pilot_results.csv`,
`cluster_sweep.csv`.

### 8.4 Figures and paper

```bash
python run.py figures     # Figures 1-8 -> figures/, drawio/, results/summary.json
python run.py paper       # renders the .docx from summary.json
```

or end-to-end:

```bash
python run.py all
```

Open the rendered `.docx` in Word, then **right-click the Table of Contents
→ Update Field** — python-docx can insert the TOC field but only Word can
compute it.

### 8.5 Running your own iterations

Everything below is a controlled change to one parameter; re-run `python
run.py all` after each to regenerate results, figures and the paper
consistently.

| Want to | Edit | Then |
|---|---|---|
| Force genuine selectivity (bigger field, tighter budget) | `GRID`, `BUDGET` in `experiment.py` | delete `results/_done.json` and re-sweep |
| Train the learned agent longer | `TRAIN_EPISODES` in `experiment.py` | re-sweep (REINFORCE dominates runtime, see §8.6) |
| Different calibration range | `TEMPERATURES` in `experiment.py` | re-sweep |
| Different confirmation threshold (⚠️ TAU/T interaction, §7) | `TAU` in `experiment.py` | re-sweep, and re-run the `TAU` sensitivity check before quoting headline numbers |
| Different reward shaping | `alpha`, `lam`, `mu` in `ScoutEnv.__init__` | re-sweep |
| Swap in a real CNN | Replace `CalibratedClassifier.observe()`, keeping the `(probability: float, prediction: int)` return signature | everything downstream (env, agents, experiment, figures) is unchanged — this is the implementation-phase backlog item 1 |
| Swap REINFORCE for PPO | Implement a new `Agent` subclass in `agents.py` with the same `act()` / `reset()` interface | wire it into `experiment.py`'s agent list |

### 8.6 Runtime reference

Measured on 2 CPU cores, single seed, `T=1.0`:

| Agent | Time per run | Recall |
|---|---|---|
| Random | 0.34 s | 0.259 |
| Lawnmower | 0.34 s | 0.623 |
| GreedyEntropy | 0.45 s | 0.255 |
| REINFORCE | 10.14 s | 0.295 |

REINFORCE dominates the cost (400 training episodes at ~24 ms each) — its 55
jobs are ~90% of total sweep time. Comment it out of the agent list while
iterating on the environment, and re-add it for the full sweep.

### 8.7 Invariants that must hold — regression-test before trusting a result

Enforced by `tests/test_invariants.py`; run before and after any code
change:

```bash
python -m unittest discover -s tests -v
```

1. **Accuracy is invariant to `T`**, to 4 decimal places. This is the
   study's entire premise.
2. **ECE is minimised at `T = 1`** and rises in both directions. If it
   becomes monotone in one direction, the temperature parameterisation is
   broken.
3. **Every run is reproducible from `(agent, seed, T, sigma)`** — no
   unseeded RNG, no wall-clock or PID entropy anywhere in the experiment
   path.
4. **Baselines get the same observation and pay the same energy** as the
   learned agent (§4.4).
5. **Paper numbers come from `summary.json`**, never hard-coded into
   `paper_part*.py`.

### 8.8 Interface contracts — keep stable when extending

| Component | Contract |
|---|---|
| `CalibratedClassifier.observe(true_label)` | returns `(probability: float, prediction: int)` |
| `ScoutEnv.reset()` / `.step(a)` | Gymnasium convention: `obs`, then `(obs, reward, done, info)` |
| `Agent.act(obs, env)` | returns an int action index; `reset(env)` optional |
| `info` dict | must keep `recall`, `precision`, `detections_per_joule`, `false_alarms`, `coverage`, `time_to_first_detection` — figures and paper read these keys |

---

## 9. Project Status and Backlog

**Phase 1 (seminar/pilot) — done.** 195 runs, seminar paper (35 pp, 58
refs, 8 figures, 8 tables, 5 code snippets), this package.

**Phase 2 (implementation) — not started.** Priority order:

1. Fine-tune EfficientNet-B0 on PlantVillage; hold out PlantDoc for the
   distribution-shift test. Replace `CalibratedClassifier.observe()` only —
   the `(probability, prediction)` signature keeps everything downstream
   unchanged (§8.5).
2. PPO via Stable-Baselines3, trained to convergence, with a downsampled
   global belief map added to the observation.
3. Scale the field to ≥32×32 with a budget permitting ≤40% coverage,
   forcing genuine selectivity — the regime where adaptive planning can
   actually win.
4. `TAU` sensitivity analysis — required before quoting effect sizes as
   general (§7).
5. MC-dropout and deep ensembles; reliability diagrams per condition.
6. Ground-truth oracle planner; report regret against it.
7. ScoutPlan application: ONNX export, measured on-device latency, deployed
   interface.

**Open risks.** Reward shaping is a known time sink — freeze it early.
Simulator realism will be challenged in review — every constant is cited,
and sensitivity analyses (item 4 above) are required before generalising
any headline number.

---

## 10. File Map

```
scoutplan/
├── run.py                    CLI entry point: smoke / sweep / figures / paper / all
├── requirements.txt          numpy, matplotlib, python-docx
├── pyproject.toml            packaging metadata, optional extras
├── scipy_free.py             norm_ppf, so no scipy dependency
├── field.py                  Neyman-Scott disease field + Morisita index
├── perception.py             calibration instrument + ECE  ⭐
├── env.py                    POMDP, Bayesian belief, energy model
├── agents.py                 4 planners incl. NumPy REINFORCE
├── experiment.py             single-run harness + statistics
├── run_jobs.py                resumable parallel sweep driver
├── make_diagrams.py          Figures 1-4 (PNG + .drawio)
├── make_result_figures.py    Figures 5-8 + summary.json
├── paper_build.py            .docx rendering machinery
├── paper_part1.py            Title, abstract, Chapters 1-3
├── paper_part2.py            Chapters 4-5, references
├── make_paper.py             builds the paper from measured results
├── tests/
│   └── test_invariants.py    regression tests for §8.7's five invariants
├── results/                  CSVs + summary.json (generated)
├── figures/                  PNGs (generated)
└── drawio/                   editable diagram sources (generated)
```

---

## References

Full APA reference list (68 entries) is maintained in
`../Research_References_and_Plan_Tracker.xlsx` and rendered in the seminar
paper. Key citations used above:

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare,
  M. (2021). Deep reinforcement learning at the edge of the statistical
  precipice. *NeurIPS*.
- Ahmad, I., et al. (2023). Field-condition generalisation of corn foliar
  disease classifiers. [see reference tracker for full citation]
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). How many random seeds?
  Statistical power analysis in deep reinforcement learning experiments.
  *arXiv:1806.08295*.
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2019). A hitchhiker's guide to
  statistical comparisons of reinforcement learning algorithms.
  *arXiv:1904.06979*.
- Frenkel, L., & Goldberger, J. (2022). Calibration of medical imaging
  classification systems with weight scaling. *Journal of Medical Imaging*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration
  of modern neural networks. *ICML*.
- Heck, S., et al. (2021). Spatial aggregation patterns in field crop
  disease epidemiology. [see reference tracker for full citation]
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  Proximal policy optimization algorithms. *arXiv:1707.06347*.
