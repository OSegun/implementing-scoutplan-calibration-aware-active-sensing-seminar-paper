"""
scoutplan.env
=============
The ScoutPlan environment: an energy-constrained aerial scouting POMDP in
which the agent's information signal is a *possibly miscalibrated* classifier.

Formulation
-----------
The true disease map is never observed, so the problem is a POMDP (Lauri et
al., 2022; Kurniawati, 2021).  It is made tractable in the standard way, by
augmenting the state with a Bayesian belief over each cell and acting on that
belief.

State      s = (position, remaining energy, belief map, visit-count map)
Action     a in {8-connected moves}
Observation  o = reported probability of disease from the classifier
Reward     r = alpha * (belief entropy reduction)
               + lam  * (newly confirmed true detection)
               - mu   * (energy consumed by the action)

The critical coupling is in `_bayes_update`.  The agent treats the reported
probability as if it were calibrated.  When it is not, the posterior is driven
to the wrong place -- an overconfident classifier collapses the posterior after
a single look and removes the agent's incentive to revisit, while an
underconfident one leaves the posterior diffuse and wastes energy on settled
cells.  Miscalibration is therefore not additive observation noise: it is a
systematic bias in the belief dynamics, which is the mechanism this study
sets out to quantify.

Energy model
------------
Translation cost is distance-proportional with a diagonal penalty of sqrt(2),
plus a fixed per-step hover/acquisition term.  Coefficients follow the
payload-and-manoeuvre decomposition used in agricultural UAV coverage planning
(Fu et al., 2025) and the varying-power CPP formulation of Theile et al.
(2020); absolute values are normalised so that a full boustrophedon sweep of
the default field consumes approximately the default budget, making the
budget-relative comparisons in Chapter 4 meaningful.
"""

from __future__ import annotations
import numpy as np

from perception import CalibratedClassifier

MOVES = [(-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1), (1, 0), (1, 1)]

E_HOVER = 1.0          # fixed acquisition + hover cost per step
E_TRANSLATE = 0.6      # per unit distance travelled


def _entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


class ScoutEnv:
    """Energy-constrained aerial scouting environment."""

    def __init__(self, field, classifier: CalibratedClassifier,
                 budget: float = 150.0, patch: int = 5, prior: float = 0.12,
                 detect_threshold: float = 0.85,
                 alpha: float = 1.0, lam: float = 6.0, mu: float = 0.15,
                 rng: np.random.Generator | None = None):
        self.field = field
        self.clf = classifier
        self.budget = budget
        self.patch = patch
        self.prior = prior
        self.tau = detect_threshold
        self.alpha, self.lam, self.mu = alpha, lam, mu
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n = field.size
        self.reset()

    # ------------------------------------------------------------------ API

    def reset(self):
        n = self.n
        self.belief = np.full((n, n), self.prior, dtype=float)
        self.visits = np.zeros((n, n), dtype=np.int32)
        self.pos = (n // 2, n // 2)
        self.energy = float(self.budget)
        self.detected = set()
        self.false_alarms = set()
        self.first_detection_step = None
        self.energy_spent = 0.0
        self.steps = 0
        self.obs_log = []          # (confidence, was_correct) for ECE audit
        self._observe_current()
        return self._obs()

    def step(self, action: int):
        dr, dc = MOVES[action]
        r, c = self.pos
        nr, nc = int(np.clip(r + dr, 0, self.n - 1)), int(np.clip(c + dc, 0, self.n - 1))

        dist = float(np.hypot(nr - r, nc - c))
        cost = E_HOVER + E_TRANSLATE * dist

        h_before = _entropy(self.belief).sum()
        n_before = len(self.detected)

        self.pos = (nr, nc)
        self.energy -= cost
        self.energy_spent += cost
        self.steps += 1
        self._observe_current()

        h_after = _entropy(self.belief).sum()
        new_hits = len(self.detected) - n_before

        reward = (self.alpha * (h_before - h_after)
                  + self.lam * new_hits
                  - self.mu * cost)

        done = self.energy <= E_HOVER + E_TRANSLATE * np.sqrt(2)
        return self._obs(), float(reward), bool(done), self.info()

    def info(self):
        total = max(1, self.field.n_diseased)
        tp, fp = len(self.detected), len(self.false_alarms)
        return {
            "detections": tp,
            "false_alarms": fp,
            "precision": tp / max(1, tp + fp),
            "recall": tp / total,
            "energy_spent": self.energy_spent,
            "steps": self.steps,
            "coverage": float((self.visits > 0).mean()),
            "detections_per_joule": tp / max(1e-9, self.energy_spent),
            "time_to_first_detection": (self.first_detection_step
                                        if self.first_detection_step is not None
                                        else self.steps),
        }

    # ------------------------------------------------------------- internals

    def _observe_current(self):
        r, c = self.pos
        y = int(self.field.labels[r, c])
        p, yhat = self.clf.observe(y)
        self.visits[r, c] += 1
        self._bayes_update(r, c, p)
        self.obs_log.append((max(p, 1 - p), float(yhat == y)))

        # A cell is "called" once the posterior crosses the confirmation
        # threshold.  Calls on healthy cells are false alarms and are recorded
        # separately: an overconfident classifier produces confidently wrong
        # beliefs, and counting only true positives would hide that cost.
        if self.belief[r, c] >= self.tau:
            if y == 1:
                if (r, c) not in self.detected:
                    self.detected.add((r, c))
                    if self.first_detection_step is None:
                        self.first_detection_step = self.steps
            else:
                self.false_alarms.add((r, c))

    def _bayes_update(self, r: int, c: int, p: float):
        """
        Bayesian belief update treating the reported probability as calibrated.

        The classifier reports P(y=1 | x).  Converting to a likelihood ratio
        under the classifier's own implied prior (the field prior) and
        multiplying into the current belief gives the posterior.  If the
        reported probability is miscalibrated the likelihood ratio is wrong by
        a systematic factor, which is exactly the failure mode under study.
        """
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        b = float(np.clip(self.belief[r, c], 1e-6, 1 - 1e-6))
        pi = float(np.clip(self.prior, 1e-6, 1 - 1e-6))
        lr = (p / (1 - p)) * ((1 - pi) / pi)      # likelihood ratio
        odds = (b / (1 - b)) * lr
        self.belief[r, c] = odds / (1.0 + odds)

    def _patch(self, arr: np.ndarray) -> np.ndarray:
        k = self.patch // 2
        r, c = self.pos
        padded = np.pad(arr, k, mode="edge")
        return padded[r:r + self.patch, c:c + self.patch]

    def _obs(self) -> np.ndarray:
        b = self._patch(self.belief).ravel()
        h = self._patch(_entropy(self.belief)).ravel()
        v = np.tanh(self._patch(self.visits.astype(float))).ravel()
        r, c = self.pos
        extra = np.array([
            self.energy / self.budget,
            r / (self.n - 1),
            c / (self.n - 1),
        ])
        return np.concatenate([b, h, v, extra]).astype(np.float64)

    @property
    def obs_dim(self) -> int:
        return 3 * self.patch * self.patch + 3

    @property
    def n_actions(self) -> int:
        return len(MOVES)
