"""
scoutplan.perception
====================
Calibration-controlled classifier surrogate and calibration metrics.

Why a surrogate rather than a fine-tuned CNN in the pilot
---------------------------------------------------------
The independent variable in RQ2 is *calibration error at fixed accuracy*.
Temperature scaling divides the logits by a scalar T before the link function.
Because the link is strictly monotone and the decision threshold sits at
logit 0, dividing by any T > 0 leaves the arg-max — and therefore the accuracy
— exactly unchanged, while moving the reported confidence.  Frenkel and
Goldberger (2022) state this property explicitly for temperature scaling.

The surrogate below reproduces that mechanism exactly: a latent logit is drawn
per observation from a class-conditional Gaussian whose separation `mu` fixes
the achievable accuracy, and the reported probability is sigmoid(z / T).
Sweeping T therefore sweeps calibration with accuracy held constant by
construction, which is precisely the experimental instrument the study needs.

In the full study this class is replaced by a fine-tuned EfficientNet-B0 whose
logits are scaled by the same T; the interface (`observe`) is unchanged.  The
surrogate's separation parameter is set from the field-condition generalisation
accuracy reported by Ahmad et al. (2023) for corn foliar disease.
"""

from __future__ import annotations
import numpy as np
from scipy_free import norm_ppf  # local minimal implementation


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def mu_for_accuracy(acc: float) -> float:
    """
    Separation of the two class-conditional logit Gaussians that yields a
    given balanced accuracy under a threshold at zero.

    With z | y=1 ~ N(+mu, 1) and z | y=0 ~ N(-mu, 1), predicting y=1 when
    z > 0 gives accuracy Phi(mu).  Hence mu = Phi^{-1}(acc).
    """
    return float(norm_ppf(acc))


class CalibratedClassifier:
    """Binary disease classifier with an explicit temperature knob."""

    def __init__(self, accuracy: float = 0.816, temperature: float = 1.0,
                 rng: np.random.Generator | None = None):
        self.accuracy = accuracy
        self.mu = mu_for_accuracy(accuracy)
        self.T = temperature
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def observe(self, true_label: int) -> tuple[float, int]:
        """
        Return (reported_probability_of_disease, hard_prediction).

        Under equal class priors with z | y=1 ~ N(+mu, 1) and z | y=0 ~
        N(-mu, 1), the exactly calibrated posterior is sigmoid(2 * mu * z).
        The reported probability is therefore sigmoid(2 * mu * z / T), which
        parameterises the family so that

            T = 1  -> perfectly calibrated
            T < 1  -> overconfident
            T > 1  -> underconfident

        The hard prediction depends only on sign(z) and is therefore
        independent of T: accuracy is invariant to the temperature, which is
        the property that makes T a clean experimental instrument.
        """
        z = self.rng.normal(self.mu if true_label == 1 else -self.mu, 1.0)
        p = sigmoid(2.0 * self.mu * z / self.T)
        return float(p), int(z > 0)


# --------------------------------------------------------------------------
# Calibration metrics
# --------------------------------------------------------------------------

def expected_calibration_error(probs: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    """
    Expected Calibration Error (Guo et al., 2017).

    ECE = sum_b (|B_b| / n) * | acc(B_b) - conf(B_b) |

    `probs` are confidences in the *predicted* class (i.e. max(p, 1-p)),
    `correct` is a 0/1 array of whether that prediction was right.
    """
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - probs[m].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Bin centres, empirical accuracy and mean confidence per bin."""
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, confs, weights = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if not m.any():
            continue
        centres.append(0.5 * (lo + hi))
        accs.append(correct[m].mean())
        confs.append(probs[m].mean())
        weights.append(m.sum())
    return (np.array(centres), np.array(accs),
            np.array(confs), np.array(weights))


def measure_calibration(accuracy: float, temperature: float,
                        n: int = 40000, seed: int = 0):
    """
    Empirically measure (accuracy, ECE) for a given temperature.

    Used to produce the calibration table that anchors the RQ2 sweep: it
    demonstrates measured accuracy invariance and measured ECE variation
    rather than assuming them.
    """
    rng = np.random.default_rng(seed)
    clf = CalibratedClassifier(accuracy=accuracy, temperature=temperature, rng=rng)
    y = rng.integers(0, 2, size=n)
    p = np.empty(n)
    yhat = np.empty(n, dtype=int)
    for i in range(n):
        p[i], yhat[i] = clf.observe(int(y[i]))
    conf = np.maximum(p, 1.0 - p)
    correct = (yhat == y).astype(float)
    return {
        "temperature": temperature,
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct),
        "mean_confidence": float(conf.mean()),
    }
