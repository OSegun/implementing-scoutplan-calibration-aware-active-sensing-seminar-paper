"""
scoutplan.field
===============
Neyman-Scott cluster process generator for spatially aggregated crop disease.

Rationale
---------
Field surveys of foliar disease consistently report *aggregated* rather than
random spatial patterns; Heck et al. (2021) found the beta-binomial (an
aggregation-reflecting distribution) best described Cercospora leaf spot
incidence across 31 table beet fields, with aggregation detected in >95% of
datasets by point-process, runs and autocorrelation methods.  A Neyman-Scott
cluster process is the standard point-process model for exactly this structure:
parent "foci" are scattered at random, and offspring (infected plants) are
scattered around each parent with dispersal scale sigma.

Reducing `sigma` tightens the clusters; increasing it approaches complete
spatial randomness.  This single parameter is therefore the independent
variable used in the RQ1 cluster-strength sweep.
"""

from __future__ import annotations
import numpy as np


class DiseaseField:
    """A square field of cells, each labelled healthy (0) or diseased (1)."""

    def __init__(self, size: int = 20, n_parents: int = 3,
                 offspring_mean: int = 14, sigma: float = 1.6,
                 rng: np.random.Generator | None = None):
        self.size = size
        self.n_parents = n_parents
        self.offspring_mean = offspring_mean
        self.sigma = sigma
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.labels = self._generate()

    def _generate(self) -> np.ndarray:
        """Draw one realisation of the Neyman-Scott cluster process."""
        n = self.size
        labels = np.zeros((n, n), dtype=np.int8)

        # Parent foci, uniform over the field.
        k = max(1, self.rng.poisson(self.n_parents))
        parents = self.rng.uniform(0, n, size=(k, 2))

        for py, px in parents:
            m = self.rng.poisson(self.offspring_mean)
            if m == 0:
                continue
            # Isotropic Gaussian dispersal around the focus.
            offs = self.rng.normal(0.0, self.sigma, size=(m, 2))
            pts = np.stack([py + offs[:, 0], px + offs[:, 1]], axis=1)
            pts = np.round(pts).astype(int)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < n) & (pts[:, 1] >= 0) & (pts[:, 1] < n)
            pts = pts[ok]
            labels[pts[:, 0], pts[:, 1]] = 1

        return labels

    # ---- descriptive statistics used for reporting -----------------------

    @property
    def prevalence(self) -> float:
        return float(self.labels.mean())

    @property
    def n_diseased(self) -> int:
        return int(self.labels.sum())

    def morisita_index(self, quadrat: int = 4) -> float:
        """
        Morisita's index of dispersion over `quadrat` x `quadrat` blocks.

        I = q * sum(n_i(n_i - 1)) / (N(N - 1))

        I == 1  -> random (Poisson)
        I >  1  -> aggregated / clustered
        I <  1  -> regular / over-dispersed

        Reported so that the degree of aggregation produced by a given sigma
        is stated quantitatively rather than asserted.
        """
        n = self.size
        step = max(1, n // quadrat)
        counts = []
        for i in range(0, n, step):
            for j in range(0, n, step):
                counts.append(int(self.labels[i:i + step, j:j + step].sum()))
        counts = np.asarray(counts, dtype=float)
        N = counts.sum()
        q = len(counts)
        if N < 2:
            return float("nan")
        return float(q * np.sum(counts * (counts - 1)) / (N * (N - 1)))
