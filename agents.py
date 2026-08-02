"""
scoutplan.agents
================
Baseline planners and a dependency-free REINFORCE policy-gradient agent.

Baselines
---------
Lawnmower   : boustrophedon sweep -- the geometric coverage plan that current
              practice and the coverage-path-planning literature optimise
              (Theile et al., 2020; Fu et al., 2025).
GreedyEntropy: myopic one-step maximisation of belief-entropy reduction -- the
              non-learned ablation of the proposed method, isolating how much
              of any gain is attributable to *sequential* planning rather than
              to using uncertainty at all.
Random      : uniform random move -- the performance floor.

Learned agent
-------------
REINFORCE with a reward-to-go estimator and a learned scalar baseline, written
directly in NumPy.  PPO (Schulman et al., 2017) is the algorithm of record for
the full study; REINFORCE is used for the pilot because it needs no deep
learning framework and therefore executes inside the constrained environment
used to produce these results, while sharing the same policy-gradient family.
"""

from __future__ import annotations
import numpy as np

from env import MOVES


# --------------------------------------------------------------------- base

class Agent:
    name = "agent"

    def reset(self, env):
        pass

    def act(self, obs, env) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    name = "Random"

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, env):
        return int(self.rng.integers(0, len(MOVES)))


class LawnmowerAgent(Agent):
    """Boustrophedon sweep, executed as a fixed waypoint queue."""
    name = "Lawnmower"

    def __init__(self, rng=None):
        self.plan = []

    def reset(self, env):
        n = env.n
        pts = []
        for r in range(n):
            cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
            for c in cols:
                pts.append((r, c))
        # Rotate the queue so it starts near the agent's start cell.
        start = env.pos
        idx = min(range(len(pts)),
                  key=lambda i: abs(pts[i][0] - start[0]) + abs(pts[i][1] - start[1]))
        self.plan = pts[idx:] + pts[:idx]
        self.k = 0

    def act(self, obs, env):
        r, c = env.pos
        # Advance past waypoints already reached.
        while self.k < len(self.plan) and self.plan[self.k] == (r, c):
            self.k += 1
        if self.k >= len(self.plan):
            self.k = 0
        tr, tc = self.plan[self.k]
        dr = int(np.sign(tr - r))
        dc = int(np.sign(tc - c))
        if dr == 0 and dc == 0:
            return 4
        return MOVES.index((dr, dc))


class GreedyEntropyAgent(Agent):
    """Move to the neighbouring cell with the highest belief entropy."""
    name = "GreedyEntropy"

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, env):
        r, c = env.pos
        best, best_h = [], -1.0
        for i, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.n and 0 <= nc < env.n):
                continue
            b = np.clip(env.belief[nr, nc], 1e-9, 1 - 1e-9)
            h = -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
            h -= 0.05 * env.visits[nr, nc]          # mild revisit discouragement
            if h > best_h + 1e-12:
                best_h, best = h, [i]
            elif abs(h - best_h) <= 1e-12:
                best.append(i)
        if not best:
            return int(self.rng.integers(0, len(MOVES)))
        return int(self.rng.choice(best))


# ------------------------------------------------------------- REINFORCE

class ReinforceAgent(Agent):
    """Two-layer tanh MLP policy trained with REINFORCE + learned baseline."""
    name = "REINFORCE"

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64,
                 lr: float = 0.01, gamma: float = 0.99,
                 rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        s1 = np.sqrt(1.0 / obs_dim)
        s2 = np.sqrt(1.0 / hidden)
        self.W1 = self.rng.normal(0, s1, (obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.normal(0, s2, (hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.baseline = 0.0
        self.n_actions = n_actions
        self._train = True

    # -- forward ---------------------------------------------------------

    def _forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max()
        e = np.exp(logits)
        return h, e / e.sum()

    def act(self, obs, env):
        h, p = self._forward(obs)
        if self._train:
            a = int(self.rng.choice(self.n_actions, p=p))
        else:
            a = int(np.argmax(p))
        self._cache = (obs, h, p, a)
        return a

    def eval_mode(self):
        self._train = False

    # -- learning --------------------------------------------------------

    def update(self, trajectory):
        """
        trajectory: list of (obs, h, probs, action, reward)

        Gradient of log pi wrt logits is (onehot(a) - probs); the advantage is
        the reward-to-go minus a running scalar baseline, which reduces
        variance without needing a separate critic network.
        """
        obs = np.array([t[0] for t in trajectory])
        hs = np.array([t[1] for t in trajectory])
        ps = np.array([t[2] for t in trajectory])
        acts = np.array([t[3] for t in trajectory])
        rews = np.array([t[4] for t in trajectory], dtype=float)

        # Discounted reward-to-go.
        g = np.zeros_like(rews)
        run = 0.0
        for i in range(len(rews) - 1, -1, -1):
            run = rews[i] + self.gamma * run
            g[i] = run

        self.baseline = 0.95 * self.baseline + 0.05 * g.mean()
        adv = g - self.baseline
        sd = adv.std()
        if sd > 1e-8:
            adv = adv / sd

        onehot = np.zeros_like(ps)
        onehot[np.arange(len(acts)), acts] = 1.0
        dlogits = (onehot - ps) * adv[:, None]         # (T, A)

        gW2 = hs.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dh = (dlogits @ self.W2.T) * (1.0 - hs ** 2)
        gW1 = obs.T @ dh
        gb1 = dh.sum(axis=0)

        # Gradient ascent with global-norm clipping for stability.
        gn = np.sqrt(sum(float((x ** 2).sum()) for x in (gW1, gb1, gW2, gb2)))
        scale = min(1.0, 10.0 / (gn + 1e-8))
        self.W1 += self.lr * scale * gW1
        self.b1 += self.lr * scale * gb1
        self.W2 += self.lr * scale * gW2
        self.b2 += self.lr * scale * gb2
