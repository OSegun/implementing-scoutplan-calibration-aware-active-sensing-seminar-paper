"""
Regression tests for the invariants declared in CLAUDE.md section 7.
Run with: python -m unittest discover -s tests   (from inside scoutplan/)
or:       python -m pytest tests                 (if pytest is installed)

These are not a general test suite -- they encode the two properties the
entire study depends on. If either fails, the calibration instrument is
broken and no downstream result is valid.
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception import measure_calibration
from field import DiseaseField
from env import ScoutEnv
from perception import CalibratedClassifier
from agents import LawnmowerAgent, RandomAgent, GreedyEntropyAgent
import numpy as np


class TestAccuracyInvariance(unittest.TestCase):
    """Invariant 1: accuracy must not move with temperature."""

    def test_accuracy_constant_across_temperature(self):
        accs = {round(measure_calibration(0.816, T, n=20000, seed=1)["accuracy"], 4)
                for T in (0.3, 0.5, 1.0, 2.0, 3.0, 4.0)}
        self.assertEqual(len(accs), 1, f"instrument broken: accuracy varies -> {accs}")


class TestCalibrationShape(unittest.TestCase):
    """Invariant 2: ECE is minimised at T=1 and rises in both directions."""

    def test_ece_minimised_at_t_equals_one(self):
        eces = {T: measure_calibration(0.816, T, n=20000, seed=1)["ece"]
                for T in (0.3, 1.0, 4.0)}
        self.assertLess(eces[1.0], eces[0.3])
        self.assertLess(eces[1.0], eces[4.0])


class TestEnvironmentContract(unittest.TestCase):
    """Invariant 4: the info dict keeps the keys the figures/paper read."""

    REQUIRED_KEYS = {"recall", "precision", "detections_per_joule",
                      "false_alarms", "coverage", "time_to_first_detection"}

    def test_info_dict_has_required_keys(self):
        rng = np.random.default_rng(0)
        fld = DiseaseField(size=12, n_parents=3, offspring_mean=10, sigma=1.6, rng=rng)
        env = ScoutEnv(fld, CalibratedClassifier(0.816, 1.0, rng), budget=190,
                        prior=0.15, detect_threshold=0.75, rng=rng)
        agent = LawnmowerAgent()
        obs = env.reset()
        agent.reset(env)
        done = False
        info = {}
        while not done:
            obs, _, done, info = env.step(agent.act(obs, env))
        missing = self.REQUIRED_KEYS - set(info.keys())
        self.assertFalse(missing, f"info dict missing required keys: {missing}")

    def test_baselines_share_the_same_interface(self):
        rng = np.random.default_rng(0)
        for AgentCls in (LawnmowerAgent, RandomAgent, GreedyEntropyAgent):
            fld = DiseaseField(size=12, n_parents=3, offspring_mean=10, sigma=1.6, rng=rng)
            env = ScoutEnv(fld, CalibratedClassifier(0.816, 1.0, rng), budget=190,
                            prior=0.15, detect_threshold=0.75, rng=rng)
            agent = AgentCls(rng) if AgentCls is not LawnmowerAgent else AgentCls()
            obs = env.reset()
            if hasattr(agent, "reset"):
                agent.reset(env)
            done = False
            while not done:
                obs, _, done, info = env.step(agent.act(obs, env))
            self.assertIn("detections_per_joule", info)


if __name__ == "__main__":
    unittest.main()
