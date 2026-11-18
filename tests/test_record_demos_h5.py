"""Teleoperasyonlar/record_demos.py: pencere açmadan, scripted politikayı cihaz yerine takarak HDF5 kaydı."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py

from mahut_kol.envs.scripted_policy import MultiShapeScriptedPolicy
from mahut_kol.robots.mahut_robot import ACTION_DIM
from Teleoperasyonlar.record_demos import record_demonstrations


class _ScriptedSource:
    def __init__(self, env):
        self.u = env.unwrapped

    def reset(self, qpos):
        self.p = MultiShapeScriptedPolicy(self.u)
        self.p.reset(qpos)

    def act(self, qpos):
        return self.p.act(qpos)

    def poll_events(self):
        return set()

    def close(self):
        pass


class TestRecordDemosH5(unittest.TestCase):
    def test_records_successful_episode_and_appends(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "demos.h5")
            self.assertEqual(record_demonstrations(out, 1, make_source=_ScriptedSource, render=False), 1)
            self.assertEqual(record_demonstrations(out, 2, make_source=_ScriptedSource, render=False), 2)
            with h5py.File(out) as h5:
                self.assertEqual(h5.attrs["total_episodes"], 2)
                g = h5["episode_1"]
                self.assertEqual(g["actions"].shape[1], ACTION_DIM)
                self.assertEqual(g["states"].shape, g["actions"].shape)
                self.assertIn("relation", json.loads(g.attrs["command"]))


if __name__ == "__main__":
    unittest.main()
