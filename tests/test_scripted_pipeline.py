"""
Aşama 4-7 uçtan uca: scripted politika, kayıt betiği (sahte cihazla), üretici, doğrulayıcı,
sabit sahne (object_poses) ve değerlendirmedeki dil varyantları.
"""

import builtins
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import mahut_kol.envs  # noqa: F401
from mahut_kol.commands import Command
from mahut_kol.envs.scripted_policy import MultiShapeScriptedPolicy
from mahut_kol.recording import SIDECAR_NAME, read_sidecar
from mahut_kol.recording.session import make_env

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScriptedPolicyActionSource:
    """Kayıt betiğine telefon yerine takılan sahte cihaz: scripted politikayı oynatır."""

    name = "fake"

    def __init__(self, env):
        self.env = env

    def reset(self, qpos):
        self.p = MultiShapeScriptedPolicy(self.env.unwrapped)
        self.p.reset(qpos)

    def act(self, qpos):
        return self.p.act(qpos)

    def poll_events(self):
        return set()

    def status_text(self):
        return "sahte"

    def publish_status(self, **kw):
        pass

    def close(self):
        pass


class TestScriptedPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(600, obs_mode="state_dict")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _run(self, cmd, seed):
        self.env.reset(seed=seed, options=cmd.to_reset_options(2))
        u = self.env.unwrapped
        pol = MultiShapeScriptedPolicy(u)
        pol.reset(u.agent.robot.get_qpos()[0].numpy())
        self.assertIsNone(pol.plan_error)
        for _ in range(len(pol)):
            _, _, _, _, info = self.env.step(torch.as_tensor(pol.act(u.agent.robot.get_qpos()[0].numpy())))
            if info["success"].item():
                return True
        return False

    def test_each_shape_and_top(self):
        cases = [
            (Command("red_cube", "green_cylinder", "RIGHT"), 1),
            (Command("blue_cylinder", "red_cube", "FRONT"), 2),
            (Command("green_triangular_prism", "blue_cube", "LEFT"), 3),
            (Command("red_cylinder", "blue_cube", "TOP"), 4),
        ]
        for cmd, seed in cases:
            self.assertTrue(self._run(cmd, seed), cmd.key)

    def test_fixed_scene_object_poses(self):
        env = self.env
        env.reset(seed=7, options=Command("red_cube", "green_cylinder", "BACK").to_reset_options(2))
        poses = env.unwrapped.get_active_object_poses()
        others = [k for k in poses if k not in ("red_cube", "green_cylinder")]
        env.reset(seed=99, options={**Command("red_cube", "green_cylinder", "BACK").to_reset_options(),
                                    "object_poses": poses, "distractor_objects": others})
        for k, (x, y, yaw) in env.unwrapped.get_active_object_poses().items():
            self.assertAlmostEqual(x, poses[k][0], places=4)
            self.assertAlmostEqual(y, poses[k][1], places=4)
        with self.assertRaises(ValueError):
            env.reset(options={**Command("red_cube", "green_cylinder", "BACK").to_reset_options(),
                               "object_poses": {"red_cube": poses["red_cube"]}})


class TestRecordingScripts(unittest.TestCase):
    def test_record_generate_validate(self):
        rec = load_script("record_vla_demos")
        gen = load_script("generate_scripted_demos")
        val = load_script("validate_dataset")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "ds"
            # 1) Teleop kayıt betiği, sahte cihaz + terminal onayı (Enter = kaydet)
            with mock.patch.object(rec, "make_action_source", lambda args: ScriptedPolicyActionSource(env_holder[0])), \
                 mock.patch.object(builtins, "input", lambda *_a, **_k: ""):
                orig_make_env = rec.make_env
                env_holder = []

                def _make_env(*a, **k):
                    e = orig_make_env(*a, **k)
                    env_holder.append(e)
                    return e

                with mock.patch.object(rec, "make_env", _make_env):
                    rec.main(["--root", str(root), "--num-episodes", "1", "--no-display", "--seed", "3"])
            rows = read_sidecar(root / SIDECAR_NAME)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "teleop")
            self.assertTrue((root / "command_queue_state.json").exists())
            # 2) Scripted üretici aynı veri kümesine ekler
            gen.main(["--root", str(root), "--num-episodes", "2", "--seed", "5"])
            rows = read_sidecar(root / SIDECAR_NAME)
            self.assertEqual([r["source"] for r in rows], ["teleop", "scripted", "scripted"])
            self.assertTrue((root / "scripted_generation_report.jsonl").exists())
            # 3) Doğrulayıcı
            out = Path(d) / "rapor.json"
            self.assertEqual(val.main(["--root", str(root), "--video-check", "all", "--json", str(out)]), 0)
            rep = json.loads(out.read_text())
            self.assertEqual(rep["episodes"], 3)
            self.assertEqual(rep["errors"], [])
            # 4) Held-out sızıntısı yakalanır (sidecar'a sahte satır yazmadan: doğrulayıcı fonksiyonu)
            from mahut_kol.commands import load_heldout

            h = load_heldout()[0]
            lines = (root / SIDECAR_NAME).read_text().splitlines()
            r0 = json.loads(lines[0])
            r0["command"] = h.to_dict()
            bad = Path(d) / "bad"
            import shutil

            shutil.copytree(root, bad)  # kopya üzerinde boz; asıl veri kümesine dokunma
            (bad / SIDECAR_NAME).write_text("\n".join([json.dumps(r0)] + lines[1:]) + "\n")
            rep2 = val.validate(bad, "local/x", "none", 0, allow_heldout=False)
            self.assertTrue(any("held-out" in e for e in rep2.errors), rep2.errors)


class TestEvalLanguageVariants(unittest.TestCase):
    def test_variants_valid_and_free(self):
        ev = load_script("eval_policy")
        env = make_env(600, obs_mode="state_dict")
        base = Command("red_cube", "green_cylinder", "LEFT")
        env.reset(seed=11, options=base.to_reset_options(2))
        poses = env.unwrapped.get_active_object_poses()
        vs = ev.language_variants(env, base, poses, 5, np.random.default_rng(0))
        self.assertTrue(vs)
        for v in vs:
            self.assertNotEqual(v.key, base.key)
            self.assertEqual({v.manipulated_object, v.reference_object} <= set(poses), True)
            others = [k for k in poses if k not in (v.manipulated_object, v.reference_object)]
            env.reset(seed=11, options={**v.to_reset_options(), "object_poses": poses, "distractor_objects": others})
        env.close()


if __name__ == "__main__":
    unittest.main()
