"""Aşama 4: LeRobot yazıcı - oluşturma, devam/ekleme, şema koruması, sidecar."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mahut_kol.commands import Command, to_english_instruction
from mahut_kol.recording import (
    SIDECAR_NAME,
    VLADatasetWriter,
    check_schema,
    episode_meta,
    read_sidecar,
    vla_features,
)

CMD = Command("blue_cube", "red_cylinder", "RIGHT")


def fake_frames(n, val=50):
    for i in range(n):
        yield {
            "observation.images.front": np.full((224, 224, 3), val, np.uint8),
            "observation.images.wrist": np.full((224, 224, 3), 255 - val, np.uint8),
            "observation.state": np.full(7, i, np.float32),
            "action": np.full(7, -i, np.float32),
            "command.slots": np.array([2, 0, 0, 1, 3]),
            "task": to_english_instruction(CMD),
        }


def snapshot(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())


class TestWriter(unittest.TestCase):
    def test_create_resume_append(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "ds"
            with VLADatasetWriter(root, image_writer_threads=0) as w:
                self.assertTrue(w.created)
                for e in range(2):
                    w.add_episode(fake_frames(10 + e), episode_meta(CMD, "x", e, "teleop", False, True, 1.0))
            with VLADatasetWriter(root, image_writer_threads=0) as w:
                self.assertFalse(w.created)
                self.assertEqual(w.num_episodes, 2)
                ep = w.add_episode(fake_frames(7, 200), episode_meta(CMD, "x", 9, "scripted", False, True, 1.0))
                self.assertEqual(ep, 2)
            rows = read_sidecar(root / SIDECAR_NAME)
            self.assertEqual([r["episode_index"] for r in rows], [0, 1, 2])
            self.assertEqual([r["num_frames"] for r in rows], [10, 11, 7])
            self.assertEqual(rows[2]["source"], "scripted")
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset("local/x", root=root, video_backend="pyav")
            self.assertEqual(ds.num_episodes, 3)
            self.assertEqual(ds.num_frames, 28)
            last = ds[27]
            self.assertEqual(int(last["episode_index"]), 2)
            self.assertAlmostEqual(float(last["observation.images.front"].mean()), 200 / 255, delta=0.03)
            np.testing.assert_array_equal(last["command.slots"].numpy(), [2, 0, 0, 1, 3])

    def test_reopen_dataset_with_zero_episodes(self):
        """Bölüm kaydedilmeden kapanan oturumdan sonra veri kümesi yeniden açılabilmeli."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "ds"
            VLADatasetWriter(root, image_writer_threads=0).close()
            with VLADatasetWriter(root, image_writer_threads=0) as w:
                self.assertEqual(w.num_episodes, 0)
                w.add_episode(fake_frames(4), episode_meta(CMD, "x", 0, "teleop", False, True, 1.0))
            with VLADatasetWriter(root, image_writer_threads=0) as w:
                self.assertEqual(w.num_episodes, 1)

    def test_schema_mismatch_is_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "ds"
            with VLADatasetWriter(root, image_writer_threads=0) as w:
                w.add_episode(fake_frames(5), episode_meta(CMD, "x", 0, "teleop", False, True, 1.0))
            info_path = root / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            info["features"]["observation.state"]["shape"] = [6]
            info_path.write_text(json.dumps(info))
            before = snapshot(root)
            with self.assertRaises(ValueError) as ctx:
                VLADatasetWriter(root, image_writer_threads=0)
            self.assertIn("observation.state", str(ctx.exception))
            self.assertEqual(before, snapshot(root))

    def test_non_dataset_dir_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "other"
            root.mkdir()
            (root / "notes.txt").write_text("önemli")
            with self.assertRaises(FileExistsError):
                VLADatasetWriter(root, image_writer_threads=0)
            self.assertEqual((root / "notes.txt").read_text(), "önemli")

    def test_check_schema(self):
        self.assertEqual(check_schema(vla_features(), 20), [])
        self.assertTrue(check_schema(vla_features(), 30))

    def test_no_rmtree_in_recording_code(self):
        here = Path(__file__).resolve().parents[1]
        for p in [here / "src/mahut_kol/recording/writer.py", here / "scripts/record_vla_demos.py"]:
            self.assertNotIn("rmtree(", p.read_text(encoding="utf-8"), p)


if __name__ == "__main__":
    unittest.main()


class TestGoalOverlay(unittest.TestCase):
    """Hedef bölge işareti yalnızca ekrana çizilir; kaydedilen gözlem değişmez."""

    def test_overlay_projects_and_leaves_obs_clean(self):
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import numpy as np

        from mahut_kol.recording.session import PygameUI, goal_overlay, goal_region_world, make_env

        env = make_env()
        obs, _ = env.reset(seed=3)
        ov = goal_overlay(env)
        self.assertIn("front", ov)
        ring, center = ov["front"]
        self.assertEqual(ring.shape, (32, 2))
        self.assertTrue(np.all((center >= 0) & (center < 224)), center)
        # Merkez, hedef konumun izdüşümüyle aynı; daire yüzeyde (TOP değil -> masa)
        pts = goal_region_world(env)
        self.assertAlmostEqual(float(pts[-1, 2]), 0.001, places=3)
        before = {c: obs["sensor_data"][c]["rgb"].clone() for c in ("front", "wrist")}
        ui = PygameUI(scale=1, control_camera="wrist")
        ui.update(obs, "test", False, overlay=ov)
        # Yeşil çerçeve yalnızca kontrolün göre olduğu (sağdaki wrist) panelde
        self.assertEqual(tuple(ui.screen.get_at((224 + 1, 200)))[:3], (60, 220, 110))
        self.assertNotEqual(tuple(ui.screen.get_at((1, 200)))[:3], (60, 220, 110))
        ui.close()
        for c in before:
            self.assertTrue(bool((obs["sensor_data"][c]["rgb"] == before[c]).all()))
        env.close()
