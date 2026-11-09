"""cloud/train_entry.py mantığı (Modal hesabı ve GPU gerektirmez)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_entry", ROOT / "cloud" / "train_entry.py")
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)


def _cfg(policy="act", **kw):
    base = dict(policy=policy, repo_id="local/x", steps=10, batch_size=2, save_freq=5, log_freq=1,
                num_workers=0, seed=1, video_backend="pyav", wandb=False, rename_map={}, policy_args={})
    base.update(kw)
    return base


def test_new_run_then_refuses_existing_without_resume(tmp_path):
    out, resume = te.prepare_run(tmp_path, "r1", "L1", resume=False)
    assert out == tmp_path / "r1" / "train" and resume is None
    with pytest.raises(FileExistsError):
        te.prepare_run(tmp_path, "r1", "L2", resume=False)


def test_same_launch_id_is_auto_restart_and_new_attempt_dir_without_checkpoint(tmp_path):
    te.prepare_run(tmp_path, "r1", "L1", resume=False)
    (tmp_path / "r1" / "train").mkdir()
    out, resume = te.prepare_run(tmp_path, "r1", "L1", resume=False)  # Modal yeniden başlatması
    assert resume is None and out == tmp_path / "r1" / "train_r2"


def test_resume_picks_highest_step_checkpoint(tmp_path):
    te.prepare_run(tmp_path, "r1", "L1", resume=False)
    for step in ("000005", "000010"):
        d = tmp_path / "r1" / "train" / "checkpoints" / step
        (d / "pretrained_model").mkdir(parents=True)
        (d / "training_state").mkdir()
        (d / "pretrained_model" / "train_config.json").write_text("{}")
    out, cfg = te.prepare_run(tmp_path, "r1", "L9", resume=True)
    assert cfg.parent.parent.name == "000010" and out == tmp_path / "r1" / "train"
    assert json.loads((tmp_path / "r1" / "launch.json").read_text())["launch_id"] == "L9"


def test_build_argv_smolvla_has_rename_and_empty_cameras():
    c = _cfg("smolvla", rename_map={"observation.images.front": "observation.images.camera1"},
             policy_args={"path": "lerobot/smolvla_base", "empty_cameras": 1}, episodes=[0, 1])
    a = te.build_argv(c, Path("/d"), Path("/o"), "r", None)
    assert "--policy.path=lerobot/smolvla_base" in a and "--policy.empty_cameras=1" in a
    assert any(x.startswith("--rename_map=") for x in a)
    assert "--dataset.episodes=[0, 1]" in a and "--dataset.video_backend=pyav" in a
    assert "--wandb.enable=false" in a and "--eval_freq=0" in a


def test_build_argv_resume_only_overrides_steps():
    a = te.build_argv(_cfg(), Path("/d"), Path("/o"), "r", Path("/x/train_config.json"))
    assert a[:2] == ["--config_path=/x/train_config.json", "--resume=true"] and "--steps=10" in a
    assert not any(x.startswith("--dataset") for x in a)


def test_human_step_parser():
    assert te._human_to_int("2K") == 2000 and te._human_to_int("320") == 320
