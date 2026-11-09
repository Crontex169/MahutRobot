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
        _write_ckpt(tmp_path, step)
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


# --- checkpoint tamlık doğrulaması (yarım checkpoint seçilmemeli) -------------------------------


def _write_ckpt(root: Path, step: str, *, scheduler=True, missing=(), run="r1", attempt="train"):
    ck = root / run / attempt / "checkpoints" / step
    (ck / "pretrained_model").mkdir(parents=True)
    (ck / "training_state").mkdir()
    files = {
        "pretrained_model/model.safetensors": b"x", "pretrained_model/train_config.json":
            json.dumps({"scheduler": {"type": "cosine"} if scheduler else None}).encode(),
        "training_state/training_step.json": b"{}", "training_state/rng_state.safetensors": b"x",
        "training_state/optimizer_state.safetensors": b"x", "training_state/optimizer_param_groups.json": b"[]",
    }
    if scheduler:
        files["training_state/scheduler_state.json"] = b"{}"
    for rel, data in files.items():
        if rel not in missing:
            (ck / rel).write_bytes(data)
    return ck


@pytest.mark.parametrize("scheduler", [True, False])  # SmolVLA zamanlayıcılı, ACT zamanlayıcısız
def test_complete_checkpoint_accepted(tmp_path, scheduler):
    ck = _write_ckpt(tmp_path, "000100", scheduler=scheduler)
    assert te.checkpoint_is_complete(ck)


@pytest.mark.parametrize("missing", [
    "pretrained_model/model.safetensors", "training_state/training_step.json",
    "training_state/rng_state.safetensors", "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json", "training_state/scheduler_state.json",
])
def test_partial_checkpoint_rejected(tmp_path, missing):
    assert not te.checkpoint_is_complete(_write_ckpt(tmp_path, "000100", missing=(missing,)))


def test_empty_file_rejected(tmp_path):
    ck = _write_ckpt(tmp_path, "000100")
    (ck / "training_state" / "optimizer_state.safetensors").write_bytes(b"")
    assert not te.checkpoint_is_complete(ck)


def test_resume_falls_back_to_previous_complete_checkpoint(tmp_path):
    te.prepare_run(tmp_path, "r1", "L1", resume=False)
    _write_ckpt(tmp_path, "000100")
    _write_ckpt(tmp_path, "000200", missing=("training_state/optimizer_state.safetensors",))  # yazarken kesilmiş
    cfg = te.last_checkpoint_config(tmp_path / "r1")
    assert cfg is not None and cfg.parent.parent.name == "000100"
    _, resume_cfg = te.prepare_run(tmp_path, "r1", "L1", resume=False)  # aynı launch_id: otomatik yeniden başlatma
    assert resume_cfg.parent.parent.name == "000100"


def test_only_partial_checkpoint_means_no_resume(tmp_path):
    te.prepare_run(tmp_path, "r1", "L1", resume=False)
    _write_ckpt(tmp_path, "000100", missing=("training_state/rng_state.safetensors",))
    assert te.last_checkpoint_config(tmp_path / "r1") is None
    out, resume_cfg = te.prepare_run(tmp_path, "r1", "L1", resume=False)
    assert resume_cfg is None and out == tmp_path / "r1" / "train_r2"  # yeni deneme klasörü, yarım olan silinmez
