"""cloud/mahut_cloud.py koruma mantığı (Modal hesabı gerektirmez: listdir/subprocess taklit edilir)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mahut_cloud", ROOT / "cloud" / "mahut_cloud.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


def test_parse_episodes():
    assert mc.parse_episodes(None) is None
    assert mc.parse_episodes("0-3") == [0, 1, 2, 3]
    assert mc.parse_episodes("0,5,7-8") == [0, 5, 7, 8]


def _train_args(extra=()):
    ap_args = ["train", "--config", str(ROOT / "configs/training/act.yaml"), "--run", "yeni_koşu", *extra]
    return ap_args


def _fake_listdir(existing):
    return lambda vol, path: [path] if (vol, path) in existing else []


def test_train_refuses_existing_run_without_resume(monkeypatch):
    monkeypatch.setattr(mc, "listdir", _fake_listdir({(mc.DATA_VOL, "/mahut_smoke_v1_r1_act"), (mc.RUNS_VOL, "/runs/yeni_koşu")}))
    monkeypatch.setattr(mc.subprocess, "run", lambda *a, **k: pytest.fail("modal çağrılmamalı"))
    with pytest.raises(SystemExit) as e:
        mc.main(_train_args())
    assert "zaten var" in str(e.value)


def test_train_resume_requires_existing_run(monkeypatch):
    monkeypatch.setattr(mc, "listdir", _fake_listdir({(mc.DATA_VOL, "/mahut_smoke_v1_r1_act")}))
    with pytest.raises(SystemExit) as e:
        mc.main(_train_args(["--resume"]))
    assert "yok" in str(e.value)


def test_train_requires_uploaded_dataset(monkeypatch):
    monkeypatch.setattr(mc, "listdir", _fake_listdir(set()))
    with pytest.raises(SystemExit) as e:
        mc.main(_train_args())
    assert "upload" in str(e.value)


def test_train_enforces_timeout_ceiling(monkeypatch):
    monkeypatch.setattr(mc.subprocess, "run", lambda *a, **k: pytest.fail("modal çağrılmamalı"))
    with pytest.raises(SystemExit) as e:
        mc.main(_train_args(["--timeout-min", "120"]))
    assert "max-timeout-min" in str(e.value)


def test_train_launches_detached_with_gpu_env(monkeypatch):
    monkeypatch.setattr(mc, "listdir", _fake_listdir({(mc.DATA_VOL, "/mahut_smoke_v1_r1_act")}))
    seen = {}

    class R:
        returncode = 0

    def fake_run(cmd, env=None, **k):
        seen["cmd"], seen["env"] = cmd, env
        return R()

    monkeypatch.setattr(mc.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as e:
        mc.main(_train_args(["--gpu", "L4,A10", "--steps", "7"]))
    assert e.value.code == 0
    assert "--detach" in seen["cmd"] and seen["env"]["MAHUT_GPU"] == "L4,A10"
    assert seen["env"]["MAHUT_TIMEOUT_MIN"] == "30"


def test_upload_refuses_existing_name(monkeypatch, tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    monkeypatch.setattr(mc, "listdir", _fake_listdir({(mc.DATA_VOL, "/var")}))
    with pytest.raises(SystemExit) as e:
        mc.main(["upload", "--src", str(tmp_path), "--name", "var"])
    assert "zaten var" in str(e.value)


def test_download_refuses_existing_destination(monkeypatch, tmp_path):
    (tmp_path / "koşu").mkdir()
    with pytest.raises(SystemExit) as e:
        mc.main(["download", "--run", "koşu", "--dest", str(tmp_path)])
    assert "zaten var" in str(e.value)
