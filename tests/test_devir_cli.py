"""cloud/devir.py komut mantığı: Modal hesabı olmadan, sahte Volume/uzak çağrılarla."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
spec = importlib.util.spec_from_file_location("devir", ROOT / "cloud" / "devir.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
L = D.L

CFG = {"policy": "act", "dataset": "veri_act_s1", "steps": 100,
       "maliyet": {"usd_per_1000_steps": 0.05, "sabit_usd": 0.05, "steps_per_s": 9.5, "startup_min": 5}}


def make_iter(dm_iter: Path, it, step, kisi, parent_it=None):
    ck = dm_iter / f"_w{it}{kisi}" / "ck"
    (ck / "pretrained_model").mkdir(parents=True)
    (ck / "training_state").mkdir()
    (ck / "pretrained_model" / "model.safetensors").write_bytes(b"w")
    (ck / "pretrained_model" / "train_config.json").write_text("{}")
    for f in ("training_step.json", "rng_state.safetensors", "optimizer_state.safetensors", "optimizer_param_groups.json"):
        (ck / "training_state" / f).write_bytes(b"x")
    out = dm_iter / L.iter_name(it, step, kisi)
    L.build_iter_tar(out, ck, {}, {"hat": "h", "iterasyon": it, "adim": step, "kisi": kisi, "parent": None,
                                  "parent_iterasyon": parent_it, "politika": "act", "git_commit": "a" * 40, "config": CFG,
                                  "sabit_surumu": 1, "paket_surumleri": {}, "gpu": "x", "tur_suresi_s": 1.0})
    return out


class FakeVol:
    def __init__(self, existing=()):
        self.existing, self.puts = set(existing), []

    def listdir(self, path):
        class E:
            def __init__(s, p):
                s.path = p
        return [E(p) for p in self.existing if p.startswith(path)]

    def batch_upload(self, force=False):
        vol = self

        class B:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def put_file(s, src, dst):
                vol.puts.append(dst)
        return B()


@pytest.fixture
def dm(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DM", tmp_path)
    monkeypatch.setattr(D, "ITER_DIR", tmp_path / "iterasyonlar")
    monkeypatch.setattr(D, "SABIT_DIR", tmp_path / "sabit")
    monkeypatch.setattr(D, "STATE_DIR", tmp_path / ".devir_state")
    monkeypatch.setattr(D, "profile_ok", lambda: True)
    return tmp_path


def test_restore_uses_same_absolute_paths_and_verifies(dm, monkeypatch):
    tar = make_iter(dm / "iterasyonlar" / "h", 1, 100, "deniz")
    man = L.verify_tar(tar, "iter")
    want = {k: v for k, v in man["dosyalar"].items() if k.split("/")[0] in L.ITER_DIRS}
    calls = iter([{"bulundu": False, "dosyalar": {}}, {"bulundu": True, "dosyalar": want}])
    monkeypatch.setattr(D, "remote_call", lambda *a: next(calls))
    vol = FakeVol()
    D.restore_checkpoint("h", tar, man, vol)
    base = "/runs/h/train/checkpoints/000100/"
    assert base + "pretrained_model/model.safetensors" in vol.puts and base + "training_state/training_step.json" in vol.puts
    assert "/runs/h/launch.json" in vol.puts                       # prepare_run ile uyumlu
    assert not any("manifest" in p for p in vol.puts)


def test_restore_skips_when_digest_same(dm, monkeypatch):
    tar = make_iter(dm / "iterasyonlar" / "h", 1, 100, "deniz")
    man = L.verify_tar(tar, "iter")
    want = {k: v for k, v in man["dosyalar"].items() if k.split("/")[0] in L.ITER_DIRS}
    monkeypatch.setattr(D, "remote_call", lambda *a: {"bulundu": True, "dosyalar": want})
    vol = FakeVol()
    D.restore_checkpoint("h", tar, man, vol)
    assert vol.puts == []


def test_restore_refuses_when_step_exists_with_different_content(dm, monkeypatch):
    tar = make_iter(dm / "iterasyonlar" / "h", 1, 100, "deniz")
    man = L.verify_tar(tar, "iter")
    monkeypatch.setattr(D, "remote_call", lambda *a: {"bulundu": True, "dosyalar": {"pretrained_model/model.safetensors": "0" * 64}})
    vol = FakeVol()
    with pytest.raises(SystemExit):
        D.restore_checkpoint("h", tar, man, vol)
    assert vol.puts == []                                           # üzerine yazılmadı


def test_restore_fails_if_post_upload_digest_differs(dm, monkeypatch):
    tar = make_iter(dm / "iterasyonlar" / "h", 1, 100, "deniz")
    man = L.verify_tar(tar, "iter")
    calls = iter([{"bulundu": False, "dosyalar": {}}, {"bulundu": True, "dosyalar": {"x": "y"}}])
    monkeypatch.setattr(D, "remote_call", lambda *a: next(calls))
    with pytest.raises(SystemExit):
        D.restore_checkpoint("h", tar, man, FakeVol())


def test_devam_stops_on_fork_before_any_remote_call(dm, monkeypatch, capsys):
    hat = dm / "iterasyonlar" / "h"
    make_iter(hat, 1, 100, "deniz")
    make_iter(hat, 2, 200, "ali", parent_it=1)
    make_iter(hat, 2, 210, "veli", parent_it=1)
    monkeypatch.setattr(D, "remote_call", lambda *a: pytest.fail("uzak çağrı yapılmamalı"))
    monkeypatch.setattr(D, "spent_now", lambda: pytest.fail("harcama okunmamalı"))
    with pytest.raises(SystemExit):
        D.main(["devam", "--hat", "h", "--kisi", "mehmet", "--adim", "10"])
    assert "ÇATALLANMA" in capsys.readouterr().err


def test_devam_without_any_iteration_explains(dm, capsys):
    with pytest.raises(SystemExit):
        D.main(["devam", "--hat", "h", "--kisi", "ali", "--adim", "10"])
    assert "iter_*.tar yok" in capsys.readouterr().err


def test_bad_kisi_rejected_without_touching_modal(dm, monkeypatch, capsys):
    monkeypatch.setattr(D, "profile_ok", lambda: pytest.fail("profil çağrılmamalı"))
    with pytest.raises(SystemExit):
        D.main(["devam", "--hat", "h", "--kisi", "ali@ornek.com", "--adim", "10"])
    assert "geçersiz" in capsys.readouterr().err


def test_no_profile_message_has_no_identity(monkeypatch, capsys):
    monkeypatch.setattr(D, "profile_ok", lambda: False)
    with pytest.raises(SystemExit):
        D.main(["kur"])
    assert "modal setup" in capsys.readouterr().err


def test_spent_now_parses_summary_and_fails_safe(monkeypatch):
    class R:
        returncode = 0
        stdout = json.dumps({"billed_cost": "0E-8", "adjustments": {"credits": "-2.5"}})

    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: R())
    assert D.spent_now() == pytest.approx(2.5)
    R.returncode = 1
    with pytest.raises(SystemExit):
        D.spent_now()


def test_timeout_and_steps_checked_before_launch(dm, monkeypatch):
    called = []
    monkeypatch.setattr(D, "launch_train", lambda *a, **k: called.append(1) or 0)
    with pytest.raises(L.DevirHata):
        L.check_steps_allowed(CFG, 10**9, 1.0, D.DEFAULT_MAX_TIMEOUT_MIN)
    assert called == []


def test_temizle_dry_run_never_deletes(dm, monkeypatch):
    hat = dm / "iterasyonlar" / "h"
    make_iter(hat, 1, 100, "deniz")

    class V(FakeVol):
        def remove_file(self, *a, **k):
            pytest.fail("silme çağrılmamalı")

    monkeypatch.setattr(D, "volumes", lambda: (V(), V()))
    monkeypatch.setattr(D, "volume_checkpoints", lambda rvol, h: [(200, "/runs/h/train/checkpoints/000200"),
                                                                 (100, "/runs/h/train/checkpoints/000100")])
    D.cmd_temizle(type("A", (), {"hat": "h", "onayla": False})())


def test_temizle_deletes_only_packaged_and_never_latest(dm, monkeypatch):
    hat = dm / "iterasyonlar" / "h"
    make_iter(hat, 1, 100, "deniz")          # yalnızca 100 paketlenmiş
    removed = []

    class V(FakeVol):
        def remove_file(self, p, recursive=False):
            removed.append(p)

    monkeypatch.setattr(D, "volumes", lambda: (V(), V()))
    monkeypatch.setattr(D, "volume_checkpoints", lambda rvol, h: [(300, "/c/000300"), (200, "/c/000200"), (100, "/c/000100")])
    D.cmd_temizle(type("A", (), {"hat": "h", "onayla": True})())
    assert removed == ["/c/000100"]            # 300 en son, 200 paketlenmedi -> dokunulmaz
