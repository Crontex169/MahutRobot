"""cloud/devir_lib.py: manifest, güvenli tar, çatallanma, git ve kredi mantığı (Modal gerektirmez)."""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("devir_lib", ROOT / "cloud" / "devir_lib.py")
L = importlib.util.module_from_spec(spec)
sys.modules["devir_lib"] = L
spec.loader.exec_module(L)

CFG = {"policy": "act", "maliyet": {"usd_per_1000_steps": 0.05, "sabit_usd": 0.05, "steps_per_s": 9.5, "startup_min": 5}}


# ---- yardımcılar -------------------------------------------------------------------------------


def make_ckpt(root: Path, step=100, scheduler=False):
    ck = root / f"ck{step}"
    (ck / "pretrained_model").mkdir(parents=True)
    (ck / "training_state").mkdir()
    (ck / "pretrained_model" / "model.safetensors").write_bytes(b"w" * 50)
    (ck / "pretrained_model" / "train_config.json").write_text(json.dumps({"scheduler": None}))
    for f, d in {"training_step.json": b"{}", "rng_state.safetensors": b"r", "optimizer_state.safetensors": b"o",
                 "optimizer_param_groups.json": b"[]"}.items():
        (ck / "training_state" / f).write_bytes(d)
    return ck


def fields(it=1, step=100, kisi="deniz", parent=None, parent_it=None):
    return {"hat": "act_v1", "iterasyon": it, "adim": step, "kisi": kisi, "parent": parent, "parent_iterasyon": parent_it,
            "politika": "act", "git_commit": "a" * 40, "config": CFG, "sabit_surumu": 1, "paket_surumleri": {"torch": "2.6.0"},
            "gpu": "NVIDIA A10", "tur_suresi_s": 60.0}


def make_iter(tmp_path, it=1, step=100, kisi="deniz", parent=None, parent_it=None) -> Path:
    ck = make_ckpt(tmp_path / f"w{it}{kisi}", step)
    log = tmp_path / f"log{it}{kisi}"
    log.write_text("log")
    att = tmp_path / f"att{it}{kisi}"
    att.write_text("{}\n")
    out = tmp_path / "hat" / L.iter_name(it, step, kisi)
    L.build_iter_tar(out, ck, {"train.log": log, "attempts.jsonl": att}, fields(it, step, kisi, parent, parent_it))
    return out


def raw_tar(path: Path, members: dict, manifest_files=None):
    """Elle tar üret; members: {ad: bytes | (tipbayrağı, linkname)}. manifest_files: sha sözlüğü (varsayılan: boş değil)."""
    mf = {"format_surumu": 1, "tur": "iter", "dosyalar": manifest_files or {"train.log": L.sha256_bytes(b"x")}}
    with tarfile.open(path, "w:") as tf:
        data = json.dumps(mf).encode()
        ti = tarfile.TarInfo("manifest.json"); ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
        for name, v in members.items():
            ti = tarfile.TarInfo(name)
            if isinstance(v, tuple):
                ti.type, ti.linkname = v
                tf.addfile(ti)
            else:
                ti.size = len(v)
                tf.addfile(ti, io.BytesIO(v))
    return path


# ---- adlar ------------------------------------------------------------------------------------


def test_names_roundtrip_and_validation():
    n = L.iter_name(2, 10000, "ali")
    assert n == "iter_0002_step_010000_ali.tar"
    assert L.parse_iter_name(n) == {"iterasyon": 2, "adim": 10000, "kisi": "ali"}
    with pytest.raises(L.DevirHata):
        L.parse_iter_name("iter_0002_step_010000_ali (1).tar")  # Drive eki
    for bad in ("a@b.com", "Ali", "ali veli", "", "../x", "x" * 21):
        with pytest.raises(L.DevirHata):
            L.check_kisi(bad)
    with pytest.raises(L.DevirHata):
        L.check_hat("Act V1")


def test_volume_names_suffix_only_from_env():
    assert L.volume_names({}) == ("mahut-datasets", "mahut-runs")
    assert L.volume_names({"MAHUT_VOLUME_SUFFIX": "_test2"}) == ("mahut-datasets_test2", "mahut-runs_test2")
    assert L.dataset_names("mahut_smoke_v1", 1, True) == {"ham": "mahut_smoke_v1_s1", "act": "mahut_smoke_v1_act_s1"}


# ---- manifest ve tar --------------------------------------------------------------------------


def test_iter_tar_manifest_and_verify(tmp_path):
    p = make_iter(tmp_path)
    m = L.verify_tar(p, "iter")
    assert m["format_surumu"] == 1 and m["hat"] == "act_v1" and m["parent"] is None
    assert set(m["dosyalar"]) == {"pretrained_model/model.safetensors", "pretrained_model/train_config.json",
                                  "training_state/training_step.json", "training_state/rng_state.safetensors",
                                  "training_state/optimizer_state.safetensors", "training_state/optimizer_param_groups.json",
                                  "train.log", "attempts.jsonl"}
    assert all(len(v) == 64 for v in m["dosyalar"].values())
    assert "@" not in json.dumps(m)                       # e-posta/kimlik yok
    with tarfile.open(p) as tf:                            # sıkıştırmasız ve düz dosya
        assert all(x.isreg() for x in tf) and not any(x.name.endswith("/") for x in tf)


def test_existing_tar_never_overwritten(tmp_path):
    p = make_iter(tmp_path)
    ck = make_ckpt(tmp_path / "again", 100)
    with pytest.raises(L.DevirHata, match="zaten var"):
        L.build_iter_tar(p, ck, {}, fields())


def test_flipped_byte_detected(tmp_path):
    p = make_iter(tmp_path)
    data = bytearray(p.read_bytes())
    i = data.index(b"w" * 50)
    data[i + 10] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(L.DevirHata, match="sha256"):
        L.verify_tar(p, "iter")


def test_truncated_tar_detected(tmp_path):
    p = make_iter(tmp_path)
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])
    with pytest.raises(L.DevirHata):
        L.verify_tar(p, "iter")


@pytest.mark.parametrize("name", ["../evil.json", "pretrained_model/../../evil.json", "/etc/x.json", "pretrained_model//x.json"])
def test_path_escape_rejected(tmp_path, name):
    p = raw_tar(tmp_path / "t.tar", {name: b"{}"})
    with pytest.raises(L.DevirHata, match="güvensiz|beklenmeyen"):
        L.inspect_tar(p, "iter")


@pytest.mark.parametrize("name", ["pretrained_model/model.pt", "pretrained_model/w.pth", "pretrained_model/w.bin",
                                  "pretrained_model/x.pkl", "pretrained_model/y.pickle"])
def test_pickle_rejected(tmp_path, name):
    with pytest.raises(L.DevirHata, match="yasaklı"):
        L.inspect_tar(raw_tar(tmp_path / "t.tar", {name: b"x"}), "iter")


@pytest.mark.parametrize("typ", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_links_rejected(tmp_path, typ):
    p = raw_tar(tmp_path / "t.tar", {"pretrained_model/model.safetensors": (typ, "/etc/passwd")})
    with pytest.raises(L.DevirHata, match="düz dosya"):
        L.inspect_tar(p, "iter")


def test_unexpected_location_or_ext_rejected(tmp_path):
    for name in ("evil.json", "pretrained_model/a/b.json", "pretrained_model/x.txt", "other/x.json", "run.sh"):
        with pytest.raises(L.DevirHata):
            L.inspect_tar(raw_tar(tmp_path / f"{abs(hash(name))}.tar", {name: b"x"}), "iter")


def test_member_not_in_manifest_rejected(tmp_path):
    p = raw_tar(tmp_path / "t.tar", {"train.log": b"x", "attempts.jsonl": b"y"},
                manifest_files={"train.log": L.sha256_bytes(b"x")})
    with pytest.raises(L.DevirHata, match="uyuşmuyor"):
        L.verify_tar(p, "iter")


def test_extract_cannot_escape_and_does_not_overwrite(tmp_path):
    p = make_iter(tmp_path)
    dest = tmp_path / "out"
    names = L.extract_tar(p, dest, "iter")
    assert "pretrained_model/model.safetensors" in names and (dest / "train.log").exists()
    with pytest.raises(FileExistsError):                  # 'x' kipi: var olanın üzerine yazmaz
        L.extract_tar(p, dest, "iter")


def test_sabit_tar_dedups_act_videos(tmp_path):
    def ds(root: Path, act=False):
        (root / "meta").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "videos" / "cam").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 3, "total_frames": 30}))
        (root / "data" / "chunk-000" / "f.parquet").write_bytes(b"A" if not act else b"B")
        (root / "videos" / "cam" / "v.mp4").write_bytes(b"VIDEO" * 100)
        (root / "mahut_episodes.jsonl").write_text("{}\n")

    ham, act = tmp_path / "ham", tmp_path / "act"
    ds(ham)
    ds(act, act=True)
    m = L.build_sabit_tar(tmp_path / "sabit_v1.tar", 1, ham, act, "veri_s1", "veri_act_s1")
    assert m["bolum_sayisi"] == 3 and m["kare_sayisi"] == 30
    assert m["act_video_ham_ile_ortak"] == {"act/videos/cam/v.mp4": "ham/videos/cam/v.mp4"}
    names = L.inspect_tar(tmp_path / "sabit_v1.tar", "sabit")
    assert "ham/videos/cam/v.mp4" in names and "act/videos/cam/v.mp4" not in names   # tek kopya
    assert "act/data/chunk-000/f.parquet" in names
    L.verify_tar(tmp_path / "sabit_v1.tar", "sabit")


# ---- çatallanma -------------------------------------------------------------------------------


def test_chain_picks_head_and_next_number(tmp_path):
    make_iter(tmp_path, 1, 100, "deniz")
    make_iter(tmp_path, 2, 200, "ali", "iter_0001_step_000100_deniz.tar", 1)
    head = L.pick_parent(tmp_path / "hat")
    assert head["iterasyon"] == 2 and L.next_iteration_number(tmp_path / "hat") == 3


def test_fork_same_parent_detected_and_sec_resolves(tmp_path):
    make_iter(tmp_path, 1, 100, "deniz")
    make_iter(tmp_path, 2, 200, "ali", "p", 1)
    make_iter(tmp_path, 2, 210, "veli", "p", 1)
    with pytest.raises(L.CatallanmaHatasi, match="ÇATALLANMA"):
        L.pick_parent(tmp_path / "hat")
    pick = L.pick_parent(tmp_path / "hat", sec="iter_0002_step_000210_veli.tar")
    assert pick["kisi"] == "veli"
    with pytest.raises(L.DevirHata):
        L.pick_parent(tmp_path / "hat", sec="yok.tar")


def test_same_iteration_number_is_fork(tmp_path):
    make_iter(tmp_path, 1, 100, "deniz")
    make_iter(tmp_path, 1, 150, "ali")
    with pytest.raises(L.CatallanmaHatasi):
        L.pick_parent(tmp_path / "hat")


def test_renamed_file_mismatch_rejected(tmp_path):
    p = make_iter(tmp_path, 1, 100, "deniz")
    p.rename(p.with_name("iter_0001_step_000100_ali.tar"))
    with pytest.raises(L.DevirHata, match="uyuşmuyor"):
        L.scan_iterations(p.parent)


# ---- git --------------------------------------------------------------------------------------


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    for d in ("cloud", "configs/training", "src/mahut_kol/commands", "docs"):
        (r / d).mkdir(parents=True)
        (r / d / "f.txt").write_text("1")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "c1")
    return r


def test_git_clean_ok_and_dirty_rejected(repo):
    head = L.git_head(repo)
    L.check_git(repo, head)
    (repo / "docs" / "f.txt").write_text("degisti")
    with pytest.raises(L.DevirHata, match="temiz değil"):
        L.check_git(repo, head)


def test_git_data_model_ignored_in_status(repo):
    (repo / "Data_Model").mkdir()
    (repo / "Data_Model" / "x.tar").write_text("x")
    assert L.git_dirty_files(repo) == []


def test_git_code_path_differs_from_manifest_commit(repo):
    old = L.git_head(repo)
    (repo / "cloud" / "f.txt").write_text("2")
    git(repo, "commit", "-qam", "c2")
    with pytest.raises(L.DevirHata, match="cloud"):
        L.check_git(repo, old)
    (repo / "docs" / "f.txt").write_text("2")             # kod yolu dışı fark sorun değil
    git(repo, "commit", "-qam", "c3")
    L.check_git(repo, L.git_head(repo))


def test_git_unknown_commit_rejected(repo):
    with pytest.raises(L.DevirHata, match="bu depoda yok"):
        L.check_git(repo, "f" * 40)


# ---- sürüm, sabit, kredi ----------------------------------------------------------------------


def test_version_and_sabit_checks():
    assert L.compare_versions({"torch": "2.6.0"}, {"torch": "2.6.0", "x": "1"}) == []
    assert L.compare_versions({"torch": "2.6.0"}, {"torch": "2.7.0"})
    m = {"hat": "act_v1", "sabit_surumu": 1, "sabit_manifest_ozeti": "abc"}
    L.check_hat_and_sabit(m, "act_v1", "abc", 1)
    with pytest.raises(L.DevirHata, match="hat"):
        L.check_hat_and_sabit(m, "smol_v1", "abc", 1)
    with pytest.raises(L.DevirHata, match="sabit sürümü"):
        L.check_hat_and_sabit(m, "act_v1", "abc", 2)
    with pytest.raises(L.DevirHata, match="manifest özeti"):
        L.check_hat_and_sabit(m, "act_v1", "zzz", 1)


def test_credit_math():
    s = {"billed_cost": "0E-8", "adjustments": {"credits": "-0.75000000"}}
    assert L.credit_used(s) == pytest.approx(0.75)
    assert L.credit_used({"billed_cost": "1.5", "adjustments": {"credits": "-30"}}) == pytest.approx(31.5)
    assert L.available_budget(0.75) == pytest.approx(24.25)
    assert L.available_budget(29) == 0.0
    # 24.25 USD, ACT: (24.25-0.05-0.10)/0.05*1000 = 482000 adım
    assert L.max_steps(CFG, 24.25) == 482000
    assert L.max_steps(CFG, 0.1) == 0


def test_steps_limit_rejected_with_max_in_message():
    with pytest.raises(L.DevirHata, match="azami adım: 0"):
        L.check_steps_allowed(CFG, 100, spent_usd=25.0, max_timeout_min=90)
    pahali = {**CFG, "maliyet": {**CFG["maliyet"], "usd_per_1000_steps": 0.5}}      # 0,5 USD/1000 adım
    # 24 USD harcandı -> 1 USD kullanılabilir; (1-0.05-0.10)/0.5*1000 = 1700 adım
    with pytest.raises(L.DevirHata, match="azami adım: 1700"):
        L.check_steps_allowed(pahali, 5000, spent_usd=24.0, max_timeout_min=90)
    assert L.check_steps_allowed(pahali, 1700, spent_usd=24.0, max_timeout_min=90) >= 1


def test_timeout_scales_and_max_timeout_enforced():
    assert L.timeout_minutes(CFG, 100) == 6                                   # 5 + 100/9.5/60*1.3 ≈ 5.2 -> 6
    assert L.timeout_minutes(CFG, 100000) > L.timeout_minutes(CFG, 1000)
    assert L.check_steps_allowed(CFG, 100, 0.75, 90) == 6
    with pytest.raises(L.DevirHata, match="max-timeout-min"):
        L.check_steps_allowed(CFG, 100000, 0.75, max_timeout_min=60)
    with pytest.raises(L.DevirHata):
        L.check_steps_allowed(CFG, 0, 0.75, 90)


def test_cost_model_requires_keys():
    with pytest.raises(L.DevirHata):
        L.cost_model({"policy": "act"})


def test_hardlinked_files_are_written_as_regular_members(tmp_path):
    import os

    ham, act = tmp_path / "ham", tmp_path / "act"
    for r in (ham, act):
        (r / "meta").mkdir(parents=True)
    (ham / "meta" / "info.json").write_text("{}")
    (ham / "mahut_episodes.jsonl").write_text("{}\n")
    os.link(ham / "mahut_episodes.jsonl", act / "mahut_episodes.jsonl")  # eski make_act_dataset'in bıraktığı bağ
    os.link(ham / "meta" / "info.json", act / "meta" / "info.json")
    L.build_sabit_tar(tmp_path / "s.tar", 1, ham, act, "a_s1", "a_act_s1")
    with tarfile.open(tmp_path / "s.tar") as tf:
        assert all(m.isreg() for m in tf)


def test_failed_build_leaves_no_final_file(tmp_path):
    bad = tmp_path / "ham"
    (bad / "meta").mkdir(parents=True)                      # info.json yok
    with pytest.raises(L.DevirHata):
        L.build_sabit_tar(tmp_path / "s.tar", 1, bad, None, "a_s1", None)
    assert not (tmp_path / "s.tar").exists()
