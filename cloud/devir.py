#!/usr/bin/env python3
"""
Eğitim devri: bir eğitimi proje ekibinden kişiler sırayla sürdürür. HERKES KENDİ Modal hesabını
kendi bilgisayarında kullanır (`modal setup` kişinin kendi işi). Model Google Drive'da durur,
dosyalar tarayıcıdan elle yüklenip indirilir.

Yalnızca `modal` + `pyyaml` + standart kütüphane gerekir (lerobot/torch/GPU gerekmez).

    python cloud/devir.py durum --hat act_v1
    python cloud/devir.py kur                                   # sabit tar -> kendi Volume'unuz (bir kez)
    python cloud/devir.py basla  --hat act_v1 --config configs/training/act.yaml --kisi deniz --adim 5000
    python cloud/devir.py paketle --hat act_v1 --kisi deniz     # -> Data_Model/iterasyonlar/act_v1/iter_....tar
    python cloud/devir.py devam  --hat act_v1 --kisi ali --adim 5000
    python cloud/devir.py sabit-olustur --surum 1 --veri ~/vla_data/mahut_smoke_v1 [--act-veri ...]
    python cloud/devir.py temizle --hat act_v1 [--onayla]       # paketlenmiş ESKİ checkpoint'leri kendi Volume'unuzdan siler

Hiçbir komut token/e-posta saklamaz ya da yazmaz; profil değiştirme/çoklu hesap özelliği YOKTUR.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import devir_lib as L  # noqa: E402
import train_entry  # noqa: E402

DM = REPO / "Data_Model"
SABIT_DIR = DM / "sabit"
ITER_DIR = DM / "iterasyonlar"
STATE_DIR = DM / ".devir_state"          # yerel, git'e girmez; kişi/launch bilgisi (kimlik içermez)
DEFAULT_MAX_TIMEOUT_MIN = 90


def die(msg: str):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------------------------
# Modal yardımcıları
# ----------------------------------------------------------------------------------------------


def modal_exe() -> str:
    exe = Path(sys.executable).parent / "modal"
    return str(exe) if exe.exists() else "modal"


def profile_ok() -> bool:
    """Modal profili var mı? Kimlik/ad GÖSTERİLMEZ."""
    return subprocess.run([modal_exe(), "profile", "current"], capture_output=True, text=True).returncode == 0


def require_profile():
    if not profile_ok():
        die("Modal profili yok. Kendi bilgisayarınızda `modal setup` çalıştırıp kendi hesabınızla giriş yapın.")


def spent_now() -> float:
    """Bu ayki gerçek harcama (USD): `modal billing summary --json` -> kredi + faturalanan."""
    r = subprocess.run([modal_exe(), "billing", "summary", "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        die("Bu ayki harcama okunamadı (modal billing summary). Güvenli tarafta kalmak için durduruldu.")
    try:
        return L.credit_used(json.loads(r.stdout))
    except ValueError:
        die("modal billing summary çıktısı çözümlenemedi.")


def volumes():
    import modal

    d, r = L.volume_names()
    return (modal.Volume.from_name(d, create_if_missing=True), modal.Volume.from_name(r, create_if_missing=True))


def vol_list(vol, path: str) -> list[str]:
    try:
        return [e.path for e in vol.listdir(path)]
    except Exception:  # noqa: BLE001  (yol yok)
        return []


def vol_read(vol, path: str) -> bytes | None:
    try:
        return b"".join(vol.read_file(path))
    except Exception:  # noqa: BLE001
        return None


def remote_call(fn: str, *args):
    sys.path.insert(0, str(HERE))
    import modal_app  # noqa: PLC0415  (yerel içe aktarma: GPU/timeout ortam değişkenleri burada okunur)

    with modal_app.app.run():
        return getattr(modal_app, fn).remote(*args)


def download_dir(vol, remote: str, local: Path) -> int:
    n = 0
    for e in vol.listdir(remote, recursive=True):
        if e.type.name != "FILE":
            continue
        rel = Path(e.path).relative_to(Path(remote))
        target = local / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
        n += 1
    return n


# ----------------------------------------------------------------------------------------------
# Yerel durum
# ----------------------------------------------------------------------------------------------


def sabit_files() -> dict[int, Path]:
    out = {}
    for p in sorted(SABIT_DIR.glob("sabit_v*.tar")) if SABIT_DIR.exists() else []:
        m = L.SABIT_RE.match(p.name)
        if m:
            out[int(m[1])] = p
    return out


def pick_sabit(arg: str | None, surum: int | None = None) -> Path:
    if arg:
        p = Path(arg).expanduser()
        if not p.exists():
            die(f"{p} yok")
        return p
    files = sabit_files()
    if surum is not None:
        if surum not in files:
            die(f"sabit_v{surum}.tar {SABIT_DIR} içinde yok. Drive'dan Data_Model/sabit/ klasörüne indirin.")
        return files[surum]
    if not files:
        die(f"Data_Model/sabit/ içinde sabit_v*.tar yok. Drive'dan indirip {SABIT_DIR} klasörüne koyun.")
    return files[max(files)]


def state_path(hat: str) -> Path:
    return STATE_DIR / f"{hat}.json"


def read_state(hat: str) -> dict | None:
    p = state_path(hat)
    return json.loads(p.read_text()) if p.exists() else None


def write_state(hat: str, st: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(hat).write_text(json.dumps(st, indent=1))


def dataset_for(cfg: dict, sabit_manifest: dict) -> str:
    names = sabit_manifest["veri_kumeleri"]
    if cfg["policy"] == "act":
        if "act" not in names:
            die("Bu sabit pakette ACT türev veri kümesi yok (sabit-olustur --act-veri ...).")
        return names["act"]
    return names["ham"]


def sabit_marker_path(name: str) -> str:
    return f"/{name}/SABIT.json"


def check_sabit_installed(dvol, sabit_manifest: dict, name: str):
    raw = vol_read(dvol, sabit_marker_path(name))
    if raw is None:
        die(f"Veri kümesi '{name}' Volume'unuzda yok. Önce `python cloud/devir.py kur` çalıştırın.")
    marker = json.loads(raw)
    if marker.get("manifest_ozeti") != L.manifest_digest(sabit_manifest):
        die(f"Volume'unuzdaki '{name}' bu sabit tar'dan farklı bir sürümden. (Eski sürüm silinmez; farklı sabit_vN kullanın.)")


# ----------------------------------------------------------------------------------------------
# sabit-olustur / kur
# ----------------------------------------------------------------------------------------------


def cmd_sabit_olustur(a):
    veri = Path(a.veri).expanduser()
    act = Path(a.act_veri).expanduser() if a.act_veri else None
    base = veri.name
    names = L.dataset_names(base, a.surum, act is not None)
    out = SABIT_DIR / L.sabit_name(a.surum)
    if out.exists():
        die(f"{out} zaten var; sabit değişmez. Veri değiştiyse yeni --surum verin.")
    m = L.build_sabit_tar(out, a.surum, veri, act, names["ham"], names.get("act"))
    print(f"✅ {out} ({out.stat().st_size / 1e6:.0f} MB)  bölüm={m['bolum_sayisi']} kare={m['kare_sayisi']} "
          f"veri kümeleri={m['veri_kumeleri']} ortak ACT videosu={len(m['act_video_ham_ile_ortak'])}")


def cmd_kur(a):
    require_profile()
    tar = pick_sabit(a.sabit)
    try:
        man = L.verify_tar(tar, "sabit")
    except L.DevirHata as e:
        die(str(e))
    dvol, _ = volumes()
    digest = L.manifest_digest(man)
    todo = {}
    for kind, name in man["veri_kumeleri"].items():
        raw = vol_read(dvol, sabit_marker_path(name))
        if raw is not None:
            if json.loads(raw).get("manifest_ozeti") == digest:
                print(f"= {name}: Volume'unuzda zaten kurulu (özet aynı), atlandı")
                continue
            die(f"'{name}' Volume'unuzda FARKLI bir sürümle kurulu; üzerine yazılmaz.")
        if vol_list(dvol, f"/{name}"):
            die(f"'{name}' Volume'unuzda işaretsiz (SABIT.json yok) duruyor; üzerine yazılmaz.")
        todo[kind] = name
    if not todo:
        return
    tmp = Path(tempfile.mkdtemp(prefix=".kur_", dir=DM if DM.exists() else None))
    try:
        L.extract_tar(tar, tmp, "sabit")
        for kind, name in todo.items():
            files = {}
            for p in sorted((tmp / kind).rglob("*")):
                if p.is_file():
                    files[p.relative_to(tmp / kind).as_posix()] = p
            if kind == "act":  # ham ile ortak videolar tar'da tek kopya
                for k_act, k_ham in man["act_video_ham_ile_ortak"].items():
                    files[k_act[len("act/"):]] = tmp / k_ham
            with dvol.batch_upload(force=False) as b:
                for rel, p in sorted(files.items()):
                    b.put_file(str(p), f"/{name}/{rel}")
                b.put_file(io.BytesIO(json.dumps({"surum": man["surum"], "manifest_ozeti": digest}).encode()),
                           sabit_marker_path(name))
            print(f"✅ {name}: {len(files)} dosya yüklendi")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # kendi oluşturduğumuz geçici klasör


# ----------------------------------------------------------------------------------------------
# Eğitim başlatma (basla / devam)
# ----------------------------------------------------------------------------------------------


def launch_train(cfg: dict, hat: str, launch_id: str, resume: bool, timeout_min: int) -> int:
    env = dict(os.environ, MAHUT_GPU=str(cfg["gpu"]), MAHUT_TIMEOUT_MIN=str(timeout_min),
               MAHUT_MAX_RETRIES=str(cfg.get("max_retries", 2)))
    cmd = [modal_exe(), "run", "--detach", f"{HERE / 'modal_app.py'}::train_cli",
           "--config-json", json.dumps(cfg), "--run", hat, "--launch-id", launch_id] + (["--resume"] if resume else [])
    print(f"▶ {hat}: politika={cfg['policy']} hedef adım={cfg['steps']} gpu={cfg['gpu']} timeout={timeout_min} dk")
    return subprocess.run(cmd, env=env).returncode


def cmd_basla(a):
    kisi, hat = L.check_kisi(a.kisi), L.check_hat(a.hat)
    require_profile()
    import yaml

    cfg = yaml.safe_load(Path(a.config).read_text())
    if (ITER_DIR / hat).exists() and L.scan_iterations(ITER_DIR / hat):
        die(f"'{hat}' hattında yerel iterasyon dosyası var; ilk tur zaten yapılmış. `devam` kullanın.")
    L.check_git(REPO, None)
    sab = pick_sabit(a.sabit)
    sman = L.verify_tar(sab, "sabit")
    dvol, rvol = volumes()
    cfg["dataset"] = dataset_for(cfg, sman)
    check_sabit_installed(dvol, sman, cfg["dataset"])
    if vol_list(rvol, f"/runs/{hat}"):
        die(f"Volume'unuzda '{hat}' koşusu zaten var (üzerine yazılmaz). Önceki turdan devam için `devam`, aksi halde başka --hat.")
    spent = spent_now()
    timeout = L.check_steps_allowed(cfg, a.adim, spent, a.max_timeout_min)
    cfg.update(steps=a.adim, timeout_min=timeout)
    launch_id = uuid.uuid4().hex
    write_state(hat, {"launch_id": launch_id, "kisi": kisi, "parent": None, "parent_iterasyon": None, "baslangic_adim": 0,
                      "hedef_adim": a.adim, "cfg": cfg, "sabit_surumu": sman["surum"],
                      "sabit_manifest_ozeti": L.manifest_digest(sman), "baslangic": time.strftime("%FT%T")})
    rc = launch_train(cfg, hat, launch_id, False, timeout)
    if rc != 0:
        die(f"Eğitim başarısız (kod {rc}). Ayrıntı yukarıda; kalıcı hatalar yeniden denenmez.")
    print(f"✅ Eğitim bitti. Şimdi: python cloud/devir.py paketle --hat {hat} --kisi {kisi}")


def restore_checkpoint(hat: str, tar: Path, manifest: dict, rvol) -> None:
    """Tar'daki checkpoint'i /runs/<hat>/train/checkpoints/<adım>/ altına geri yükler (aynı yol -> train_config.json
    içindeki mutlak yollar geçerli kalır). Aynı adım varsa özet aynıysa atlanır, farklıysa hata verilir."""
    step = manifest["adim"]
    want = {k: v for k, v in manifest["dosyalar"].items() if k.split("/")[0] in L.ITER_DIRS}
    cur = remote_call("volume_sha", hat, step)
    if cur["bulundu"]:
        if cur["dosyalar"] == want:
            print(f"= adım {step}: Volume'unuzda zaten var (özet aynı), geri yükleme atlandı")
            return
        die(f"Volume'unuzda '{hat}' adım {step} FARKLI içerikle zaten var; üzerine yazılmaz.")
    tmp = Path(tempfile.mkdtemp(prefix=".restore_", dir=DM))
    try:
        L.extract_tar(tar, tmp, "iter")
        base = f"/runs/{hat}/train/checkpoints/{step:06d}"
        with rvol.batch_upload(force=False) as b:
            for rel in sorted(want):
                b.put_file(str(tmp / rel), f"{base}/{rel}")
            if not vol_list(rvol, f"/runs/{hat}/launch.json"):
                b.put_file(io.BytesIO(json.dumps({"launch_id": "restore"}).encode()), f"/runs/{hat}/launch.json")
            for extra in ("train.log", "attempts.jsonl"):
                if (tmp / extra).exists():
                    b.put_file(str(tmp / extra), f"/runs/{hat}/restored/{step:06d}_{extra}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    back = remote_call("volume_sha", hat, step)
    if back["dosyalar"] != want:
        die("Geri yükleme sonrası özet doğrulaması BAŞARISIZ (Volume'daki dosyalar tar ile aynı değil). Eğitim başlatılmadı.")
    print(f"✅ adım {step} checkpoint'i Volume'a geri yüklendi ve özetleri doğrulandı ({len(want)} dosya)")


def cmd_devam(a):
    kisi, hat = L.check_kisi(a.kisi), L.check_hat(a.hat)
    require_profile()
    hat_dir = ITER_DIR / hat
    try:
        parent = L.pick_parent(hat_dir, a.sec)
    except L.DevirHata as e:
        die(str(e))
    if parent is None:
        die(f"{hat_dir} içinde iter_*.tar yok. Drive'dan son iterasyonu indirip bu klasöre koyun (ilk tur için `basla`).")
    try:
        man = L.verify_tar(parent["yol"], "iter")          # sha256 + üye güvenliği
        L.check_git(REPO, man.get("git_commit"))
    except L.DevirHata as e:
        die(str(e))
    sab = pick_sabit(a.sabit, man.get("sabit_surumu"))
    sman = L.verify_tar(sab, "sabit")
    try:
        L.check_hat_and_sabit(man, hat, L.manifest_digest(sman), sman["surum"])
    except L.DevirHata as e:
        die(str(e))
    cfg = dict(man["config"])
    dvol, rvol = volumes()
    check_sabit_installed(dvol, sman, cfg["dataset"])
    diffs = L.compare_versions(man.get("paket_surumleri", {}), remote_call("env_info"))
    if diffs:
        die("Modal imajındaki paket sürümleri manifest ile aynı değil:\n  " + "\n  " .join(diffs)
            + "\nAynı git commit'inde olduğunuzdan emin olun; sürümler değiştiyse sabit sürümü/commit güncellenmeli.")
    cur = man["adim"]
    target = cur + a.adim
    spent = spent_now()
    timeout = L.check_steps_allowed(cfg, a.adim, spent, a.max_timeout_min)
    if target <= cur:
        die("hedef adım mevcut adımdan büyük olmalı")
    restore_checkpoint(hat, parent["yol"], man, rvol)
    cfg.update(steps=target, timeout_min=timeout)
    launch_id = uuid.uuid4().hex
    write_state(hat, {"launch_id": launch_id, "kisi": kisi, "parent": parent["ad"], "parent_iterasyon": man["iterasyon"],
                      "baslangic_adim": cur, "hedef_adim": target, "cfg": cfg, "sabit_surumu": sman["surum"],
                      "sabit_manifest_ozeti": L.manifest_digest(sman), "baslangic": time.strftime("%FT%T")})
    rc = launch_train(cfg, hat, launch_id, True, timeout)
    if rc != 0:
        die(f"Eğitim başarısız (kod {rc}).")
    print(f"✅ Eğitim bitti (adım {cur} → {target}). Şimdi: python cloud/devir.py paketle --hat {hat} --kisi {kisi}")


# ----------------------------------------------------------------------------------------------
# paketle / temizle / durum
# ----------------------------------------------------------------------------------------------


def volume_checkpoints(rvol, hat: str) -> list[tuple[int, str]]:
    """[(adım, uzak_yol)] büyükten küçüğe, tüm deneme klasörlerinden."""
    out = []
    for att in vol_list(rvol, f"/runs/{hat}"):
        if Path(att).name.startswith("train") and "." not in Path(att).name:
            for ck in vol_list(rvol, f"{att}/checkpoints"):
                n = Path(ck).name
                if n.isdigit():
                    out.append((int(n), ck))
    return sorted(out, reverse=True)


def cmd_paketle(a):
    kisi, hat = L.check_kisi(a.kisi), L.check_hat(a.hat)
    require_profile()
    st = read_state(hat)
    if not st:
        die(f"'{hat}' için yerel durum yok; bu bilgisayarda `basla`/`devam` çalıştırılmamış.")
    try:
        L.check_git(REPO, None)
    except L.DevirHata as e:
        die(str(e))
    dvol, rvol = volumes()
    rd = vol_read(rvol, f"/runs/{hat}/round_{st['launch_id']}.json")
    if rd is None:
        die("Bu turun bitiş kaydı (round_*.json) Volume'da yok: eğitim bitmemiş ya da başarısız olmuş. Paketlenmedi.")
    rnd = json.loads(rd)
    cks = volume_checkpoints(rvol, hat)
    if not cks:
        die("Volume'da checkpoint yok")
    tmp = Path(tempfile.mkdtemp(prefix=".paket_", dir=DM))
    try:
        chosen = None
        for step, remote in cks:
            local = tmp / f"{step:06d}"
            download_dir(rvol, f"{remote}/pretrained_model", local / "pretrained_model")
            download_dir(rvol, f"{remote}/training_state", local / "training_state")
            if train_entry.checkpoint_is_complete(local):
                chosen = (step, remote, local)
                break
            print(f"⚠️ adım {step} checkpoint'i eksik/yarım, bir öncekine düşülüyor")
        if chosen is None:
            die("Tam (doğrulanmış) checkpoint bulunamadı.")
        step, remote, local = chosen
        if step <= st["baslangic_adim"]:
            die(f"En yeni tam checkpoint (adım {step}) turun başlangıcından ({st['baslangic_adim']}) ileri değil.")
        extra = {}
        for f in ("train.log", "attempts.jsonl"):
            data = vol_read(rvol, f"/runs/{hat}/{f}")
            if data is not None:
                (tmp / f).write_bytes(data)
                extra[f] = tmp / f
        it = (st["parent_iterasyon"] or 0) + 1
        out = ITER_DIR / hat / L.iter_name(it, step, kisi)
        fields = {
            "hat": hat, "iterasyon": it, "adim": step, "kisi": kisi, "parent": st["parent"],
            "parent_iterasyon": st["parent_iterasyon"], "politika": st["cfg"]["policy"], "git_commit": L.git_head(REPO),
            "config": st["cfg"], "sabit_surumu": st["sabit_surumu"], "sabit_manifest_ozeti": st["sabit_manifest_ozeti"],
            "paket_surumleri": rnd["paket_surumleri"], "gpu": rnd["gpu"], "tur_suresi_s": rnd["tur_suresi_s"],
        }
        try:
            L.build_iter_tar(out, local, extra, fields)
        except L.DevirHata as e:
            die(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"✅ {out} ({out.stat().st_size / 1e6:.0f} MB) — iterasyon {it}, adım {step}, ebeveyn: {st['parent']}")
    print(f"   Bu dosyayı tarayıcıdan Drive'daki Data_Model/iterasyonlar/{hat}/ klasörüne yükleyin.")


def cmd_temizle(a):
    hat = L.check_hat(a.hat)
    require_profile()
    _, rvol = volumes()
    cks = volume_checkpoints(rvol, hat)
    if len(cks) < 2:
        print("Silinecek eski checkpoint yok (en son checkpoint'e dokunulmaz).")
        return
    packaged = {r["adim"] for r in L.scan_iterations(ITER_DIR / hat)}
    for r in L.scan_iterations(ITER_DIR / hat):
        L.verify_tar(r["yol"], "iter")  # yalnızca doğrulanmış paketler "paketlenmiş" sayılır
    latest = cks[0][0]
    old = [(s, p) for s, p in cks[1:] if s in packaged and s != latest]
    skipped = [s for s, _ in cks[1:] if s not in packaged]
    print(f"En son (korunur): adım {latest}. Silinebilir (paketlenmiş+doğrulanmış): {[s for s, _ in old]}. "
          f"Paketlenmediği için korunan: {skipped}")
    if not a.onayla:
        print("Kuru çalıştırma. Gerçekten silmek için --onayla ekleyin (yalnızca KENDİ Volume'unuz; yerel tar ve Drive etkilenmez).")
        return
    for s, p in old:
        rvol.remove_file(p, recursive=True)
        print(f"🗑  silindi: {p}")


def cmd_durum(a):
    hat = L.check_hat(a.hat)
    print(f"Modal profili: {'var' if profile_ok() else 'YOK (kendi bilgisayarınızda `modal setup`)'}")
    sab = sabit_files()
    print(f"sabit (yerel): {[p.name for p in sab.values()] or 'YOK'}")
    its = []
    try:
        its = L.scan_iterations(ITER_DIR / hat)
    except L.DevirHata as e:
        print(f"⚠️ {e}")
    if its:
        last = its[-1]
        print(f"yerel son iterasyon: {last['ad']} (adım {last['adim']})")
    else:
        print("yerel iterasyon: YOK (ilk tur için `basla`)")
    fork = None
    try:
        L.pick_parent(ITER_DIR / hat)
    except L.CatallanmaHatasi as e:
        fork = str(e)
    if fork:
        print(f"⚠️ {fork}")
    nxt = (its[-1]["iterasyon"] + 1) if its else 1
    print(f"sıradaki iterasyon no: {nxt}")
    if profile_ok():
        spent = spent_now()
        print(f"bu ay harcanan (kredi): {spent:.2f} USD / {L.CREDIT_USD:.0f} USD, güvenlik payı {L.CREDIT_MARGIN_USD:.0f} USD")
        import yaml

        for pol in ("act", "smolvla"):
            cfg = yaml.safe_load((REPO / "configs" / "training" / f"{pol}.yaml").read_text())
            print(f"  azami --adim ({pol}): {L.max_steps(cfg, L.available_budget(spent))}")
        if sab and its:
            sman = L.read_manifest(sab[max(sab)], "sabit")
            dvol, _ = volumes()
            name = dataset_for(its[-1]["manifest"]["config"], sman)
            print(f"sabit Volume'unuzda: {'kurulu' if vol_read(dvol, sabit_marker_path(name)) else 'KURULU DEĞİL (`kur`)'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("durum"); s.add_argument("--hat", required=True); s.set_defaults(fn=cmd_durum)
    s = sub.add_parser("sabit-olustur")
    s.add_argument("--surum", type=int, required=True); s.add_argument("--veri", required=True)
    s.add_argument("--act-veri"); s.set_defaults(fn=cmd_sabit_olustur)
    s = sub.add_parser("kur"); s.add_argument("--sabit"); s.set_defaults(fn=cmd_kur)
    s = sub.add_parser("basla")
    s.add_argument("--hat", required=True); s.add_argument("--config", required=True); s.add_argument("--kisi", required=True)
    s.add_argument("--adim", type=int, required=True); s.add_argument("--sabit")
    s.add_argument("--max-timeout-min", type=int, default=DEFAULT_MAX_TIMEOUT_MIN); s.set_defaults(fn=cmd_basla)
    s = sub.add_parser("devam")
    s.add_argument("--hat", required=True); s.add_argument("--kisi", required=True); s.add_argument("--adim", type=int, required=True)
    s.add_argument("--sec", help="çatallanmada devam edilecek iter_*.tar dosyası"); s.add_argument("--sabit")
    s.add_argument("--max-timeout-min", type=int, default=DEFAULT_MAX_TIMEOUT_MIN); s.set_defaults(fn=cmd_devam)
    s = sub.add_parser("paketle"); s.add_argument("--hat", required=True); s.add_argument("--kisi", required=True)
    s.set_defaults(fn=cmd_paketle)
    s = sub.add_parser("temizle"); s.add_argument("--hat", required=True); s.add_argument("--onayla", action="store_true")
    s.set_defaults(fn=cmd_temizle)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except L.DevirHata as e:
        die(str(e))


if __name__ == "__main__":
    main()
