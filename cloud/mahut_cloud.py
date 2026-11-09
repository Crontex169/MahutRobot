#!/usr/bin/env python3
"""
Modal sarmalayıcısı: veri yükleme, eğitim başlatma, checkpoint indirme.
(Yerel venv: ~/.venvs/lerobot/bin/python; Modal kimlik doğrulaması `modal setup` ile KULLANICIDA.)

    python cloud/mahut_cloud.py check                       # Modal profili + Volume'lar
    python cloud/mahut_cloud.py upload --src ~/vla_data/mahut_smoke_v1 --name mahut_smoke_v1_r1
    python cloud/mahut_cloud.py train  --config configs/training/act.yaml --run act_duman_01 \
           [--dataset mahut_smoke_v1_r1_act] [--steps 300] [--episodes 0-9] [--gpu L4] [--resume]
    python cloud/mahut_cloud.py download --run act_duman_01 [--step 300]
    python cloud/mahut_cloud.py ls [--run act_duman_01]

Silme/üzerine yazma yoktur: var olan veri adı veya koşu adı HATA verir (koşuyu sürdürmek için --resume).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY_BIN = Path(sys.executable).parent
DATA_VOL, RUNS_VOL = "mahut-datasets", "mahut-runs"
UPLOAD_DIRS = ("meta", "data", "videos")           # images/ (geçici kare PNG'leri) yüklenmez
UPLOAD_FILES = ("mahut_episodes.jsonl", "DERIVED_FROM.json")


def modal_cli() -> str:
    exe = PY_BIN / "modal"
    return str(exe) if exe.exists() else "modal"


def volume():
    import modal

    return modal


def listdir(vol_name: str, path: str):
    import modal

    vol = modal.Volume.from_name(vol_name, create_if_missing=True)
    try:
        return [e.path for e in vol.listdir(path)]
    except Exception:  # noqa: BLE001  (yol yok)
        return []


def parse_episodes(s: str | None):
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_check(a):
    r = subprocess.run([modal_cli(), "profile", "current"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("Modal profili yok. Lütfen kendiniz çalıştırın:  modal setup")
    print("Modal profili:", r.stdout.strip())
    for v in (DATA_VOL, RUNS_VOL):
        print(v, "kökü:", listdir(v, "/"))


def cmd_upload(a):
    import modal

    src = Path(a.src).expanduser()
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"{src} LeRobot veri kümesi değil")
    if listdir(DATA_VOL, f"/{a.name}"):
        sys.exit(f"Volume '{DATA_VOL}' içinde '{a.name}' zaten var; üzerine yazılmaz. Yeni sürümlü ad verin (ör. ..._r2).")
    vol = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
    n = 0
    with vol.batch_upload(force=False) as b:
        for d in UPLOAD_DIRS:
            if (src / d).exists():
                b.put_directory(str(src / d), f"/{a.name}/{d}")
        for f in UPLOAD_FILES:
            if (src / f).exists():
                b.put_file(str(src / f), f"/{a.name}/{f}")
    print(f"✅ yüklendi: {DATA_VOL}:/{a.name}  ({', '.join(e for e in listdir(DATA_VOL, f'/{a.name}'))})")
    n += 1


def cmd_train(a):
    cfg = yaml.safe_load(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    for k in ("dataset", "steps", "batch_size", "gpu", "save_freq", "timeout_min"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.episodes is not None:
        cfg["episodes"] = parse_episodes(a.episodes)
    if a.repo_id:
        cfg["repo_id"] = a.repo_id
    if cfg["timeout_min"] > a.max_timeout_min:
        sys.exit(f"timeout_min={cfg['timeout_min']} > --max-timeout-min={a.max_timeout_min}. "
                 "Uzun koşu için bilerek --max-timeout-min yükseltin (kullanıcı onayı gerekir).")
    if not listdir(DATA_VOL, f"/{cfg['dataset']}"):
        sys.exit(f"Volume '{DATA_VOL}' içinde '{cfg['dataset']}' yok (önce `upload`).")
    exists = bool(listdir(RUNS_VOL, f"/runs/{a.run}"))
    if exists and not a.resume:
        sys.exit(f"Koşu '{a.run}' zaten var; üzerine yazılmaz. Yeni ad verin ya da --resume kullanın.")
    if a.resume and not exists:
        sys.exit(f"--resume verildi ama koşu '{a.run}' yok.")
    env = dict(os.environ, MAHUT_GPU=str(cfg["gpu"]), MAHUT_TIMEOUT_MIN=str(cfg["timeout_min"]),
               MAHUT_MAX_RETRIES=str(cfg.get("max_retries", 2)))
    cmd = [modal_cli(), "run"] + ([] if a.no_detach else ["--detach"]) + [
        f"{HERE / 'modal_app.py'}::train_cli",
        "--config-json", json.dumps(cfg), "--run", a.run, "--launch-id", uuid.uuid4().hex,
    ] + (["--resume"] if a.resume else [])
    print(f"▶ {a.run}: gpu={cfg['gpu']} timeout={cfg['timeout_min']}dk adım={cfg['steps']} veri={cfg['dataset']}")
    sys.exit(subprocess.run(cmd, env=env).returncode)


def cmd_download(a):
    dest = Path(a.dest).expanduser() / a.run
    if dest.exists():
        sys.exit(f"{dest} zaten var; üzerine yazılmaz. Başka --dest verin.")
    remote = f"/runs/{a.run}"
    attempts = [p for p in listdir(RUNS_VOL, remote) if Path(p).name.startswith("train") and "." not in Path(p).name]
    best = None
    for att in attempts:
        for ck in listdir(RUNS_VOL, f"{att}/checkpoints"):
            name = Path(ck).name
            if name.isdigit() and (a.step is None or int(name) == a.step):
                best = max(best, (int(name), ck)) if best else (int(name), ck)
    if not best:
        sys.exit(f"{remote} altında checkpoint bulunamadı")
    step, ck = best
    dest.mkdir(parents=True)
    out = dest / "checkpoints" / f"{step:06d}"
    out.parent.mkdir(parents=True)
    subs = ["pretrained_model"] + (["training_state"] if a.with_training_state else [])
    import modal

    vol = modal.Volume.from_name(RUNS_VOL)
    for sub in subs:
        n = 0
        for e in vol.listdir(f"{ck}/{sub}", recursive=True):
            if e.type.name != "FILE":
                continue
            rel = Path(e.path).relative_to(Path(ck) / sub)
            target = out / sub / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)
            n += 1
        print(f"  {sub}: {n} dosya")
    for f in ("train.log", "attempts.jsonl", "launch_history.jsonl"):
        subprocess.run([modal_cli(), "volume", "get", RUNS_VOL, f"{remote}/{f}", str(dest / f)], capture_output=True)
    (dest / "checkpoints" / "last").symlink_to(out.name)
    print(f"✅ {dest}\n   değerlendirme: python scripts/eval_policy.py --checkpoint {dest}/checkpoints/last/pretrained_model")


def cmd_ls(a):
    print("\n".join(listdir(RUNS_VOL, f"/runs/{a.run}" if a.run else "/runs") or ["(boş)"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    u = sub.add_parser("upload")
    u.add_argument("--src", required=True)
    u.add_argument("--name", required=True, help="Volume'daki sürümlü ad")
    u.set_defaults(fn=cmd_upload)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--run", required=True, help="benzersiz koşu adı")
    t.add_argument("--dataset"), t.add_argument("--repo-id")
    t.add_argument("--steps", type=int), t.add_argument("--batch-size", type=int, dest="batch_size")
    t.add_argument("--save-freq", type=int, dest="save_freq"), t.add_argument("--gpu")
    t.add_argument("--timeout-min", type=int, dest="timeout_min")
    t.add_argument("--episodes", help="ör. 0-9 ya da 0,3,5")
    t.add_argument("--set", action="append", default=[], metavar="ANAHTAR=JSON")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--no-detach", action="store_true")
    t.add_argument("--max-timeout-min", type=int, default=30, dest="max_timeout_min")
    t.set_defaults(fn=cmd_train)
    d = sub.add_parser("download")
    d.add_argument("--run", required=True)
    d.add_argument("--step", type=int, help="varsayılan: en son checkpoint")
    d.add_argument("--dest", default="~/vla_runs")
    d.add_argument("--with-training-state", action="store_true")
    d.set_defaults(fn=cmd_download)
    l = sub.add_parser("ls")
    l.add_argument("--run")
    l.set_defaults(fn=cmd_ls)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
