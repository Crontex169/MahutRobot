#!/usr/bin/env python3
"""
Eğitim giriş noktası: YAML/JSON koşu ayarından `lerobot-train` komutunu kurar, koşu
klasörünü yönetir ve kesilen koşuyu AYNI koşu adıyla checkpoint'ten sürdürür.

Hem Modal konteynerinde (cloud/modal_app.py çağırır) hem yerelde hata ayıklama için çalışır;
Modal'a bağımlı değildir.

Koşu klasörü düzeni (<out_root>/<run_name>/):
    launch.json          koşuyu başlatan launch_id (otomatik yeniden başlatmayı tanımak için)
    train/               lerobot çıktı klasörü (deneme 1)   | checkpoints/<adım>/pretrained_model, last
    train_r2/, ...       checkpoint'siz kesilen denemeden sonra açılan yeni deneme klasörleri
    attempts.jsonl       her denemenin süresi, adım/sn, çıkış kodu
    train.log            tüm denemelerin birleşik logu

Kurallar (veri/çıktı asla silinmez veya ezilmez):
    * Koşu klasörü yoksa: yeni koşu.
    * Koşu klasörü varsa ve launch_id aynıysa (Modal yeniden başlatması: preemption/timeout)
      ya da --resume verildiyse: son checkpoint'ten sürdür.
    * Aksi halde HATA (üzerine yazılmaz).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOG_RE = re.compile(r"step:(?P<step>\S+).*?loss:(?P<loss>\S+).*?updt_s:(?P<updt>\S+)\s+data_s:(?P<data>\S+)")


def _human_to_int(s: str) -> int:
    """lerobot adımları '1K', '20K' gibi kısaltır."""
    s = s.strip()
    mult = {"K": 1_000, "M": 1_000_000}.get(s[-1:].upper(), 1)
    return int(float(s[:-1] if mult != 1 else s) * mult)


def load_config(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


REQUIRED_STATE_FILES = ("training_step.json", "rng_state.safetensors", "optimizer_state.safetensors",
                        "optimizer_param_groups.json")


def checkpoint_is_complete(ck: Path) -> bool:
    """Checkpoint yazılırken kesilmişse (yarım) geçersiz say.

    Gerekli: pretrained_model/{model.safetensors, train_config.json} ve training_state/ içinde
    training_step.json, rng_state.safetensors, optimizer_state.safetensors, optimizer_param_groups.json;
    ayrıca eğitimde zamanlayıcı kullanılıyorsa (train_config.json'da `scheduler` boş değil)
    scheduler_state.json. (ACT zamanlayıcı kullanmaz, bu dosya onda yoktur.) Dosyalar boş olamaz.
    """
    pm, st = ck / "pretrained_model", ck / "training_state"
    cfg = pm / "train_config.json"
    needed = [pm / "model.safetensors", cfg] + [st / f for f in REQUIRED_STATE_FILES]
    try:
        if not all(p.is_file() and p.stat().st_size > 0 for p in needed):
            return False
        if json.loads(cfg.read_text()).get("scheduler"):
            sched = st / "scheduler_state.json"
            return sched.is_file() and sched.stat().st_size > 0
        return True
    except (OSError, ValueError):
        return False


def last_checkpoint_config(run_dir: Path) -> Path | None:
    """Tüm denemelerdeki TAM checkpoint'lerden en büyük adımlısının train_config.json'u.

    `checkpoints/last` sembolik bağına güvenilmez (Volume'da sembolik bağ garanti değil);
    sayısal adlı checkpoint klasörleri taranır, yarım olanlar atlanıp bir öncekine düşülür.
    """
    best = None  # (adım, deneme mtime, yol)
    for d in run_dir.glob("train*"):
        for ck in (d / "checkpoints").glob("[0-9]*"):
            if checkpoint_is_complete(ck):
                key = (int(ck.name), d.stat().st_mtime, ck / "pretrained_model" / "train_config.json")
                best = max(best, key) if best else key
    return best[2] if best else None


def next_attempt_dir(run_dir: Path) -> Path:
    if not (run_dir / "train").exists():
        return run_dir / "train"
    n = 2
    while (run_dir / f"train_r{n}").exists():
        n += 1
    return run_dir / f"train_r{n}"


def build_argv(cfg: dict, data_root: Path, out_dir: Path, run_name: str, resume_cfg: Path | None) -> list[str]:
    """`lerobot-train` argümanları."""
    if resume_cfg is not None:
        # Sürdürürken yapılandırma checkpoint'ten gelir; yalnızca (uzatılabilen) adım sayısı ve
        # kaydetme/log sıklığı CLI ile geçersiz kılınır.
        return [
            f"--config_path={resume_cfg}", "--resume=true", f"--steps={cfg['steps']}",
            f"--save_freq={cfg['save_freq']}", f"--log_freq={cfg['log_freq']}",
        ]
    pa = dict(cfg.get("policy_args") or {})
    argv = []
    if cfg["policy"] == "smolvla":
        argv.append(f"--policy.path={pa.pop('path', 'lerobot/smolvla_base')}")
    else:
        argv.append(f"--policy.type={cfg['policy']}")
    argv += ["--policy.push_to_hub=false", "--policy.device=cuda"]
    argv += [f"--policy.{k}={json.dumps(v) if isinstance(v, (list, dict, bool)) else v}" for k, v in pa.items()]
    if cfg.get("rename_map"):
        argv.append(f"--rename_map={json.dumps(cfg['rename_map'])}")
    argv += [
        f"--dataset.repo_id={cfg['repo_id']}", f"--dataset.root={data_root}",
        f"--dataset.video_backend={cfg.get('video_backend', 'pyav')}",
    ]
    if cfg.get("episodes") is not None:
        argv.append(f"--dataset.episodes={json.dumps(list(cfg['episodes']))}")
    argv += [
        f"--batch_size={cfg['batch_size']}", f"--steps={cfg['steps']}", f"--save_freq={cfg['save_freq']}",
        f"--log_freq={cfg['log_freq']}", f"--num_workers={cfg['num_workers']}", f"--seed={cfg.get('seed', 1000)}",
        "--eval_freq=0", f"--wandb.enable={'true' if cfg.get('wandb') else 'false'}",
        f"--output_dir={out_dir}", f"--job_name={run_name}",
    ]
    return argv


def prepare_run(out_root: Path, run_name: str, launch_id: str, resume: bool) -> tuple[Path, Path | None]:
    """(deneme çıktı klasörü, sürdürülecek train_config.json | None). Var olanı ezmez."""
    run_dir = out_root / run_name
    lj = run_dir / "launch.json"
    if run_dir.exists():
        prev = json.loads(lj.read_text()).get("launch_id") if lj.exists() else None
        auto_restart = prev is not None and prev == launch_id
        if not (auto_restart or resume):
            raise FileExistsError(
                f"Koşu '{run_name}' zaten var ({run_dir}); üzerine yazılmaz. Yeni bir koşu adı verin "
                "ya da aynı koşuyu sürdürmek için --resume kullanın."
            )
        if not auto_restart:  # kullanıcı sürdürüyor: yeni launch_id'yi geçmişe ekle
            with open(run_dir / "launch_history.jsonl", "a") as fh:
                fh.write(json.dumps({"launch_id": launch_id, "t": time.time(), "resume": True}) + "\n")
            lj.write_text(json.dumps({"launch_id": launch_id}))
        resume_cfg = last_checkpoint_config(run_dir)
        if resume_cfg is not None:
            return resume_cfg.parents[3], resume_cfg
        return next_attempt_dir(run_dir), None  # henüz checkpoint yok: yeni deneme klasörü
    run_dir.mkdir(parents=True)
    lj.write_text(json.dumps({"launch_id": launch_id}))
    return run_dir / "train", None


def run_training(argv: list[str], run_dir: Path, on_tick=None, tick_s: float = 30.0) -> dict:
    """lerobot-train alt süreci; çıktıyı train.log'a ekler, adım/sn ölçer, düzenli on_tick() çağırır."""
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *argv]
    t0 = time.time()
    steps, upd, dat, last_step, first_step, losses = 0, 0.0, 0.0, 0, None, []
    log = open(run_dir / "train.log", "a")
    log.write(f"\n===== deneme {time.strftime('%F %T')} =====\n{' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_tick = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        m = LOG_RE.search(line)
        if m:
            try:
                s = _human_to_int(m["step"])
                first_step = s if first_step is None else first_step
                last_step = s
                losses.append((s, float(m["loss"])))
                upd, dat = float(m["updt"]), float(m["data"])
            except ValueError:
                pass
        if on_tick and time.time() - last_tick > tick_s:
            on_tick()
            last_tick = time.time()
    rc = proc.wait()
    log.close()
    if on_tick:
        on_tick()
    wall = time.time() - t0
    return {
        "returncode": rc, "wall_s": round(wall, 1), "first_logged_step": first_step, "last_logged_step": last_step,
        "last_updt_s": upd, "last_data_s": dat,
        "steps_per_s_from_log": round(1.0 / (upd + dat), 3) if (upd + dat) > 0 else None,
        "losses": losses[:: max(1, len(losses) // 40)] + losses[-1:],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="koşu ayarı YAML/JSON")
    ap.add_argument("--run", required=True, help="benzersiz koşu adı")
    ap.add_argument("--data-root", required=True, help="LeRobot veri kümesi kökü")
    ap.add_argument("--out-root", required=True, help="koşuların yazıldığı kök (Volume bağlama noktası/runs)")
    ap.add_argument("--launch-id", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="yalnızca komutu yazdır")
    ap.add_argument("--set", action="append", default=[], metavar="ANAHTAR=DEĞER", help="ayar geçersiz kılma (JSON değer)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    out_root, data_root = Path(a.out_root), Path(a.data_root)
    if not (data_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"veri kümesi yok: {data_root}")
    out_dir, resume_cfg = prepare_run(out_root, a.run, a.launch_id or str(time.time()), a.resume)
    argv_l = build_argv(cfg, data_root, out_dir, a.run, resume_cfg)
    print(f"▶ koşu={a.run} {'SÜRDÜR' if resume_cfg else 'YENİ'} çıktı={out_dir}\n  {' '.join(argv_l)}", flush=True)
    if a.dry_run:
        return 0
    return_info = run_training(argv_l, out_root / a.run)
    return_info.update({"run": a.run, "launch_id": a.launch_id, "resumed": resume_cfg is not None, "out_dir": str(out_dir),
                        "config": cfg, "finished": time.strftime("%FT%T")})
    with open(out_root / a.run / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(return_info, default=str) + "\n")
    print("SONUÇ", json.dumps({k: v for k, v in return_info.items() if k not in ("losses", "config")}), flush=True)
    return return_info["returncode"]


if __name__ == "__main__":
    sys.exit(main())
