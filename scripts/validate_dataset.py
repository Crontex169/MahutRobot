#!/usr/bin/env python3
"""
VLA veri kümesi doğrulama ve kapsama raporu (LeRobot v3.0 + mahut_episodes.jsonl).

Kontroller:
  1. Şema: özellikler/şekiller/fps beklenen VLA şemasıyla aynı mı
  2. Kare sayıları: meta bölüm uzunlukları = parquet satırları = sidecar num_frames; indeksler ardışık
  3. NaN/Inf: observation.state ve action
  4. Video okunabilirliği: örneklenen (ya da tüm) bölümlerin ilk/orta/son karesi çözülebiliyor mu
  5. Sidecar tutarlılığı: satır sayısı, episode_index sırası, talimat = bölüm görevi,
     slotlar = komut = parquet'teki command.slots, komut geçerli (TOP+prizma yok)
  6. Kapsama: (manipüle nesne × yön) ve (referans nesne × yön) tabloları (toplam ve kaynağa göre)
  7. Held-out sızıntısı: held-out listesindeki komutlar veri kümesinde var mı

Kullanım:
    ~/.venvs/lerobot/bin/python scripts/validate_dataset.py [--root ~/vla_data/mahut_multishape]
        [--video-check sample|all|none] [--video-samples 20] [--allow-heldout] [--json rapor.json]
Çıkış kodu: 0 = sorun yok, 1 = hata var.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.commands import Command, load_heldout, to_slots  # noqa: E402
from mahut_kol.envs.multi_shape_task import OBJECT_IDS, RELATIONS  # noqa: E402
from mahut_kol.recording import DEFAULT_REPO_ID, DEFAULT_ROOT, IMAGE_KEYS, SIDECAR_NAME, SLOTS_KEY, check_schema, read_sidecar  # noqa: E402


class Report:
    def __init__(self):
        self.errors, self.warnings, self.info = [], [], {}

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def _read_parquets(pattern):
    import pandas as pd

    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def coverage_table(rows, field):
    """{(nesne, yön): sayı}"""
    c = Counter()
    for r in rows:
        cmd = r["command"]
        c[(cmd[field], cmd["relation"])] += 1
    return c


def format_table(counter, title, valid_cells):
    lines = [title, f"{'':24s}" + "".join(f"{r:>7s}" for r in RELATIONS)]
    for o in OBJECT_IDS:
        cells = []
        for rel in RELATIONS:
            if (o, rel) not in valid_cells:
                cells.append(f"{'-':>7s}")
            else:
                cells.append(f"{counter.get((o, rel), 0):7d}")
        lines.append(f"{o:24s}" + "".join(cells))
    vals = [counter.get(c, 0) for c in valid_cells]
    if vals:
        lines.append(f"  en az {min(vals)}, en çok {max(vals)}, boş hücre {sum(v == 0 for v in vals)}/{len(vals)}")
    return "\n".join(lines)


def validate(root: Path, repo_id: str, video_check: str, video_samples: int, allow_heldout: bool) -> Report:
    rep = Report()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        rep.err(f"meta/info.json yok: {root}")
        return rep
    info = json.loads(info_path.read_text(encoding="utf-8"))

    # 1. Şema
    for p in check_schema(info.get("features", {}), info.get("fps", -1)):
        rep.err(f"şema: {p}")

    # 2. Kare sayıları
    eps = _read_parquets(str(root / "meta" / "episodes" / "*" / "*.parquet"))
    data = _read_parquets(str(root / "data" / "*" / "*.parquet"))
    if eps is None or data is None:
        rep.err("bölüm meta verisi ya da veri parquet'i yok")
        return rep
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    n_eps = len(eps)
    rep.info["episodes"] = n_eps
    rep.info["frames"] = int(len(data))
    if info.get("total_episodes") != n_eps:
        rep.err(f"info.total_episodes={info.get('total_episodes')} != meta bölüm {n_eps}")
    if info.get("total_frames") != len(data):
        rep.err(f"info.total_frames={info.get('total_frames')} != parquet satır {len(data)}")
    if list(eps["episode_index"]) != list(range(n_eps)):
        rep.err("bölüm indeksleri ardışık değil")
    per_ep = data.groupby("episode_index").size()
    for _, e in eps.iterrows():
        ei = int(e["episode_index"])
        n = int(per_ep.get(ei, 0))
        if n != int(e["length"]):
            rep.err(f"bölüm {ei}: meta uzunluk {e['length']} != parquet {n}")
        fi = data.loc[data["episode_index"] == ei, "frame_index"].to_numpy()
        if not np.array_equal(np.sort(fi), np.arange(n)):
            rep.err(f"bölüm {ei}: frame_index ardışık değil")
    if not np.array_equal(np.sort(data["index"].to_numpy()), np.arange(len(data))):
        rep.err("global index ardışık değil")

    # 3. NaN/Inf
    for col in ("observation.state", "action"):
        arr = np.stack(data[col].to_numpy()).astype(np.float64)
        bad = ~np.isfinite(arr)
        if bad.any():
            rows = np.unique(data.loc[bad.any(axis=1), "episode_index"])
            rep.err(f"{col}: {int(bad.sum())} NaN/Inf değer (bölümler {rows[:10].tolist()})")
        if arr.shape[1] != 7:
            rep.err(f"{col} boyutu {arr.shape[1]} != 7")

    # 5. Sidecar
    rows = read_sidecar(root / SIDECAR_NAME)
    rep.info["sidecar_rows"] = len(rows)
    if len(rows) != n_eps:
        rep.err(f"sidecar satırı {len(rows)} != bölüm {n_eps}")
    tasks = _read_parquets(str(root / "meta" / "tasks.parquet"))
    slots_by_ep = {}
    if SLOTS_KEY in data.columns:
        for ei, g in data.groupby("episode_index"):
            s = np.stack(g[SLOTS_KEY].to_numpy())
            if not (s == s[0]).all():
                rep.err(f"bölüm {ei}: command.slots bölüm içinde değişiyor")
            slots_by_ep[int(ei)] = s[0].tolist()
    ep_tasks = {int(e["episode_index"]): list(e["tasks"]) for _, e in eps.iterrows()}
    for i, r in enumerate(rows):
        ei = r.get("episode_index")
        if ei != i:
            rep.err(f"sidecar satır {i}: episode_index={ei} (beklenen {i})")
            continue
        try:
            cmd = Command(r["command"]["manipulated_object"], r["command"]["reference_object"], r["command"]["relation"])
        except (KeyError, ValueError) as e:
            rep.err(f"sidecar bölüm {ei}: geçersiz komut ({e})")
            continue
        if i < n_eps and r.get("num_frames") != int(eps.loc[i, "length"]):
            rep.err(f"sidecar bölüm {ei}: num_frames {r.get('num_frames')} != {int(eps.loc[i, 'length'])}")
        if ep_tasks.get(ei) != [r.get("instruction")]:
            rep.err(f"sidecar bölüm {ei}: talimat '{r.get('instruction')}' != görev {ep_tasks.get(ei)}")
        if to_slots(cmd).tolist() != r.get("slots"):
            rep.err(f"sidecar bölüm {ei}: slotlar komutla uyuşmuyor")
        if slots_by_ep.get(ei) is not None and slots_by_ep[ei] != r.get("slots"):
            rep.err(f"bölüm {ei}: parquet command.slots {slots_by_ep[ei]} != sidecar {r.get('slots')}")
        if r.get("source") not in ("teleop", "scripted"):
            rep.warn(f"sidecar bölüm {ei}: bilinmeyen source '{r.get('source')}'")
        if not r.get("success", False):
            rep.warn(f"sidecar bölüm {ei}: success=False")
    if tasks is not None:
        rep.info["unique_instructions"] = int(len(tasks))

    # 4. Video okunabilirliği
    if video_check != "none" and n_eps:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        try:
            ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
        except Exception as e:  # noqa: BLE001
            rep.err(f"LeRobotDataset açılamadı: {e}")
            ds = None
        if ds is not None:
            ep_ids = list(range(n_eps))
            if video_check == "sample" and n_eps > video_samples:
                ep_ids = sorted(np.random.default_rng(0).choice(n_eps, video_samples, replace=False).tolist())
            checked = 0
            for ei in ep_ids:
                a, b = int(eps.loc[ei, "dataset_from_index"]), int(eps.loc[ei, "dataset_to_index"])
                for idx in sorted({a, (a + b) // 2, b - 1}):
                    try:
                        item = ds[idx]
                        for k in IMAGE_KEYS:
                            img = item[k]
                            if tuple(img.shape) != (3, 224, 224) or not np.isfinite(img.numpy()).all():
                                rep.err(f"bölüm {ei} kare {idx}: {k} şekil {tuple(img.shape)}")
                            elif float(img.std()) < 1e-3:
                                rep.warn(f"bölüm {ei} kare {idx}: {k} düz renk görüntü")
                    except Exception as e:  # noqa: BLE001
                        rep.err(f"bölüm {ei} kare {idx}: video okunamadı ({type(e).__name__}: {e})")
                checked += 1
            rep.info["video_episodes_checked"] = checked

    # 6. Kapsama
    valid_m = {(m, rel) for m in OBJECT_IDS for rel in RELATIONS}
    valid_r = {(r, rel) for r in OBJECT_IDS for rel in RELATIONS if not (rel == "TOP" and r.endswith("triangular_prism"))}
    cov = {}
    for src in ["all"] + sorted({r.get("source", "?") for r in rows}):
        sub = rows if src == "all" else [r for r in rows if r.get("source") == src]
        cov[src] = (coverage_table(sub, "manipulated_object"), coverage_table(sub, "reference_object"), len(sub))
    rep.info["coverage"] = cov
    rep.info["valid_cells"] = (valid_m, valid_r)
    rep.info["sources"] = dict(Counter(r.get("source", "?") for r in rows))

    # 7. Held-out sızıntısı
    held = {c.key for c in load_heldout()}
    leaked = defaultdict(list)
    for r in rows:
        c = r["command"]
        key = f"{c['manipulated_object']}|{c['reference_object']}|{c['relation']}"
        if key in held:
            leaked[key].append(r["episode_index"])
        if bool(r.get("heldout")) != (key in held):
            rep.err(f"bölüm {r['episode_index']}: heldout bayrağı {r.get('heldout')} ama listede {key in held}")
    rep.info["heldout_leak"] = dict(leaked)
    if leaked:
        msg = f"held-out komutları veri kümesinde: {len(leaked)} komut, {sum(map(len, leaked.values()))} bölüm"
        (rep.warn if allow_heldout else rep.err)(msg)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--video-check", choices=["sample", "all", "none"], default="sample")
    ap.add_argument("--video-samples", type=int, default=20)
    ap.add_argument("--allow-heldout", action="store_true", help="held-out sızıntısını hata değil uyarı say")
    ap.add_argument("--json", default=None, help="raporu JSON olarak da yaz")
    args = ap.parse_args(argv)
    root = Path(args.root).expanduser()

    rep = validate(root, args.repo_id, args.video_check, args.video_samples, args.allow_heldout)
    print(f"📁 {root}")
    print(f"   bölüm {rep.info.get('episodes')}, kare {rep.info.get('frames')}, sidecar {rep.info.get('sidecar_rows')}, "
          f"kaynak {rep.info.get('sources')}, farklı talimat {rep.info.get('unique_instructions')}, "
          f"video kontrol edilen bölüm {rep.info.get('video_episodes_checked', 0)}")
    if "coverage" in rep.info:
        vm, vr = rep.info["valid_cells"]
        for src, (cm, cr, n) in rep.info["coverage"].items():
            print(f"\n=== Kapsama ({src}, {n} bölüm) ===")
            print(format_table(cm, "(manipüle nesne × yön)", vm))
            print(format_table(cr, "(referans nesne × yön)", vr))
    print("\n=== Held-out sızıntısı ===")
    print("   yok" if not rep.info.get("heldout_leak") else f"   {rep.info['heldout_leak']}")
    for w in rep.warnings:
        print(f"⚠️  {w}")
    for e in rep.errors:
        print(f"❌ {e}")
    print("\n" + ("✅ Doğrulama başarılı" if not rep.errors else f"❌ {len(rep.errors)} hata"))
    if args.json:
        out = {
            "root": str(root), "errors": rep.errors, "warnings": rep.warnings,
            **{k: v for k, v in rep.info.items() if k not in ("coverage", "valid_cells")},
            "coverage": {
                src: {"manipulated_x_relation": {f"{o}|{r}": v for (o, r), v in cm.items()},
                      "reference_x_relation": {f"{o}|{r}": v for (o, r), v in cr.items()}, "episodes": n}
                for src, (cm, cr, n) in rep.info.get("coverage", {}).items()
            },
        }
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not rep.errors else 1


if __name__ == "__main__":
    sys.exit(main())
