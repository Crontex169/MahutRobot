#!/usr/bin/env python3
"""
ACT (dilsiz kıyas) için TÜREV veri kümesi: kaynak veri kümesine hiç dokunmadan,
`command.slots` (5 slot) -> `observation.environment_state` (17 boyutlu one-hot) ekler.

Neden türev veri kümesi (eğitim sarmalayıcısı / lerobot yaması yerine):
  * lerobot 0.4.4 `dataset_to_policy_features` yalnızca `observation.environment_state` anahtarını
    ENV girdisi sayar; ACT bunu dışarıdan hazır bir sütun olarak bekler.
  * lerobot-train'i sarmalamak `make_dataset`/`LeRobotDataset.__getitem__` üzerinde monkeypatch
    gerektirirdi (lerobot sürümüne kırılgan). Türev küme ise standart `lerobot-train` ile
    değişiklik OLMADAN eğitilir; sürümlü ad kaynağın hangi anında türetildiğini belgeler.
  * Maliyet: yalnızca veri parquet dosyaları (7+7+5+17 sayı/kare) yeniden yazılır; videolar
    sabit bağlantıyla (aynı disk) paylaşılır, değilse kopyalanır. Kaynak asla değiştirilmez.

Hedef kök zaten varsa HATA verir (üzerine yazmaz, silmez).

Örnek:
    ~/.venvs/lerobot/bin/python scripts/make_act_dataset.py \
        --src ~/vla_data/mahut_smoke_v1 --dst ~/vla_data/mahut_smoke_v1_act
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.commands.onehot import ENV_STATE_KEY, ONEHOT_DIM, slots_to_onehot  # noqa: E402

SLOTS_KEY = "command.slots"
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _env_stats(x: np.ndarray) -> dict:
    s = {
        "min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(), "count": [int(x.shape[0])],
    }
    for k, q in QUANTILES.items():
        s[k] = np.quantile(x, q, axis=0).tolist()
    return s


def build(src: Path, dst: Path) -> dict:
    src, dst = src.expanduser().resolve(), dst.expanduser()
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{src} bir LeRobot veri kümesi değil (meta/info.json yok)")
    if dst.exists():
        raise FileExistsError(f"{dst} zaten var; üzerine yazılmaz. Yeni bir ad verin (ör. ..._act_v2).")
    info = json.loads((src / "meta" / "info.json").read_text())
    if SLOTS_KEY not in info["features"]:
        raise ValueError(f"{SLOTS_KEY} özelliği yok")
    if ENV_STATE_KEY in info["features"]:
        raise ValueError(f"{ENV_STATE_KEY} kaynakta zaten var")

    dst.mkdir(parents=True)
    linked = copied = 0
    # 1) olduğu gibi taşınanlar: meta (tasks/episodes), videolar, sidecar'lar
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or rel.parts[0] in ("data", "images") or rel == Path("meta/info.json") \
                or rel == Path("meta/stats.json"):
            continue
        kind = _link_or_copy(p, dst / rel)
        linked += kind == "link"
        copied += kind == "copy"

    # 2) data parquet: one-hot sütunu ekle
    all_env = []
    for p in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(p)
        slots = np.asarray(t.column(SLOTS_KEY).to_pylist(), dtype=np.int64)
        env = slots_to_onehot(slots)
        all_env.append(env)
        col = pa.FixedSizeListArray.from_arrays(pa.array(env.reshape(-1), type=pa.float32()), ONEHOT_DIM)
        t = t.append_column(pa.field(ENV_STATE_KEY, col.type), col)
        meta = dict(t.schema.metadata or {})
        if b"huggingface" in meta:  # HF datasets özellik şeması
            hf = json.loads(meta[b"huggingface"])
            hf["info"]["features"][ENV_STATE_KEY] = {
                "feature": {"dtype": "float32", "_type": "Value"}, "length": ONEHOT_DIM, "_type": "List"}
            meta[b"huggingface"] = json.dumps(hf).encode()
            t = t.replace_schema_metadata(meta)
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    # 3) info.json + stats.json
    (dst / "meta").mkdir(exist_ok=True)
    names = ["manipulated_color_" + c for c in ("red", "green", "blue")] \
        + ["manipulated_shape_" + c for c in ("cube", "cylinder", "triangular_prism")] \
        + ["reference_color_" + c for c in ("red", "green", "blue")] \
        + ["reference_shape_" + c for c in ("cube", "cylinder", "triangular_prism")] \
        + ["relation_" + c for c in ("FRONT", "BACK", "LEFT", "RIGHT", "TOP")]
    info["features"][ENV_STATE_KEY] = {"dtype": "float32", "shape": [ONEHOT_DIM], "names": names}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    stats = json.loads((src / "meta" / "stats.json").read_text())
    stats[ENV_STATE_KEY] = _env_stats(np.concatenate(all_env))
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    (dst / "DERIVED_FROM.json").write_text(json.dumps(
        {"source": str(src), "added": ENV_STATE_KEY, "from": SLOTS_KEY, "frames": int(sum(len(e) for e in all_env))},
        indent=2))
    return {"dst": str(dst), "frames": int(sum(len(e) for e in all_env)), "linked": linked, "copied": copied}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    print(build(Path(a.src), Path(a.dst)))


if __name__ == "__main__":
    main()
