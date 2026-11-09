#!/usr/bin/env python3
"""
Açık çevrim (ezber) kontrolü: checkpoint, eğitim verisinin kareleri üzerinde, eylem parçası
(chunk) tahminini gerçek eylemlerle karşılaştırır. Ezber testinde veri/normalizasyon/eylem
eşlemesi hatasını simülasyon çalıştırmadan gösterir.

Girdiler eğitimdekiyle birebir: veri kümesinden gelen kareler (video_backend=pyav) ->
checkpoint'in ön işlemcisi -> politika -> son işlemci. Ölçüt: ilk `--horizon` adımda eklem
başına ortalama mutlak hata (rad) ve "sıfır eylem / ortalama eylem" tabanıyla karşılaştırma.

    python scripts/open_loop_check.py --checkpoint ~/vla_runs/act_ezber_01/checkpoints/last/pretrained_model \
        --root ~/vla_data/mahut_smoke_v1_act --episodes 0-9
(ACT checkpoint'i için türev veri kümesi, SmolVLA için ham veri kümesi kullanılır.)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="local/ds")
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--stride", type=int, default=25, help="her bölümde kaç karede bir örnekle")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    lo, _, hi = a.episodes.partition("-")
    eps = list(range(int(lo), int(hi or lo) + 1))
    ds = LeRobotDataset(a.repo_id, root=Path(a.root).expanduser(), episodes=eps, video_backend="pyav")
    ckpt = os.path.realpath(os.path.expanduser(a.checkpoint))
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = a.device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(a.device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": a.device}})

    ep_idx = np.asarray(ds.hf_dataset["episode_index"])
    all_actions = np.asarray(ds.hf_dataset["action"])
    errs, base_zero, base_mean = [], [], []
    mean_action = all_actions.mean(0)
    for ep in eps:
        frames = np.flatnonzero(ep_idx == ep)
        for i in frames[:: a.stride]:
            if i + a.horizon > frames[-1]:
                continue
            item = ds[int(i)]
            batch = {k: v.unsqueeze(0) for k, v in item.items()
                     if k.startswith("observation.") and isinstance(v, torch.Tensor)}
            batch["task"] = [item["task"]]
            policy.reset()
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(pre(batch))  # (1, T, D) normalize edilmiş çıktı
                pred = post(chunk)[0, : a.horizon].float().cpu().numpy()
            gt = all_actions[i: i + a.horizon]
            errs.append(np.abs(pred[:, :7] - gt).mean(0))
            base_zero.append(np.abs(gt).mean(0))
            base_mean.append(np.abs(gt - mean_action).mean(0))
    e, bz, bm = (np.mean(x, axis=0) for x in (errs, base_zero, base_mean))
    out = {
        "checkpoint": ckpt, "samples": len(errs), "horizon": a.horizon,
        "mae_per_joint": e.round(4).tolist(), "mae_mean": round(float(e.mean()), 4),
        "baseline_mean_action_mae": round(float(bm.mean()), 4), "baseline_zero_mae": round(float(bz.mean()), 4),
        "ratio_model_to_mean_baseline": round(float(e.mean() / bm.mean()), 3),
    }
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
