#!/usr/bin/env python3
"""
Eğitilmiş LeRobot politikasını MahutKol-MultiShapeRelational-v1'de kapalı çevrim değerlendirir.

Kayıt şemasıyla aynı girdiler: observation.images.front + observation.images.wrist (224x224),
observation.state = 7 boyutlu qpos (5 kol + 2 parmak), task = İngilizce talimat (kayıttaki
şablonlar). Komutlar reset(options) ile komut JSON'undan verilir.

Çıktı (terminal + --out JSON):
  1. Nesne ve yön bazında başarı (eğitim komutlarından örneklem)
  2. Held-out komutlarda başarı
  3. Dil duyarlılığı: AYNI sahne (object_poses ile birebir) farklı komutlarla çalıştırılır.
     Politika dili yok sayıyorsa komut değişse de aynı nesneyi aynı yere götürür. Ölçülen:
     her komutta doğru nesneyi hareket ettirme oranı, başarı ve "diğer komutun hedefine
     yerleştirme" (yanlış dil izleme) oranı.

Örnekler:
    python3 scripts/eval_policy.py --checkpoint ~/vla_runs/x/checkpoints/last/pretrained_model
    python3 scripts/eval_policy.py --checkpoint latest --episodes 40 --heldout-episodes 30 --language-scenes 10
    # Boru hattı duman testi (model yok): betik kendi oracle'ı ile çalışır
    python3 scripts/eval_policy.py --policy scripted --episodes 3 --heldout-episodes 2 --language-scenes 1
Bu betik yalnızca değerlendirme altyapısıdır; model değerlendirmesi burada koşulmaz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.commands import BalancedCommandQueue, Command, load_heldout, to_english_instruction, to_slots  # noqa: E402
from mahut_kol.commands.onehot import ENV_STATE_KEY, slots_to_onehot  # noqa: E402
from mahut_kol.envs.multi_shape_task import (  # noqa: E402
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_NUM_DISTRACTORS,
    RELATIONS,
    compute_relational_target_position,
    is_valid_command,
)
from mahut_kol.recording import IMAGE_KEYS  # noqa: E402
from mahut_kol.recording.session import make_env  # noqa: E402

DEFAULT_CKPT = "~/vla_runs/smolvla_mahut_kol/checkpoints/last/pretrained_model"  # eski; `latest` için bkz. --checkpoint


# ---------------------------------------------------------------------------
# Politika sarmalayıcıları
# ---------------------------------------------------------------------------


class LeRobotPolicy:
    """lerobot 0.4.4 politika + ön/son işlemciler. Gözlem anahtarları EĞİTİMDEKİ (kayıt şeması) ile aynı.

    * Kamera adı eşlemesi (SmolVLA: front->camera1, wrist->camera2) ve `empty_cameras` doldurması
      checkpoint'teki ön işlemci/yapılandırmadan gelir; burada yeniden uygulanmaz (çift eşleme olmasın).
      `--rename` yalnızca eski checkpoint'lerde ön işlemci eşlemesi eksikse içindir.
    * ACT (dilsiz kıyas): yapılandırmada `observation.environment_state` varsa komutun 5 slotu eğitimdeki
      `commands.onehot.slots_to_onehot` ile 17 boyutlu one-hot'a çevrilip verilir (birebir aynı kod).
    * Dil girdisi (`task`): SmolVLA gibi dil kullanan politikalara İngilizce talimat verilir; ACT yok sayar.
    """

    def __init__(self, ckpt: str, device: str | None, rename: dict):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = device or cfg.device
        self.policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(cfg.device).eval()
        overrides = {"device_processor": {"device": str(cfg.device)}}
        if rename:
            overrides["rename_observations_processor"] = {"rename_map": rename}
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides=overrides)
        self.device = cfg.device
        self.cfg_type = cfg.type
        self.uses_env_state = ENV_STATE_KEY in cfg.input_features
        self.onehot = None
        self.act_lat: list[float] = []
        self.act_min = np.full(7, np.inf)
        self.act_max = np.full(7, -np.inf)
        self.n_nan = 0
        print(f"[politika] tür={cfg.type} cihaz={cfg.device} env_state={self.uses_env_state} "
              f"girdiler={sorted(cfg.input_features)}", flush=True)

    def reset(self, env, instruction, cmd=None):
        self.policy.reset()
        self.instruction = instruction
        if self.uses_env_state:
            if cmd is None:
                raise ValueError("Bu politika yapılandırılmış komut (one-hot) bekliyor ama komut verilmedi")
            self.onehot = torch.as_tensor(slots_to_onehot(to_slots(cmd)))[None]

    def act(self, obs, env):
        batch = {
            key: obs["sensor_data"][cam]["rgb"].permute(0, 3, 1, 2).float() / 255.0
            for key, cam in IMAGE_KEYS.items()
        }
        batch["observation.state"] = obs["agent"]["qpos"][:, :7].float()  # 5 kol + 2 parmak
        batch["task"] = [self.instruction]
        if self.uses_env_state:
            batch[ENV_STATE_KEY] = self.onehot.clone()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.act_lat.append(time.time() - t0)
        a = action.detach().float().cpu()
        if not torch.isfinite(a).all():
            self.n_nan += 1
        self.act_min = np.minimum(self.act_min, a[0].numpy())
        self.act_max = np.maximum(self.act_max, a[0].numpy())
        return a

    def stats(self) -> dict:
        lat = np.array(self.act_lat) if self.act_lat else np.zeros(1)
        out = {
            "act_calls": len(self.act_lat), "nonfinite_actions": self.n_nan,
            "action_min": self.act_min.round(3).tolist(), "action_max": self.act_max.round(3).tolist(),
            "act_latency_ms_mean": round(float(lat.mean() * 1000), 2),
            "act_latency_ms_p95": round(float(np.percentile(lat, 95) * 1000), 2),
            "act_latency_ms_max": round(float(lat.max() * 1000), 2),
        }
        if str(self.device).startswith("cuda"):
            out["gpu_peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            out["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        return out


class ScriptedOraclePolicy:
    """Duman testi: dili ve sahneyi bilen scripted politika (model yerine)."""

    def reset(self, env, instruction, cmd=None):
        from mahut_kol.envs.scripted_policy import MultiShapeScriptedPolicy

        u = env.unwrapped
        self.p = MultiShapeScriptedPolicy(u)
        self.p.reset(u.agent.robot.get_qpos()[0].cpu().numpy())

    def act(self, obs, env):
        q = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
        return torch.as_tensor(self.p.act(q))[None]


# ---------------------------------------------------------------------------
# Bölüm çalıştırma
# ---------------------------------------------------------------------------


def run(env, policy, cmd: Command, seed: int, max_steps: int, paraphrase: bool, num_distractors: int,
        object_poses=None):
    opts = cmd.to_reset_options(num_distractors)
    if object_poses is not None:
        opts["object_poses"] = object_poses
        opts["distractor_objects"] = [k for k in object_poses if k not in (cmd.manipulated_object, cmd.reference_object)]
    obs, _ = env.reset(seed=seed, options=opts)
    u = env.unwrapped
    start = {k: u.manipulation_objects[k].pose.p[0].cpu().numpy().copy()
             for k in [u.manipulated_object_id, u.reference_object_id] + u.distractor_object_ids}
    instruction = to_english_instruction(cmd, seed=seed if paraphrase else None)
    policy.reset(env, instruction, cmd)
    success, steps = False, 0
    for steps in range(1, max_steps + 1):
        obs, _, _, truncated, info = env.step(policy.act(obs, env))
        if info["success"].item():
            success = True
            break
        if bool(truncated.item() if hasattr(truncated, "item") else truncated):
            break
    end = {k: u.manipulation_objects[k].pose.p[0].cpu().numpy().copy() for k in start}
    moved = {k: float(np.linalg.norm(end[k][:2] - start[k][:2])) for k in start}
    return {"command": cmd.key, "instruction": instruction, "seed": seed, "success": success,
            "steps": steps, "moved": moved, "end": {k: v.tolist() for k, v in end.items()}}


def rate(xs):
    return f"{sum(xs)}/{len(xs)} (%{100 * sum(xs) / max(len(xs), 1):.0f})"


# ---------------------------------------------------------------------------
# Dil duyarlılığı testi
# ---------------------------------------------------------------------------


def language_variants(env, base: Command, poses: dict, k: int, rng) -> list:
    """Aynı 4 nesneyle, hedefi geçerli ve boş olan farklı komutlar (manipüle nesnesi farklı olanlar öncelikli)."""
    ids = list(poses)
    u = env.unwrapped
    cands = []
    for m in ids:
        for r in ids:
            for rel in RELATIONS:
                if (m, r, rel) == (base.manipulated_object, base.reference_object, base.relation):
                    continue
                if not is_valid_command(m, r, rel):
                    continue
                ref = np.array(poses[r][:2])
                tgt = compute_relational_target_position(np.r_[ref, 0.0], rel, delta=u.relational_delta)[:2]
                x0, x1 = u.workspace_x_bounds
                y0, y1 = u.workspace_y_bounds
                if not (x0 <= tgt[0] <= x1 and y0 <= tgt[1] <= y1):
                    continue
                if rel != "TOP" and any(np.linalg.norm(np.array(poses[o][:2]) - tgt) < u.min_pairwise_distance
                                        for o in ids if o != r):
                    continue
                cands.append(Command(m, r, rel))
    rng.shuffle(cands)
    cands.sort(key=lambda c: c.manipulated_object == base.manipulated_object)  # farklı nesne önce
    return cands[:k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="pretrained_model klasörü ya da 'latest'")
    ap.add_argument("--policy", choices=["lerobot", "scripted"], default="lerobot")
    ap.add_argument("--device", default=os.environ.get("EVAL_DEVICE"))
    ap.add_argument("--rename", action="append", default=[], metavar="KAYNAK=HEDEF",
                    help="gözlem anahtarı yeniden adlandırma (ör. observation.images.front=observation.images.camera1)")
    ap.add_argument("--episodes", type=int, default=45, help="eğitim komutlarından bölüm (dengeli kuyruk)")
    ap.add_argument("--heldout-episodes", type=int, default=30, help="held-out komutlarından bölüm")
    ap.add_argument("--language-scenes", type=int, default=10, help="dil duyarlılığı için sahne sayısı")
    ap.add_argument("--variants", type=int, default=3, help="sahne başına alternatif komut")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    ap.add_argument("--num-distractors", type=int, default=DEFAULT_NUM_DISTRACTORS)
    ap.add_argument("--paraphrase", action="store_true", help="kanonik yerine seed'li paraphrase talimat")
    ap.add_argument("--seed", type=int, default=12345, help="değerlendirme tohumu (kayıt tohumlarından farklı)")
    ap.add_argument("--out", default=None, help="JSON rapor yolu (varsayılan: eval_results/<adım>_<zaman>.json)")
    args = ap.parse_args(argv)

    if args.policy == "lerobot":
        if not args.checkpoint:
            ap.error("--checkpoint gerekli (ya da duman testi için --policy scripted)")
        ckpt = os.path.realpath(os.path.expanduser(DEFAULT_CKPT if args.checkpoint == "latest" else args.checkpoint))
        rename = dict(r.split("=", 1) for r in args.rename)
        policy = LeRobotPolicy(ckpt, args.device, rename)
        tag = os.path.basename(os.path.dirname(ckpt))
    else:
        policy = ScriptedOraclePolicy()
        tag = "scripted_oracle"

    env = make_env(args.max_steps)
    rng = np.random.default_rng(args.seed)
    results = {"train": [], "heldout": [], "language": []}
    t0 = time.time()

    # 1. Eğitim komutları (dengeli örneklem, held-out hariç)
    q = BalancedCommandQueue(seed=args.seed)
    for i, cmd in enumerate(q.take(args.episodes)):
        r = run(env, policy, cmd, args.seed + i, args.max_steps, args.paraphrase, args.num_distractors)
        results["train"].append(r)
        print(f"[eğitim {i + 1}/{args.episodes}] {'✅' if r['success'] else '❌'} {r['instruction']}", flush=True)

    # 2. Held-out
    held = load_heldout()
    for i in range(args.heldout_episodes):
        cmd = held[i % len(held)]
        r = run(env, policy, cmd, args.seed + 100_000 + i, args.max_steps, args.paraphrase, args.num_distractors)
        results["heldout"].append(r)
        print(f"[held-out {i + 1}/{args.heldout_episodes}] {'✅' if r['success'] else '❌'} {r['instruction']}", flush=True)

    # 3. Dil duyarlılığı: aynı sahne, farklı komutlar
    pool = BalancedCommandQueue(seed=args.seed + 7).take(max(args.language_scenes * 3, 1))
    scenes_done = 0
    for j, base in enumerate(pool):
        if scenes_done >= args.language_scenes:
            break
        seed = args.seed + 200_000 + j
        env.reset(seed=seed, options=base.to_reset_options(args.num_distractors))
        poses = env.unwrapped.get_active_object_poses()
        variants = language_variants(env, base, poses, args.variants, rng)
        if not variants:
            continue
        scene = {"seed": seed, "poses": poses, "runs": []}
        for cmd in [base] + variants:
            r = run(env, policy, cmd, seed, args.max_steps, args.paraphrase, args.num_distractors, object_poses=poses)
            # Hangi nesne en çok hareket etti? Doğru nesne mi?
            mover = max(r["moved"], key=r["moved"].get)
            r["moved_correct_object"] = mover == cmd.manipulated_object and r["moved"][mover] > 0.02
            # Diğer komutlardan birinin hedefine mi koydu? (dili yok sayma işareti)
            wrong = False
            for other in [base] + variants:
                if other.key == cmd.key or r["success"]:
                    continue
                ref_end = np.array(r["end"][other.reference_object])
                tgt = compute_relational_target_position(ref_end, other.relation, delta=env.unwrapped.relational_delta)
                obj_end = np.array(r["end"][other.manipulated_object])
                if np.linalg.norm(obj_end[:2] - tgt[:2]) < 0.025 and abs(obj_end[2] - tgt[2]) < 0.012:
                    wrong = True
            r["placed_for_other_command"] = wrong
            scene["runs"].append(r)
            print(f"[dil sahne {scenes_done + 1}] {'✅' if r['success'] else '❌'} "
                  f"doğru nesne={'evet' if r['moved_correct_object'] else 'hayır'} {r['instruction']}", flush=True)
        results["language"].append(scene)
        scenes_done += 1
    env.close()

    # ---------------- Rapor ----------------
    def by(rows, f):
        d = defaultdict(list)
        for r in rows:
            d[f(Command.from_key(r["command"]))].append(r["success"])
        return {k: rate(v) for k, v in sorted(d.items())}

    tr, ho = results["train"], results["heldout"]
    lang_runs = [r for s in results["language"] for r in s["runs"]]
    summary = {
        "policy": tag,
        "train_success": rate([r["success"] for r in tr]),
        "train_by_manipulated_object": by(tr, lambda c: c.manipulated_object),
        "train_by_reference_object": by(tr, lambda c: c.reference_object),
        "train_by_relation": by(tr, lambda c: c.relation),
        "train_by_shape": by(tr, lambda c: c.manipulated_object.split("_", 1)[1]),
        "heldout_success": rate([r["success"] for r in ho]),
        "heldout_by_relation": by(ho, lambda c: c.relation),
        "language_scenes": len(results["language"]),
        "language_success": rate([r["success"] for r in lang_runs]),
        "language_correct_object_moved": rate([r["moved_correct_object"] for r in lang_runs]),
        "language_placed_for_other_command": rate([r["placed_for_other_command"] for r in lang_runs]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if hasattr(policy, "stats"):
        summary["policy_stats"] = policy.stats()
    print("\n===== ÖZET =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    out = Path(args.out) if args.out else Path("eval_results") / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"📄 Rapor: {out}")


if __name__ == "__main__":
    main()
