#!/usr/bin/env python3
"""
Otomatik (scripted) VLA demo üretici - MultiShapeRelationalTask.

Teleop kayıtlarıyla AYNI şema ve aynı LeRobot veri kümesi; sidecar'da `source: scripted`.
Yalnızca başarılı bölümler kaydedilir. Komutlar dengeli kuyruktan gelir (held-out hariç);
kuyruk durumu ayrı dosyada tutulur (<root>/command_queue_state_scripted.json) ki teleop
kuyruğunun dengesini bozmasın. Kesilirse aynı komutla tekrar çalıştırınca kaldığı yerden
devam eder (veri kümesi ve kuyruk durumu eklemeli).

Örnekler:
    ~/.venvs/lerobot/bin/python scripts/generate_scripted_demos.py --num-episodes 2000
    ~/.venvs/lerobot/bin/python scripts/generate_scripted_demos.py --num-episodes 5 --root /tmp/deneme
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.commands import BalancedCommandQueue, load_heldout, to_english_instruction  # noqa: E402
from mahut_kol.envs.multi_shape_task import DEFAULT_MAX_EPISODE_STEPS, DEFAULT_NUM_DISTRACTORS  # noqa: E402
from mahut_kol.envs.scripted_policy import MultiShapeScriptedPolicy  # noqa: E402
from mahut_kol.recording import DEFAULT_REPO_ID, DEFAULT_ROOT, FPS, VLADatasetWriter, build_frame, episode_meta  # noqa: E402
from mahut_kol.recording.session import make_env, run_episode  # noqa: E402

QUEUE_STATE_SCRIPTED = "command_queue_state_scripted.json"
REPORT_NAME = "scripted_generation_report.json"


def classify_failure(policy, res, probes) -> str:
    """Başarısız bölümün nedeni."""
    if res.aborted:
        return f"aborted_{res.aborted}"
    if not probes.get("grasped_after_lift", False):
        return "grasp_failed"
    if not probes.get("grasped_before_place", False):
        return "dropped_in_transport"
    info = res.info or {}
    if not info.get("is_placed", False):
        return "placement_miss"
    if not info.get("is_released", False):
        return "not_released"
    if not info.get("is_static", False):
        return "not_settled"
    return "timeout"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-episodes", type=int, default=100, help="bu çalıştırmada KAYDEDİLECEK başarılı bölüm")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--queue-state", default=None, help=f"varsayılan: <root>/{QUEUE_STATE_SCRIPTED}")
    ap.add_argument("--seed", type=int, default=1, help="kuyruk ve sahne tohum tabanı (teleoptan farklı tutun)")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    ap.add_argument("--num-distractors", type=int, default=DEFAULT_NUM_DISTRACTORS)
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--max-attempts-factor", type=float, default=3.0, help="deneme sınırı = faktör x bölüm")
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--render-backend", default=None, help="ör. 'cpu' (varsayılan: ManiSkill seçimi)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).expanduser()
    heldout = load_heldout()
    heldout_keys = {c.key for c in heldout}
    qpath = Path(args.queue_state) if args.queue_state else root / QUEUE_STATE_SCRIPTED

    writer = VLADatasetWriter(root, args.repo_id)
    root.mkdir(parents=True, exist_ok=True)
    queue = BalancedCommandQueue(qpath, seed=args.seed, include_heldout=args.include_heldout, heldout=heldout)
    print(f"📁 {root} ({'yeni' if writer.created else 'devam'}: {writer.num_episodes} bölüm) | "
          f"kuyruk: {queue.n_committed} kayıt, {queue.n_failures} başarısız deneme")

    env = make_env(args.max_steps, render_backend=args.render_backend)
    u = env.unwrapped
    reasons: Counter = Counter()
    per_shape_rel: Counter = Counter()
    per_shape_rel_ok: Counter = Counter()
    saved = attempts = 0
    t_start = time.time()
    max_attempts = int(args.num_episodes * args.max_attempts_factor) + 10
    try:
        while saved < args.num_episodes and attempts < max_attempts:
            cmd = queue.peek()
            seed = args.seed * 10_000_000 + queue.n_committed + queue.n_failures
            attempts += 1
            obs, _ = env.reset(seed=seed, options=cmd.to_reset_options(args.num_distractors))
            instruction = to_english_instruction(cmd, seed=None if args.no_paraphrase else seed)
            policy = MultiShapeScriptedPolicy(u)
            policy.reset(u.agent.robot.get_qpos()[0].cpu().numpy())
            key = (cmd.manipulated_object.split("_", 1)[1], cmd.relation)
            per_shape_rel[key] += 1
            if policy.plan_error:
                reasons[policy.plan_error] += 1
                queue.report_failure(cmd)
                continue

            probes = {}
            lift_end, place_start = policy.phase_end("lift"), policy.phase_end("transport")

            def probe(step, info, _p=probes, _le=lift_end, _ps=place_start):
                obj_z = float(u.get_manipulated_object().pose.p[0, 2])
                grasped = bool(info["is_grasped"].item()) and obj_z > 0.04
                if step == _le:
                    _p["grasped_after_lift"] = grasped
                if step == _ps:
                    _p["grasped_before_place"] = grasped

            res = run_episode(
                env, obs,
                act_fn=policy.act,
                frame_fn=lambda o, a: build_frame(o, a, instruction, cmd),
                max_steps=min(args.max_steps, len(policy)),
                step_cb=probe,
            )
            if not res.success:
                reasons[classify_failure(policy, res, probes)] += 1
                queue.report_failure(cmd)
                continue
            ep = writer.add_episode(
                res.frames,
                episode_meta(
                    cmd, instruction, seed, source="scripted", heldout=cmd.key in heldout_keys,
                    success=True, duration_s=res.steps / FPS, steps=res.steps, fps=FPS,
                    grasp_tilt_deg=round(float(policy.tilt * 180 / 3.141592653589793), 1),
                ),
            )
            queue.commit(cmd)
            reasons["success"] += 1
            per_shape_rel_ok[key] += 1
            saved += 1
            if saved % 10 == 0 or saved == args.num_episodes:
                el = time.time() - t_start
                print(f"  {saved}/{args.num_episodes} kaydedildi | deneme {attempts} | "
                      f"başarı %{100 * saved / attempts:.0f} | {el / saved:.1f} s/bölüm | son: bölüm {ep}",
                      flush=True)
    except KeyboardInterrupt:
        print("\n⛔ Ctrl+C: kayıtlı bölümler ve kuyruk durumu korunuyor.")
    finally:
        writer.close()
        env.close()

    rate = saved / max(attempts, 1)
    report = {
        "attempts": attempts,
        "saved": saved,
        "success_rate": round(rate, 4),
        "failure_reasons": {k: v for k, v in reasons.items() if k != "success"},
        "success_by_shape_relation": {
            f"{s}|{r}": f"{per_shape_rel_ok[(s, r)]}/{n}" for (s, r), n in sorted(per_shape_rel.items())
        },
        "dataset_episodes": writer.num_episodes,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Rapor: çalıştırma başına bir satır eklenir (eski raporlar silinmez)
    with open(root / REPORT_NAME.replace(".json", ".jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, ensure_ascii=False) + "\n")
    print("\n📊 Scripted üretim raporu")
    print(f"   deneme {attempts}, kaydedilen {saved}, başarı oranı %{100 * rate:.1f}")
    for k, v in sorted(report["failure_reasons"].items(), key=lambda kv: -kv[1]):
        print(f"   ❌ {k}: {v}")
    print(f"   şekil|yön başarı: {report['success_by_shape_relation']}")
    print(f"   veri kümesi: {writer.num_episodes} bölüm -> {root}")


if __name__ == "__main__":
    main()
