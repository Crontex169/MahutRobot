#!/usr/bin/env python3
"""
Teleoperasyonla VLA demo kaydı (MultiShapeRelationalTask, LeRobot v0.4.4 formatı).

Her bölüm: komut seç -> reset(options) -> İngilizce talimatı göster -> teleop ->
operatör hazır olunca (telefonda ilk debriyaj) süre başlar ->
başarı otomatik algılanır -> operatör onaylar (Enter kaydet, R at ve yeniden dene, Q çık).

Veri kümesi var ise AÇILIR ve sonuna eklenir; hiçbir şey silinmez. Şema uyuşmazsa hata verir.
Sidecar: <root>/mahut_episodes.jsonl (source: teleop). Komut kuyruğu durumu:
<root>/command_queue_state.json (devam ettirilebilir).

Örnekler:
    # Telefon (önce: adb reverse tcp:8000 tcp:8000; telefonda http://localhost:8000)
    ~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device phone
    # Klavye / PS4 (pygame penceresi odakta)
    ~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device gamepad
    # Komutu Türkçe yazarak ya da sesle ver
    ~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device phone --command-source text
    ~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device phone --command-source voice
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.commands import (  # noqa: E402
    BalancedCommandQueue,
    QueueCommandSource,
    TextCommandSource,
    VoiceCommandSource,
    load_heldout,
    to_english_instruction,
)
from mahut_kol.envs.multi_shape_task import DEFAULT_NUM_DISTRACTORS  # noqa: E402
from mahut_kol.recording import (  # noqa: E402
    DEFAULT_REPO_ID,
    DEFAULT_ROOT,
    FPS,
    QUEUE_STATE_NAME,
    VLADatasetWriter,
    build_frame,
    episode_meta,
)
TELEOP_MAX_STEPS = 1200

from mahut_kol.recording.session import (  # noqa: E402
    PygameUI,
    TerminalUI,
    make_env,
    run_episode,
    wait_for_start,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["phone", "gamepad", "keyboard"], default="phone")
    ap.add_argument("--command-source", choices=["queue", "text", "voice"], default="queue")
    ap.add_argument("--voice-mode", choices=["push_to_talk", "vad"], default="push_to_talk")
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="LeRobot veri kümesi kökü")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--queue-state", default=None, help=f"varsayılan: <root>/{QUEUE_STATE_NAME}")
    ap.add_argument("--num-episodes", type=int, default=10, help="bu oturumda kaydedilecek bölüm sayısı")
    ap.add_argument("--seed", type=int, default=0, help="komut kuyruğu ve sahne tohumu tabanı")
    # İnsan teleopu scripted politikadan yavaş: varsayılan 60 s (ortamın 600 adımı = 30 s dar kalıyor)
    ap.add_argument("--max-steps", type=int, default=TELEOP_MAX_STEPS,
                    help=f"bölüm adım sınırı ({FPS} Hz; varsayılan {TELEOP_MAX_STEPS})")
    ap.add_argument("--num-distractors", type=int, default=DEFAULT_NUM_DISTRACTORS)
    ap.add_argument("--include-heldout", action="store_true", help="held-out komutları da kaydet")
    ap.add_argument("--no-display", action="store_true", help="pygame penceresi yerine terminal")
    ap.add_argument("--port", type=int, default=8000, help="telefon sunucusu portu")
    ap.add_argument("--control-frame", choices=["wrist", "world"], default="wrist",
                    help="telefon hareketi hangi kameraya göre: wrist = kıskaç kamerası, world = front kamera")
    ap.add_argument("--no-paraphrase", action="store_true", help="hep kanonik talimat")
    return ap.parse_args(argv)


def make_action_source(args):
    if args.device == "phone":
        from mahut_kol.teleop.phone_webxr import PhoneActionSource

        return PhoneActionSource(config={"port": args.port, "control_frame": args.control_frame})
    from mahut_kol.teleop.action_source import GamepadKeyboardActionSource

    return GamepadKeyboardActionSource()


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).expanduser()
    heldout = load_heldout()
    heldout_keys = {c.key for c in heldout}

    writer = VLADatasetWriter(root, args.repo_id)
    print(f"📁 Veri kümesi: {root}  ({'yeni' if writer.created else 'devam'}; "
          f"{writer.num_episodes} bölüm, {writer.num_frames} kare)")

    if args.command_source == "queue":
        qpath = Path(args.queue_state) if args.queue_state else root / QUEUE_STATE_NAME
        source = QueueCommandSource(
            BalancedCommandQueue(qpath, seed=args.seed, include_heldout=args.include_heldout, heldout=heldout)
        )
    elif args.command_source == "text":
        source = TextCommandSource(heldout_keys=heldout_keys, include_heldout=args.include_heldout)
    else:
        source = VoiceCommandSource(
            mode=args.voice_mode, heldout_keys=heldout_keys, include_heldout=args.include_heldout,
            model_size=args.whisper_model,
        )

    env = make_env(args.max_steps)
    actions = make_action_source(args)
    ui = TerminalUI() if args.no_display else PygameUI()
    saved = 0
    attempt = 0
    try:
        while saved < args.num_episodes:
            cmd, info = source.next_command()
            if cmd is None:
                print(f"❌ Komut kabul edilmedi, bölüm başlatılmadı: {info.get('error')}")
                continue
            # Sahne tohumu: veri kümesindeki bölüm sayısı + deneme -> oturumlar arası tekrar yok
            seed = args.seed * 1_000_003 + writer.num_episodes * 101 + attempt
            attempt += 1
            obs, _ = env.reset(seed=seed, options=cmd.to_reset_options(args.num_distractors))
            instruction = to_english_instruction(cmd, seed=None if args.no_paraphrase else seed)
            ui.show_instruction(instruction, f"[{cmd.key}] seed={seed}")
            actions.reset(env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy())
            actions.publish_status(instruction=instruction, success=False, episode=writer.num_episodes)

            # Süre sayacı ve kayıt, operatör hazır olunca başlar (telefonda ilk debriyaj).
            hint = getattr(actions, "start_hint", lambda: "")()
            if hint:
                print(f"⏳ {hint}", flush=True)
            start = wait_for_start(obs, actions, ui)
            if start == "quit":
                print("👋 Çıkılıyor.")
                break
            if start == "reset":
                print("↩️  Yeni sahne istendi.")
                continue
            print(f"▶️  Bölüm başladı ({args.max_steps} adım = {args.max_steps / FPS:.0f} s).", flush=True)

            res = run_episode(
                env, obs,
                act_fn=actions.act,
                frame_fn=lambda o, a: build_frame(o, a, instruction, cmd),
                max_steps=args.max_steps,
                ui=ui,
                status_fn=actions.status_text,
                events_fn=actions.poll_events,
                on_success=lambda s: actions.publish_status(instruction=instruction, success=True),
                realtime_hz=FPS,
            )
            if res.aborted == "quit":
                print("👋 Çıkılıyor (bu bölüm kaydedilmedi).")
                break
            if res.aborted == "reset":
                print("↩️  Bölüm atıldı, aynı komut yeniden denenecek.")
                continue
            if not res.success:
                print(f"⏱️  {res.steps} adımda başarı yok; bölüm atıldı.")
                source.on_failed(cmd)
                # Pencereyi dondurup tuş beklemek yerine doğrudan sonraki bölüme geç.
                continue
            choice = ui.confirm(f"✅ Başarılı ({res.steps} adım, {res.duration_s:.1f} s).")
            if choice == "quit":
                print("👋 Çıkılıyor (bu bölüm kaydedilmedi).")
                break
            if choice != "save":
                print("↩️  Bölüm atıldı.")
                continue
            ep = writer.add_episode(
                res.frames,
                episode_meta(
                    cmd, instruction, seed, source="teleop", heldout=cmd.key in heldout_keys,
                    success=True, duration_s=res.duration_s, device=args.device,
                    command_source=args.command_source, steps=res.steps, fps=FPS,
                    command_input=info.get("raw_text") or info.get("transcript"),
                ),
            )
            source.on_saved(cmd)
            saved += 1
            print(f"💾 Bölüm {ep} kaydedildi ({len(res.frames)} kare). Oturum: {saved}/{args.num_episodes}")
    except KeyboardInterrupt:
        print("\n⛔ Ctrl+C: kayıtlı bölümler korunuyor, kapatılıyor...")
    finally:
        writer.close()
        actions.close()
        ui.close()
        env.close()
    print(f"✅ Toplam {writer.num_episodes} bölüm, {writer.num_frames} kare: {root}")


if __name__ == "__main__":
    main()
