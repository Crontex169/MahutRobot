#!/usr/bin/env python3
"""
MahutKol Sesli Komut ve Doğal Dil Çözücü Test Aracı.

Kullanım Örnekleri:
1. Doğrudan metin testi (mikrofonsuz hızlı doğrulama):
   python3 scripts/test_voice_command.py --mode text --input-text "Kırmızı küpü al yeşil silindirin sağına koy"

2. Mikrofondan Bas-Konuş (Push-to-Talk) canlı test:
   python3 scripts/test_voice_command.py --mode push_to_talk --model base

3. Mikrofondan Sessizlik Tespiti (VAD) ile test:
   python3 scripts/test_voice_command.py --mode vad --model base

4. Ses dosyası üzerinden test:
   python3 scripts/test_voice_command.py --mode file --audio-file path/to/sample.wav
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Repo kök dizinini sys.path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mahut_kol.speech import (
    AudioRecorder,
    CommandParser,
    STTEngine,
    VoiceCommander,
)

# Renkli terminal çıktıları
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner():
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 65)
    print("   🤖 MAHUTKOL TÜRKÇE SESLİ KOMUT & ROBOT ÇÖZÜCÜ TEST ARACI   ")
    print("=" * 65)
    print(f"{Colors.END}")


def print_command_summary(transcript: str, cmd: dict):
    print(f"\n{Colors.BOLD}--- Komut Değerlendirme Raporu ---{Colors.END}")
    print(f"{Colors.BLUE}📝 Ham / Algılanan Metin:{Colors.END} \"{transcript}\"")
    print(f"{Colors.CYAN}🔤 Normalize Edilmiş Metin:{Colors.END} \"{cmd.get('normalized_text', '')}\"")

    if cmd.get("valid"):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ KOMUT BAŞARIYLA ÇÖZÜMLENDİ:{Colors.END}")
        print(f"  🔹 Eylem (Action):              {Colors.BOLD}{cmd.get('action')}{Colors.END}")
        print(f"  🔹 Taşınacak Nesne (Manipulated): {Colors.BOLD}{cmd.get('manipulated_object')}{Colors.END}")
        print(f"  🔹 Referans Nesne (Reference):   {Colors.BOLD}{cmd.get('reference_object') or 'Yok'}{Colors.END}")
        print(f"  🔹 Bağıl Yön (Relation):         {Colors.BOLD}{cmd.get('relation') or 'Yok'}{Colors.END}")
        print(f"  🔹 Hedef Slot (Target Slot):     {Colors.BOLD}{cmd.get('target_slot') or 'Yok'}{Colors.END}")
        print(f"  🔹 Güven Skoru (Confidence):     {cmd.get('confidence', 1.0) * 100:.0f}%")
    elif cmd.get("needs_confirmation"):
        print(f"\n{Colors.YELLOW}{Colors.BOLD}🤔 ONAY GEREKLİ (düşük güvenli bulanık eşleşme):{Colors.END}")
        print(f"  ❓ {Colors.BOLD}{cmd.get('confirmation_prompt')}{Colors.END}")
        print(f"  🔹 Önerilen: {cmd.get('action')} | {cmd.get('manipulated_object')} | {cmd.get('target_slot') or 'Yok'}")
        print(f"  🔹 Güven Skoru (Confidence):     {cmd.get('confidence', 0.0) * 100:.0f}%")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ GEÇERSİZ VEYA EKSİK KOMUT:{Colors.END}")
        print(f"  ⚠️  Hata Nedeni: {Colors.YELLOW}{cmd.get('error', 'Bilinmeyen hata')}{Colors.END}")

    fuzzy = (cmd.get("details") or {}).get("fuzzy_matches")
    if fuzzy:
        print(f"\n{Colors.CYAN}🔎 Bulanık Eşleşmeler:{Colors.END}")
        for fm in fuzzy:
            extra = " (belirsiz)" if fm.get("ambiguous") else ""
            print(f"  '{fm['token']}' → '{fm['matched']}' ({fm['category']}, skor {fm['score']}){extra}")

    print(f"\n{Colors.BOLD}JSON Çıktısı (Simülasyona İletilen):{Colors.END}")
    print(json.dumps(cmd, indent=2, ensure_ascii=False))
    print("-" * 65)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MahutKol Türkçe STT ve Robot Komut Çözücü Canlı Doğrulama Aracı"
    )
    parser.add_argument(
        "--mode",
        choices=["push_to_talk", "vad", "text", "file"],
        default="push_to_talk",
        help="Girdi modu: push_to_talk (varsayılan), vad (sessizlik tespiti), text (metin), file (ses dosyası)"
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium"],
        default="base",
        help="Faster-Whisper model boyutu (varsayılan: base)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Kullanılacak donanım: auto (varsayılan CUDA varsa cuda), cuda, cpu"
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Hesaplama tipi (float16, int8, int8_float16 vb.)"
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default="Kırmızı küpü al yeşil silindirin sağına koy",
        help="--mode text seçildiğinde çözümlenecek Türkçe cümle"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="--mode file seçildiğinde okunacak ses dosyasının yolu"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Sürekli dinleme döngüsü modunda çalıştır"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    device = None if args.device == "auto" else args.device

    if args.mode == "text":
        print(f"{Colors.CYAN}ℹ️  Metin Çözümleme Modu (--mode text){Colors.END}")
        parser = CommandParser()
        cmd = parser.parse(args.input_text)
        print_command_summary(args.input_text, cmd)
        return

    # STT ve Mikrofon gerektiren modlar
    print(f"{Colors.CYAN}⚙️  STT Modeli Başlatılıyor (Boyut: {args.model}, Donanım: {args.device})...{Colors.END}")
    commander = VoiceCommander(
        model_size=args.model,
        device=device,
        compute_type=args.compute_type,
        language="tr",
    )
    print(f"{Colors.GREEN}✓ STT Motoru hazır: {commander.stt.device} ({commander.stt.compute_type}){Colors.END}")

    if args.mode == "file":
        if not args.audio_file:
            print(f"{Colors.RED}Hata: --mode file için --audio-file parametresi gereklidir.{Colors.END}")
            sys.exit(1)
        if not os.path.exists(args.audio_file):
            print(f"{Colors.RED}Hata: Ses dosyası bulunamadı: {args.audio_file}{Colors.END}")
            sys.exit(1)
        print(f"\n🎵 Ses dosyası işleniyor: {args.audio_file}")
        transcript, cmd = commander.process_audio_file(args.audio_file)
        print_command_summary(transcript, cmd)
        return

    if args.interactive:
        commander.run_interactive_loop(mode=args.mode)
    else:
        transcript, cmd = commander.listen_and_parse(mode=args.mode)
        print_command_summary(transcript, cmd)


if __name__ == "__main__":
    main()
