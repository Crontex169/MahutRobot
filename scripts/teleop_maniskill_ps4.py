#!/usr/bin/env python3
"""
ManiSkill MahutKol PS4 / Gamepad Teleoperasyon Betiği Yönlendiricisi.

NOT: Bu betik 'Teleoperasyonlar/teleop_maniskill_ps4.py' konumuna taşınmıştır.
Doğrudan çalıştırmak için:
    python Teleoperasyonlar/teleop_maniskill_ps4.py
"""

import os
import sys

_TARGET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Teleoperasyonlar/teleop_maniskill_ps4.py"))

if __name__ == "__main__":
    print(f"ℹ️  [YÖNLENDİRME] Betik 'Teleoperasyonlar/teleop_maniskill_ps4.py' konumundan çalıştırılıyor...")
    os.execv(sys.executable, [sys.executable, _TARGET] + sys.argv[1:])
