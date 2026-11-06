"""
MahutKol PS4 Kontrolcüsü (DualShock 4) ve Gamepad Teleoperasyon Modülü.

NOT: Bu modül 'Teleoperasyonlar/teleop_ps4.py' konumuna taşınmıştır.
Geriye dönük uyumluluk için buradan dışa aktarılmaktadır.
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Teleoperasyonlar.teleop_ps4 import MahutPS4Controller

__all__ = ["MahutPS4Controller"]
