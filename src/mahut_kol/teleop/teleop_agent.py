"""
PS4 Kolu ve Klavye ile Teleoperasyon Denetleyicisi.

NOT: Bu modül 'Teleoperasyonlar/teleop_agent.py' konumuna taşınmıştır.
Geriye dönük uyumluluk için buradan dışa aktarılmaktadır.
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Teleoperasyonlar.teleop_agent import TeleopController

__all__ = ["TeleopController"]
