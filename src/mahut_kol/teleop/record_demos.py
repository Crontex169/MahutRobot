#!/usr/bin/env python3
"""
İnsan Gösterimi (Demonstration) Kaydedici Şablonu.

NOT: Bu betik 'Teleoperasyonlar/record_demos.py' konumuna taşınmıştır.
Geriye dönük uyumluluk için buradan çalıştırılabilir veya dışa aktarılabilir.
"""

import sys
import os

_TARGET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Teleoperasyonlar/record_demos.py"))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Teleoperasyonlar.record_demos import *

if __name__ == "__main__":
    import runpy
    runpy.run_path(_TARGET, run_name="__main__")
