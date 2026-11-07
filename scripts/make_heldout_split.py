#!/usr/bin/env python3
"""
Held-out komut listesini üretir (configs/commands/heldout_commands.json).
Var olan dosyanın üzerine yazmaz; yalnızca ilk kurulumda bir kez çalıştırılır.

    python3 scripts/make_heldout_split.py [--out yol] [--seed N]
"""

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.commands.queue import DEFAULT_HELDOUT_PATH, HELDOUT_SEED, coverage_tables, load_heldout, write_heldout  # noqa: E402
from mahut_kol.commands.schema import all_valid_commands  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_HELDOUT_PATH))
    ap.add_argument("--seed", type=int, default=HELDOUT_SEED)
    args = ap.parse_args()
    path = write_heldout(args.out, args.seed)
    held = load_heldout(path)
    train = [c for c in all_valid_commands() if c.key not in {h.key for h in held}]
    cm, cr = coverage_tables(train)
    print(f"✅ {len(held)} held-out komut yazıldı: {path}")
    print(f"   Eğitim komutları: {len(train)}; hücre başına en az komut: "
          f"manipüle×yön={min(cm.values())}, referans×yön={min(cr.values())}")


if __name__ == "__main__":
    main()
