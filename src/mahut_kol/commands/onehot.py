"""
5 sayısal komut slotu -> 17 boyutlu one-hot vektör (dil almayan politikalar için, ör. ACT).

Düzen (sırayla bitişik bloklar): manipüle renk (3) | manipüle şekil (3) | referans renk (3) |
referans şekil (3) | yön (5)  =  17. Slot indeksleri `schema.to_slots` ile aynıdır
(renk COLORS, şekil SHAPES, yön RELATIONS sırası).

Bu modül bilerek yalnızca numpy'a bağlıdır (ortam/sim içe aktarmaz): eğitim sarmalayıcısı,
türetilmiş veri kümesi üreticisi ve `eval_policy.py` aynı fonksiyonu paylaşır; böylece
eğitim ve çıkarım girdisi birebir aynı kalır.
"""

from __future__ import annotations

import numpy as np

# (renk, şekil, renk, şekil, yön) blok boyutları; schema.py ile testte karşılaştırılır.
SLOT_SIZES = (3, 3, 3, 3, 5)
ONEHOT_DIM = int(sum(SLOT_SIZES))  # 17
ENV_STATE_KEY = "observation.environment_state"


def slots_to_onehot(slots) -> np.ndarray:
    """(5,) ya da (N, 5) tamsayı slotlar -> (17,) ya da (N, 17) float32 one-hot."""
    s = np.asarray(slots)
    single = s.ndim == 1
    s = np.atleast_2d(s).astype(np.int64)
    if s.shape[1] != len(SLOT_SIZES):
        raise ValueError(f"slot sayısı {s.shape[1]} != {len(SLOT_SIZES)}")
    out = np.zeros((s.shape[0], ONEHOT_DIM), dtype=np.float32)
    offset = 0
    for j, size in enumerate(SLOT_SIZES):
        col = s[:, j]
        if col.min() < 0 or col.max() >= size:
            raise ValueError(f"slot {j} aralık dışı [0, {size}): {col.min()}..{col.max()}")
        out[np.arange(s.shape[0]), offset + col] = 1.0
        offset += size
    return out[0] if single else out
