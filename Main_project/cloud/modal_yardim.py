"""mahut_cloud.py ve devir.py'nin ortak Modal yardımcıları (CLI yolu, Volume okuma/indirme).

`modal` yalnızca Volume açılırken içe aktarılır; böylece CLI'lar Modal kurulmadan da yardım metni gösterir.
"""

from __future__ import annotations

import sys
from pathlib import Path


def modal_exe() -> str:
    """Çalışan venv'deki `modal` CLI'si, yoksa PATH'teki."""
    exe = Path(sys.executable).parent / "modal"
    return str(exe) if exe.exists() else "modal"


def open_volume(name: str, create: bool = True):
    import modal

    return modal.Volume.from_name(name, create_if_missing=create)


def vol_list(vol, path: str) -> list[str]:
    try:
        return [e.path for e in vol.listdir(path)]
    except Exception:  # noqa: BLE001  (yol yok)
        return []


def vol_read(vol, path: str) -> bytes | None:
    try:
        return b"".join(vol.read_file(path))
    except Exception:  # noqa: BLE001
        return None


def download_dir(vol, remote: str, local: Path) -> int:
    """`remote` altındaki tüm dosyaları `local`'a indirir; var olan dosyanın üzerine yazmaz. Dosya sayısını döner."""
    n = 0
    for e in vol.listdir(remote, recursive=True):
        if e.type.name != "FILE":
            continue
        target = local / Path(e.path).relative_to(Path(remote))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
        n += 1
    return n
