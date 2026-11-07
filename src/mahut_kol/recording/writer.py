"""
VLA veri kümesi yazıcı: LeRobotDataset (lerobot 0.4.4, codebase v3.0) + sidecar JSONL.

Kurallar:
- Var olan veri kümesi her zaman AÇILIR ve sonuna eklenir; hiçbir dosya silinmez ya da
  üzerine yazılmaz (`shutil.rmtree` yok). Şema/fps uyuşmazlığında hata verilir, dokunulmaz.
- Kök dizin var ama LeRobot veri kümesi değilse (meta/info.json yok) ve boş değilse hata.
- Her bölüm `save_episode()` başarılı olduktan SONRA sidecar'a (<root>/mahut_episodes.jsonl)
  bir satır eklenir; satırdaki `episode_index` LeRobot bölüm indeksiyle aynıdır.
- `close()` (ya da `with`) LeRobot `finalize()` çağırır; parquet yazıcıları kapanmadan
  çıkılırsa son dosyanın meta verisi eksik kalabilir.

Teleop (`scripts/record_vla_demos.py`) ve scripted (`scripts/generate_scripted_demos.py`)
kayıtları aynı şemayı ve aynı yazıcıyı kullanır; ayrım sidecar'daki `source` alanıdır.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from mahut_kol.commands.schema import SLOT_NAMES, Command, to_slots
from mahut_kol.envs.multi_shape_task import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FRONT_CAMERA_NAME,
    WRIST_CAMERA_NAME,
)

FPS = 20
ROBOT_TYPE = "mahut_kol"
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "finger_left", "finger_right"]
DEFAULT_ROOT = Path(os.path.expanduser("~/vla_data/mahut_multishape"))
DEFAULT_REPO_ID = "local/mahut_multishape"
SIDECAR_NAME = "mahut_episodes.jsonl"
QUEUE_STATE_NAME = "command_queue_state.json"

IMAGE_KEYS = {
    f"observation.images.{FRONT_CAMERA_NAME}": FRONT_CAMERA_NAME,
    f"observation.images.{WRIST_CAMERA_NAME}": WRIST_CAMERA_NAME,
}
SLOTS_KEY = "command.slots"


def vla_features() -> Dict[str, Dict[str, Any]]:
    """LeRobot özellik şeması (kayıt ve doğrulamada tek kaynak)."""
    img = {
        "dtype": "video",
        "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        **{k: dict(img) for k in IMAGE_KEYS},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (7,), "names": list(JOINT_NAMES)},
        # 5 sayısal komut slotu (manipüle renk/şekil, referans renk/şekil, yön). "observation."
        # öneki yok: politikalar varsayılan olarak bunu girdi saymaz; isteğe bağlı kullanılır.
        SLOTS_KEY: {"dtype": "int64", "shape": (5,), "names": list(SLOT_NAMES)},
    }


def build_frame(obs: Mapping[str, Any], action, instruction: str, cmd: Command) -> Dict[str, Any]:
    """ManiSkill gözlemi (num_envs = 1) + eylem -> LeRobot karesi."""

    def _np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    frame: Dict[str, Any] = {}
    for key, cam in IMAGE_KEYS.items():
        frame[key] = _np(obs["sensor_data"][cam]["rgb"])[0].astype(np.uint8)
    frame["observation.state"] = _np(obs["agent"]["qpos"])[0].astype(np.float32)
    frame["action"] = np.asarray(_np(action), dtype=np.float32).reshape(7)
    frame[SLOTS_KEY] = to_slots(cmd)
    frame["task"] = instruction
    return frame


def _normalize_feature(f: Mapping[str, Any]) -> Dict[str, Any]:
    return {"dtype": f["dtype"], "shape": tuple(f["shape"]), "names": list(f["names"]) if f.get("names") else None}


def check_schema(existing: Mapping[str, Any], fps: int) -> List[str]:
    """Var olan veri kümesi özellikleri ile beklenen şema arasındaki farklar (boş = uyumlu)."""
    problems = []
    if int(fps) != FPS:
        problems.append(f"fps {fps} != {FPS}")
    want = vla_features()
    for k, f in want.items():
        if k not in existing:
            problems.append(f"eksik özellik: {k}")
            continue
        a, b = _normalize_feature(existing[k]), _normalize_feature(f)
        if a != b:
            problems.append(f"{k}: {a} != {b}")
    for k in existing:
        if k not in want and k not in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            problems.append(f"beklenmeyen özellik: {k}")
    return problems


class VLADatasetWriter:
    """LeRobot veri kümesini açar/oluşturur, bölüm ekler, sidecar tutar."""

    def __init__(
        self,
        root: os.PathLike = DEFAULT_ROOT,
        repo_id: str = DEFAULT_REPO_ID,
        vcodec: str = "libsvtav1",
        image_writer_threads: int = 4,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root).expanduser()
        self.repo_id = repo_id
        self.sidecar_path = self.root / SIDECAR_NAME
        info = self.root / "meta" / "info.json"
        if info.exists():
            meta = json.loads(info.read_text(encoding="utf-8"))
            problems = check_schema(meta.get("features", {}), meta.get("fps", -1))
            if problems:
                raise ValueError(
                    f"Var olan veri kümesi şeması uyumsuz ({self.root}); dokunulmadı:\n  - "
                    + "\n  - ".join(problems)
                )
            self.ds = LeRobotDataset(repo_id, root=self.root, vcodec=vcodec)
            self.created = False
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError(
                    f"{self.root} boş değil ama LeRobot veri kümesi değil (meta/info.json yok); dokunulmadı."
                )
            if self.root.exists():
                self.root.rmdir()  # yalnızca BOŞ dizin; LeRobotDataset.create dizini kendisi açar
            self.ds = LeRobotDataset.create(
                repo_id,
                FPS,
                features=vla_features(),
                root=self.root,
                robot_type=ROBOT_TYPE,
                use_videos=True,
                image_writer_threads=image_writer_threads,
                vcodec=vcodec,
            )
            self.created = True
        if not self.created and image_writer_threads > 0:
            self.ds.start_image_writer(num_processes=0, num_threads=image_writer_threads)
        self._check_sidecar_consistency()
        self._closed = False

    # ------------------------------------------------------------------
    @property
    def num_episodes(self) -> int:
        return int(self.ds.meta.total_episodes)

    @property
    def num_frames(self) -> int:
        return int(self.ds.meta.total_frames)

    def _check_sidecar_consistency(self) -> None:
        rows = read_sidecar(self.sidecar_path)
        if len(rows) != self.num_episodes:
            print(
                f"⚠️  Sidecar satır sayısı ({len(rows)}) LeRobot bölüm sayısıyla ({self.num_episodes}) "
                f"uyuşmuyor: {self.sidecar_path}. Yeni bölümler yine de doğru indeksle eklenecek; "
                "scripts/validate_dataset.py ile kontrol edin."
            )

    def add_episode(self, frames: Iterable[Dict[str, Any]], meta: Dict[str, Any]) -> int:
        """Kareleri yazar, bölümü kaydeder, sidecar'a satır ekler. Bölüm indeksini döner."""
        if self._closed:
            raise RuntimeError("Yazıcı kapatıldı")
        episode_index = self.num_episodes
        n = 0
        for f in frames:
            self.ds.add_frame(dict(f))
            n += 1
        if n == 0:
            raise ValueError("Boş bölüm kaydedilemez")
        self.ds.save_episode()
        row = {"episode_index": episode_index, "num_frames": n, **meta}
        row.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.sidecar_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return episode_index

    def discard_buffer(self) -> None:
        """add_frame ile birikmiş ama kaydedilmemiş kareleri bırakır (yalnızca geçici görüntüler)."""
        if self.ds.episode_buffer is not None and self.ds.episode_buffer.get("size", 0) > 0:
            self.ds.clear_episode_buffer()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ds.finalize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_sidecar(path: os.PathLike) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_meta(
    cmd: Command,
    instruction: str,
    seed: int,
    source: str,
    heldout: bool,
    success: bool,
    duration_s: float,
    **extra: Any,
) -> Dict[str, Any]:
    """Sidecar satırı için ortak alanlar."""
    return {
        "seed": int(seed),
        "command": cmd.to_dict(),
        "instruction": instruction,
        "slots": [int(v) for v in to_slots(cmd)],
        "heldout": bool(heldout),
        "source": source,
        "success": bool(success),
        "duration_s": round(float(duration_s), 3),
        **extra,
    }
