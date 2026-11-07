"""
Cihazdan bağımsız teleoperasyon eylem kaynağı arayüzü.

Kayıt betiği (`scripts/record_vla_demos.py`) yalnızca `ActionSource` ile konuşur; telefon
(WebXR), PS4 gamepad ya da klavye aynı betiğe takılır.

Sözleşme:
- Eylem her zaman 7 boyutlu mutlak `pd_joint_pos` hedefidir:
  [joint0..joint4 (rad), finger_left, finger_right (m)]. Parmak: 0.0 = açık, 0.042 = kapalı.
- `reset(qpos)` bölüm başında ölçülen eklem konumuyla çağrılır; kaynak hedefini buna eşitler.
- `act(qpos)` her kontrol adımında (20 Hz) çağrılır ve sıradaki hedefi döner.
- `poll_events()` cihazdan gelen olayları ("reset" gibi) döner ve temizler.
- `ready()` False iken kayıt betiği bölümü başlatmaz (ör. telefon bağlanıp debriyaja basılana dek).
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence, Set

import numpy as np

ARM_DOF = 5
FINGER_OPEN = 0.0
FINGER_CLOSED = 0.042
# MahutKol URDF eklem sınırları (joint0..4)
ARM_Q_MIN = np.array([-1.5708, 0.0, -1.0, -3.141593, -1.570796])
ARM_Q_MAX = np.array([1.5708, 3.14159, 1.5708, 0.0, 1.570796])


def clip_action(action: Sequence[float]) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    a[:ARM_DOF] = np.clip(a[:ARM_DOF], ARM_Q_MIN, ARM_Q_MAX)
    a[ARM_DOF:] = np.clip(a[ARM_DOF:], FINGER_OPEN, FINGER_CLOSED)
    return a


def gripper_open_ratio_to_finger(ratio: float) -> float:
    """Kıskaç açıklık oranı (0 = kapalı, 1 = tam açık; telefon kaydırıcısı) -> parmak eklemi."""
    r = float(np.clip(ratio, 0.0, 1.0))
    return FINGER_CLOSED * (1.0 - r)


class ActionSource(abc.ABC):
    """Teleoperasyon eylem kaynağı."""

    name: str = "base"

    def __init__(self):
        self.target: Optional[np.ndarray] = None

    def reset(self, qpos: Sequence[float]) -> None:
        self.target = clip_action(np.asarray(qpos, dtype=np.float32)[:7])

    @abc.abstractmethod
    def act(self, qpos: Sequence[float]) -> np.ndarray:
        """Sıradaki 7 boyutlu pd_joint_pos hedefi."""

    def poll_events(self) -> Set[str]:
        return set()

    def ready(self) -> bool:
        """Operatör başlamaya hazır mı? Bölüm (ve süre sayacı) ancak True olunca başlar."""
        return True

    def start_hint(self) -> str:
        """Bölüm başlamadan önce gösterilecek ipucu."""
        return ""

    def status_text(self) -> str:
        return ""

    def publish_status(self, **status) -> None:
        """Cihaza durum bilgisi gönder (ör. telefona talimat). Varsayılan: yok."""

    def close(self) -> None:
        pass


class GamepadKeyboardActionSource(ActionSource):
    """
    Mevcut `Teleoperasyonlar/` PS4/klavye denetleyicisini (MahutPS4Controller) mutlak
    pd_joint_pos hedefine çevirir: her adımda hedef += artım * ölçek.

    Klavye (pygame penceresi odakta olmalı): A/D joint0, W/S joint1, ↑/↓ joint2, ←/→ joint3,
    Q/E joint4, SPACE kapat, F aç, R sıfırla.
    """

    name = "gamepad"

    def __init__(self, controller=None, arm_step: float = 0.06, gripper_step: float = 0.004):
        super().__init__()
        if controller is None:
            from mahut_kol.utils.teleop_ps4 import MahutPS4Controller

            controller = MahutPS4Controller()
        self.controller = controller
        self.arm_step = arm_step  # rad / adım (tam eğik çubukta, arm_scale ile çarpılır)
        self.gripper_step = gripper_step

    def act(self, qpos: Sequence[float]) -> np.ndarray:
        if self.target is None:
            self.reset(qpos)
        delta = np.asarray(self.controller.poll_action(), dtype=np.float32)
        scale = max(getattr(self.controller, "arm_scale", 1.0), 1e-6)
        gscale = max(getattr(self.controller, "gripper_scale", 1.0), 1e-6)
        t = self.target.copy()
        t[:ARM_DOF] += delta[:ARM_DOF] / scale * self.arm_step
        t[ARM_DOF:] += delta[ARM_DOF:] / gscale * self.gripper_step
        self.target = clip_action(t)
        return self.target.copy()

    def poll_events(self) -> Set[str]:
        try:
            return {"reset"} if self.controller.poll_reset() else set()
        except Exception:
            return set()


class ScriptedActionSource(ActionSource):
    """Önceden üretilmiş eylem dizisini oynatır (test ve scripted demolar için)."""

    name = "scripted"

    def __init__(self, actions: Sequence[Sequence[float]]):
        super().__init__()
        self.actions = [clip_action(a) for a in actions]
        self.i = 0

    def act(self, qpos: Sequence[float]) -> np.ndarray:
        if self.i < len(self.actions):
            self.target = self.actions[self.i]
            self.i += 1
        elif self.target is None:
            self.reset(qpos)
        return self.target.copy()

    @property
    def done(self) -> bool:
        return self.i >= len(self.actions)
