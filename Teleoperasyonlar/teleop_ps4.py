"""
MahutKol PS4 Kontrolcüsü (DualShock 4) ve Gamepad Teleoperasyon Modülü.

Bu modül, PS4 gamepad analog ve tetik verilerini MahutKol robotunun
eklem uzayı artımsal (delta pos) eylemlerine dönüştürür.
5-DOF kol + 2 parmak kıskaç (7 aksiyon boyutu).
"""

import os
from typing import List, Optional
import numpy as np

# Arka plan joystick olaylarını destekle
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"


class MahutPS4Controller:
    """
    MahutKol için PS4 / Gamepad teleoperasyon denetleyicisi.
    """

    def __init__(
        self,
        deadzone: float = 0.12,
        arm_scale: float = 0.25,
        gripper_scale: float = 0.5,
    ):
        self.deadzone = deadzone
        self.arm_scale = arm_scale
        self.gripper_scale = gripper_scale
        self.joystick = None
        self._pygame_initialized = False

    def _deadzone(self, val: float) -> float:
        """Analog çubuk için ölü bölge filtresi uygular."""
        if abs(val) < self.deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - self.deadzone) / (1.0 - self.deadzone)

    def compute_action_from_inputs(
        self,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        l2: float,
        r2: float,
        droll: float = 0.0,
    ) -> np.ndarray:
        """
        Ham girdi değerlerinden 7 boyutlu eklem aksiyonu hesaplar:
        [joint0, joint1, joint2, joint3, joint4, finger_left, finger_right]
        """
        lx_filt = self._deadzone(lx)
        ly_filt = self._deadzone(ly)
        rx_filt = self._deadzone(rx)
        ry_filt = self._deadzone(ry)
        droll_filt = self._deadzone(droll)

        # 5 Kol eklemi:
        # joint0 (Taban Yaw): Sol stick X (- sol, + sağ)
        # joint1 (Omuz Pitch): Sol stick Y (ileri itince -ly pozitiftir)
        # joint2 (Dirsek Pitch): Sağ stick Y (ileri itince -ry pozitiftir)
        # joint3 (Bilek Pitch): Sağ stick X (sağa itince rx pozitiftir)
        # joint4 (Bilek Roll): D-Pad X (- sol, + sağ) veya droll
        j0 = float(-lx_filt * self.arm_scale)
        j1 = float(-ly_filt * self.arm_scale)
        j2 = float(-ry_filt * self.arm_scale)
        j3 = float(rx_filt * self.arm_scale)
        j4 = float(droll_filt * self.arm_scale)

        # Tutucu (Prismatic):
        # URDF sınırları: lower=0.0 (açık), upper=0.042 (kapalı).
        # R2 tetiği basıldıkça kapatır (+delta).
        # L2 tetiği basıldıkça açar (-delta).
        val_r2 = float(r2) if float(r2) > 0.05 else 0.0
        val_l2 = float(l2) if float(l2) > 0.05 else 0.0
        grip_delta = float((val_r2 - val_l2) * self.gripper_scale)

        return np.array([j0, j1, j2, j3, j4, grip_delta, grip_delta], dtype=np.float32)

    def is_reset_requested(
        self,
        pressed_buttons: List[int],
        pressed_keys: List[str],
    ) -> bool:
        """
        Üçgen (Button 2), Options (Button 9), Button 3 veya klavyede 'r'
        tuşuna basılıp basılmadığını denetler.
        """
        reset_buttons = {2, 3, 9}
        if any(b in reset_buttons for b in pressed_buttons):
            return True
        if any(k.lower() == "r" for k in pressed_keys):
            return True
        return False

    def init_joystick(self) -> bool:
        """
        Pygame joystick alt sistemini başlatır ve bağlı kontrolcüyü seçer.
        Bağlı gamepad varsa True, yoksa False döner.
        """
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()

            self._pygame_initialized = True
            count = pygame.joystick.get_count()
            if count > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                return True
            return False
        except Exception as e:
            print(f"[UYARI] Pygame joystick başlatılamadı: {e}")
            return False

    def get_controller_name(self) -> Optional[str]:
        """Bağlı kontrolcünün adını döner."""
        if self.joystick is not None:
            return self.joystick.get_name()
        return None

    def poll_action(self) -> np.ndarray:
        """
        Bağlı gamepad veya klavyeden canlı olayları okuyup 7 boyutlu aksiyon üretir.
        """
        if not self._pygame_initialized:
            if not self.init_joystick():
                return np.zeros(7, dtype=np.float32)

        import pygame
        pygame.event.pump()

        if self.joystick is not None:
            num_axes = self.joystick.get_numaxes()
            raw_lx = self.joystick.get_axis(0) if num_axes > 0 else 0.0
            raw_ly = self.joystick.get_axis(1) if num_axes > 1 else 0.0

            if num_axes >= 6:
                # Standart Linux PS4 / DualShock haritası
                raw_l2 = self.joystick.get_axis(2)
                raw_rx = self.joystick.get_axis(3)
                raw_ry = self.joystick.get_axis(4)
                raw_r2 = self.joystick.get_axis(5)

                l2 = max(0.0, (raw_l2 + 1.0) / 2.0)
                r2 = max(0.0, (raw_r2 + 1.0) / 2.0)
            else:
                raw_rx = self.joystick.get_axis(2) if num_axes > 2 else 0.0
                raw_ry = self.joystick.get_axis(3) if num_axes > 3 else 0.0
                l2 = 0.0
                r2 = 0.0

            droll = 0.0
            if self.joystick.get_numhats() > 0:
                droll = float(self.joystick.get_hat(0)[0])

            return self.compute_action_from_inputs(raw_lx, raw_ly, raw_rx, raw_ry, l2, r2, droll=droll)

        # Gamepad yoksa klavye yedek kontrolü
        keys = pygame.key.get_pressed()
        lx, ly, rx, ry, l2, r2, droll = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if keys[pygame.K_a]: lx = 1.0
        if keys[pygame.K_d]: lx = -1.0
        if keys[pygame.K_w]: ly = -1.0
        if keys[pygame.K_s]: ly = 1.0
        if keys[pygame.K_UP]: ry = -1.0
        if keys[pygame.K_DOWN]: ry = 1.0
        if keys[pygame.K_LEFT]: rx = -1.0
        if keys[pygame.K_RIGHT]: rx = 1.0
        if keys[pygame.K_q]: droll = -1.0
        if keys[pygame.K_e]: droll = 1.0
        if keys[pygame.K_SPACE]: r2 = 1.0  # Kapat
        if keys[pygame.K_f]: l2 = 1.0      # Aç

        return self.compute_action_from_inputs(lx, ly, rx, ry, l2, r2, droll=droll)

    def poll_reset(self) -> bool:
        """Canlı olarak reset basılıp basılmadığını kontrol eder."""
        if not self._pygame_initialized:
            return False

        import pygame
        pygame.event.pump()

        pressed_buttons = []
        if self.joystick is not None:
            num_buttons = self.joystick.get_numbuttons()
            for b in range(num_buttons):
                if self.joystick.get_button(b):
                    pressed_buttons.append(b)

        pressed_keys = []
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            pressed_keys.append("r")

        return self.is_reset_requested(pressed_buttons, pressed_keys)
