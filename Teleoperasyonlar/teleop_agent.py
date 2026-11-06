import os
from typing import Dict, Optional, Tuple
import numpy as np

# Pygame pencereli olmadan çalışabilmesi ve joystick olaylarını dinlemesi için
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
import pygame


class TeleopController:
    """
    PS4 Kolu (DualShock 4), Genel Gamepad ve Klavye ile teleoperasyon denetleyicisi.
    
    PS4 Kolu Eşleştirmesi (End-Effector Kartezyen Kontrolü):
    - Sol Analog Çubuk:
        - İleri / Geri -> X ekseni (dx)
        - Sol / Sağ     -> Y ekseni (dy)
    - Sağ Analog Çubuk:
        - Yukarı / Aşağı -> Z ekseni (dz)
        - Sol / Sağ       -> Yaw dönüşü (dyaw)
    - D-Pad (Yön Tuşları):
        - Yukarı / Aşağı -> Pitch açısı (dpitch)
        - Sol / Sağ       -> Roll açısı (droll)
    - L2 / R2 Tetikleri:
        - R2: Gripper Kapat (tut)
        - L2: Gripper Aç (bırak)
    - Özel Tuşlar:
        - Options / Triangle: Bölümü Sıfırla (Reset)
        - Cross (X): Epizotu Başarılı Kaydet
        - Circle (O): Epizotu İptal Et / Atla
    """

    def __init__(
        self,
        device_type: str = "gamepad",
        deadzone: float = 0.08,
        action_scale: float = 0.8,
    ):
        """
        Parametreler:
        - device_type: 'gamepad' (PS4/Xbox), 'keyboard', 'spacemouse' veya 'vr'
        - deadzone: Analog çubuk ölü bölge eşiği (drift engelleme)
        - action_scale: Hız ve hassasiyet çarpanı
        """
        self.device_type = device_type.lower()
        self.deadzone = deadzone
        self.action_scale = action_scale
        self.joystick: Optional[pygame.joystick.Joystick] = None

        pygame.init()

        if self.device_type in ["gamepad", "ps4"]:
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            if joystick_count == 0:
                print("[UYARI] Bağlı gamepad/PS4 kolu bulunamadı! Klavyeye geçiliyor.")
                self.device_type = "keyboard"
            else:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"[BİLGİ] Gamepad bağlandı: {self.joystick.get_name()}")

        self.gripper_state = 0.0  # -1.0: tamamen açık, 1.0: tamamen kapalı

    def _apply_deadzone(self, value: float) -> float:
        """Analog çubuk titremelerini ve drifti önler."""
        if abs(value) < self.deadzone:
            return 0.0
        # Ölü bölge sonrası doğrusal ölçekleme
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def get_action(self) -> np.ndarray:
        """
        Anlık girişlere göre normalleştirilmiş [-1, 1] eylem vektörü üretir.
        
        Çıktı Formatı: 7 elemanlı np.ndarray:
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        pygame.event.pump()

        if self.device_type in ["gamepad", "ps4"] and self.joystick is not None:
            return self._get_gamepad_action()
        elif self.device_type == "keyboard":
            return self._get_keyboard_action()
        else:
            return np.zeros(7, dtype=np.float32)

    def _get_gamepad_action(self) -> np.ndarray:
        """PS4 (DualShock 4) kontrolcüsünden gelen eksen ve butonları eyleme dönüştürür."""
        num_axes = self.joystick.get_numaxes()

        # Linux üzerinde PS4 (DualShock 4) ve evdev eksen haritası:
        # Axis 0: Sol Analog X
        # Axis 1: Sol Analog Y
        # Axis 2: L2 Tetik (boştayken -1.0, basılınca +1.0)
        # Axis 3: Sağ Analog X
        # Axis 4: Sağ Analog Y
        # Axis 5: R2 Tetik (boştayken -1.0, basılınca +1.0)
        raw_lx = self.joystick.get_axis(0) if num_axes > 0 else 0.0
        raw_ly = self.joystick.get_axis(1) if num_axes > 1 else 0.0

        if num_axes >= 6:
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

        dx = -self._apply_deadzone(raw_ly) * self.action_scale
        dy = self._apply_deadzone(raw_lx) * self.action_scale

        # Sağ analog Y ekseni -> Z yukarı/aşağı (terslenir)
        dz = -self._apply_deadzone(raw_ry) * self.action_scale
        # Sağ analog X ekseni -> Yaw rotasyonu
        dyaw = self._apply_deadzone(raw_rx) * self.action_scale

        # D-Pad (Hat) ile Roll & Pitch
        droll, dpitch = 0.0, 0.0
        if self.joystick.get_numhats() > 0:
            hat_x, hat_y = self.joystick.get_hat(0)
            droll = float(hat_x) * self.action_scale * 0.5
            dpitch = float(hat_y) * self.action_scale * 0.5

        # Tetikler ile Gripper Kontrolü
        if r2 > 0.1:
            self.gripper_state = min(1.0, self.gripper_state + r2 * 0.1)  # Kapat
        elif l2 > 0.1:
            self.gripper_state = max(-1.0, self.gripper_state - l2 * 0.1)  # Aç

        return np.array([dx, dy, dz, droll, dpitch, dyaw, self.gripper_state], dtype=np.float32)

    def _get_keyboard_action(self) -> np.ndarray:
        """Klavye tuş haritası (W/S, A/D, Q/E, J/L, I/K, Space)."""
        keys = pygame.key.get_pressed()
        dx, dy, dz = 0.0, 0.0, 0.0
        droll, dpitch, dyaw = 0.0, 0.0, 0.0

        if keys[pygame.K_w]: dx += self.action_scale
        if keys[pygame.K_s]: dx -= self.action_scale
        if keys[pygame.K_a]: dy += self.action_scale
        if keys[pygame.K_d]: dy -= self.action_scale
        if keys[pygame.K_q]: dz += self.action_scale
        if keys[pygame.K_e]: dz -= self.action_scale

        if keys[pygame.K_j]: dyaw += self.action_scale * 0.5
        if keys[pygame.K_l]: dyaw -= self.action_scale * 0.5
        if keys[pygame.K_i]: dpitch += self.action_scale * 0.5
        if keys[pygame.K_k]: dpitch -= self.action_scale * 0.5
        if keys[pygame.K_u]: droll += self.action_scale * 0.5
        if keys[pygame.K_o]: droll -= self.action_scale * 0.5

        if keys[pygame.K_SPACE]:
            self.gripper_state = 1.0 if self.gripper_state < 0 else -1.0

        return np.array([dx, dy, dz, droll, dpitch, dyaw, self.gripper_state], dtype=np.float32)

    def is_reset_requested(self) -> bool:
        """Kullanıcının bölümü sıfırlamak isteyip istemediğini kontrol eder (Triangle / Options / R tuşu)."""
        pygame.event.pump()
        if self.device_type in ["gamepad", "ps4"] and self.joystick is not None:
            num_buttons = self.joystick.get_numbuttons()
            # PS4 Linux: Button 2: Üçgen, Button 9: Options
            # Generic gamepad: Button 3: Üçgen/Y, Button 6/7/8/9: Back/Start
            if num_buttons > 9:
                if self.joystick.get_button(2) or self.joystick.get_button(9):
                    return True
            elif num_buttons > 3:
                if self.joystick.get_button(2) or self.joystick.get_button(3):
                    return True
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_r])

    def is_save_requested(self) -> bool:
        """Başarılı gösterimi onaylayıp kaydetme tuşu (Cross / X veya Enter tuşu)."""
        pygame.event.pump()
        if self.device_type in ["gamepad", "ps4"] and self.joystick is not None:
            # Button 0: Cross (X)
            if self.joystick.get_numbuttons() > 0:
                return bool(self.joystick.get_button(0))
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_RETURN])

    def close(self):
        """Kaynakları serbest bırakır."""
        if self.joystick is not None:
            self.joystick.quit()
        pygame.quit()

