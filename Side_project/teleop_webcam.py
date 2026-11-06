#!/usr/bin/env python3
"""
MuJoCo SO-100 Robot Kolu İçin Web Kamerası Tabanlı (MediaPipe Hands) Teleoperasyon Modülü
(Yüksek Performanslı, Düşük Gecikmeli ve Gelişmiş Ergonomik Versiyon)

Temel Optimizasyonlar ve İyileştirmeler:
  1. Sıfır Tampon Gecikmesi (Zero-Latency Threaded Video Capture):
     - ThreadedCamera sınıfı arka planda sürekli en son kamerayı okur ve Linux V4L2
       tamponundaki birikmeyi (buffer queue lag) tamamen yok eder (<1ms kare erişimi).
  2. Gerçek Zamanlı Fizik Alt-Adımları (Real-Time Physics Sub-Stepping):
     - Kamera kareleri arası geçen süreye göre (~33ms) her döngüde dinamik olarak
       yaklaşık 16 MuJoCo fizik adımı (dt=0.002s) çalıştırılır. Böylece robot kolu
       simülasyonda 1:1 gerçek zaman hızında hareket eder ve motorlar gecikmesiz tepki verir.
  3. Çift Modlu Debriyaj (Clutch) Sistemi:
     - Toggle Modu (Varsayılan - Önerilen): [Space] tuşuna bir kez basıldığında debriyaj
       açılır ve el rahatça hareket ettirilir. Tekrar [Space] basıldığında robot donar.
     - Hold Modu: [Space] tuşuna basılı tutularak kullanılır; 750ms tolerans ile Linux
       tuş tekrarı (500ms repeat delay) kaynaklı titreme/sıfırlanmalar engellenmiştir.
     - [M] tuşu ile Toggle ve Hold modları arasında anında geçiş yapılabilir.
     - Fare Sol Tık desteği: Pencereye tıklandığında odak sağlanır ve debriyaj tetiklenir.
  4. Kararlı 3B Derinlik (İleri/Geri) Haritalaması:
     - Elin eğilip bükülmesinden etkilenmeyen hipotenüs tabanlı avuç ölçeği (palm scale)
       ve %3 ölü bölge (deadband) ile bilek dönüşlerinde istenmeyen ileri/geri hareketler önlenir.
  5. Bilek Açısı (Pitch & Roll) İkincil Görev Koruması ve Aç/Kapa:
     - [W] tuşu ile bilek açı takibi açılıp kapatılabilir. Kapalıyken sadece 3B konum takip edilir.
  6. Optimize Edilmiş EMA Filtresi ve Güvenlik Sınırları:
     - alpha_pos=0.60 ile yüksek tepki hızı sağlanırken piksel titremesi filtrelenir.
"""

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Güvenli İçe Aktarımı
try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
except (ImportError, AttributeError):
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except (ImportError, AttributeError):
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing
        import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# MuJoCo İçe Aktarımı
try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("HATA: 'mujoco' kütüphanesi bulunamadı! 'pip install mujoco' çalıştırın.")
    sys.exit(1)

# Yerel IK Çözücüyü İçe Aktar
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ik_solver import SO100IKSolver


# ==============================================================================
# SİSTEM VE KONTROL YAPILANDIRMASI (CONFIG)
# ==============================================================================
CONFIG = {
    # Kamera Ayarları
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "flip_camera": True,             # Ayna görüntüsü modu

    # MediaPipe Hands Ayarları
    "max_num_hands": 1,
    "min_detection_confidence": 0.70,
    "min_tracking_confidence": 0.70,

    # Hassasiyet ve Ölçek Çarpanları
    "scale_lateral": 0.50,           # Kamera X farkı -> Robot X (Sağ/Sol)
    "scale_vertical": 0.50,          # Kamera Y farkı -> Robot Z (Yukarı/Aşağı)
    "scale_depth": 0.55,             # El boyutu oranı -> Robot Y (İleri/Geri)
    "scale_pitch": 1.0,              # Avuç eğimi -> Robot Wrist_Pitch
    "scale_roll": 1.0,               # Avuç dönüşü -> Robot Wrist_Roll
    "depth_deadband": 0.03,          # El boyutunda %3'lük ölü bölge (titreşim engelleme)

    # Eksen Yönleri (MuJoCo SO-100 Koordinatları)
    "sign_lateral": 1.0,             # Sağa el hareketi -> Robot +X
    "sign_vertical": 1.0,            # Yukarı el hareketi -> Robot +Z
    "sign_depth": -1.0,              # El yaklaşınca (büyüyünce) -> Robot İleri (-Y)

    # Gripper Eşikleri
    "pinch_min": 0.20,               # Bu oranın altında gripper = -1.0 (Kapalı)
    "pinch_max": 0.72,               # Bu oranın üstünde gripper = 1.0 (Tam Açık)

    # Titreme Önleme ve Hızlı Tepki (Low Latency Smoothing)
    "ema_alpha_pos": 0.60,           # Pozisyon EMA filtresi (hızlı ve net tepki)
    "ema_alpha_rot": 0.50,           # Açı EMA filtresi
    "ema_alpha_gripper": 0.55,       # Gripper EMA filtresi
    "max_step_pos": 0.060,           # Tek karede izin verilen maksimum konum sıçraması (m)

    # Çalışma Hacmi Sınırları (Workspace Bounding Box)
    "workspace_limits": {
        "x": (-0.28, 0.28),          # Robot Sağ/Sol sınırları
        "y": (-0.48, -0.05),         # Robot İleri/Geri sınırları
        "z": (0.015, 0.42),          # Zemin üstü çalışma yüksekliği
    },

    # Debriyaj (Clutch) Ayarları
    "clutch_mode": "toggle",         # "toggle" (tıkla-aç/tıkla-kapa) veya "hold" (basılı tut)
    "clutch_space_timeout": 0.75,    # Hold modunda Linux 500ms tuş gecikmesini tolere eden süre (sn)

    # Bilek Açı Takibi
    "wrist_tracking": True,          # Bilek pitch/roll açısını robota aktar (W ile aç/kapa)

    # Ters Kinematik (IK) Ayarları
    "ik_iterations": 6,
    "ik_tolerance": 1e-4,
    "ik_damping": 1e-3,
}


# ==============================================================================
# SIFIR GECİKMELİ İŞ PARÇACIKLI KAMERA AKIŞI (THREADED CAMERA)
# ==============================================================================
class ThreadedCamera:
    """
    Arka planda bağımsız bir thread ile kamerayı sürekli okur.
    Linux V4L2 sürücü tamponunda kare birikmesini (lag) sıfırlar ve ana döngüye
    daima en son gerçek zamanlı kareyi mikrosaniye mertebesinde sunar.
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Web kamerası açılamadı (Kamera ID: {camera_id})!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Linux V4L2 için donanımsal tampon boyutunu 1'e ayarla
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.running = True
        self.ret, self.frame = self.cap.read()

        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """En güncel kamera karesini gecikmesiz döner."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


# ==============================================================================
# MATEMATİK VE HESAPLAMA YARDIMCI FONKSİYONLARI
# ==============================================================================
def compute_pinch_ratio(
    thumb_tip: np.ndarray,
    index_tip: np.ndarray,
    palm_ref_1: np.ndarray,
    palm_ref_2: np.ndarray,
) -> float:
    """
    Başparmak ucu ve işaret parmağı ucu arasındaki mesafeyi
    avuç genişliğiyle normalize ederek boyut bağımsız pinch oranını döner.
    """
    pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
    palm_size = float(np.linalg.norm(palm_ref_1 - palm_ref_2))
    if palm_size < 1e-4:
        return 1.0
    return pinch_dist / palm_size


def compute_gripper_value(
    pinch_ratio: float,
    pinch_min: float = 0.20,
    pinch_max: float = 0.72,
) -> float:
    """
    Pinch oranını [-1.0, 1.0] aralığına yumuşak interpolasyon (Smoothstep) ile eşler.
      - pinch_ratio <= pinch_min: -1.0 (Kapalı)
      - pinch_ratio >= pinch_max:  1.0 (Açık)
    """
    if pinch_ratio <= pinch_min:
        return -1.0
    if pinch_ratio >= pinch_max:
        return 1.0

    t = (pinch_ratio - pinch_min) / (pinch_max - pinch_min)
    t_smooth = 3.0 * (t ** 2) - 2.0 * (t ** 3)
    return float(-1.0 + 2.0 * t_smooth)


def gripper_to_jaw_angle(gripper_val: float, jaw_min: float, jaw_max: float) -> float:
    """[-1.0, 1.0] gripper değerini MuJoCo jaw eklem açısına (radyan) dönüştürür."""
    norm_val = 0.5 * (gripper_val + 1.0)
    norm_val = np.clip(norm_val, 0.0, 1.0)
    return float(jaw_min + norm_val * (jaw_max - jaw_min))


def compute_hand_orientation(
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    pinky_mcp: np.ndarray,
) -> Tuple[float, float]:
    """
    3B el landmark noktalarından avuç yüzey normalini ve Pitch/Roll açılarını (radyan) hesaplar.
    """
    v_long = middle_mcp - wrist
    v_lat = index_mcp - pinky_mcp

    normal = np.cross(v_lat, v_long)
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-6:
        return 0.0, 0.0
    normal = normal / norm_val

    # Pitch açısı (El eğimi: Yukarı / Aşağı yönelim)
    pitch = math.atan2(float(normal[1]), math.sqrt(float(normal[0]) ** 2 + float(normal[2]) ** 2))

    # Roll açısı (Eksenel dönüş: Avuç içinin sağa/sola dönmesi)
    roll = math.atan2(float(normal[0]), float(normal[2]))

    return pitch, roll


def apply_ema(new_val: np.ndarray, prev_val: np.ndarray, alpha: float) -> np.ndarray:
    """Üstel Hareketli Ortalama (EMA / Low-pass filter) uygular."""
    alpha = float(np.clip(alpha, 0.01, 1.0))
    return alpha * new_val + (1.0 - alpha) * prev_val


def apply_step_limiter(candidate_pos: np.ndarray, current_pos: np.ndarray, max_step: float) -> np.ndarray:
    """Tek bir karede izin verilen maksimum konum sıçramasını sınırlar."""
    step_vec = candidate_pos - current_pos
    dist = float(np.linalg.norm(step_vec))
    if dist > max_step and dist > 1e-6:
        return current_pos + step_vec * (max_step / dist)
    return candidate_pos.copy()


def clamp_to_workspace(pos: np.ndarray, limits: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Verilen 3B konumu çalışma hacmi (workspace) sınırları içine kırpar."""
    clamped = pos.copy()
    clamped[0] = np.clip(clamped[0], limits["x"][0], limits["x"][1])
    clamped[1] = np.clip(clamped[1], limits["y"][0], limits["y"][1])
    clamped[2] = np.clip(clamped[2], limits["z"][0], limits["z"][1])
    return clamped


def map_camera_deltas_to_robot(
    dx_cam: float,
    dy_cam: float,
    hand_width: float,
    anchor_hand_width: float,
    config: dict,
) -> np.ndarray:
    """
    Kamera koordinat farklarını ve el boyutu değişimini MuJoCo robot koordinat farklarına eşler.
    Ölü bölge (deadband) ile el titremeleri filtrelenir.
    """
    delta_x_robot = config["sign_lateral"] * dx_cam * config["scale_lateral"]
    delta_z_robot = config["sign_vertical"] * (-dy_cam) * config["scale_vertical"]

    width_ratio = (hand_width - anchor_hand_width) / max(anchor_hand_width, 1e-4)

    # Ölü bölge (Deadband) kontrolü
    deadband = config.get("depth_deadband", 0.0)
    if abs(width_ratio) < deadband:
        effective_ratio = 0.0
    else:
        effective_ratio = math.copysign(abs(width_ratio) - deadband, width_ratio)

    delta_y_robot = config["sign_depth"] * effective_ratio * config["scale_depth"]
    return np.array([delta_x_robot, delta_y_robot, delta_z_robot], dtype=np.float64)


# ==============================================================================
# OPENCV GÖRSELLEŞTİRME VE HUD ARAYÜZÜ
# ==============================================================================
class WebcamTeleopHUD:
    """Kamera görüntüsü üzerine iskelet, debriyaj ve gripper telemetrisini çizen sınıf."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        clutch_active: bool,
        gripper_val: float,
        target_pos: np.ndarray,
        fps: float,
        hand_detected: bool,
        ref_pt_px: Optional[Tuple[int, int]] = None,
        thumb_px: Optional[Tuple[int, int]] = None,
        index_px: Optional[Tuple[int, int]] = None,
        clutch_mode: str = "toggle",
        wrist_active: bool = True,
    ) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. Pinch Çizgisi ve Parmak Uçları
        if thumb_px is not None and index_px is not None:
            pct = 0.5 * (gripper_val + 1.0)
            line_color = (
                int(255 * (1.0 - pct)),
                int(255 * pct),
                int(100 + 155 * pct),
            )
            cv2.line(frame, thumb_px, index_px, line_color, 3, cv2.LINE_AA)
            cv2.circle(frame, thumb_px, 6, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, index_px, 6, (0, 255, 255), -1, cv2.LINE_AA)

        # 2. Avuç İçi Referans Noktası (Landmark 9)
        if ref_pt_px is not None:
            ref_color = (0, 255, 0) if clutch_active else (140, 140, 140)
            cv2.circle(frame, ref_pt_px, 8, ref_color, -1, cv2.LINE_AA)
            cv2.circle(frame, ref_pt_px, 14, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. Sol Üst Debriyaj (Clutch) Durum Rozeti
        badge_x, badge_y = 15, 15
        badge_w, badge_h = 330, 44
        if clutch_active:
            cv2.rectangle(
                frame,
                (badge_x, badge_y),
                (badge_x + badge_w, badge_y + badge_h),
                (15, 135, 15),
                -1,
            )
            cv2.rectangle(
                frame,
                (badge_x, badge_y),
                (badge_x + badge_w, badge_y + badge_h),
                (60, 255, 60),
                2,
            )
            mode_lbl = "TOGGLE" if clutch_mode == "toggle" else "HOLD"
            cv2.putText(
                frame,
                f"CLUTCH: AKTIF ({mode_lbl})",
                (badge_x + 12, badge_y + 29),
                self.font,
                0.64,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(
                frame,
                (badge_x, badge_y),
                (badge_x + badge_w, badge_y + badge_h),
                (25, 25, 160),
                -1,
            )
            cv2.rectangle(
                frame,
                (badge_x, badge_y),
                (badge_x + badge_w, badge_y + badge_h),
                (60, 60, 255),
                2,
            )
            cv2.putText(
                frame,
                "CLUTCH: PASIF (Space Bas)",
                (badge_x + 12, badge_y + 29),
                self.font,
                0.64,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Gripper Açıklık Barı ve Yüzdesi
        bar_x, bar_y = 15, 72
        bar_w, bar_h = 220, 20
        grip_pct = int(np.clip(0.5 * (gripper_val + 1.0) * 100.0, 0.0, 100.0))

        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_w + 105, bar_y + bar_h + 5), (30, 30, 30), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        fill_w = int(bar_w * (grip_pct / 100.0))
        bar_color = (0, int(2.55 * grip_pct), int(255 - 1.55 * grip_pct))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        grip_label = "KAPALI" if grip_pct < 15 else "ACIK" if grip_pct > 80 else "TUTUYOR"
        cv2.putText(
            frame,
            f"{grip_pct}% {grip_label}",
            (bar_x + bar_w + 8, bar_y + 15),
            self.font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 5. Telemetri ve Durum Göstergeleri
        telem_text = f"Hedef EE: X:{target_pos[0]:+.2f}  Y:{target_pos[1]:+.2f}  Z:{target_pos[2]:+.2f} m"
        cv2.putText(frame, telem_text, (15, 120), self.font, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

        wrist_txt = "Bilek: ACIK [W]" if wrist_active else "Bilek: SABIT [W]"
        wrist_col = (0, 255, 255) if wrist_active else (150, 150, 150)
        cv2.putText(frame, wrist_txt, (15, 142), self.font, 0.46, wrist_col, 1, cv2.LINE_AA)

        if not hand_detected:
            cv2.putText(
                frame,
                "UYARI: El Bulunamadi! Kameraya elinizi gosterin.",
                (15, 166),
                self.font,
                0.48,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        # 6. Alt Bilgi Çubuğu (Gecikmesiz FPS ve Tuş Rehberi)
        bottom_bar_y = h - 25
        cv2.rectangle(frame, (0, h - 35), (w, h), (18, 18, 18), -1)
        guide_text = "Space/Tik: Debriyaj | M: Mod | W: Bilek | R: Sifirla | Q: Cikis"
        cv2.putText(frame, guide_text, (12, bottom_bar_y + 16), self.font, 0.43, (180, 180, 180), 1, cv2.LINE_AA)

        fps_text = f"FPS: {fps:.0f} (Dusuk Ping)"
        cv2.putText(frame, fps_text, (w - 170, bottom_bar_y + 16), self.font, 0.43, (0, 255, 100), 1, cv2.LINE_AA)

        return frame


# ==============================================================================
# ANA TELEOPERASYON MOTORU
# ==============================================================================
def run_webcam_teleoperation(scene_file: Path, camera_id: int = 0, flip_cam: bool = True):
    """
    Sıfır tampon gecikmeli ve gerçek zamanlı fizik alt-adımlı teleoperasyon döngüsü.
    """
    print("=" * 70)
    print("   MuJoCo SO-100 Robot Kolu Webcam (MediaPipe Hands) Teleoperasyonu")
    print("   [Ultra Düşük Gecikmeli & Çoklu Fizik Adımlı Sürüm]")
    print("=" * 70)
    print(f"[Ayar] Sahne Dosyası: {scene_file}")
    print(f"[Ayar] Kamera ID: {camera_id}")

    # 1. MuJoCo Model ve Veri Yükleme
    if not scene_file.exists():
        raise FileNotFoundError(f"MuJoCo sahne dosyası bulunamadı: {scene_file}")

    model = mujoco.MjModel.from_xml_path(str(scene_file))
    data = mujoco.MjData(model)

    mocap_id = -1
    for i in range(model.nbody):
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if b_name == "target_mocap":
            mocap_id = model.body_mocapid[i]
            break

    ik_solver = SO100IKSolver(
        model=model,
        site_name="ee_site",
        body_name="Fixed_Jaw",
        damping_pos=CONFIG["ik_damping"],
    )

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    home_ee_pos, home_ee_rot = ik_solver.get_ee_pose(data)
    home_qpos = data.qpos[:5].copy()

    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = home_ee_pos.copy()

    jaw_min = float(model.jnt_range[5, 0])
    jaw_max = float(model.jnt_range[5, 1])

    # 2. Sıfır Gecikmeli Threaded Kamera Başlatma
    cam = ThreadedCamera(
        camera_id=camera_id,
        width=CONFIG["frame_width"],
        height=CONFIG["frame_height"],
    )

    mp_hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=CONFIG["max_num_hands"],
        min_detection_confidence=CONFIG["min_detection_confidence"],
        min_tracking_confidence=CONFIG["min_tracking_confidence"],
    )

    hud = WebcamTeleopHUD()

    # 3. Teleoperasyon Kontrol Değişkenleri
    anchor_robot_pos = home_ee_pos.copy()
    anchor_robot_pitch = float(home_qpos[3])
    anchor_robot_roll = float(home_qpos[4])

    anchor_hand_ref = np.zeros(2, dtype=np.float64)
    anchor_hand_size = 100.0
    anchor_hand_pitch = 0.0
    anchor_hand_roll = 0.0

    smooth_target_pos = home_ee_pos.copy()
    smooth_target_pitch = float(anchor_robot_pitch)
    smooth_target_roll = float(anchor_robot_roll)
    smooth_gripper = -1.0

    # Debriyaj Durumları
    clutch_mode = CONFIG["clutch_mode"]     # "toggle" veya "hold"
    clutch_active = False
    prev_clutch = False
    last_space_press_time = 0.0
    wrist_tracking_active = CONFIG["wrist_tracking"]

    mouse_clicked_flag = False

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_clicked_flag, clutch_active
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked_flag = True

    window_name = "SO-100 Kamera Teleoperasyonu"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    ws_limits = CONFIG["workspace_limits"]
    alpha_pos = CONFIG["ema_alpha_pos"]
    alpha_rot = CONFIG["ema_alpha_rot"]
    alpha_grip = CONFIG["ema_alpha_gripper"]
    max_step = CONFIG["max_step_pos"]

    t_prev_frame = time.time()
    fps_val = 30.0

    print("\n[Hazır] Ultra Düşük Gecikmeli Teleoperasyon Başlatıldı!")
    print("Kontrol Kısayolları:")
    print("  - [Space] veya [Sol Fare Tık]: Debriyaj Aç / Kapa (Toggle Modu)")
    print("  - [M] tuşu: Debriyaj Modunu Değiştir (Toggle / Hold)")
    print("  - [W] tuşu: Bilek Açısı Takibini Aç / Kapat (Pitch/Roll)")
    print("  - [R] tuşu: Robotu başlangıç pozisyonuna sıfırla")
    print("  - [Q] veya [ESC]: Çıkış yap\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # En Güncel Kareyi Oku (Sıfır Tampon Gecikmesi)
                ret, frame = cam.read_latest()
                if not ret or frame is None:
                    time.sleep(0.002)
                    continue

                if flip_cam:
                    frame = cv2.flip(frame, 1)

                h_px, w_px, _ = frame.shape

                # MediaPipe El Tespiti
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = mp_hands_detector.process(rgb_frame)
                rgb_frame.flags.writeable = True

                hand_detected = False
                ref_pt_px = None
                thumb_px = None
                index_px = None

                current_hand_ref = np.zeros(2, dtype=np.float64)
                current_hand_size = 100.0
                current_hand_pitch = 0.0
                current_hand_roll = 0.0
                current_raw_gripper = smooth_gripper

                if results.multi_hand_landmarks:
                    hand_detected = True
                    hand_lms = results.multi_hand_landmarks[0]

                    # İskelet Çizimi
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    lms = hand_lms.landmark
                    p_wrist = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float64)
                    p_thumb = np.array([lms[4].x, lms[4].y, lms[4].z], dtype=np.float64)
                    p_idx_mcp = np.array([lms[5].x, lms[5].y, lms[5].z], dtype=np.float64)
                    p_idx_tip = np.array([lms[8].x, lms[8].y, lms[8].z], dtype=np.float64)
                    p_mid_mcp = np.array([lms[9].x, lms[9].y, lms[9].z], dtype=np.float64)
                    p_pinky_mcp = np.array([lms[17].x, lms[17].y, lms[17].z], dtype=np.float64)

                    thumb_px = (int(p_thumb[0] * w_px), int(p_thumb[1] * h_px))
                    index_px = (int(p_idx_tip[0] * w_px), int(p_idx_tip[1] * h_px))
                    ref_pt_px = (int(p_mid_mcp[0] * w_px), int(p_mid_mcp[1] * h_px))

                    # 1. Referans Noktası (Landmark 9 - Avuç Merkezi)
                    current_hand_ref = np.array([p_mid_mcp[0], p_mid_mcp[1]], dtype=np.float64)

                    # 2. Dönmeye Karşı Kararlı Hipotenüs Avuç Ölçeği (Derinlik Hesabı)
                    p0_px = np.array([p_wrist[0] * w_px, p_wrist[1] * h_px])
                    p5_px = np.array([p_idx_mcp[0] * w_px, p_idx_mcp[1] * h_px])
                    p9_px = np.array([p_mid_mcp[0] * w_px, p_mid_mcp[1] * h_px])
                    p17_px = np.array([p_pinky_mcp[0] * w_px, p_pinky_mcp[1] * h_px])

                    palm_len = float(np.linalg.norm(p9_px - p0_px))
                    palm_width = float(np.linalg.norm(p5_px - p17_px))
                    current_hand_size = math.hypot(palm_len, palm_width)

                    # 3. Kıskaç (Gripper) Oranı ve Değeri
                    p4_px = np.array([thumb_px[0], thumb_px[1]], dtype=np.float64)
                    p8_px = np.array([index_px[0], index_px[1]], dtype=np.float64)
                    pinch_ratio = compute_pinch_ratio(p4_px, p8_px, p5_px, p17_px)
                    current_raw_gripper = compute_gripper_value(
                        pinch_ratio=pinch_ratio,
                        pinch_min=CONFIG["pinch_min"],
                        pinch_max=CONFIG["pinch_max"],
                    )

                    # 4. Avuç Normalinden Pitch ve Roll Açısı
                    current_hand_pitch, current_hand_roll = compute_hand_orientation(
                        wrist=p_wrist,
                        index_mcp=p_idx_mcp,
                        middle_mcp=p_mid_mcp,
                        pinky_mcp=p_pinky_mcp,
                    )

                # Klavye ve Kullanıcı Girişleri
                key = cv2.waitKey(1) & 0xFF
                now = time.time()

                # Space Tuşu veya Fare Tıklaması
                space_event = (key == ord(' ') or key == 32)
                mouse_event = mouse_clicked_flag
                mouse_clicked_flag = False

                if space_event or mouse_event:
                    last_space_press_time = now
                    if clutch_mode == "toggle":
                        clutch_active = not clutch_active
                        print(f"[Debriyaj] {'AÇIK' if clutch_active else 'KAPALI'} (Toggle)")

                if key == ord('m') or key == ord('M'):
                    clutch_mode = "hold" if clutch_mode == "toggle" else "toggle"
                    print(f"[Debriyaj Modu] {clutch_mode.upper()} olarak ayarlandı.")

                elif key == ord('w') or key == ord('W'):
                    wrist_tracking_active = not wrist_tracking_active
                    print(f"[Bilek Takibi] {'AÇIK' if wrist_tracking_active else 'KAPALI'}")

                elif key == ord('r') or key == ord('R'):
                    if model.nkey > 0:
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                    else:
                        mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    home_ee_pos, _ = ik_solver.get_ee_pose(data)
                    anchor_robot_pos = home_ee_pos.copy()
                    smooth_target_pos = home_ee_pos.copy()
                    anchor_robot_pitch = float(data.qpos[3])
                    anchor_robot_roll = float(data.qpos[4])
                    smooth_target_pitch = anchor_robot_pitch
                    smooth_target_roll = anchor_robot_roll
                    clutch_active = False
                    if mocap_id >= 0:
                        data.mocap_pos[mocap_id] = home_ee_pos.copy()
                    print("[Sistem] Robot sıfırlandı.")

                elif key == 27 or key == ord('q') or key == ord('Q'):
                    print("[Sistem] Çıkış isteği alındı.")
                    break

                # Hold Modu Süre Değerlendirmesi
                if clutch_mode == "hold":
                    clutch_active = (now - last_space_press_time) < CONFIG["clutch_space_timeout"]

                clutch_just_pressed = clutch_active and not prev_clutch
                prev_clutch = clutch_active

                # Debriyaj Başladığında Çapa (Anchor) Kilitle
                if clutch_just_pressed:
                    cur_robot_pos, _ = ik_solver.get_ee_pose(data)
                    anchor_robot_pos = cur_robot_pos.copy()
                    smooth_target_pos = cur_robot_pos.copy()
                    anchor_robot_pitch = float(data.qpos[3])
                    anchor_robot_roll = float(data.qpos[4])
                    smooth_target_pitch = anchor_robot_pitch
                    smooth_target_roll = anchor_robot_roll

                    anchor_hand_ref = current_hand_ref.copy()
                    anchor_hand_size = max(current_hand_size, 10.0)
                    anchor_hand_pitch = current_hand_pitch
                    anchor_hand_roll = current_hand_roll

                # Debriyaj Aktifken Hedefleri Güncelle
                if clutch_active and hand_detected:
                    dx_cam = current_hand_ref[0] - anchor_hand_ref[0]
                    dy_cam = current_hand_ref[1] - anchor_hand_ref[1]

                    delta_robot_xyz = map_camera_deltas_to_robot(
                        dx_cam=dx_cam,
                        dy_cam=dy_cam,
                        hand_width=current_hand_size,
                        anchor_hand_width=anchor_hand_size,
                        config=CONFIG,
                    )
                    candidate_pos = anchor_robot_pos + delta_robot_xyz
                    candidate_pos = clamp_to_workspace(candidate_pos, ws_limits)
                    candidate_pos = apply_step_limiter(candidate_pos, smooth_target_pos, max_step)
                    smooth_target_pos = apply_ema(candidate_pos, smooth_target_pos, alpha_pos)

                    if wrist_tracking_active:
                        d_pitch = np.clip(current_hand_pitch - anchor_hand_pitch, -1.2, 1.2)
                        d_roll = np.clip(current_hand_roll - anchor_hand_roll, -1.5, 1.5)
                        target_pitch = anchor_robot_pitch + d_pitch * CONFIG["scale_pitch"]
                        target_roll = anchor_robot_roll + d_roll * CONFIG["scale_roll"]
                        smooth_target_pitch = float(apply_ema(np.array(target_pitch), np.array(smooth_target_pitch), alpha_rot))
                        smooth_target_roll = float(apply_ema(np.array(target_roll), np.array(smooth_target_roll), alpha_rot))
                    else:
                        smooth_target_pitch = anchor_robot_pitch
                        smooth_target_roll = anchor_robot_roll

                    if mocap_id >= 0:
                        data.mocap_pos[mocap_id] = smooth_target_pos

                    q_cmd, _ = ik_solver.solve(
                        target_pos=smooth_target_pos,
                        target_pitch=smooth_target_pitch,
                        target_roll=smooth_target_roll,
                        q_init=data.qpos[:5],
                        n_iters=CONFIG["ik_iterations"],
                        tolerance=CONFIG["ik_tolerance"],
                    )
                    data.ctrl[:5] = q_cmd

                # Gripper Filtreleme
                if hand_detected:
                    smooth_gripper = float(apply_ema(
                        np.array(current_raw_gripper),
                        np.array(smooth_gripper),
                        alpha_grip
                    ))
                target_jaw = gripper_to_jaw_angle(smooth_gripper, jaw_min, jaw_max)
                data.ctrl[5] = target_jaw

                # Gerçek Zamanlı Çoklu Fizik Alt-Adımları (1:1 Physics Timing)
                t_now = time.time()
                dt_frame = t_now - t_prev_frame
                t_prev_frame = t_now

                dt_sim = min(dt_frame, 0.10)
                n_substeps = max(1, int(round(dt_sim / model.opt.timestep)))

                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                viewer.sync()

                # FPS Hesabı
                if dt_frame > 0:
                    fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt_frame)

                # OpenCV HUD Gösterimi
                hud_frame = hud.render(
                    frame=frame,
                    clutch_active=clutch_active,
                    gripper_val=smooth_gripper,
                    target_pos=smooth_target_pos,
                    fps=fps_val,
                    hand_detected=hand_detected,
                    ref_pt_px=ref_pt_px,
                    thumb_px=thumb_px,
                    index_px=index_px,
                    clutch_mode=clutch_mode,
                    wrist_active=wrist_tracking_active,
                )
                cv2.imshow(window_name, hud_frame)

    except KeyboardInterrupt:
        print("\n[Sistem] Kullanıcı tarafından durduruldu.")
    finally:
        print("[Sistem] Kapatılıyor...")
        cam.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SO-100 Robot Kolu İçin Web Kamerası Teleoperasyonu (Düşük Gecikmeli)")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Kullanılacak web kamerası indeksi (varsayılan: 0)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="so_arm100/teleop_scene.xml",
        help="Yüklenecek MuJoCo sahne XML dosyası",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Kamera aynalama (mirror) modunu devre dışı bırakır",
    )
    parser.add_argument(
        "--hold-clutch",
        action="store_true",
        help="Debriyajı varsayılan olarak Hold modunda başlatır (varsayılan: Toggle)",
    )
    args = parser.parse_args()

    if args.hold_clutch:
        CONFIG["clutch_mode"] = "hold"

    scene_path = CURRENT_DIR / args.scene
    if not scene_path.exists():
        fallback = CURRENT_DIR / "so_arm100" / "scene.xml"
        if fallback.exists():
            scene_path = fallback

    run_webcam_teleoperation(
        scene_file=scene_path,
        camera_id=args.camera,
        flip_cam=not args.no_mirror,
    )


if __name__ == "__main__":
    main()
