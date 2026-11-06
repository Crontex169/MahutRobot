#!/usr/bin/env python3
"""
MuJoCo SO-100 Robot Kolu İçin WebXR (ARCore) Tabanlı Telefon Teleoperasyon Sistemi

Bu sistem, Android telefonu (örn. Poco X7 Pro) 6-DoF uzamsal kumanda olarak kullanarak
SO-100 robot kolunu MuJoCo fizik motorunda gerçek zamanlı ve düşük gecikmeyle kontrol eder.

Temel Özellikler:
  - WebXR Device API (immersive-ar / ARCore VIO) üzerinden telefon 6-DoF takibi.
  - USB ADB port köprüleme (`adb reverse tcp:8000 tcp:8000`) ile SSL'siz ve ultra düşük gecikmeli (<5ms) iletişim.
  - Daemon thread içinde bağımsız çalışan FastAPI + Uvicorn WebSocket sunucusu.
  - 5-DoF Nullspace Projeksiyonlu Damped Least Squares (DLS) Ters Kinematik (IK) çözücü.
  - Debriyaj (Clutch) mekanizması ile ergonomik göreceli hareket kontrolü.
  - Üstel Hareketli Ortalama (EMA) yumuşatma filtresi ve çalışma hacmi (workspace) güvenlik sınırlandırması.

Kullanım:
  1. Telefonu USB ile PC'ye bağlayın (USB Hata Ayıklama açık olmalıdır).
  2. Terminalde: adb reverse tcp:8000 tcp:8000
  3. Bu scripti çalıştırın: python3 webxr_teleop.py
  4. Telefonda Chrome ile http://localhost:8000 adresine gidin.
  5. 'AR Teleop Başlat' butonuna basın ve debriyaja basılı tutarak robotu yönlendirin.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("HATA: 'mujoco' kütüphanesi bulunamadı! 'pip install mujoco' çalıştırın.")
    sys.exit(1)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    print("HATA: 'fastapi' veya 'uvicorn' bulunamadı! 'pip install fastapi uvicorn' çalıştırın.")
    sys.exit(1)

# Yerel IK çözücüyü içe aktar
from ik_solver import SO100IKSolver


# ==============================================================================
# SİSTEM AYARLARI VE YAPILANDIRMA (CONFIG)
# ==============================================================================
CONFIG = {
    # Ağ Ayarları
    "host": "0.0.0.0",
    "port": 8000,

    # Hassasiyet ve Ölçek Çarpanları
    "pos_scale": 1.0,           # Telefon yer değiştirmesi (m) -> Robot hedefi (m)
    "rot_scale": 1.0,           # Telefon eğimi (rad) -> Robot bilek açısı (rad)

    # WebXR Kamera Koordinatından MuJoCo Dünya Koordinatına Eksen Eşleşmesi
    # WebXR Telefon Standart Ekseni: 0: +X (Sağ), 1: +Y (Yukarı), 2: +Z (Geri / -Z İleri)
    # MuJoCo Robot Temel Eşleşmesi:
    #   ΔX_robot = -ΔZ_telefon (İleri/Geri)
    #   ΔY_robot = -ΔX_telefon (Sağ/Sol)
    #   ΔZ_robot = +ΔY_telefon (Yukarı/Aşağı)
    "axis_map": {
        "x": 2,  # Robot X ekseni <- Telefon Z ekseni
        "y": 0,  # Robot Y ekseni <- Telefon X ekseni
        "z": 1,  # Robot Z ekseni <- Telefon Y ekseni
    },
    "axis_signs": {
        "x": -1.0,  # -Z_telefon -> +X_robot
        "y": -1.0,  # -X_telefon -> +Y_robot
        "z": 1.0,   # +Y_telefon -> +Z_robot
    },

    # Titreme Önleme (Smoothing) ve Güvenlik Sınırları
    "ema_alpha_pos": 0.30,      # Pozisyon EMA filtresi (0.0: tam donuk, 1.0: filtresiz anlık)
    "ema_alpha_rot": 0.30,      # Açı EMA filtresi
    "max_step_pos": 0.035,      # Tek bir karede izin verilen azami konum sıçraması (m)
    "workspace_limits": {
        "x": (-0.28, 0.28),     # Robot sağ/sol sınırları (metre)
        "y": (-0.48, -0.05),    # Robot ileri/geri sınırları (metre)
        "z": (0.015, 0.42),     # Zemin üstü çalışma yüksekliği (metre)
    },

    # Fizik ve Kontrol Döngüsü
    "ik_iterations": 4,         # Her adımda DLS IK çözücü iterasyonu
    "loop_rate_hz": 120.0,      # Simülasyon döngü frekansı (Hz)
}


# ==============================================================================
# THREAD-SAFE VERİ PAYLAŞIM SINIFI
# ==============================================================================
class TeleopSharedState:
    """FastAPI WebSocket sunucusu ile MuJoCo simülasyon döngüsü arasında güvenli veri köprüsü."""

    def __init__(self):
        self.lock = threading.Lock()
        self.client_connected = False
        self.clutch_active = False
        self.prev_clutch = False

        # Telefon koordinat farkları
        self.delta_pos = np.zeros(3, dtype=np.float64)       # [dx, dy, dz] telefon ekseninde
        self.delta_euler = np.zeros(3, dtype=np.float64)     # [dpitch, droll, dyaw] radyan
        self.gripper_norm = 0.0                             # 0.0 (kapalı) - 1.0 (açık)

        # Durum ve kontrol bayrakları
        self.reset_requested = False
        self.last_packet_time = 0.0
        self.packet_count = 0

    def update_teleop(self, data: dict):
        """Telefondan gelen teleoperasyon paketini kilit altında günceller."""
        with self.lock:
            self.clutch_active = bool(data.get("clutch", False))
            delta_pos = data.get("delta_pos", [0.0, 0.0, 0.0])
            delta_euler = data.get("delta_euler", [0.0, 0.0, 0.0])
            self.delta_pos = np.array(delta_pos, dtype=np.float64)
            self.delta_euler = np.array(delta_euler, dtype=np.float64)
            self.gripper_norm = float(data.get("gripper", 0.0))
            self.last_packet_time = time.time()
            self.packet_count += 1

    def request_reset(self):
        """Simülasyon sıfırlama bayrağını ayarlar."""
        with self.lock:
            self.reset_requested = True

    def get_state(self) -> Tuple[bool, bool, np.ndarray, np.ndarray, float, bool]:
        """Döngü tarafından okunmak üzere güncel durumu atomik olarak döner."""
        with self.lock:
            clutch = self.clutch_active
            clutch_just_pressed = clutch and not self.prev_clutch
            clutch_just_released = not clutch and self.prev_clutch
            self.prev_clutch = clutch

            delta_pos = self.delta_pos.copy()
            delta_euler = self.delta_euler.copy()
            gripper = self.gripper_norm
            reset = self.reset_requested
            self.reset_requested = False

            return clutch, clutch_just_pressed, delta_pos, delta_euler, gripper, reset


# ==============================================================================
# FASTAPI & WEBSOCKET SUNUCUSU
# ==============================================================================
def create_app(state: TeleopSharedState, html_file_path: Path) -> FastAPI:
    """FastAPI uygulamasını oluşturur ve WebSocket uç noktasını yapılandırır."""
    app = FastAPI(title="SO-100 WebXR Teleop Server")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        if html_file_path.exists():
            return HTMLResponse(content=html_file_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h2>HATA: index.html dosyası bulunamadı!</h2>", status_code=404)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        with state.lock:
            state.client_connected = True
        print("[Sunucu] Telefon bağlandı (WebSocket aktif).")

        try:
            while True:
                msg_text = await websocket.receive_text()
                try:
                    payload = json.loads(msg_text)
                    msg_type = payload.get("type", "")

                    if msg_type == "teleop":
                        state.update_teleop(payload)
                    elif msg_type == "reset":
                        state.request_reset()
                        print("[Sunucu] Telefonda sıfırlama (reset) butonuna basıldı.")
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            print("[Sunucu] Telefon bağlantısı kesildi.")
        finally:
            with state.lock:
                state.client_connected = False
                state.clutch_active = False

    return app


class BackgroundServer:
    """FastAPI sunucusunu ana MuJoCo döngüsünü engellemeden arka plan daemon thread'inde çalıştırır."""

    def __init__(self, app: FastAPI, host: str, port: int):
        self.config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config=self.config)
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def stop(self):
        self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)


# ==============================================================================
# ANA TELEOPERASYON SİMÜLASYON DÖNGÜSÜ
# ==============================================================================
def run_teleoperation(scene_file: Path, html_file: Path):
    """MuJoCo simülasyonunu ve WebXR teleoperasyon döngüsünü yürütür."""
    print("=" * 70)
    print("SO-100 Robot Kolu WebXR (ARCore) Teleoperasyon Sistemi")
    print("=" * 70)
    print(f"Model Sahnesi : {scene_file}")
    print(f"Web Arayüzü   : {html_file}")
    print(f"Sunucu Adresi : http://{CONFIG['host']}:{CONFIG['port']}")
    print("-" * 70)
    print("Adımlar:")
    print("  1. Telefonu USB kablosuyla bilgisayara bağlayın.")
    print("  2. Farklı bir terminalde şu komutu çalıştırın:")
    print(f"     adb reverse tcp:{CONFIG['port']} tcp:{CONFIG['port']}")
    print(f"  3. Android Chrome ile şu adresi açın: http://localhost:{CONFIG['port']}")
    print("  4. 'AR Teleop Başlat' butonuna dokunun ve debriyajla kontrol edin.")
    print("=" * 70)

    # Modeli Yükle
    if not scene_file.exists():
        raise FileNotFoundError(f"Sahne XML dosyası bulunamadı: {scene_file}")

    model = mujoco.MjModel.from_xml_path(str(scene_file))
    data = mujoco.MjData(model)

    # Hedef Mocap Gövdesi tespiti
    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
    mocap_id = model.body_mocapid[mocap_body_id] if mocap_body_id >= 0 else -1

    # SO100 IK Çözücüyü Başlat
    ik_solver = SO100IKSolver(
        model=model,
        site_name="ee_site",
        body_name="Fixed_Jaw",
        damping_pos=CONFIG.get("ik_damping", 1e-3),
    )

    # Başlangıç pozuna (home keyframe) sıfırla
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Başlangıç uç nokta ve eklem değerleri
    home_ee_pos, home_ee_rot = ik_solver.get_ee_pose(data)
    home_qpos = data.qpos[:5].copy()

    # Mocap hedef göstergesini başlangıç noktasına hizala
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = home_ee_pos.copy()

    # Ortak Durum ve Arka Plan Sunucusu
    state = TeleopSharedState()
    app = create_app(state, html_file)
    server = BackgroundServer(app, host=CONFIG["host"], port=CONFIG["port"])
    server.start()
    print("[Sistem] Arka plan HTTP/WebSocket sunucusu hazır.")

    # Teleoperasyon Kontrol Değişkenleri
    anchor_ee_pos = home_ee_pos.copy()
    anchor_pitch = home_qpos[3]
    anchor_roll = home_qpos[4]

    smooth_target_pos = home_ee_pos.copy()
    smooth_target_pitch = float(anchor_pitch)
    smooth_target_roll = float(anchor_roll)

    # Kıskaç açı sınırları
    jaw_min = model.jnt_range[5, 0]
    jaw_max = model.jnt_range[5, 1]

    # Çalışma hacmi sınırları
    ws_x = CONFIG["workspace_limits"]["x"]
    ws_y = CONFIG["workspace_limits"]["y"]
    ws_z = CONFIG["workspace_limits"]["z"]

    dt_target = 1.0 / CONFIG["loop_rate_hz"]

    print("\nMuJoCo Simülasyon penceresi açılıyor. Çıkmak için pencereyi kapatın...\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t_start = time.time()

                # Durumu al
                clutch, clutch_pressed, delta_pos, delta_euler, gripper, reset_req = state.get_state()

                # 1. Sıfırlama İsteği (Reset)
                if reset_req:
                    if model.nkey > 0:
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                    else:
                        mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)

                    home_ee_pos, _ = ik_solver.get_ee_pose(data)
                    anchor_ee_pos = home_ee_pos.copy()
                    smooth_target_pos = home_ee_pos.copy()
                    anchor_pitch = data.qpos[3]
                    anchor_roll = data.qpos[4]
                    smooth_target_pitch = float(anchor_pitch)
                    smooth_target_roll = float(anchor_roll)

                    if mocap_id >= 0:
                        data.mocap_pos[mocap_id] = home_ee_pos.copy()

                # 2. Debriyaj (Clutch) Yeni Basıldığında Anchor Pozu Kaydet
                if clutch_pressed:
                    current_ee_pos, _ = ik_solver.get_ee_pose(data)
                    anchor_ee_pos = current_ee_pos.copy()
                    smooth_target_pos = current_ee_pos.copy()
                    anchor_pitch = data.qpos[3]
                    anchor_roll = data.qpos[4]
                    smooth_target_pitch = float(anchor_pitch)
                    smooth_target_roll = float(anchor_roll)

                # 3. Debriyaj Aktifken Hedef Hesaplama
                if clutch:
                    # Eksen Eşleme: WebXR -> MuJoCo Robot
                    axis_map = CONFIG["axis_map"]
                    axis_signs = CONFIG["axis_signs"]
                    p_scale = CONFIG["pos_scale"]
                    r_scale = CONFIG["rot_scale"]

                    dx_robot = axis_signs["x"] * delta_pos[axis_map["x"]] * p_scale
                    dy_robot = axis_signs["y"] * delta_pos[axis_map["y"]] * p_scale
                    dz_robot = axis_signs["z"] * delta_pos[axis_map["z"]] * p_scale

                    candidate_pos = anchor_ee_pos + np.array([dx_robot, dy_robot, dz_robot])

                    # Güvenlik 1: Çalışma Hacmi Sınırlandırması (Workspace Bounding Box)
                    candidate_pos[0] = np.clip(candidate_pos[0], ws_x[0], ws_x[1])
                    candidate_pos[1] = np.clip(candidate_pos[1], ws_y[0], ws_y[1])
                    candidate_pos[2] = np.clip(candidate_pos[2], ws_z[0], ws_z[1])

                    # Güvenlik 2: Sıçrama Kırpma (Step Limiter)
                    step_vec = candidate_pos - smooth_target_pos
                    step_dist = np.linalg.norm(step_vec)
                    max_step = CONFIG["max_step_pos"]
                    if step_dist > max_step:
                        candidate_pos = smooth_target_pos + step_vec * (max_step / step_dist)

                    # Güvenlik 3: Üstel Hareketli Ortalama (EMA) Yumuşatma
                    alpha_p = CONFIG["ema_alpha_pos"]
                    smooth_target_pos = alpha_p * candidate_pos + (1.0 - alpha_p) * smooth_target_pos

                    # Açı Hedefleri (Pitch ve Roll)
                    alpha_r = CONFIG["ema_alpha_rot"]
                    target_pitch = anchor_pitch + delta_euler[0] * r_scale
                    target_roll = anchor_roll + delta_euler[1] * r_scale

                    smooth_target_pitch = alpha_r * target_pitch + (1.0 - alpha_r) * smooth_target_pitch
                    smooth_target_roll = alpha_r * target_roll + (1.0 - alpha_r) * smooth_target_roll

                    # Mocap Görsel Hedefini Güncelle
                    if mocap_id >= 0:
                        data.mocap_pos[mocap_id] = smooth_target_pos

                    # 4. Ters Kinematik (IK) Çözümü
                    q_cmd, _ = ik_solver.solve(
                        target_pos=smooth_target_pos,
                        target_pitch=smooth_target_pitch,
                        target_roll=smooth_target_roll,
                        q_init=data.qpos[:5],
                        n_iters=CONFIG["ik_iterations"],
                    )

                    # Aktüatörlere Uygula
                    data.ctrl[:5] = q_cmd

                # 5. Kıskaç (Gripper) Kontrolü
                target_jaw = jaw_min + gripper * (jaw_max - jaw_min)
                data.ctrl[5] = target_jaw

                # 6. Fizik Adımı ve Görüntü Senkronizasyonu
                mujoco.mj_step(model, data)
                viewer.sync()

                # Döngü Hızını Koru
                elapsed = time.time() - t_start
                sleep_time = dt_target - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Sistem] Kullanıcı tarafından durduruldu.")
    finally:
        print("[Sistem] Kapatılıyor...")
        server.stop()
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="SO-100 WebXR Teleoperasyon Sistemi")
    parser.add_argument(
        "--scene",
        type=str,
        default="so_arm100/teleop_scene.xml",
        help="Yüklenecek MuJoCo sahne dosyası",
    )
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    scene_path = current_dir / args.scene
    html_path = current_dir / "index.html"

    # Sahne fallback kontrolü
    if not scene_path.exists():
        fallback_scene = current_dir / "so_arm100" / "scene.xml"
        if fallback_scene.exists():
            scene_path = fallback_scene

    run_teleoperation(scene_file=scene_path, html_file=html_path)


if __name__ == "__main__":
    main()
