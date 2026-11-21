"""
Telefon (WebXR) teleopunun WebSocket sunucusu: telefondan gelen paketleri kontrol döngüsüne aktarır.

Protokol `Side_project/webxr_teleop.py` ile aynıdır; `web/index.html` değişmeden bağlanır.

Protokol (telefon -> sunucu, JSON):
    {"type": "teleop", "clutch": bool, "delta_pos": [dx, dy, dz], "delta_euler": [pitch, roll, yaw],
     "gripper": 0..1, ...}       delta'lar debriyaja basıldığı andaki telefon pozuna göredir
    {"type": "reset"}
Sunucu -> telefon (isteğe bağlı): {"type": "status", "instruction": str, "success": bool, ...}
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # sunucu bağımlılıkları isteğe bağlı: eşleyici/testler bunlarsız da çalışır
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover
    FastAPI = WebSocket = WebSocketDisconnect = HTMLResponse = None  # type: ignore

DEFAULT_HTML = Path(__file__).resolve().parent / "web" / "index.html"


def quat_rel_rotvec(q_cur: Sequence[float], q_anchor: Sequence[float]) -> np.ndarray:
    """
    WebXR kuaterniyonları [x, y, z, w]: q_rel = q_cur * q_anchor^-1 dönüşünün dönme vektörü
    (rad, telefon 'local' uzayı eksenlerinde). Euler'den farklı olarak gimbal kilidi yoktur: telefon
    ekrana bakarken neredeyse yatay tutulduğunda (pitch ≈ -80..-90°) sayfanın Euler açıları
    sıçrıyor, kol da yanlış bileğe dönüp masa korumasına takılıyordu. Küçük açılarda sayfanın
    [pitch(X), roll(Y), yaw(Z)] değerleriyle aynı indeks ve işarettedir.
    """
    x1, y1, z1, w1 = (float(v) for v in q_cur)
    x2, y2, z2, w2 = -float(q_anchor[0]), -float(q_anchor[1]), -float(q_anchor[2]), float(q_anchor[3])
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    if w < 0:  # kısa yol
        x, y, z, w = -x, -y, -z, -w
    v = np.array([x, y, z])
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v * (2.0 * np.arctan2(n, w) / n)


class TeleopSharedState:
    """WebSocket sunucusu (arka plan thread) ile kontrol döngüsü arasındaki köprü."""

    def __init__(self, max_packet_jump_m: float = 0.08):
        self.lock = threading.Lock()
        self.max_packet_jump_m = max_packet_jump_m
        self._jump_offset = np.zeros(3)   # yok sayılan sıçramaların toplamı (debriyaj vuruşu içinde)
        self._last_raw_delta: Optional[np.ndarray] = None
        self.jump_count = 0
        self.client_connected = False
        self.clutch_active = False
        self.prev_clutch = False
        self.delta_pos = np.zeros(3)
        self.delta_euler = np.zeros(3)
        self.gripper_ratio = 1.0  # 1 = tam açık
        self.reset_requested = False
        self.last_packet_time = 0.0
        self.packet_count = 0
        self.outbox: List[Dict[str, Any]] = []
        self.msg_count = 0          # gelen tüm WebSocket mesajları
        self.bad_count = 0          # çözülemeyen / tanınmayan mesajlar
        self.msg_types: Dict[str, int] = {}
        self.first_teleop_raw: Optional[str] = None
        self.last_error: Optional[str] = None
        self._pkt_clutch = False          # son paketteki debriyaj (paket tarafı kenar algılama)
        self._anchor_quat: Optional[np.ndarray] = None

    def update_teleop(self, data: Dict[str, Any]) -> None:
        with self.lock:
            clutch = bool(data.get("clutch", False))
            raw_delta = np.asarray(data.get("delta_pos", [0, 0, 0]), dtype=np.float64)[:3]
            if not clutch or not self._pkt_clutch or self._last_raw_delta is None:
                self._jump_offset = np.zeros(3)  # yeni vuruş: telefon sayfası çapasını sıfırladı
            elif np.linalg.norm(raw_delta - self._last_raw_delta) > self.max_packet_jump_m:
                self._jump_offset += raw_delta - self._last_raw_delta  # takip sıçraması: yok say
                self.jump_count += 1
            self._last_raw_delta = raw_delta if clutch else None
            self.delta_pos = raw_delta - self._jump_offset
            raw_quat = data.get("raw_quat")
            if clutch and raw_quat is not None and len(raw_quat) == 4:
                # Açı farkını ham kuaterniyondan hesapla (bkz. quat_rel_rotvec); çapa = debriyajın
                # ilk paketi. Sayfanın delta_euler'i yalnızca raw_quat yoksa kullanılır.
                q = np.asarray(raw_quat, dtype=np.float64)
                if not self._pkt_clutch or self._anchor_quat is None:
                    self._anchor_quat = q
                self.delta_euler = quat_rel_rotvec(q, self._anchor_quat)
            else:
                self.delta_euler = np.asarray(data.get("delta_euler", [0, 0, 0]), dtype=np.float64)[:3]
            self.clutch_active = clutch
            self._pkt_clutch = clutch
            if "gripper" in data:
                self.gripper_ratio = float(np.clip(float(data["gripper"]), 0.0, 1.0))
            self.last_packet_time = time.time()
            self.packet_count += 1

    def handle_message(self, text: str) -> Optional[str]:
        """Ham WebSocket metnini işler; mesaj tipini döner (geçersizse None). Asla hata atmaz."""
        with self.lock:
            self.msg_count += 1
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            self._bad(f"JSON değil: {str(text)[:120]}")
            return None
        if not isinstance(payload, dict):
            self._bad(f"sözlük değil: {str(text)[:120]}")
            return None
        t = str(payload.get("type", ""))
        with self.lock:
            self.msg_types[t] = self.msg_types.get(t, 0) + 1
        if t == "teleop":
            if self.first_teleop_raw is None:
                self.first_teleop_raw = str(text)[:400]
            try:
                self.update_teleop(payload)
            except Exception as e:  # bozuk alan (ör. null) bağlantıyı düşürmesin
                self._bad(f"teleop paketi işlenemedi: {e!r}: {str(text)[:200]}")
                return None
        elif t == "reset":
            with self.lock:
                self.reset_requested = True
        else:
            return None
        return t

    def _bad(self, err: str) -> None:
        with self.lock:
            self.bad_count += 1
            self.last_error = err

    def set_connected(self, connected: bool) -> None:
        with self.lock:
            self.client_connected = connected
            if not connected:
                self.clutch_active = False

    def clutch_now(self, timeout_s: Optional[float] = None) -> bool:
        """Debriyaj şu an basılı mı (kenar algılama durumunu değiştirmeden okur)."""
        with self.lock:
            if timeout_s is not None and time.time() - self.last_packet_time > timeout_s:
                return False
            return self.clutch_active

    def get_state(self, timeout_s: Optional[float] = None) -> Tuple[bool, bool, np.ndarray, np.ndarray, float, bool]:
        """(debriyaj, yeni_basıldı, delta_pos, delta_euler, kıskaç_oranı, reset) - atomik okuma."""
        with self.lock:
            clutch = self.clutch_active
            if timeout_s is not None and clutch and time.time() - self.last_packet_time > timeout_s:
                clutch = False  # telefon donduysa kol hareket etmesin
            pressed = clutch and not self.prev_clutch
            self.prev_clutch = clutch
            reset = self.reset_requested
            self.reset_requested = False
            return clutch, pressed, self.delta_pos.copy(), self.delta_euler.copy(), self.gripper_ratio, reset

    def push_status(self, msg: Dict[str, Any]) -> None:
        with self.lock:
            self.outbox = [msg]  # yalnızca en güncel durum

    def pop_status(self) -> List[Dict[str, Any]]:
        with self.lock:
            out, self.outbox = self.outbox, []
            return out


# ---------------------------------------------------------------------------
# FastAPI + uvicorn sunucusu
# ---------------------------------------------------------------------------


def create_app(state: TeleopSharedState, html_file: Path = DEFAULT_HTML):
    import asyncio

    if FastAPI is None:
        raise ImportError("Telefon teleopu için: pip install fastapi uvicorn")
    app = FastAPI(title="MahutKol WebXR Teleop (ManiSkill)")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        if Path(html_file).exists():
            return HTMLResponse(Path(html_file).read_text(encoding="utf-8"))
        return HTMLResponse("<h2>HATA: index.html bulunamadı</h2>", status_code=404)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        state.set_connected(True)
        print("[Sunucu] Telefon bağlandı.")

        async def sender():
            while True:
                for msg in state.pop_status():
                    await websocket.send_text(json.dumps(msg, ensure_ascii=False))
                await asyncio.sleep(0.1)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                text = await websocket.receive_text()
                if state.handle_message(text) == "reset":
                    print("[Sunucu] Telefondan sıfırlama isteği.")
        except WebSocketDisconnect as e:
            print(f"[Sunucu] Telefon bağlantısı kesildi (kod {getattr(e, 'code', '?')}).")
        except Exception as e:  # pragma: no cover - teşhis için
            print(f"[Sunucu] WebSocket hatası: {e!r}")
        finally:
            send_task.cancel()
            state.set_connected(False)

    return app


class BackgroundServer:
    """uvicorn'u daemon thread'de çalıştırır (simülasyon döngüsünü engellemez)."""

    def __init__(self, app, host: str, port: int):
        import uvicorn

        self.server = uvicorn.Server(uvicorn.Config(app=app, host=host, port=port, log_level="warning"))
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def stop(self) -> None:
        self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
