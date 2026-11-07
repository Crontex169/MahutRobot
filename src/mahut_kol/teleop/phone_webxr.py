"""
Telefon (WebXR / ARCore) ile MahutKol teleoperasyonu - ManiSkill sürümü.

`Side_project/webxr_teleop.py` (MuJoCo + SO-100) referans alınarak ManiSkill ve MahutKol için
yeniden yazıldı; orijinal dosyaya dokunulmadı. Aynı WebSocket protokolü kullanılır, böylece
`src/mahut_kol/teleop/web/index.html` (orijinalin kopyası) değişmeden bağlanabilir.

Protokol (telefon -> sunucu, JSON):
    {"type": "teleop", "clutch": bool, "delta_pos": [dx, dy, dz], "delta_euler": [pitch, roll, yaw],
     "gripper": 0..1, ...}       delta'lar debriyaja basıldığı andaki telefon pozuna göredir
    {"type": "reset"}
Sunucu -> telefon (isteğe bağlı): {"type": "status", "instruction": str, "success": bool, ...}

Kurulum (telefon USB ile bağlı, USB hata ayıklama açık):
    adb reverse tcp:8000 tcp:8000
    python3 scripts/record_vla_demos.py --device phone
    Telefonda Chrome: http://localhost:8000 -> "AR Teleop Başlat"

Eksen eşlemesi (ManiSkill dünyası: robot +x'e bakar; operatör x > 0 tarafında robota, yani -x
yönüne bakar; sabit "front" kamera da oradadır, görüntünün sağı = +y):
    WebXR telefon ekseni: +X sağ, +Y yukarı, +Z kullanıcıya doğru (ileri = -Z).
    Telefonu ileri (robota doğru) itmek  -> TCP -x   : Δx_dünya = +ΔZ_telefon
    Telefonu sağa götürmek               -> TCP +y   : Δy_dünya = +ΔX_telefon
    Telefonu yukarı kaldırmak            -> TCP +z   : Δz_dünya = +ΔY_telefon
    Telefonun burnunu aşağı eğmek (X etrafında pitch) -> kıskaç daha dik (eğim azalır)
    Telefonu direksiyon gibi çevirmek (Z etrafında)   -> bilek roll (joint4)
WebXR 'local' referans uzayı oturum başladığı andaki yönelime sabitlenir: AR oturumunu
robota/ekrana bakarken başlatın.

Güvenlik (orijinaldeki gibi): çalışma hacmi kırpma, adım sınırlama, EMA yumuşatma, debriyaj
(bırakılınca kol son hedefte sabit kalır). Ek olarak: IK çözümü parmakları masaya sokuyorsa
güncelleme reddedilir; eklem başına adım sınırı vardır.

IK seçimi: `mahut_kol.robots.kinematics` (URDF'ten saf numpy DLS, eğim kısıtlı). Karşılaştırma:
- `pick_cube_task.py` fk/ik: MuJoCo XML modelini kullanır (simülasyondaki URDF değil), yalnızca
  sabit -45° yaklaşma ve joint4 = 0 için; her çağrıda 25^3 ızgara + Newton (~yüzlerce ms) ->
  20 Hz teleop için yavaş ve esnek değil.
- `Side_project/ik_solver.py` nullspace'li DLS: hızlı ve artımlı, fakat MuJoCo'ya bağlı ve bilek
  pitch'ini eklem hedefi olarak nullspace'te izler; MahutKol'da konum için 4 eklem kullanılınca
  nullspace 1 boyutlu kalır ve yaklaşma açısı kontrolü zayıflar.
- Seçilen: aynı DLS fikri, ancak yaklaşma eğimini birincil görevin parçası (tek skaler kısıt)
  yapan, URDF ile birebir FK'li sürüm. Önceki çözümden ısıtılmış başlangıçla adım başına ~1-3 ms.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # sunucu bağımlılıkları isteğe bağlı: eşleyici/testler bunlarsız da çalışır
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover
    FastAPI = WebSocket = WebSocketDisconnect = HTMLResponse = None  # type: ignore

from mahut_kol.robots.kinematics import MahutKinematics, finger_min_z, get_kinematics
from mahut_kol.teleop.action_source import (
    ARM_DOF,
    ActionSource,
    clip_action,
    gripper_open_ratio_to_finger,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_HTML = WEB_DIR / "index.html"

CONFIG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    # Telefon yer değiştirmesi (m) -> TCP hedefi (m); açı (rad) -> açı (rad)
    "pos_scale": 1.0,
    "rot_scale": 1.0,
    # Δdünya[i] = işaret[i] * Δtelefon[eksen[i]]  (bkz. modül açıklaması)
    "axis_map": {"x": 2, "y": 0, "z": 1},
    "axis_signs": {"x": 1.0, "y": 1.0, "z": 1.0},
    # delta_euler indeksleri: 0 = X etrafında (pitch), 1 = Y etrafında, 2 = Z etrafında
    "tilt_from": (0, 1.0),
    "roll_from": (2, -1.0),
    "ema_alpha_pos": 0.35,
    "ema_alpha_rot": 0.35,
    "max_step_pos": 0.02,          # m / kontrol adımı (20 Hz -> 0.4 m/s)
    "max_step_joint": 0.15,        # rad / kontrol adımı
    # TCP çalışma hacmi (dünya, m). Masa üstü çalışma alanını (x 0.10-0.22, y -0.16..0.12)
    # kaldırma/yaklaşma payıyla kapsar.
    "workspace_limits": {"x": (0.02, 0.30), "y": (-0.26, 0.22), "z": (0.02, 0.30)},
    "tilt_limits_deg": (25.0, 85.0),
    "ik_iterations": 30,
    "packet_timeout_s": 0.5,       # bu süre paket gelmezse debriyaj bırakılmış sayılır
}


def phone_to_world_delta(delta_pos: Sequence[float], config: Dict[str, Any] = CONFIG) -> np.ndarray:
    d = np.asarray(delta_pos, dtype=np.float64)
    am, sg = config["axis_map"], config["axis_signs"]
    return config["pos_scale"] * np.array(
        [sg["x"] * d[am["x"]], sg["y"] * d[am["y"]], sg["z"] * d[am["z"]]]
    )


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


# ---------------------------------------------------------------------------
# Thread-safe paylaşılan durum
# ---------------------------------------------------------------------------


class TeleopSharedState:
    """WebSocket sunucusu (arka plan thread) ile kontrol döngüsü arasındaki köprü."""

    def __init__(self):
        self.lock = threading.Lock()
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
            self.delta_pos = np.asarray(data.get("delta_pos", [0, 0, 0]), dtype=np.float64)[:3]
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


# ---------------------------------------------------------------------------
# Telefon -> eklem hedefi
# ---------------------------------------------------------------------------


class PhoneTeleopMapper:
    """
    Telefon delta pozunu MahutKol eklem hedefine çevirir (sunucudan bağımsız, test edilebilir).

    Debriyaja basılınca: çapa = o anki hedef TCP konumu, eğim ve roll.
    Basılıyken: hedef TCP = çapa + eksen_eşleme(Δtelefon); kırp, adım sınırla, EMA, IK.
    Bırakılınca: son eklem hedefi korunur (kol sabit).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, kin: Optional[MahutKinematics] = None):
        self.cfg = dict(CONFIG)
        if config:
            self.cfg.update(config)
        self.K = kin or get_kinematics()
        self.q_cmd: Optional[np.ndarray] = None  # 5 kol eklemi
        self.anchor_pos = np.zeros(3)
        self.anchor_tilt = 0.0
        self.anchor_roll = 0.0
        self.smooth_pos = np.zeros(3)
        self.smooth_tilt = 0.0
        self.smooth_roll = 0.0
        self.delta_offset = np.zeros(3)
        self.euler_offset = np.zeros(3)
        self.last_ik_ok = True
        self.guard_rejects = 0  # parmak-masa koruması yüzünden reddedilen adım sayısı
        self.ik_fails = 0

    def reset(self, arm_q: Sequence[float]) -> None:
        self.q_cmd = np.asarray(arm_q, dtype=np.float64)[:ARM_DOF].copy()
        self._set_anchor()

    def _set_anchor(self) -> None:
        p, _ = self.K.fk(self.q_cmd)
        self.anchor_pos = p.copy()
        self.smooth_pos = p.copy()
        self.anchor_tilt = self.K.tilt_of(self.q_cmd)
        self.smooth_tilt = self.anchor_tilt
        self.anchor_roll = float(self.q_cmd[4])
        self.smooth_roll = self.anchor_roll

    def tcp_target(self) -> np.ndarray:
        return self.smooth_pos.copy()

    def step(
        self, clutch: bool, clutch_pressed: bool, delta_pos: Sequence[float], delta_euler: Sequence[float]
    ) -> np.ndarray:
        """Bir kontrol adımı; 5 kol eklemi hedefi döner."""
        if self.q_cmd is None:
            raise RuntimeError("Önce reset(qpos) çağrılmalı")
        if clutch_pressed:
            self._set_anchor()
            # Telefon sayfası debriyajda kendi çapasını sıfırlar (delta ≈ 0). Gecikmiş/eski bir
            # paket gelirse sıçrama olmasın diye basıldığı andaki delta da çapaya katılır.
            self.delta_offset = np.asarray(delta_pos, dtype=np.float64).copy()
            self.euler_offset = np.asarray(delta_euler, dtype=np.float64).copy()
        if not clutch:
            return self.q_cmd.copy()

        c = self.cfg
        delta_pos = np.asarray(delta_pos, dtype=np.float64) - self.delta_offset
        delta_euler = np.asarray(delta_euler, dtype=np.float64) - self.euler_offset
        cand = self.anchor_pos + phone_to_world_delta(delta_pos, c)
        lim = c["workspace_limits"]
        cand = np.array([np.clip(cand[i], *lim[k]) for i, k in enumerate("xyz")])
        step = cand - self.smooth_pos
        n = np.linalg.norm(step)
        if n > c["max_step_pos"]:
            cand = self.smooth_pos + step * (c["max_step_pos"] / n)
        a = c["ema_alpha_pos"]
        new_pos = a * cand + (1 - a) * self.smooth_pos

        de = np.asarray(delta_euler, dtype=np.float64)
        ti, ts = c["tilt_from"]
        ri, rs = c["roll_from"]
        tlo, thi = np.radians(c["tilt_limits_deg"])
        tilt_t = float(np.clip(self.anchor_tilt + ts * c["rot_scale"] * de[ti], tlo, thi))
        roll_t = self.anchor_roll + rs * c["rot_scale"] * de[ri]
        ar = c["ema_alpha_rot"]
        new_tilt = ar * tilt_t + (1 - ar) * self.smooth_tilt
        new_roll = ar * roll_t + (1 - ar) * self.smooth_roll

        q, ok, _ = self.K.ik(
            new_pos, tilt=new_tilt, roll=new_roll, q_init=self.q_cmd, n_iters=c["ik_iterations"]
        )
        if not ok:
            # Eğim tutturulamıyorsa yalnızca konumu dene (kullanıcı sınıra itmiş olabilir)
            q2, ok2, _ = self.K.ik(new_pos, roll=new_roll, q_init=self.q_cmd, n_iters=c["ik_iterations"])
            if ok2:
                q, ok = q2, True
        if not ok:
            self.ik_fails += 1
        # Eklem adım sınırı; parmaklar masaya girecekse adımı küçült (en fazla 3 kez yarıla)
        # Eklem adım sınırı: tüm adımı ORANTILI küçült. Eklem eklem kırpmak hareketin yönünü
        # bozuyordu (ör. masadan yukarı kalkarken bilek önce aşağı dönüp parmağı masaya sokuyordu).
        dq = q - self.q_cmd
        big = float(np.max(np.abs(dq)))
        if big > c["max_step_joint"]:
            dq *= c["max_step_joint"] / big
        # Kol zaten masaya değecek kadar alçaktaysa (ör. kavrama sırasında) her adımı reddetmek
        # kolu kilitliyordu (gerçek telefon denemesinde yukarı bile çıkamadı). Kural: parmaklar
        # masanın altına inemez; zaten alttaysa daha da aşağı inemez, yukarı her zaman serbest.
        floor = min(0.0, finger_min_z(self.K, self.q_cmd))
        for frac in (1.0, 0.5, 0.25, 0.125):
            q_new = self.q_cmd + frac * dq
            if finger_min_z(self.K, q_new) >= floor - 1e-6:
                break
        else:
            self.guard_rejects += 1
            self.last_ik_ok = False  # masaya girerdi: güncellemeyi reddet, kol yerinde kalır
            return self.q_cmd.copy()
        self.last_ik_ok = ok and frac == 1.0
        self.q_cmd = q_new
        if frac == 1.0:
            self.smooth_pos = new_pos
            self.smooth_tilt = new_tilt
            self.smooth_roll = new_roll
        else:
            # Kısmi adım: yumuşatılmış hedefi gerçekleşen TCP'ye çek (birikmiş hata olmasın)
            self.smooth_pos = self.K.fk(self.q_cmd)[0]
            self.smooth_tilt = self.K.tilt_of(self.q_cmd)
        return self.q_cmd.copy()


class PhoneActionSource(ActionSource):
    """Telefonu `ActionSource` olarak sunar. `start_server=False` ile sahte mesajlarla test edilir."""

    name = "phone"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        html_file: Path = DEFAULT_HTML,
        start_server: bool = True,
        state: Optional[TeleopSharedState] = None,
    ):
        super().__init__()
        self.cfg = dict(CONFIG)
        if config:
            self.cfg.update(config)
        self.state = state or TeleopSharedState()
        self.mapper = PhoneTeleopMapper(self.cfg)
        self._events: Set[str] = set()
        self.debug_period_s = float(self.cfg.get("debug_period_s", 1.0))  # 0 = kapalı
        self._last_qpos: Optional[np.ndarray] = None
        self._monitor_stop = threading.Event()
        if start_server and self.debug_period_s > 0:  # sahte mesajlı testlerde izleyici yok
            threading.Thread(target=self._monitor, daemon=True).start()
        self.server: Optional[BackgroundServer] = None
        if start_server:
            app = create_app(self.state, html_file)
            self.server = BackgroundServer(app, self.cfg["host"], self.cfg["port"])
            self.server.start()
            print(f"[Telefon] Sunucu hazır: http://localhost:{self.cfg['port']}  "
                  f"(önce: adb reverse tcp:{self.cfg['port']} tcp:{self.cfg['port']})")

    def reset(self, qpos: Sequence[float]) -> None:
        super().reset(qpos)
        self.mapper.reset(self.target[:ARM_DOF])

    def act(self, qpos: Sequence[float]) -> np.ndarray:
        if self.target is None:
            self.reset(qpos)
        clutch, pressed, dpos, deul, grip, reset = self.state.get_state(self.cfg["packet_timeout_s"])
        if reset:
            self._events.add("reset")
        arm = self.mapper.step(clutch, pressed, dpos, deul)
        finger = gripper_open_ratio_to_finger(grip)
        self.target = clip_action(np.r_[arm, finger, finger])
        self._last_qpos = np.asarray(qpos, dtype=np.float64).copy()
        return self.target.copy()

    def _monitor(self) -> None:
        """
        Arka planda saniyede bir tanı satırı: telefondan gerçekten ne geliyor ve kola ulaşıyor mu?
        Kontrol döngüsünden bağımsızdır (bölüm başlamadan beklerken de yazar).
        """
        s, m = self.state, self.mapper
        last_msgs, last_t, printed_raw, last_err, idle_n = 0, time.time(), False, None, 0
        while not self._monitor_stop.wait(self.debug_period_s):
            now = time.time()
            with s.lock:
                msgs, types, bad, err = s.msg_count, dict(s.msg_types), s.bad_count, s.last_error
                clutch, dpos, deul = s.clutch_active, s.delta_pos.copy(), s.delta_euler.copy()
                age = now - s.last_packet_time if s.last_packet_time else float("inf")
                connected, raw = s.client_connected, s.first_teleop_raw
            rate = (msgs - last_msgs) / max(now - last_t, 1e-6)
            idle = msgs == last_msgs and not clutch
            last_msgs, last_t = msgs, now
            if msgs == 0 and not connected:
                continue
            # Boştayken (paket yok, debriyaj yok) yalnızca 10 saniyede bir yaz
            idle_n = idle_n + 1 if idle else 0
            if idle and idle_n % 10 != 1:
                continue
            if raw and not printed_raw:
                print(f"[Telefon] İlk teleop paketi: {raw}", flush=True)
                printed_raw = True
            if err and err != last_err:
                print(f"[Telefon] HATALI MESAJ: {err}", flush=True)
                last_err = err
            line = (f"[Telefon] {'bağlı' if connected else 'KOPUK'} {rate:4.0f} mesaj/s türler={types} "
                    f"hatalı={bad} debriyaj={'BASILI' if clutch else 'yok'} son paket {age:4.1f} s önce "
                    f"Δtel(cm)={np.round(dpos * 100, 1)} Δaçı(°)={np.round(np.degrees(deul), 0)}")
            if m.q_cmd is not None:
                line += f" hedefTCP={np.round(m.K.fk(m.q_cmd)[0], 3)}"
                if self._last_qpos is not None:
                    line += f" gerçekTCP={np.round(m.K.fk(self._last_qpos[:ARM_DOF])[0], 3)}"
                line += f" ikHata={m.ik_fails} masaRed={m.guard_rejects}"
            print(line, flush=True)

    def poll_events(self) -> Set[str]:
        with self.state.lock:
            if self.state.reset_requested:  # bölüm başlamadan basılan "Sıfırla" da sayılsın
                self.state.reset_requested = False
                self._events.add("reset")
        ev, self._events = self._events, set()
        return ev

    def ready(self) -> bool:
        # Bölüm, telefonda debriyaja ilk basıldığında başlar: bağlanma/AR başlatma süresi
        # bölüm süresinden yenmez ve kaydın başında boş (hareketsiz) kareler birikmez.
        return self.state.clutch_now(self.cfg["packet_timeout_s"])

    def start_hint(self) -> str:
        if not self.state.client_connected:
            return f"Telefon bekleniyor: Chrome'da http://localhost:{self.cfg['port']} açın"
        return "Başlamak için telefonda DEBRİYAJ'a basılı tutun"

    def status_text(self) -> str:
        s = self.state
        conn = "bağlı" if s.client_connected else "BAĞLI DEĞİL"
        clutch = "AKTİF" if s.clutch_active else "bırakıldı"
        ik = "" if self.mapper.last_ik_ok else "  [IK sınırda]"
        return f"telefon {conn}, debriyaj {clutch}, paket {s.packet_count}{ik}"

    def publish_status(self, **status) -> None:
        self.state.push_status({"type": "status", **status})

    def close(self) -> None:
        self._monitor_stop.set()
        if self.server is not None:
            self.server.stop()
