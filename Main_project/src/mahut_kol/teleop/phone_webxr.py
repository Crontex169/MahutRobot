"""
Telefon (WebXR / ARCore) ile MahutKol teleoperasyonu - ManiSkill sürümü.

`Side_project/webxr_teleop.py` (MuJoCo + SO-100) referans alınarak ManiSkill ve MahutKol için
yeniden yazıldı; orijinal dosyaya dokunulmadı. Aynı WebSocket protokolü kullanılır, böylece
`src/mahut_kol/teleop/web/index.html` (orijinalin kopyası) değişmeden bağlanabilir.

Protokol ve WebSocket sunucusu: `phone_server.py`.

Kurulum (telefon USB ile bağlı, USB hata ayıklama açık):
    adb reverse tcp:8000 tcp:8000
    python3 scripts/record_vla_demos.py --device phone
    Telefonda Chrome: http://localhost:8000 -> "AR Teleop Başlat"

Varsayılan konum eşlemesi (CONFIG["control_frame"] = "wrist"): kıskaç kamerası görüntüsüne göre;
telefon sağa -> görüntüde sağa, ileri -> görüntünün üstü, yukarı -> +z (bkz. wrist_control_basis).

"world" eksen eşlemesi (ManiSkill dünyası: robot +x'e bakar; operatör x > 0 tarafında robota, yani -x
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
- `Side_project/ik_solver.py` nullspace'li DLS: hızlı ve artımlı, fakat MuJoCo'ya bağlı ve bilek
  pitch'ini eklem hedefi olarak nullspace'te izler; MahutKol'da konum için 4 eklem kullanılınca
  nullspace 1 boyutlu kalır ve yaklaşma açısı kontrolü zayıflar.
- Seçilen: aynı DLS fikri, ancak yaklaşma eğimini birincil görevin parçası (tek skaler kısıt)
  yapan, URDF ile birebir FK'li sürüm. Önceki çözümden ısıtılmış başlangıçla adım başına ~1-3 ms.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np

from mahut_kol.robots.kinematics import MahutKinematics, finger_min_z, get_kinematics
from mahut_kol.teleop.action_source import (
    ARM_DOF,
    ActionSource,
    clip_action,
    gripper_open_ratio_to_angle,
)
from mahut_kol.teleop.phone_server import DEFAULT_HTML, BackgroundServer, TeleopSharedState, create_app

CONFIG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    # Telefon yer değiştirmesi (m) -> TCP hedefi (m); açı (rad) -> açı (rad)
    "pos_scale": 1.0,
    "rot_scale": 1.0,
    # Konum kontrol çerçevesi:
    #   "wrist": bilek (kıskaç) kamerası görüntüsüne göre - telefon sağa -> görüntüde sağa,
    #            ileri itmek -> görüntünün üstüne doğru (kameranın baktığı yatay yön), yukarı -> +z.
    #            Çerçeve debriyaja basıldığı anda sabitlenir (vuruş boyunca dönmez).
    #   "world": sabit front kameraya göre (axis_map / axis_signs).
    "control_frame": "wrist",
    # Δdünya[i] = işaret[i] * Δtelefon[eksen[i]]  ("world" çerçevesi; bkz. modül açıklaması)
    "axis_map": {"x": 2, "y": 0, "z": 1},
    "axis_signs": {"x": 1.0, "y": 1.0, "z": 1.0},
    # delta_euler indeksleri: 0 = X etrafında (pitch), 1 = Y etrafında, 2 = Z etrafında
    # Debriyajdan beri kamera yönü bu kadar döndüyse pencerede "bırakıp yeniden bas" uyarısı
    "frame_drift_warn_deg": 30.0,
    "tilt_from": (0, 1.0),
    "roll_from": (2, -1.0),
    "ema_alpha_pos": 0.35,
    "ema_alpha_rot": 0.35,
    "max_step_pos": 0.02,          # m / kontrol adımı (20 Hz -> 0.4 m/s)
    "max_step_joint": 0.15,        # rad / kontrol adımı
    # TCP çalışma hacmi (dünya, m). Masa üstü çalışma alanını (WORKSPACE_*) kaldırma/yaklaşma
    # payıyla kapsar. x >= 0.08: daha yakında kol tabanın üstüne katlanıyor ve bilek kamerası
    # geriye bakıyordu (operatöre yönler ters geliyordu).
    "workspace_limits": {"x": (0.08, 0.30), "y": (-0.22, 0.22), "z": (0.02, 0.30)},
    "tilt_limits_deg": (25.0, 85.0),
    "ik_iterations": 30,
    "packet_timeout_s": 0.5,
    # Tek pakette (≈30 Hz) bundan büyük konum sıçraması ARCore takip kaybı/yeniden konumlanma
    # sayılır ve yok edilir (gerçek denemede bir anda 4-7 m sıçradı). 0.08 m/paket ≈ 2.4 m/s.
    "max_packet_jump_m": 0.08,       # bu süre paket gelmezse debriyaj bırakılmış sayılır
}


def phone_to_world_delta(delta_pos: Sequence[float], config: Dict[str, Any] = CONFIG) -> np.ndarray:
    d = np.asarray(delta_pos, dtype=np.float64)
    am, sg = config["axis_map"], config["axis_signs"]
    return config["pos_scale"] * np.array(
        [sg["x"] * d[am["x"]], sg["y"] * d[am["y"]], sg["z"] * d[am["z"]]]
    )


def wrist_camera_rotation(R_tcp: np.ndarray) -> np.ndarray:
    """Bilek kamerasının dünya dönüşü (SAPIEN kamera: x ileri, y sol, z yukarı)."""
    from mahut_kol.envs.multi_shape_task import WRIST_CAMERA_PITCH_DEG

    a = np.radians(WRIST_CAMERA_PITCH_DEG)  # TCP y ekseni etrafında (ortamdaki montajla aynı)
    Ry = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    return np.asarray(R_tcp) @ Ry


def wrist_control_basis(R_tcp: np.ndarray) -> Optional[np.ndarray]:
    """
    Bilek kamerası görüntüsüne göre yatay kontrol tabanı: sütunlar [sağ, ileri, yukarı] (dünya).
    sağ = görüntünün sağının yatay izdüşümü; ileri = yukarı × sağ (kamera aşağı bakarken
    görüntünün üstü, öne bakarken görüş yönü); yukarı = dünya +z. Dejenere ise None.
    """
    Rc = wrist_camera_rotation(R_tcp)
    right = -Rc[:, 1]
    right = np.array([right[0], right[1], 0.0])
    n = float(np.linalg.norm(right))
    if n < 0.2:  # kamera yana yatmış: görüntü sağı neredeyse düşey
        return None
    right /= n
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.cross(up, right)
    return np.stack([right, fwd, up], axis=1)


def phone_to_wrist_delta(delta_pos: Sequence[float], basis: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Telefon (WebXR: +X sağ, +Y yukarı, +Z kullanıcıya doğru) -> dünya, bilek kamerası tabanında."""
    d = np.asarray(delta_pos, dtype=np.float64)
    return scale * (basis @ np.array([d[0], -d[2], d[1]]))


# ---------------------------------------------------------------------------
# Thread-safe paylaşılan durum
# ---------------------------------------------------------------------------


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
        self.frame_drift_deg = 0.0
        self.basis: Optional[np.ndarray] = None  # bilek kamerası kontrol tabanı (debriyajda sabitlenir)
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
        self.basis = None
        if self.cfg.get("control_frame", "wrist") == "wrist":
            self.basis = wrist_control_basis(self.K.fk(self.q_cmd)[1])
        self.frame_drift_deg = 0.0

    def _update_frame_drift(self) -> None:
        """
        Debriyajda sabitlenen kontrol çerçevesi ile kameranın şu anki yönü arasındaki açı (°).
        Uzun yana harekette kol tabanda döner; fark büyüyünce kontrol ters/çapraz hissettirir.
        Çerçeve bilerek sabit tutulur (her adımda güncellemek denendi, daha kötü bulundu);
        bunun yerine operatöre "debriyajı bırakıp yeniden bas" uyarısı gösterilir.
        """
        if self.basis is None:
            self.frame_drift_deg = 0.0
            return
        cur = wrist_control_basis(self.K.fk(self.q_cmd)[1])
        if cur is None:
            return
        c = float(np.clip(cur[:, 0] @ self.basis[:, 0], -1.0, 1.0))
        self.frame_drift_deg = float(np.degrees(np.arccos(c)))

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
        if self.basis is not None:
            cand = self.anchor_pos + phone_to_wrist_delta(delta_pos, self.basis, c["pos_scale"])
        else:
            cand = self.anchor_pos + phone_to_world_delta(delta_pos, c)
        lim = c["workspace_limits"]
        clipped = np.array([np.clip(cand[i], *lim[k]) for i, k in enumerate("xyz")])
        # Kayan debriyaj: sınır dışındaki fazlalık birikmesin, geri dönüş anında başlasın
        self.anchor_pos += clipped - cand
        cand = clipped
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
            # Erişilemeyen hedef: fazlalığı at (kayan debriyaj), hedefi ulaşılan noktaya çek
            reached = self.K.fk(q)[0]
            self.anchor_pos += reached - new_pos
            new_pos = reached
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
        self._update_frame_drift()
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
        self.state = state or TeleopSharedState(self.cfg["max_packet_jump_m"])
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
        self.target = clip_action(np.r_[arm, gripper_open_ratio_to_angle(grip)])
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
                    f"hatalı={bad} takipSıçraması={s.jump_count} debriyaj={'BASILI' if clutch else 'yok'} son paket {age:4.1f} s önce "
                    f"Δtel(cm)={np.round(dpos * 100, 1)} Δaçı(°)={np.round(np.degrees(deul), 0)}")
            if m.q_cmd is not None:
                line += f" hedefTCP={np.round(m.K.fk(m.q_cmd)[0], 3)}"
                if self._last_qpos is not None:
                    line += f" gerçekTCP={np.round(m.K.fk(self._last_qpos[:ARM_DOF])[0], 3)}"
                line += f" ikHata={m.ik_fails} masaRed={m.guard_rejects} yönKaydı={m.frame_drift_deg:.0f}°"
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
        drift = ""
        if s.clutch_active and self.mapper.frame_drift_deg > self.cfg["frame_drift_warn_deg"]:
            drift = f"  ⚠ YÖN KAYDI {self.mapper.frame_drift_deg:.0f}°: debriyajı bırakıp tekrar basın"
        return f"telefon {conn}, debriyaj {clutch}, paket {s.packet_count}{ik}{drift}"

    def publish_status(self, **status) -> None:
        self.state.push_status({"type": "status", **status})

    def close(self) -> None:
        self._monitor_stop.set()
        if self.server is not None:
            self.server.stop()
