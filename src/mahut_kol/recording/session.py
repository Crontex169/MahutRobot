"""
Kayıt oturumu yardımcıları: ortam kurulumu, bölüm döngüsü ve operatör arayüzü (pygame/terminal).
`scripts/record_vla_demos.py` ve `scripts/generate_scripted_demos.py` tarafından paylaşılır.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from mahut_kol.envs.multi_shape_task import DEFAULT_MAX_EPISODE_STEPS, SUCCESS_XY_TOLERANCE

ENV_ID = "MahutKol-MultiShapeRelational-v1"


def make_env(max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS, render_backend: Optional[str] = None, **kw):
    import gymnasium as gym

    import mahut_kol.envs  # noqa: F401
    import mahut_kol.robots  # noqa: F401

    kwargs = dict(
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        max_episode_steps=max_episode_steps,
    )
    if render_backend:
        kwargs["render_backend"] = render_backend
    kwargs.update(kw)
    return gym.make(ENV_ID, **kwargs)


@dataclass
class EpisodeResult:
    frames: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    success_step: Optional[int] = None
    steps: int = 0
    duration_s: float = 0.0
    aborted: Optional[str] = None  # "reset" | "quit" | None
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hedef bölge işareti (yalnızca operatör ekranı)
# ---------------------------------------------------------------------------

Overlay = Dict[str, List[np.ndarray]]  # kamera adı -> [poligon (N, 2) piksel, merkez (1, 2) piksel]


def goal_region_world(env, n: int = 32) -> Optional[np.ndarray]:
    """
    Kabul bölgesinin dünya noktaları (n, 3): hedefin oturma yüzeyinde (masa ya da TOP'ta
    referansın üstü), yarıçapı SUCCESS_XY_TOLERANCE olan daire. Son satır merkezdir.
    """
    u = env.unwrapped
    try:
        tgt = u.get_relational_target_position()[0].cpu().numpy().astype(np.float64)
    except Exception:
        return None
    z = tgt[2] - float(getattr(u, "object_height", 0.0)) / 2.0 + 0.001
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([tgt[0] + SUCCESS_XY_TOLERANCE * np.cos(a), tgt[1] + SUCCESS_XY_TOLERANCE * np.sin(a),
                     np.full(n, z)], axis=1)
    return np.vstack([ring, [tgt[0], tgt[1], z]])


def project_points(params: Dict[str, Any], pts: np.ndarray) -> Optional[np.ndarray]:
    """OpenCV kamerası: dünya noktaları -> piksel (N, 2); kameranın arkasındaysa None."""

    def _np(x):
        x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
        return x[0] if x.ndim == 3 else x

    E, K = _np(params["extrinsic_cv"]), _np(params["intrinsic_cv"])
    pc = (E[:, :3] @ pts.T + E[:, 3:4]).T
    if np.any(pc[:, 2] <= 1e-4):
        return None
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def goal_overlay(env, cameras=("front", "wrist")) -> Overlay:
    """
    Hedef bölgenin kamera görüntülerindeki pikselleri. Bu işaret SAHNEYE EKLENMEZ, yalnızca
    pygame penceresine çizilir: kaydedilen gözlemler işaretsiz kalır (aksi halde politika
    işarete bakmayı öğrenir ve gerçek kullanımda işaret olmadığı için başarısız olurdu).
    """
    pts = goal_region_world(env)
    if pts is None:
        return {}
    params = env.unwrapped.get_sensor_params()
    out: Overlay = {}
    for cam in cameras:
        if cam in params:
            uv = project_points(params[cam], pts)
            if uv is not None:
                out[cam] = [uv[:-1], uv[-1:]]
    return out


# ---------------------------------------------------------------------------
# Operatör arayüzü
# ---------------------------------------------------------------------------


class TerminalUI:
    """Ekransız arayüz: talimatı yazdırır, onayı input() ile alır."""

    def __init__(self, auto_confirm: Optional[str] = None):
        self.auto_confirm = auto_confirm  # "save" | "discard" | None

    def show_instruction(self, text: str, extra: str = "") -> None:
        print(f"\n🗣️  TALİMAT: {text}  {extra}", flush=True)

    def update(self, obs, status: str, success: bool, overlay: Optional[Overlay] = None) -> Set[str]:
        return set()

    def confirm(self, prompt: str) -> str:
        if self.auto_confirm:
            return self.auto_confirm
        try:
            ans = input(f"{prompt} [Enter=kaydet / r=at ve yeniden dene / q=çık]: ").strip().lower()
        except EOFError:
            return "quit"
        return {"": "save", "r": "discard", "q": "quit"}.get(ans, "discard")

    def close(self) -> None:
        pass


class PygameUI:
    """
    Pencere: front + wrist kamera görüntüsü, talimat ve durum. Tuşlar:
    Enter = kaydet (başarı sonrası), R = at ve yeniden dene, Q/Esc = çık.
    Klavye teleopu (GamepadKeyboardActionSource) da bu pencerenin odağını kullanır.
    """

    def __init__(self, scale: int = 3, control_camera: Optional[str] = None):
        # Telefon hareketinin göre olduğu kamera ("front" | "wrist"); None = etiket yok
        self.control_camera = control_camera
        # SDL varsayılan olarak SIGINT/SIGTERM'i yakalayıp pencereye QUIT olayı gönderir; bu da
        # betiğin "çıkılıyor" deyip sessizce kapanmasına yol açıyordu. Sinyaller Python'da kalsın.
        os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
        import pygame

        self.pg = pygame
        pygame.init()
        self.scale = scale
        self.W, self.H = 224 * scale * 2, 224 * scale + 90
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("MahutKol VLA kayıt")
        self.font = pygame.font.SysFont("dejavusans", 20)
        self.small = pygame.font.SysFont("dejavusans", 16)
        self.instruction = ""

    def show_instruction(self, text: str, extra: str = "") -> None:
        self.instruction = text
        print(f"\n🗣️  TALİMAT: {text}  {extra}", flush=True)

    def _events(self) -> Set[str]:
        ev = set()
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT:
                print("[Arayüz] Pencere kapatıldı (QUIT).", flush=True)
                ev.add("quit")
            elif e.type == self.pg.KEYDOWN:
                if e.key in (self.pg.K_RETURN, self.pg.K_KP_ENTER):
                    ev.add("save")
                elif e.key == self.pg.K_r:
                    ev.add("discard")
                elif e.key in (self.pg.K_q, self.pg.K_ESCAPE):
                    print(f"[Arayüz] '{self.pg.key.name(e.key)}' tuşu: çıkış.", flush=True)
                    ev.add("quit")
        return ev

    def update(self, obs, status: str, success: bool, overlay: Optional[Overlay] = None) -> Set[str]:
        pg = self.pg
        self.screen.fill((20, 20, 24))
        side = 224 * self.scale
        for i, cam in enumerate(("front", "wrist")):
            img = obs["sensor_data"][cam]["rgb"][0].cpu().numpy()
            surf = pg.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            surf = pg.transform.scale(surf, (side, side))
            if overlay and cam in overlay:
                self._draw_goal(surf, overlay[cam], img.shape[1], img.shape[0], success)
            self.screen.blit(surf, (i * side, 0))
            self._draw_control_label(cam, i * side, side)
        y0 = 224 * self.scale + 8
        col = (80, 230, 120) if success else (235, 235, 235)
        self.screen.blit(self.font.render(self.instruction, True, col), (10, y0))
        self.screen.blit(self.small.render(status, True, (170, 170, 190)), (10, y0 + 32))
        hint = "BAŞARILI - Enter: kaydet, R: at" if success else "R: at ve yeniden dene   Q: çık"
        self.screen.blit(self.small.render(hint, True, col), (10, y0 + 56))
        pg.display.flip()
        return self._events()

    def _draw_control_label(self, cam: str, x0: int, side: int) -> None:
        """
        İki kamerada sağ-sol ve ileri-geri birbirinin tersidir; operatör yanlış görüntüye bakınca
        kontrol ters sanılıyordu. Yalnızca pencereye çizilir, kayda girmez.
        """
        if self.control_camera is None:
            return
        pg = self.pg
        active = cam == self.control_camera
        color = (60, 220, 110) if active else (235, 170, 60)
        text = "KONTROL BU GÖRÜNTÜYE GÖRE" if active else "SADECE BAKIŞ - YÖNLER TERS"
        label = self.small.render(text, True, (0, 0, 0))
        pg.draw.rect(self.screen, color, (x0 + 6, 6, label.get_width() + 12, label.get_height() + 6))
        self.screen.blit(label, (x0 + 12, 9))
        if active:
            pg.draw.rect(self.screen, color, (x0, 0, side, side), width=4)

    def _draw_goal(self, surf, polys, w: int, h: int, success: bool) -> None:
        """Kabul bölgesi: yarı saydam siyah daire + kenar + merkez noktası (yalnızca ekranda)."""
        pg = self.pg
        sx, sy = surf.get_width() / w, surf.get_height() / h
        ring = [(float(x * sx), float(y * sy)) for x, y in polys[0]]
        cx, cy = float(polys[1][0][0] * sx), float(polys[1][0][1] * sy)
        layer = pg.Surface(surf.get_size(), pg.SRCALPHA)
        pg.draw.polygon(layer, (0, 0, 0, 120), ring)
        edge = (40, 200, 90, 255) if success else (0, 0, 0, 255)
        pg.draw.polygon(layer, edge, ring, width=2)
        pg.draw.circle(layer, (255, 255, 255, 220), (cx, cy), 3)
        surf.blit(layer, (0, 0))

    def confirm(self, prompt: str) -> str:
        print(f"{prompt}  (pencerede Enter=kaydet, R=at, Q=çık)", flush=True)
        while True:
            ev = self._events()
            for k in ("save", "discard", "quit"):
                if k in ev:
                    return k
            time.sleep(0.02)

    def close(self) -> None:
        self.pg.quit()


# ---------------------------------------------------------------------------
# Bölüm döngüsü
# ---------------------------------------------------------------------------


def wait_for_start(
    obs,
    actions,
    ui=None,
    poll_s: float = 0.05,
    timeout_s: Optional[float] = None,
    overlay: Optional[Overlay] = None,
) -> str:
    """
    Bölümden önce operatörü bekler; simülasyon adımlanmaz (sahne sabit, kare kaydedilmez).
    Döner: "start" (actions.ready() oldu), "reset" (yeni sahne istendi), "quit" ya da "timeout".
    """
    ready = getattr(actions, "ready", lambda: True)  # eski/duck-typed kaynaklar: hemen başla
    start_hint = getattr(actions, "start_hint", lambda: "")
    t0 = time.time()
    last_hint = None
    while True:
        if ready():
            return "start"
        hint = start_hint()
        if hint != last_hint and ui is None:
            print(f"⏳ {hint}", flush=True)
        last_hint = hint
        ev: Set[str] = set()
        if ui is not None:
            ev |= ui.update(obs, f"⏳ {hint}", False, overlay=overlay) if overlay else \
                ui.update(obs, f"⏳ {hint}", False)
        ev |= actions.poll_events()
        if "quit" in ev:
            return "quit"
        if "discard" in ev or "reset" in ev:
            return "reset"
        if timeout_s is not None and time.time() - t0 > timeout_s:
            return "timeout"
        time.sleep(poll_s)


def run_episode(
    env,
    obs,
    act_fn: Callable[[np.ndarray], np.ndarray],
    frame_fn: Callable[[Any, np.ndarray], Dict[str, Any]],
    max_steps: int,
    ui=None,
    status_fn: Optional[Callable[[], str]] = None,
    events_fn: Optional[Callable[[], Set[str]]] = None,
    on_success: Optional[Callable[[int], None]] = None,
    realtime_hz: Optional[float] = None,
    stop_on_success: bool = True,
    step_cb: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    overlay_fn: Optional[Callable[[], Overlay]] = None,
) -> EpisodeResult:
    """
    Gözlem -> eylem -> kare kaydı -> env.step döngüsü. Kare, eylemden ÖNCEKİ gözlem ile o
    gözlemde verilen eylemi eşler (LeRobot kuralı). Başarı ilk görüldüğünde durur.
    """
    import torch

    res = EpisodeResult()
    t0 = time.time()
    u = env.unwrapped
    dt = 1.0 / realtime_hz if realtime_hz else 0.0
    for step in range(max_steps):
        t_step = time.time()
        qpos = u.agent.robot.get_qpos()[0].cpu().numpy()
        action = np.asarray(act_fn(qpos), dtype=np.float32)
        res.frames.append(frame_fn(obs, action))
        obs, _, terminated, truncated, info = env.step(torch.as_tensor(action))
        res.steps = step + 1
        if step_cb is not None:
            step_cb(step + 1, info)
        success = bool(info["success"].item())
        ev: Set[str] = set()
        if ui is not None:
            ov = overlay_fn() if overlay_fn else None
            ev |= ui.update(obs, status_fn() if status_fn else "", success, overlay=ov) if ov else \
                ui.update(obs, status_fn() if status_fn else "", success)
        if events_fn is not None:
            ev |= events_fn()
        if "quit" in ev:
            res.aborted = "quit"
            break
        if "discard" in ev or "reset" in ev:
            res.aborted = "reset"
            break
        if success:
            res.success = True
            res.success_step = step + 1
            if on_success:
                on_success(step + 1)
            if stop_on_success:
                break
        if bool(truncated.item() if hasattr(truncated, "item") else truncated):
            break
        if dt:
            sleep = dt - (time.time() - t_step)
            if sleep > 0:
                time.sleep(sleep)
    res.duration_s = time.time() - t0
    res.info = {k: (v.item() if hasattr(v, "item") and getattr(v, "numel", lambda: 2)() == 1 else None)
                for k, v in info.items()} if res.steps else {}
    return res
