"""
Kayıt oturumu yardımcıları: ortam kurulumu, bölüm döngüsü ve operatör arayüzü (pygame/terminal).
`scripts/record_vla_demos.py` ve `scripts/generate_scripted_demos.py` tarafından paylaşılır.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from mahut_kol.envs.multi_shape_task import DEFAULT_MAX_EPISODE_STEPS

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
# Operatör arayüzü
# ---------------------------------------------------------------------------


class TerminalUI:
    """Ekransız arayüz: talimatı yazdırır, onayı input() ile alır."""

    def __init__(self, auto_confirm: Optional[str] = None):
        self.auto_confirm = auto_confirm  # "save" | "discard" | None

    def show_instruction(self, text: str, extra: str = "") -> None:
        print(f"\n🗣️  TALİMAT: {text}  {extra}", flush=True)

    def update(self, obs, status: str, success: bool) -> Set[str]:
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

    def __init__(self, scale: int = 3):
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
                ev.add("quit")
            elif e.type == self.pg.KEYDOWN:
                if e.key in (self.pg.K_RETURN, self.pg.K_KP_ENTER):
                    ev.add("save")
                elif e.key == self.pg.K_r:
                    ev.add("discard")
                elif e.key in (self.pg.K_q, self.pg.K_ESCAPE):
                    ev.add("quit")
        return ev

    def update(self, obs, status: str, success: bool) -> Set[str]:
        pg = self.pg
        self.screen.fill((20, 20, 24))
        for i, cam in enumerate(("front", "wrist")):
            img = obs["sensor_data"][cam]["rgb"][0].cpu().numpy()
            surf = pg.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            surf = pg.transform.scale(surf, (224 * self.scale, 224 * self.scale))
            self.screen.blit(surf, (i * 224 * self.scale, 0))
        y0 = 224 * self.scale + 8
        col = (80, 230, 120) if success else (235, 235, 235)
        self.screen.blit(self.font.render(self.instruction, True, col), (10, y0))
        self.screen.blit(self.small.render(status, True, (170, 170, 190)), (10, y0 + 32))
        hint = "BAŞARILI - Enter: kaydet, R: at" if success else "R: at ve yeniden dene   Q: çık"
        self.screen.blit(self.small.render(hint, True, col), (10, y0 + 56))
        pg.display.flip()
        return self._events()

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
            ev |= ui.update(obs, status_fn() if status_fn else "", success)
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
