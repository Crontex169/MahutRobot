"""
Kayıt oturumu: ortam kurulumu ve bölüm döngüsü. Operatör arayüzü ve hedef işareti: `ui.py`.
`scripts/record_vla_demos.py` ve `scripts/generate_scripted_demos.py` tarafından paylaşılır.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from mahut_kol.envs.multi_shape_task import DEFAULT_MAX_EPISODE_STEPS
from mahut_kol.recording.ui import Overlay

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
