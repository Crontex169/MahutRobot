"""
MultiShapeRelationalTask için otomatik (scripted) gösterim politikası.

`pick_cube_task.scripted_actions` mantığının çok şekilli ilişkisel göreve genişletilmiş hali:
yaklaş -> kavra -> kaldır -> hedefin üstüne taşı -> indir -> bırak -> geri çekil -> bekle.

- Kinematik: `mahut_kol.robots.kinematics` (URDF ile birebir FK, eğim kısıtlı DLS IK).
- Kıskaç düşeyden eğik yaklaşır (MahutKol tam aşağı bakarak çalışma alanına erişemez). Eğim
  hem alma hem bırakma noktasında çözülebilen {50°, 45°, 55°} içinden seçilir; taşıma boyunca
  sabit tutulur, böylece nesne dik bırakılır.
- Şekle göre: kavrama noktası nesne merkezinin biraz üstü; parmak ekseni kol düzlemine dik
  (ortam yaw'ı buna göre hizalı spawn eder, ± 15° gürültü). Prizmada taban kenarı parmak
  eksenine paralel: pedler iki eğik yüze basar. Kalan yaw sapmasını bilek roll'u (joint4) ile
  telafi etme seçeneği (use_roll=True) denendi: eğik yaklaşımda roll parmakları masaya
  indirdiği için erişilebilirliği düşürüyor (30 bölümde 29 -> 20 başarı); varsayılan kapalı,
  ±15° sapmayı sürtünme ve kıskaç sıkması tolere ediyor.
- Kıskaç bırakmada yalnızca kısmen açılır (komşu referansa sürtmemek için).
- Distractor'lar: taşıma yüksekliği tüm nesnelerin üstünden geçecek kadar; ortam hedef
  noktasını boş tutar.

Kullanım: `policy = MultiShapeScriptedPolicy(env.unwrapped); policy.reset(qpos); a = policy.act(qpos)`.
`policy.plan_error` boş değilse (ör. IK erişilemez) bölüm atlanmalıdır.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from mahut_kol.envs.multi_shape_task import arm_heading, object_shape
from mahut_kol.robots.kinematics import MahutKinematics, finger_min_z, get_kinematics, solve_grasp_ik

FINGER_PREGRASP = 0.010   # yaklaşırken parmak açıklığı (0 = tam açık): komşu nesnelere sürtmesin
FINGER_CLOSE = 0.042      # kapat (kuvvet sınırlı; temas ~0.025-0.030'da)
FINGER_RELEASE = 0.016    # bırakma: < GRIPPER_RELEASED_QPOS (0.022) ama referansa sürtmeyecek kadar
GRASP_DZ = 0.0045         # kavrama noktası = nesne merkezi + 4.5 mm
CARRY_Z_MIN = 0.085       # taşıma: kavrama noktası en az bu yükseklikte (nesne altı ~6 cm,
                          # 35 mm'lik nesnelerin ~2.8 cm üstünden geçer)
CARRY_CLEARANCE = 0.03    # taşıma: alma/bırakma noktalarının en az bu kadar üstü
PLACE_DZ = 0.004          # bırakma anında nesne alt yüzü hedefin bu kadar üstünde
PREGRASP_BACK = 0.055     # yaklaşma ekseni boyunca geri çekilme
PREGRASP_UP = 0.02
SPEED = 0.006             # m / kontrol adımı (20 Hz -> 12 cm/s)
TILT_ORDER_DEG = (50, 45, 55)
MAX_ROLL = np.radians(30.0)


def _yaw_of(q_wxyz: Sequence[float]) -> float:
    w, x, y, z = q_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wrap(a: float, period: float) -> float:
    """a'yı [-period/2, period/2) aralığına indirger."""
    return float((a + period / 2) % period - period / 2)


class MultiShapeScriptedPolicy:
    name = "scripted"

    def __init__(self, env, kin: Optional[MahutKinematics] = None, use_roll: bool = False):
        self.env = env  # unwrapped MultiShapeRelationalTask
        self.K = kin or get_kinematics()
        self.use_roll = use_roll
        self.plan: List[np.ndarray] = []
        self.phases: List[Tuple[str, int, int]] = []  # (ad, başlangıç, bitiş)
        self.i = 0
        self.plan_error: Optional[str] = None
        self.tilt: Optional[float] = None

    # ------------------------------------------------------------------
    def _state(self):
        e = self.env
        man = e.get_manipulated_object()
        p_obj = man.pose.p[0].cpu().numpy().astype(np.float64)
        yaw = _yaw_of(man.pose.q[0].cpu().numpy())
        tgt = e.get_relational_target_position()[0].cpu().numpy().astype(np.float64)
        return p_obj, yaw, tgt

    def _roll_for(self, shape: str, obj_yaw: float, x: float, y: float, tilt: float) -> float:
        """Parmak ekseni ile nesne yüzleri arasındaki yaw farkını joint4 ile telafi eder."""
        if not self.use_roll or shape == "cylinder":
            return 0.0
        heading = arm_heading(x, y)
        if shape == "cube":
            err = _wrap(obj_yaw - heading, np.pi / 2)
        else:  # prizma: taban kenarı (yerel x) parmak eksenine (heading + 90°) paralel olmalı
            err = _wrap(obj_yaw - (heading + np.pi / 2), 2 * np.pi / 3)
        # Eğik yaklaşımda roll φ, parmak ekseninin yatay yönünü ~atan(tan φ · cos tilt) kadar
        # döndürür -> gereken φ = atan(tan(err) / cos(tilt)). İşaret: +joint4 ekseni yaklaşma yönü.
        phi = float(np.arctan2(np.tan(err), np.cos(tilt)))
        return float(np.clip(-phi, -MAX_ROLL, MAX_ROLL))

    def _ik(self, G, tilt, q_prev, roll) -> Tuple[np.ndarray, bool]:
        q, ok = solve_grasp_ik(G, tilt, q_init=q_prev, roll=roll, K=self.K)
        return q, ok

    def _line(self, G0, G1, tilt, q_prev, roll0, roll1, finger, min_steps=1):
        """Kavrama noktasını G0'dan G1'e düz çizgide götüren eklem hedefleri."""
        G0, G1 = np.asarray(G0), np.asarray(G1)
        n = max(min_steps, int(np.ceil(np.linalg.norm(G1 - G0) / SPEED)))
        out = []
        q = q_prev
        for k in range(1, n + 1):
            s = k / n
            s = 0.5 - 0.5 * np.cos(np.pi * s)  # yumuşak hız profili
            q, ok = self._ik(G0 + (G1 - G0) * s, tilt, q, roll0 + (roll1 - roll0) * s)
            if not ok:
                return None, q
            out.append(np.r_[q, finger, finger])
        return out, q

    # ------------------------------------------------------------------
    def reset(self, qpos: Sequence[float]) -> None:
        self.i = 0
        self.plan, self.phases, self.plan_error = [], [], None
        q0 = np.asarray(qpos, dtype=np.float64)[:5]
        p_obj, yaw, tgt = self._state()
        shape = object_shape(self.env.manipulated_object_id)
        G_pick = p_obj + [0, 0, GRASP_DZ]
        # hedef: nesne merkezi hedefte olacak şekilde, biraz yukarıdan bırak
        G_place = tgt + [0, 0, GRASP_DZ + PLACE_DZ]

        for tilt_deg in TILT_ORDER_DEG:
            tilt = np.radians(tilt_deg)
            plan, phases, err = self._build(q0, G_pick, G_place, yaw, shape, tilt)
            if err is None:
                self.plan, self.phases, self.tilt = plan, phases, tilt
                return
        self.plan_error = f"ik_unreachable ({err})"

    def _build(self, q0, G_pick, G_place, yaw, shape, tilt):
        plan: List[np.ndarray] = []
        phases = []

        def add(name, seg):
            phases.append((name, len(plan), len(plan) + len(seg)))
            plan.extend(seg)

        roll_pick = self._roll_for(shape, yaw, G_pick[0], G_pick[1], tilt)
        # Taşıma sırasında nesne kıskaçla birlikte döner; bırakmada roll'u sıfıra döndürmek
        # nesneyi dünyaya göre -roll kadar çevirir (yaw sapması küçük olduğundan sorun değil).
        roll_place = roll_pick

        q_pg, ok = self._ik(G_pick, tilt, q0, roll_pick)
        if not ok:
            return None, None, "pick"
        a = self.K.approach_of(self.K.fk(q_pg)[1])
        G_pre = G_pick - PREGRASP_BACK * a + [0, 0, PREGRASP_UP]
        q_pre, ok = self._ik(G_pre, tilt, q_pg, roll_pick)
        if not ok:
            return None, None, "pregrasp"

        # 1. Dinlenmeden ön-kavrama noktasına (eklem uzayında yumuşak)
        n = 40
        seg = []
        start = np.r_[q0, 0.0, 0.0]
        end = np.r_[q_pre, FINGER_PREGRASP, FINGER_PREGRASP]
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * k / n)
            seg.append(start + (end - start) * s)
        add("approach", seg)
        # 2. Yaklaşma ekseni boyunca nesneye in
        seg, q = self._line(G_pre, G_pick, tilt, q_pre, roll_pick, roll_pick, FINGER_PREGRASP, 15)
        if seg is None:
            return None, None, "descend_pick"
        add("descend_pick", seg + [seg[-1]] * 6)
        # 3. Kapat
        add("close", [np.r_[q, f, f] for f in np.linspace(FINGER_PREGRASP, FINGER_CLOSE, 8)] + [np.r_[q, FINGER_CLOSE, FINGER_CLOSE]] * 10)
        # 4. Kaldır
        carry_z = max(CARRY_Z_MIN, G_pick[2] + CARRY_CLEARANCE, G_place[2] + CARRY_CLEARANCE)
        G_lift = np.r_[G_pick[:2], carry_z]
        seg, q = self._line(G_pick, G_lift, tilt, q, roll_pick, roll_pick, FINGER_CLOSE, 15)
        if seg is None:
            return None, None, "lift"
        add("lift", seg)
        # 5. Hedefin üstüne taşı
        G_above = np.r_[G_place[:2], carry_z]
        seg, q = self._line(G_lift, G_above, tilt, q, roll_pick, roll_place, FINGER_CLOSE, 20)
        if seg is None:
            return None, None, "transport"
        add("transport", seg + [seg[-1]] * 6)
        # 6. İndir
        seg, q = self._line(G_above, G_place, tilt, q, roll_place, roll_place, FINGER_CLOSE, 20)
        if seg is None:
            return None, None, "descend_place"
        add("descend_place", seg + [seg[-1]] * 8)
        # 7. Bırak (kısmen aç)
        add("release", [np.r_[q, f, f] for f in np.linspace(FINGER_CLOSE, FINGER_RELEASE, 6)] + [np.r_[q, FINGER_RELEASE, FINGER_RELEASE]] * 8)
        # 8. Geri çekil: yukarı (parmaklar nesneden dikey sıyrılsın), sonra yaklaşma ekseninin tersine
        a = self.K.approach_of(self.K.fk(q)[1])
        G_up = G_place + [0, 0, 0.045]
        seg, q = self._line(G_place, G_up, tilt, q, roll_place, roll_place, FINGER_RELEASE, 12)
        if seg is None:
            return None, None, "retreat"
        G_back = G_up - 0.03 * a + [0, 0, 0.02]
        seg2, q2 = self._line(G_up, G_back, tilt, q, roll_place, roll_place, FINGER_RELEASE, 8)
        if seg2 is not None:
            seg, q = seg + seg2, q2
        add("retreat", seg)
        # 9. Bekle (başarı için nesnenin durağanlaşması gerekir)
        add("hold", [seg[-1]] * 40)
        # Masaya girme son kontrolü
        for t in plan:
            if finger_min_z(self.K, t[:5], float(t[5])) < 0.0:
                return None, None, "table_collision"
        return [np.asarray(t, dtype=np.float32) for t in plan], phases, None

    def act(self, qpos: Sequence[float]) -> np.ndarray:
        if not self.plan:
            return np.asarray(qpos, dtype=np.float32)[:7]
        a = self.plan[min(self.i, len(self.plan) - 1)]
        self.i += 1
        return a.copy()

    def phase_end(self, name: str) -> Optional[int]:
        for n, s, e in self.phases:
            if n == name:
                return e
        return None

    def __len__(self) -> int:
        return len(self.plan)
