"""
MahutKol 5-DoF kol için URDF tabanlı saf numpy kinematiği (FK, Jacobian, IK).

Neden ayrı modül:
- `pick_cube_task.py` içindeki fk/ik MuJoCo modelini (`mahut_kol_robot.xml`) kullanır ve
  yalnızca joint0 = joint4 = 0 düzleminde çalışır. Simülasyonda yüklenen model ise URDF'tir.
- Bu modül doğrudan `assets/urdf/mahut_kol.urdf` zincirini okur; ManiSkill'deki
  `link_gripper_center` (TCP) pozuyla birebir aynı sonucu verir (bkz. tests/test_kinematics.py).
- Telefon teleopu (Aşama 3), scripted demo üretici (Aşama 5) ve çalışma alanı
  hesaplaması (Aşama 1) aynı çözücüyü paylaşır.

Eklemler: joint0 (taban yaw, +z), joint1-3 (birbirine paralel pitch eksenleri), joint4 (bilek roll).
Kıskacın "yaklaşma" (parmak ucu) yönü joint4 eksenidir: URDF'te bu eksen link4 +x'ine göre
~10 derece eğiktir ve TCP tam bu eksen üzerindedir. joint4 dönünce TCP konumu ve yaklaşma
yönü değişmez, yalnızca parmakların açılma doğrultusu (TCP yerel y) döner.

Serbestlik sayımı: joint1-3 paralel olduğundan kol, joint0 ile döndürülen düşey bir düzlemde
çalışır. joint0-3 (4 serbestlik) = TCP konumu (3) + yaklaşma yönünün düzlem içi eğimi (1).
Yani "tam aşağı bakan" kavrama tam belirli bir problemdir; joint4 kavrama yaw'ını ayarlar.

IK yaklaşımı: Sönümlü en küçük kareler (DLS). Birincil görev 3B konum; ikincil görev
yaklaşma yönü (ağırlıklı, 2 serbestlik) ve isteğe bağlı bilek roll hedefi (joint4,
konum ve yaklaşma yönünü etkilemediği için doğrudan atanır). Side_project/ik_solver.py
içindeki nullspace'li DLS fikri temel alındı; MahutKol'da joint4 ekseni yaklaşma eksenine
paralel olduğundan roll hedefi nullspace yerine doğrudan eklem değeri olarak verilir.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_DEFAULT_URDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../assets/urdf/mahut_kol.urdf")
)

# Robotun dünya çerçevesindeki taban pozu (ortamların _load_agent'ında kullanılan değer).
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.0558])

ARM_JOINTS = ("joint0", "joint1", "joint2", "joint3", "joint4")
TCP_LINK = "link_gripper_center"


def _rpy_to_mat(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_R: np.ndarray
    origin_p: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _parse_urdf(path: str) -> List[_Joint]:
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
        lim = j.find("limit")
        lo = float(lim.get("lower", 0)) if lim is not None else 0.0
        hi = float(lim.get("upper", 0)) if lim is not None else 0.0
        joints.append(
            _Joint(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                origin_R=_rpy_to_mat(rpy),
                origin_p=np.array(xyz),
                axis=axis,
                lower=lo,
                upper=hi,
            )
        )
    return joints


def _chain(joints: List[_Joint], tip: str) -> List[_Joint]:
    by_child = {j.child: j for j in joints}
    out = []
    link = tip
    while link in by_child:
        j = by_child[link]
        out.append(j)
        link = j.parent
    return out[::-1]


class MahutKinematics:
    """URDF zincirinden TCP ileri/ters kinematiği (dünya çerçevesinde)."""

    def __init__(self, urdf_path: Optional[str] = None, base_pos: Sequence[float] = ROBOT_BASE_POS):
        self.chain = _chain(_parse_urdf(urdf_path or _DEFAULT_URDF), TCP_LINK)
        names = [j.name for j in self.chain if j.type != "fixed"]
        if tuple(names) != ARM_JOINTS:
            raise ValueError(f"Beklenmeyen kinematik zincir: {names}")
        self.base_pos = np.asarray(base_pos, dtype=np.float64)
        self.q_min = np.array([j.lower for j in self.chain if j.type != "fixed"])
        self.q_max = np.array([j.upper for j in self.chain if j.type != "fixed"])
        j4 = [j for j in self.chain if j.name == "joint4"][0]
        tcp_joint = self.chain[-1]
        if tcp_joint.type != "fixed" or not np.allclose(tcp_joint.origin_R, np.eye(3)):
            raise ValueError("TCP sabit eklemi dönüşsüz olmalı")
        # TCP çerçevesi link4 ile aynı yönelimde; joint4 ekseni link4 çerçevesinde sabit
        self.roll_axis_local = j4.axis / np.linalg.norm(j4.axis)

    # ------------------------------------------------------------------
    def fk_all(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """TCP pozu + her hareketli eklemin dünya ekseni ve konumu (Jacobian için)."""
        R = np.eye(3)
        p = self.base_pos.copy()
        axes, origins = [], []
        i = 0
        for j in self.chain:
            p = p + R @ j.origin_p
            R = R @ j.origin_R
            if j.type != "fixed":
                a = R @ (j.axis / np.linalg.norm(j.axis))
                axes.append(a)
                origins.append(p.copy())
                R = R @ _axis_angle(j.axis, float(q[i]))
                i += 1
        return p, R, axes, origins

    def fk_batch(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vektörize FK. Q: (N, 5) -> (TCP konumları (N, 3), yaklaşma yönleri (N, 3))."""
        Q = np.asarray(Q, dtype=np.float64)
        n = Q.shape[0]
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        p = np.broadcast_to(self.base_pos, (n, 3)).copy()
        i = 0
        for j in self.chain:
            p = p + R @ j.origin_p
            R = R @ j.origin_R
            if j.type != "fixed":
                a = j.axis / np.linalg.norm(j.axis)
                k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                th = Q[:, i][:, None, None]
                R = R @ (np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k))
                i += 1
        return p, R @ self.roll_axis_local

    def fk(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """(TCP konumu, TCP dönüş matrisi). q: 5 kol eklemi."""
        p, R, _, _ = self.fk_all(q)
        return p, R

    def approach_of(self, R: np.ndarray) -> np.ndarray:
        """Kıskacın parmak ucu yönü (joint4 ekseni, dünya çerçevesinde)."""
        return R @ self.roll_axis_local

    @staticmethod
    def finger_axis_of(R: np.ndarray) -> np.ndarray:
        """Parmakların açılıp kapandığı doğrultu (TCP yerel y)."""
        return R[:, 1]

    def jacobian(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(konum Jacobian 3x5, açısal Jacobian 3x5, TCP p, TCP R)."""
        p, R, axes, origins = self.fk_all(q)
        Jp = np.stack([np.cross(a, p - o) for a, o in zip(axes, origins)], axis=1)
        Jw = np.stack(axes, axis=1)
        return Jp, Jw, p, R

    # ------------------------------------------------------------------
    def ik(
        self,
        target_pos: Sequence[float],
        tilt: Optional[float] = None,
        approach: Optional[Sequence[float]] = None,
        roll: Optional[float] = None,
        q_init: Optional[Sequence[float]] = None,
        n_iters: int = 150,
        damping: float = 1e-3,
        w_rot: float = 0.05,
        pos_tol: float = 1e-3,
        rot_tol_deg: float = 3.0,
        step: float = 1.0,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        DLS IK. Dönüş: (q[5], yakınsadı_mı, konum_hatası_m).

        tilt: yaklaşma yönünün düşeyden (aşağı) açısı, radyan. Önerilen kısıt budur:
            kol düzlemi taban ekseninden yanal ~4.4 cm kaçık olduğundan yaklaşma yönünün
            yatay bileşeni serbest bırakılır; tek skaler kısıt (a_z = -cos(tilt)).
            joint0-3 (4 serbestlik) = konum (3) + tilt (1) -> tam belirli.
        approach: tam yaklaşma vektörü (birim). Yalnızca kol düzleminde kalan vektörler
            ulaşılabilirdir; genel kullanımda tilt tercih edilir.
        roll: joint4 hedefi (radyan). joint4 TCP konumunu ve yaklaşma yönünü değiştirmez,
            bu yüzden çözümden sonra doğrudan atanır.
        """
        tgt = np.asarray(target_pos, dtype=np.float64)
        q = np.array(q_init if q_init is not None else DEFAULT_IK_SEEDS[0], dtype=np.float64)[:5]
        q = np.clip(q, self.q_min, self.q_max)
        app = None if approach is None else np.asarray(approach, dtype=np.float64) / np.linalg.norm(approach)
        az_target = None if tilt is None else -np.cos(tilt)
        for _ in range(n_iters):
            Jp, Jw, p, R = self.jacobian(q)
            a = self.approach_of(R)
            rows, errs = [Jp], [tgt - p]
            if app is not None:
                # a'yı app'e döndüren küçük açı vektörü; a boyunca açısal hız a'yı değiştirmez
                P = np.eye(3) - np.outer(a, a)
                rows.append(w_rot * (P @ Jw))
                errs.append(w_rot * np.cross(a, app))
            if az_target is not None:
                # d(a)/dq_i = w_i x a  ->  z bileşeni
                daz = np.cross(Jw.T, a)[:, 2]
                rows.append(w_rot * daz[None, :])
                errs.append(np.array([w_rot * (az_target - a[2])]))
            J = np.vstack(rows)[:, :4]  # joint4 konum/yaklaşmayı etkilemez
            e = np.concatenate(errs)
            if np.linalg.norm(e) < 1e-5:
                break
            dq4 = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
            q[:4] = np.clip(q[:4] + step * dq4, self.q_min[:4], self.q_max[:4])
        if roll is not None:
            q[4] = float(np.clip(roll, self.q_min[4], self.q_max[4]))
        p, R = self.fk(q)
        a = self.approach_of(R)
        err = float(np.linalg.norm(tgt - p))
        ok = err < pos_tol
        if app is not None:
            ok = ok and float(a @ app) > np.cos(np.radians(rot_tol_deg))
        if az_target is not None:
            ok = ok and abs(np.arccos(np.clip(-a[2], -1, 1)) - tilt) < np.radians(rot_tol_deg)
        return q, ok, err

    def ik_multi_start(
        self,
        target_pos: Sequence[float],
        q_init: Optional[Sequence[float]] = None,
        seeds: Optional[Sequence[Sequence[float]]] = None,
        **kw,
    ) -> Tuple[np.ndarray, bool, float]:
        """Önce q_init'ten, olmazsa sabit tohumlardan (yaw hedefe çevrilmiş) dener."""
        tgt = np.asarray(target_pos, dtype=np.float64)
        yaw = float(np.arctan2(tgt[1] - self.base_pos[1], tgt[0] - self.base_pos[0]))
        cands = []
        if q_init is not None:
            cands.append(np.asarray(q_init, dtype=np.float64)[:5])
        for s in seeds or DEFAULT_IK_SEEDS:
            s = np.array(s, dtype=np.float64)
            s[0] = np.clip(yaw, self.q_min[0], self.q_max[0])
            cands.append(s)
        best = (cands[0], False, np.inf)
        for c in cands:
            q, ok, err = self.ik(tgt, q_init=c, **kw)
            if ok:
                return q, ok, err
            if err < best[2]:
                best = (q, ok, err)
        return best

    def tilt_of(self, q: Sequence[float]) -> float:
        """Yaklaşma yönünün düşeyden açısı (radyan)."""
        _, R = self.fk(q)
        return float(np.arccos(np.clip(-self.approach_of(R)[2], -1.0, 1.0)))


# Başlangıç duruşları: joint1-3 eklem aralıklarını kaba bir ızgarayla kaplar (yerel minimumlara
# karşı çoklu başlangıç). İlk eleman tipik "dirsek yukarı" duruşudur.
DEFAULT_IK_SEEDS = tuple(
    [(0.0, 1.0, 0.5, -1.0, 0.0)]
    + [
        (0.0, j1, j2, j3, 0.0)
        for j1 in (0.4, 1.0, 1.6, 2.2)
        for j2 in (-0.6, 0.2, 1.0)
        for j3 in (-0.5, -1.4, -2.3)
    ]
)

DOWN = np.array([0.0, 0.0, -1.0])

_DEFAULT: Optional[MahutKinematics] = None


def get_kinematics() -> MahutKinematics:
    """Paylaşılan (önbellekli) kinematik nesnesi."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MahutKinematics()
    return _DEFAULT


# ---------------------------------------------------------------------------
# Kavrama noktası tabanlı IK (çalışma alanı hesabı, scripted demolar, teleop)
# ---------------------------------------------------------------------------

# Nesne merkezinin TCP'den parmak yönünde uzaklığı. Pedler TCP'nin 0.05 m ötesine uzanır;
# 0.035 m'de 35 mm nesnenin ~%80'i pedler arasında kalır ve eğik yaklaşmada parmak uçları
# masadan ~5-8 mm yukarıda kalır.
GRASP_DEPTH = 0.035
# Kavrama eğimleri (yaklaşma yönünün düşeyle açısı, derece). Tabana yakın noktalar 45°,
# uzak noktalar 55° ile erişilebilir; 50° ortada en geniş alanı kapsar (scripts/compute_workspace.py).
GRASP_TILTS_DEG = (45, 50, 55)
TABLE_CLEARANCE = 0.002


_FINGER_HULLS: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


def _finger_hulls(urdf_path: str = _DEFAULT_URDF) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parmak çarpışma mesh'lerinin dışbükey zarf köşeleri link4 (= TCP yönelimi) çerçevesinde,
    parmak eklemi 0 iken; ve parmağın kayma ekseni. SAPIEN de çarpışmada dışbükey zarf kullanır.
    """
    global _FINGER_HULLS
    if _FINGER_HULLS is not None:
        return _FINGER_HULLS
    import trimesh

    root = ET.parse(urdf_path).getroot()
    joints = {j.name: j for j in _parse_urdf(urdf_path)}
    out = []
    for jname in ("finger_left", "finger_right"):
        j = joints[jname]
        link = root.find(f"link[@name='{j.child}']")
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            o = col.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            path = os.path.normpath(os.path.join(os.path.dirname(urdf_path), mesh.get("filename")))
            v = trimesh.load(path, force="mesh").convex_hull.vertices * scale
            v_link = v @ _rpy_to_mat(rpy).T + xyz
            v_l4 = v_link @ j.origin_R.T + j.origin_p
            out.append((v_l4, j.origin_R @ (j.axis / np.linalg.norm(j.axis))))
    _FINGER_HULLS = out
    return out


def finger_points_world(K: MahutKinematics, q: Sequence[float], finger_q: float = 0.0) -> np.ndarray:
    """Parmak zarf köşeleri (dünya çerçevesi). link4 ile TCP aynı yönelimde, TCP ofseti çıkarılır."""
    p_tcp, R = K.fk(q)
    tcp_off = K.chain[-1].origin_p  # link4 -> TCP öteleme (link4 çerçevesinde)
    pts = []
    for v_l4, axis in _finger_hulls():
        pts.append((v_l4 + finger_q * axis - tcp_off) @ R.T + p_tcp)
    return np.vstack(pts)


def finger_min_z(K: MahutKinematics, q: Sequence[float], finger_q: float = 0.0) -> float:
    """Parmakların dünya çerçevesindeki en alçak noktası (masa üstü z = 0)."""
    return float(finger_points_world(K, q, finger_q)[:, 2].min())


def solve_grasp_ik(
    grasp_point: Sequence[float],
    tilt: float,
    q_init: Optional[Sequence[float]] = None,
    roll: Optional[float] = None,
    depth: float = GRASP_DEPTH,
    K: Optional[MahutKinematics] = None,
    seeds: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Kavrama noktası G (nesne merkezi) için eklem açıları: TCP = G - depth * yaklaşma.
    Yaklaşmanın yatay yönü kol düzlemine bağlı olduğundan sabit nokta yinelemesiyle çözülür.
    Başarı: IK yakınsadı ve parmaklar masaya girmiyor.
    """
    K = K or get_kinematics()
    G = np.asarray(grasp_point, dtype=np.float64)
    tcp = G.copy()
    q = None if q_init is None else np.asarray(q_init, dtype=np.float64)
    ok = False
    for _ in range(5):
        q, ok, _ = K.ik_multi_start(tcp, tilt=tilt, roll=roll, q_init=q, seeds=seeds)
        _, R = K.fk(q)
        new_tcp = G - depth * K.approach_of(R)
        if np.linalg.norm(new_tcp - tcp) < 1e-4:
            break
        tcp = new_tcp
    q, ok, _ = K.ik_multi_start(tcp, tilt=tilt, roll=roll, q_init=q, seeds=seeds)
    ok = ok and finger_min_z(K, q) > TABLE_CLEARANCE
    return q, ok
