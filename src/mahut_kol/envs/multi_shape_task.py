"""
MultiShapeRelationalTask: MahutKol (5-DoF + paralel kıskaç) için çok şekilli ilişkisel
yerleştirme görevi (ManiSkill 3 / SAPIEN).

Özellikler:
- Nesne havuzu: 9 nesne (3 şekil x 3 renk) bir kez yüklenir; aktif olmayanlar z = -10 m'ye
  park edilir ve çarpışmaları kapatılır (PhysX zemin fırlatmasını önlemek için).
- Her bölümde: 1 manipüle nesne, 1 referans nesne, `num_distractors` adet distractor
  (kayıtta 2 -> toplam 4 aktif nesne).
- Çakışmasız yerleşim (rejection sampling, merkezler arası >= MIN_PAIRWISE_DISTANCE).
  Hedef noktası her zaman çalışma alanı içinde ve boş tutulur.
- Yön tanımları (dünya ekseni; robot tabanı orijinde, +x'e bakar; sabit "front" kamera
  operatör tarafında, x > 0'dan -x yönüne bakar, böylece görüntünün sağı = +y):
    FRONT: (x - Δ, y, z)   referansın robota bakan tarafı
    BACK:  (x + Δ, y, z)   referansın operatöre/kameraya bakan tarafı
    LEFT:  (x, y - Δ, z)   görüntüde referansın solu
    RIGHT: (x, y + Δ, z)   görüntüde referansın sağı
    TOP:   (x, y, z + h)   referansın üstü (referans üçgen prizma ise geçersiz)
  Δ = 0.06 m, h = 0.035 m.
- Başarı: nesne hedefte (xy toleransı + z toleransı + dik) VE bırakılmış (kıskaç açık, temas
  yok) VE ardışık SETTLE_STEPS kontrol adımı boyunca durağan.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor

from mahut_kol.envs.shape_objects import (
    DEFAULT_PARKING_Z,
    OBJECT_COLORS,
    OBJECT_SHAPES,
    build_manipulation_objects,
    disable_actor_collision,
    enable_actor_collision,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

RELATIONS = ("FRONT", "BACK", "LEFT", "RIGHT", "TOP")
OBJECT_IDS = tuple(f"{c}_{s}" for c in OBJECT_COLORS for s in OBJECT_SHAPES)

# Hedef mesafesi. 0.045 m'de yan yana iki 35 mm nesne arasında ~1 cm boşluk kalıyordu;
# kıskaç parmakları (her biri 16 mm kalın) LEFT/RIGHT yerleştirmede referansa sürtüyordu.
DEFAULT_RELATIONAL_DELTA = 0.06   # metre
DEFAULT_OBJECT_HEIGHT = 0.035     # metre (35 mm)
TABLETOP_HALF_HEIGHT = 0.0175     # metre (masa üstündeki nesne merkezinin z'si)

# Başarı toleransları (gerekçe):
# - Komşu iki yön hedefi (ör. RIGHT ve FRONT) arası mesafe Δ·√2 = 0.085 m. Başarı daireleri
#   örtüşmesin diye yarıçap < 0.042 m olmalı.
# - Hedefe r kadar uzaktaki nesne referans merkezine en az Δ - r uzaktadır. İki 35 mm nesne
#   iç içe geçmesin diye Δ - r >= 0.035 -> r <= 0.025 m.
# - r = 0.025 m'de bir nesne yalnızca kendi yönünün ±asin(0.025/0.06) ≈ ±25° dilimindeyse
#   başarılı sayılır; iki yönün ortasında (45°) duran nesne hiçbir yön için başarılı değildir.
SUCCESS_XY_TOLERANCE = 0.025      # metre
SUCCESS_Z_TOLERANCE = 0.012       # metre (TOP'ta referansın üstünde, diğerlerinde masada)
UPRIGHT_COS_THRESHOLD = float(np.cos(np.radians(20.0)))  # dik durma eşiği

# "Bırakıldı" tanımı: parmaklar 35 mm nesneye temas noktasından (~0.0245) daha açık ve temas yok.
# Parmak eklemi: 0.0 = tam açık, 0.042 = tam kapalı.
GRIPPER_RELEASED_QPOS = 0.022
# "Durağan" tanımı: nesne hızları eşik altında, ardışık SETTLE_STEPS kontrol adımı (20 Hz).
STATIC_LIN_VEL = 0.02             # m/s
STATIC_ANG_VEL = 0.5              # rad/s
SETTLE_STEPS = 3

# Nesne merkezleri arası en küçük mesafe. 35 mm iki küp en kötü yönelimde 0.0495 m'de temas
# eder; açık kıskaç parmakları nesne merkezinden ~0.058 m'ye uzanır. 0.065 m hem çakışmayı
# önler hem kavrama sırasında komşu nesneye sürtünme payı bırakır. Hedef noktası da (TOP hariç)
# aynı mesafeyle boş tutulur. Δ = 0.06 < 0.065 olduğu için referansın komşusu yalnızca hedef
# olabilir; distractor hedefle referans arasına giremez.
MIN_PAIRWISE_DISTANCE = 0.065     # metre

# MahutKol çalışma alanı (nesne merkezleri ve hedefler için, metre).
# FK/IK ile örneklenerek hesaplandı (scripts/compute_workspace.py): kıskaç düşeyden
# 45/50/55° eğik yaklaşırken nesne kavrama (z=0.022), referans üstüne bırakma (+0.038) ve
# taşıma (+0.065) yüksekliklerinin üçü de IK ile çözülebilen ve parmakların masaya girmediği
# hücrelerin içinden güvenli bir dikdörtgen seçildi. MahutKol tam aşağı bakarak yalnızca
# tabana çok yakın ve alçakta erişebildiği için eğik kavrama kullanılır. Kol düzlemi tabandan
# yanal ~-4.4 cm kaçık olduğundan alan y'de simetrik değildir. Eski SO-100 alanı:
# x 0.12-0.22, y -0.10..0.10.
WORKSPACE_X_MIN = 0.10
WORKSPACE_X_MAX = 0.22
WORKSPACE_Y_MIN = -0.16
WORKSPACE_Y_MAX = 0.12

# Bölüm başında kol (joint0..4, finger_left, finger_right): TCP ≈ (0.14, -0.02, 0.11), kıskaç
# düşeyden 55° eğik, parmaklar tam açık (0.0 = açık, 0.042 = kapalı). Kavrama duruşlarıyla aynı
# IK dalında (dirsek yukarı) olduğu için ilk harekette kol savrulmaz. IK ile hesaplanıp yuvarlandı.
ROBOT_REST_QPOS = [0.16, 0.86, 0.33, -1.52, 0.0, 0.0, 0.0]

# Kol düzlemi geometrisi (URDF'ten): joint0 ekseni dünyada (-0.0037, -0.0323)'te, kol düzlemi bu
# eksenden yanal -0.0114 m kaçık. Bir (x, y) noktasına uzanırken kolun yatay yönü arm_heading().
ARM_YAW_PIVOT_XY = (-0.00365706, -0.0322927)
ARM_PLANE_LATERAL_OFFSET = -0.0114

# Nesnelerin başlangıç yaw'ı: kavranabilir yönelim + gürültü. Küp 90°, prizma 120° simetrik;
# kolun o noktadaki yönüne hizalanır (parmak ekseni kol düzlemine diktir). Silindir serbest.
SPAWN_YAW_NOISE_DEG = 15.0
DEFAULT_NUM_DISTRACTORS = 2

# Bölüm uzunluğu (20 Hz'de 30 sn). gym.make(..., max_episode_steps=N) ile değiştirilebilir.
DEFAULT_MAX_EPISODE_STEPS = 600

# ---------------------------------------------------------------------------
# Kameralar (geçici; donanım netleşince güncellenecek) - tek yerden ayarlanır
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
# Sabit kamera: operatör tarafında (x > 0), robota (-x) bakar -> görüntü sağı = +y = RIGHT.
FRONT_CAMERA_NAME = "front"
FRONT_CAMERA_EYE = (0.46, -0.03, 0.36)
FRONT_CAMERA_TARGET = (0.15, -0.03, 0.02)
FRONT_CAMERA_FOV_DEG = 52.0
# Bilek kamerası: link_gripper_center'a (TCP) monte, kıskacın üst tarafında, parmak uçlarına bakar.
# Poz TCP çerçevesinde (x = parmak yönü, z = kıskacın üst tarafı). Kamera ekseni: +x ileri, +z yukarı.
WRIST_CAMERA_NAME = "wrist"
WRIST_CAMERA_POS = (-0.02, 0.0, 0.045)
WRIST_CAMERA_PITCH_DEG = 35.0     # +x'ten aşağı (TCP -z yönüne) eğim
WRIST_CAMERA_FOV_DEG = 80.0


# ---------------------------------------------------------------------------
# Komut geçerliliği
# ---------------------------------------------------------------------------


def object_shape(object_id: str) -> str:
    """'blue_triangular_prism' -> 'triangular_prism'."""
    return object_id.split("_", 1)[1]


def validate_command(manipulated_object: str, reference_object: str, relation: str) -> None:
    """Geçersiz komutta açık Türkçe mesajla ValueError fırlatır."""
    if manipulated_object not in OBJECT_IDS:
        raise ValueError(f"Bilinmeyen manipüle nesne '{manipulated_object}'. Geçerli: {OBJECT_IDS}")
    if reference_object not in OBJECT_IDS:
        raise ValueError(f"Bilinmeyen referans nesne '{reference_object}'. Geçerli: {OBJECT_IDS}")
    if relation not in RELATIONS:
        raise ValueError(f"Bilinmeyen yön '{relation}'. Geçerli: {RELATIONS}")
    if manipulated_object == reference_object:
        raise ValueError(
            f"Manipüle nesne ile referans nesne aynı olamaz ('{manipulated_object}')."
        )
    if relation == "TOP" and object_shape(reference_object) == "triangular_prism":
        raise ValueError(
            f"Üçgen prizmanın üstüne nesne konulamaz: referans '{reference_object}' bir üçgen "
            "prizma ve yön TOP. Prizmanın tepesi sivri olduğundan nesne üstünde duramaz; "
            "başka bir yön (FRONT/BACK/LEFT/RIGHT) ya da başka bir referans seçin."
        )


def is_valid_command(manipulated_object: str, reference_object: str, relation: str) -> bool:
    try:
        validate_command(manipulated_object, reference_object, relation)
        return True
    except ValueError:
        return False


def valid_command_combinations() -> List[Tuple[str, str, str]]:
    """Tüm geçerli (manipüle, referans, yön) üçlüleri: 9*8*5 - 24 = 336."""
    return [
        (m, r, rel)
        for m, r, rel in itertools.product(OBJECT_IDS, OBJECT_IDS, RELATIONS)
        if is_valid_command(m, r, rel)
    ]


# ---------------------------------------------------------------------------
# İlişkisel hedef hesaplayıcı
# ---------------------------------------------------------------------------

_REL_OFFSET_SIGN = {"FRONT": (0, -1.0), "BACK": (0, 1.0), "LEFT": (1, -1.0), "RIGHT": (1, 1.0)}


def compute_relational_target_position(
    ref_pos: Union[Sequence[float], np.ndarray, torch.Tensor],
    relation: str,
    delta: float = DEFAULT_RELATIONAL_DELTA,
    height: float = DEFAULT_OBJECT_HEIGHT,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Referans konumuna (x, y, z) göre hedef konumu. Tek vektör (3,) ya da yığın (B, 3).

    - FRONT: (x - delta, y, z)   - BACK: (x + delta, y, z)
    - LEFT:  (x, y - delta, z)   - RIGHT: (x, y + delta, z)
    - TOP:   (x, y, z + height)
    """
    relation = relation.upper()
    if relation not in RELATIONS:
        raise ValueError(f"Unknown relation '{relation}'. Must be one of {RELATIONS}")
    if isinstance(ref_pos, torch.Tensor):
        out = ref_pos.clone()
    else:
        out = np.array(ref_pos, dtype=np.float32).copy()
    if relation == "TOP":
        out[..., 2] += height
    else:
        axis, sign = _REL_OFFSET_SIGN[relation]
        out[..., axis] += sign * delta
    return out


def compute_all_relational_target_positions(
    ref_pos: Union[Sequence[float], np.ndarray, torch.Tensor],
    delta: float = DEFAULT_RELATIONAL_DELTA,
    height: float = DEFAULT_OBJECT_HEIGHT,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """5 yönün hepsi için hedefler."""
    return {
        rel: compute_relational_target_position(ref_pos, rel, delta=delta, height=height)
        for rel in RELATIONS
    }


def arm_heading(x: float, y: float) -> float:
    """Kolun (x, y) noktasına uzanırken joint0 yönü (radyan, dünya z etrafında)."""
    dx, dy = x - ARM_YAW_PIVOT_XY[0], y - ARM_YAW_PIVOT_XY[1]
    r = max(float(np.hypot(dx, dy)), abs(ARM_PLANE_LATERAL_OFFSET) + 1e-6)
    return float(np.arctan2(dy, dx) - np.arcsin(ARM_PLANE_LATERAL_OFFSET / r))


def _spawn_yaw_symmetry(shape: str) -> Optional[float]:
    return {"cube": np.pi / 2, "triangular_prism": 2 * np.pi / 3}.get(shape)


# ---------------------------------------------------------------------------
# Ortam
# ---------------------------------------------------------------------------


@register_env("MahutKol-MultiShapeRelational-v1", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
@register_env("MultiShapeRelationalTask-v0", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class MultiShapeRelationalTask(BaseEnv):
    """
    MahutKol için çok şekilli ilişkisel yerleştirme ortamı.

    reset(options=...) anahtarları:
      manipulated_object, reference_object, relation : komut (birlikte verilmeli ya da hiç)
      num_distractors (varsayılan 2), distractor_objects (isteğe bağlı açık liste),
      yaw_noise_deg (varsayılan SPAWN_YAW_NOISE_DEG),
      object_poses {nesne_id: (x, y, yaw)} (isteğe bağlı: sahneyi aynen kur; değerlendirme için)
    Komut verilmezse 336 geçerli kombinasyondan biri (bölüm RNG'si ile) seçilir.
    """

    SUPPORTED_ROBOTS = ["mahut_kol", "robot_arm"]
    SUPPORTED_REWARD_MODES = ["sparse", "none", "dense"]

    workspace_x_bounds: Tuple[float, float] = (WORKSPACE_X_MIN, WORKSPACE_X_MAX)
    workspace_y_bounds: Tuple[float, float] = (WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
    table_z: float = TABLETOP_HALF_HEIGHT
    min_pairwise_distance: float = MIN_PAIRWISE_DISTANCE
    relational_delta: float = DEFAULT_RELATIONAL_DELTA
    object_height: float = DEFAULT_OBJECT_HEIGHT

    def __init__(self, *args, robot_uids: str = "mahut_kol", **kwargs):
        self.manipulated_object_id: Optional[str] = None
        self.reference_object_id: Optional[str] = None
        self.distractor_object_ids: List[str] = []
        self.current_relation: str = "FRONT"
        self._static_steps: Optional[torch.Tensor] = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Kameralar
    # ------------------------------------------------------------------
    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        front_pose = sapien_utils.look_at(eye=list(FRONT_CAMERA_EYE), target=list(FRONT_CAMERA_TARGET))
        pitch = np.radians(WRIST_CAMERA_PITCH_DEG)
        # Kamera +x ekseni TCP +x'ten aşağı (TCP -z) doğru eğik: y ekseni etrafında +pitch
        wrist_pose = sapien.Pose(
            p=list(WRIST_CAMERA_POS), q=[np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        )
        return [
            CameraConfig(
                FRONT_CAMERA_NAME, front_pose, CAMERA_WIDTH, CAMERA_HEIGHT,
                np.deg2rad(FRONT_CAMERA_FOV_DEG), 0.01, 100,
            ),
            CameraConfig(
                WRIST_CAMERA_NAME, wrist_pose, CAMERA_WIDTH, CAMERA_HEIGHT,
                np.deg2rad(WRIST_CAMERA_FOV_DEG), 0.005, 100, mount=self.agent.tcp,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(eye=[0.55, -0.40, 0.45], target=[0.15, -0.02, 0.04])
        return CameraConfig("render_camera", pose, 512, 512, np.deg2rad(55), 0.01, 100)

    # ------------------------------------------------------------------
    # Sahne
    # ------------------------------------------------------------------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0558]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.manipulation_objects: Dict[str, Actor] = build_manipulation_objects(
            self.scene,
            initial_parking_z=DEFAULT_PARKING_Z,
        )

    # ------------------------------------------------------------------
    # Yerleşim örnekleme
    # ------------------------------------------------------------------
    def _reference_bounds(self, relation: str) -> Tuple[float, float, float, float]:
        """Hedef çalışma alanında kalacak şekilde referansın örneklenebileceği dikdörtgen."""
        x0, x1 = self.workspace_x_bounds
        y0, y1 = self.workspace_y_bounds
        d = self.relational_delta
        if relation == "FRONT":
            x0 += d
        elif relation == "BACK":
            x1 -= d
        elif relation == "LEFT":
            y0 += d
        elif relation == "RIGHT":
            y1 -= d
        if x0 > x1 or y0 > y1:
            raise ValueError(f"Çalışma alanı '{relation}' hedefi için çok küçük")
        return x0, x1, y0, y1

    def sample_layout(
        self,
        rng: np.random.RandomState,
        relation: str,
        num_others: int,
        max_layout_attempts: int = 200,
        max_object_attempts: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bir sahne için (referans_xy, hedef_xy, diğer_xy[num_others]) örnekler.
        diğer_xy[0] manipüle nesne, gerisi distractor. Kurallar:
        - Hedef çalışma alanı içinde (referans buna göre kısıtlı bölgeden seçilir).
        - Tüm nesneler birbirinden >= min_pairwise_distance.
        - Manipüle nesne ve distractor'lar hedef noktasından da >= min_pairwise_distance
          (hedef boş; manipüle nesne bölüm başında hedefte olmasın). Çakışma olursa
          distractor'lar yeniden örneklenir; olmazsa referans da yeniden seçilir.
        """
        x0, x1 = self.workspace_x_bounds
        y0, y1 = self.workspace_y_bounds
        rx0, rx1, ry0, ry1 = self._reference_bounds(relation)
        min_d = self.min_pairwise_distance
        for _ in range(max_layout_attempts):
            ref = np.array([rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)])
            tgt = compute_relational_target_position(
                np.r_[ref, 0.0], relation, delta=self.relational_delta
            )[:2]
            obstacles = [ref] if relation == "TOP" else [ref, tgt]
            others: List[np.ndarray] = []
            ok = True
            for _i in range(num_others):
                for _a in range(max_object_attempts):
                    c = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                    if all(np.linalg.norm(c - o) >= min_d for o in obstacles + others):
                        others.append(c)
                        break
                else:
                    ok = False
                    break
            if ok:
                return ref, tgt, np.array(others).reshape(num_others, 2)
        raise RuntimeError("Çakışmasız yerleşim bulunamadı; çalışma alanı/nesne sayısını kontrol edin.")

    def _sample_collision_free_positions(self, b: int, num_objects: int, max_attempts: int = 200) -> torch.Tensor:
        """Geriye dönük uyumluluk: num_objects adet çakışmasız (x, y, table_z) konum (b, n, 3)."""
        out = torch.zeros((b, num_objects, 3), device=self.device, dtype=torch.float32)
        out[:, :, 2] = self.table_z
        for env_i in range(b):
            rng = self._batched_episode_rng[env_i] if self._batched_episode_rng is not None else np.random
            ref, _, others = self.sample_layout(rng, "TOP", max(num_objects - 1, 0))
            pts = np.vstack([ref[None], others])[:num_objects]
            out[env_i, :, :2] = torch.as_tensor(pts, dtype=torch.float32)
        return out

    def _fixed_layout(self, fixed: Dict[str, Sequence[float]], active_keys: List[str]) -> np.ndarray:
        """
        options["object_poses"] = {nesne_id: (x, y, yaw)}: sahneyi aynen kurar (değerlendirmede
        aynı sahneyi farklı komutlarla çalıştırmak için). Kümede tam olarak aktif nesneler olmalı;
        hedef çalışma alanında ve boş olmalı, aksi halde ValueError.
        """
        if set(fixed) != set(active_keys):
            raise ValueError(f"object_poses nesneleri {sorted(fixed)} != aktif nesneler {sorted(active_keys)}")
        pts = np.array([[float(fixed[k][0]), float(fixed[k][1])] for k in active_keys])
        ref = pts[1]
        tgt = compute_relational_target_position(np.r_[ref, 0.0], self.current_relation, delta=self.relational_delta)
        x0, x1 = self.workspace_x_bounds
        y0, y1 = self.workspace_y_bounds
        if not (x0 - 1e-6 <= tgt[0] <= x1 + 1e-6 and y0 - 1e-6 <= tgt[1] <= y1 + 1e-6):
            raise ValueError(f"Bu sahnede {self.current_relation} hedefi çalışma alanı dışında: {tgt[:2]}")
        if self.current_relation != "TOP":
            for k, p in zip(active_keys, pts):
                if k != active_keys[1] and np.linalg.norm(p - tgt[:2]) < self.min_pairwise_distance - 1e-6:
                    raise ValueError(f"Bu sahnede hedef dolu ('{k}' hedefe çok yakın)")
        return pts

    # ------------------------------------------------------------------
    # Bölüm başlatma
    # ------------------------------------------------------------------
    def _choose_command(self, rng: np.random.RandomState, options: dict) -> Tuple[str, str, str]:
        has = [k in options for k in ("manipulated_object", "reference_object", "relation")]
        if all(has):
            cmd = (options["manipulated_object"], options["reference_object"], options["relation"].upper())
            validate_command(*cmd)
            return cmd
        # Kısmi komut: verilen alanlara uyan geçerli kombinasyonlardan seç
        combos = [
            c for c in valid_command_combinations()
            if ("manipulated_object" not in options or c[0] == options["manipulated_object"])
            and ("reference_object" not in options or c[1] == options["reference_object"])
            and ("relation" not in options or c[2] == options["relation"].upper())
        ]
        if not combos:
            m = options.get("manipulated_object", "?")
            r = options.get("reference_object", "?")
            rel = options.get("relation", "?")
            validate_command(
                m if m != "?" else "red_cube", r if r != "?" else "green_cube", rel if rel != "?" else "TOP"
            )
            raise ValueError(f"Geçerli komut bulunamadı: {m}, {r}, {rel}")
        return combos[rng.randint(len(combos))]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        options = options or {}
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            rng0 = self._batched_episode_rng[int(env_idx[0])]

            # 1. Komut (tüm paralel ortamlar aynı nesne kimliklerini kullanır)
            self.manipulated_object_id, self.reference_object_id, self.current_relation = (
                self._choose_command(rng0, options)
            )
            all_keys = list(self.manipulation_objects.keys())
            remaining = [k for k in all_keys if k not in (self.manipulated_object_id, self.reference_object_id)]
            if options.get("distractor_objects") is not None:
                distractors = list(options["distractor_objects"])
                bad = [d for d in distractors if d not in remaining]
                if bad:
                    raise ValueError(f"Geçersiz distractor(lar): {bad}")
            else:
                n = int(options.get("num_distractors", DEFAULT_NUM_DISTRACTORS))
                n = max(0, min(n, len(remaining)))
                distractors = [remaining[i] for i in rng0.choice(len(remaining), n, replace=False)]
            self.distractor_object_ids = distractors

            active_keys = [self.manipulated_object_id, self.reference_object_id] + distractors
            inactive_keys = [k for k in all_keys if k not in active_keys]

            # 2. Aktif olmayanları park et
            zero_vel = torch.zeros((b, 3), device=self.device)
            underground = Pose.create_from_pq(
                p=torch.tensor([0.0, 0.0, DEFAULT_PARKING_Z], device=self.device).repeat(b, 1)
            )
            for k in inactive_keys:
                disable_actor_collision(self.manipulation_objects[k])
                self.manipulation_objects[k].set_pose(underground)
                self.manipulation_objects[k].set_linear_velocity(zero_vel)
                self.manipulation_objects[k].set_angular_velocity(zero_vel)

            # 3. Aktifleri yerleştir: [manipüle, referans, distractor...]
            yaw_noise = np.radians(float(options.get("yaw_noise_deg", SPAWN_YAW_NOISE_DEG)))
            xyz = torch.zeros((b, len(active_keys), 3))
            xyz[..., 2] = self.table_z
            yaws = torch.zeros((b, len(active_keys)))
            fixed = options.get("object_poses")
            for i, env_i in enumerate(env_idx.tolist()):
                rng = self._batched_episode_rng[env_i]
                if fixed is not None:
                    pts = self._fixed_layout(fixed, active_keys)
                    xyz[i, :, :2] = torch.as_tensor(pts, dtype=torch.float32)
                    for j, key in enumerate(active_keys):
                        yaws[i, j] = float(fixed[key][2])
                    continue
                ref, _tgt, others = self.sample_layout(rng, self.current_relation, len(active_keys) - 1)
                pts = np.vstack([others[:1], ref[None], others[1:]])
                xyz[i, :, :2] = torch.as_tensor(pts, dtype=torch.float32)
                for j, key in enumerate(active_keys):
                    sym = _spawn_yaw_symmetry(object_shape(key))
                    if sym is None:
                        yaw = rng.uniform(0, 2 * np.pi)
                    else:
                        bearing = arm_heading(pts[j, 0], pts[j, 1])
                        # prizma: taban kenarı radyale dik (yerel +x radyale dik -> +90°)
                        base = bearing + (np.pi / 2 if sym > np.pi / 2 else 0.0)
                        yaw = base + rng.randint(4) * sym + rng.uniform(-yaw_noise, yaw_noise)
                    yaws[i, j] = float(yaw)

            for j, key in enumerate(active_keys):
                enable_actor_collision(self.manipulation_objects[key])
                yaw = yaws[:, j]
                quat = torch.stack(
                    [torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], dim=1
                )
                self.manipulation_objects[key].set_pose(Pose.create_from_pq(p=xyz[:, j], q=quat))
                self.manipulation_objects[key].set_linear_velocity(zero_vel)
                self.manipulation_objects[key].set_angular_velocity(zero_vel)

            # 4. Robot
            self.agent.reset(torch.tensor(ROBOT_REST_QPOS, dtype=torch.float32).repeat(b, 1))
            if self._static_steps is None or self._static_steps.shape[0] != self.num_envs:
                self._static_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self._static_steps[env_idx] = 0

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def get_manipulated_object(self) -> Optional[Actor]:
        if self.manipulated_object_id:
            return self.manipulation_objects[self.manipulated_object_id]
        return None

    def get_reference_object(self) -> Optional[Actor]:
        if self.reference_object_id:
            return self.manipulation_objects[self.reference_object_id]
        return None

    def get_distractor_objects(self) -> List[Actor]:
        return [self.manipulation_objects[k] for k in self.distractor_object_ids]

    def get_active_object_poses(self) -> Dict[str, Tuple[float, float, float]]:
        """Aktif nesnelerin {id: (x, y, yaw)} pozları (ilk paralel ortam). object_poses ile geri kurulur."""
        out = {}
        for k in [self.manipulated_object_id, self.reference_object_id] + self.distractor_object_ids:
            p = self.manipulation_objects[k].pose.p[0].cpu().numpy()
            w, x, y, z = self.manipulation_objects[k].pose.q[0].cpu().numpy()
            out[k] = (float(p[0]), float(p[1]), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
        return out

    def get_command(self) -> Dict[str, str]:
        """Geçerli bölümün komutu (CommandParser JSON alanlarıyla aynı adlar)."""
        return {
            "manipulated_object": self.manipulated_object_id,
            "reference_object": self.reference_object_id,
            "relation": self.current_relation,
        }

    def get_relational_target_position(self, relation: Optional[str] = None) -> torch.Tensor:
        """Referansa göre hedef konum, (b, 3)."""
        ref_obj = self.get_reference_object()
        if ref_obj is None:
            raise RuntimeError("Reference object not initialized.")
        rel = relation if relation is not None else self.current_relation
        return compute_relational_target_position(
            ref_obj.pose.p, rel, delta=self.relational_delta, height=self.object_height
        )

    def get_all_relational_target_positions(self) -> Dict[str, torch.Tensor]:
        ref_obj = self.get_reference_object()
        if ref_obj is None:
            raise RuntimeError("Reference object not initialized.")
        return compute_all_relational_target_positions(
            ref_obj.pose.p, delta=self.relational_delta, height=self.object_height
        )

    def _is_upright(self, actor: Actor, shape: str) -> torch.Tensor:
        q = actor.pose.q  # wxyz
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # Yerel eksenlerin dünya z bileşenleri (dönüş matrisinin 3. satırı)
        zx = 2 * (x * z - w * y)
        zy = 2 * (y * z + w * x)
        zz = 1 - 2 * (x * x + y * y)
        if shape == "cube":
            m = torch.max(torch.stack([zx.abs(), zy.abs(), zz.abs()], dim=1), dim=1).values
            return m > UPRIGHT_COS_THRESHOLD
        return zz > UPRIGHT_COS_THRESHOLD

    def _after_control_step(self):
        super()._after_control_step()
        obj = self.get_manipulated_object()
        if obj is None or self._static_steps is None:
            return
        static = obj.is_static(lin_thresh=STATIC_LIN_VEL, ang_thresh=STATIC_ANG_VEL)
        self._static_steps = torch.where(
            static, self._static_steps + 1, torch.zeros_like(self._static_steps)
        )

    # ------------------------------------------------------------------
    # Değerlendirme ve gözlem
    # ------------------------------------------------------------------
    def evaluate(self) -> Dict[str, torch.Tensor]:
        manip = self.get_manipulated_object()
        if manip is None:
            return {"success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}
        target = self.get_relational_target_position()
        p = manip.pose.p
        dist = torch.linalg.norm(p - target, dim=-1)
        xy_dist = torch.linalg.norm(p[:, :2] - target[:, :2], dim=-1)
        dz = (p[:, 2] - target[:, 2]).abs()
        upright = self._is_upright(manip, object_shape(self.manipulated_object_id))
        is_placed = (xy_dist < SUCCESS_XY_TOLERANCE) & (dz < SUCCESS_Z_TOLERANCE) & upright
        finger = self.agent.robot.get_qpos()[:, 5:7].mean(dim=1)
        is_grasped = self.agent.is_grasping(manip)
        is_released = (finger < GRIPPER_RELEASED_QPOS) & ~is_grasped
        static_steps = (
            self._static_steps
            if self._static_steps is not None
            else torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        )
        is_static = static_steps >= SETTLE_STEPS
        return {
            "success": is_placed & is_released & is_static,
            "is_placed": is_placed,
            "is_released": is_released,
            "is_grasped": is_grasped,
            "is_static": is_static,
            "distance_to_target": dist,
            "xy_distance_to_target": xy_dist,
        }

    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        obs = {}
        if getattr(self.agent, "tcp", None) is not None:
            obs["tcp_pose"] = self.agent.tcp.pose.raw_pose
        manip_obj = self.get_manipulated_object()
        ref_obj = self.get_reference_object()
        if manip_obj is not None:
            obs["manipulated_pose"] = manip_obj.pose.raw_pose
        if ref_obj is not None:
            obs["reference_pose"] = ref_obj.pose.raw_pose
            obs["target_position"] = self.get_relational_target_position()
        return obs
