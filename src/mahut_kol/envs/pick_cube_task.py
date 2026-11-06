"""
PickCubeTask: Yeni MahutKol modeli için Küp Alma ve Kaldırma Görevi (Pick & Place).

Özellikler:
- FK / IK (İleri ve Ters Kinematik) çözücü
- Otonom gösterim üretici (scripted demo policy)
- Kamera yapılandırması (base_camera, render_camera)
- ManiSkill ve LeRobot veri toplama uyumluluğu
"""

import os
from typing import Dict, List, Optional
import mujoco
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from mahut_kol.robots.mahut_robot import MahutRobot

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))
_ROBOT_XML_PATH = os.path.join(_ASSETS_DIR, "mahut_kol_robot.xml")

mj = mujoco.MjModel.from_xml_path(_ROBOT_XML_PATH)

HALF = 0.02  # Küp yarım boyutu (4 cm küp)
REST = [0, 0.4, 0, 0, 0, 0.02, 0.02]  # joint0-4, finger_left, finger_right
PITCH = np.radians(-45)
APPROACH = np.array([np.cos(PITCH), 0, np.sin(PITCH)])
CUBE_R, CUBE_DR = 0.155, 0.025
CUBE_ARC = np.radians(45)
CUBE_YAW = np.radians(10)
GRASP_Z = 0.045

# Link4 koordinat sisteminde kavrama noktası ve yönü
_d = mujoco.MjData(mj)
mujoco.mj_forward(mj, _d)
_L4 = mj.body("link4").id
_clamps = [
    g for g in range(mj.ngeom)
    if mj.geom_contype[g] and mj.geom_bodyid[g] in (mj.body("finger_left").id, mj.body("finger_right").id)
]
_grasp_world = np.mean([_d.geom_xpos[g] for g in _clamps], axis=0) if _clamps else np.zeros(3)
GRASP_LOCAL = _d.xmat[_L4].reshape(3, 3).T @ (_grasp_world - _d.xpos[_L4])
APPROACH_LOCAL = mj.joint("joint4").axis.copy()
YAW_XY = _d.xanchor[mj.joint("joint0").id][:2].copy()
SIDE = _grasp_world[1] - YAW_XY[1]


def fk(q):
    """joint1-3 = q, joint0 = joint4 = 0 için kavrama noktası ve yaklaşma yönü."""
    _d.qpos[:] = mj.qpos0
    for i, v in enumerate(q):
        _d.qpos[mj.joint(f"joint{i + 1}").qposadr[0]] = v
    mujoco.mj_kinematics(mj, _d)
    R = _d.xmat[_L4].reshape(3, 3)
    return _d.xpos[_L4] + R @ GRASP_LOCAL, R @ APPROACH_LOCAL


def ik_plane(x, z):
    """joint0 = 0 düzleminde kavrama noktasını (x, z)'ye getiren joint1-3 açıları."""
    lo = np.array([mj.joint(f"joint{i}").range[0] for i in (1, 2, 3)])
    hi = np.array([mj.joint(f"joint{i}").range[1] for i in (1, 2, 3)])

    def res(q):
        p, a = fk(q)
        return np.array([p[0] - x, p[2] - z, (a[0] * APPROACH[2] - a[2] * APPROACH[0]) * 0.1])

    grid = np.stack(np.meshgrid(*[np.linspace(l, h, 25) for l, h in zip(lo, hi)]), -1).reshape(-1, 3)
    grid = sorted(grid, key=lambda g: np.linalg.norm(res(g)) if fk(g)[1] @ APPROACH > 0 else np.inf)
    for q in grid[:20]:
        for _ in range(100):
            r = res(q)
            if np.linalg.norm(r) < 1e-5:
                break
            J = np.stack([(res(q + e) - r) / 1e-6 for e in np.eye(3) * 1e-6], axis=1)
            q = np.clip(q - J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), r), lo, hi)
        if np.linalg.norm(res(q)) < 1e-3 and fk(q)[1] @ APPROACH > 0.99:
            return q
    raise ValueError(f"unreachable: x={x:.3f} z={z:.3f}")


def ik(x, y, z):
    """Dünya (x, y, z) kavrama noktası için joint0-3 açıları."""
    dx, dy = x - YAW_XY[0], y - YAW_XY[1]
    reach = np.sqrt(dx ** 2 + dy ** 2 - SIDE ** 2)
    yaw = np.arctan2(dy, dx) - np.arctan2(SIDE, reach)
    return np.r_[yaw, ik_plane(YAW_XY[0] + reach, z)]


def scripted_actions(cube_xyz):
    """Küpü kavrayıp kaldıran adım adım pd_joint_pos eylemleri üretir."""
    q_grasp = ik(cube_xyz[0], cube_xyz[1], GRASP_Z)
    for swing in np.arange(0.4, q_grasp[1] + 1e-9, 0.05):
        if fk(q_grasp[1:] - [swing, 0, 0])[0][2] > GRASP_Z + 0.08:
            break
    q_up = q_grasp - [0, swing, 0, 0]
    # plan: (eklem hedefleri, kıskaç konumu, adım sayısı)
    plan = [(q_up, 0.0, 40), (q_grasp, 0.0, 30), (q_grasp, 0.042, 20), (q_up, 0.042, 40)]
    target = np.array(REST, float)
    for q, finger, steps in plan:
        start, end = target, np.r_[q, 0, finger, finger]
        for k in range(1, steps + 1):
            target = start + (end - start) * k / steps
            yield target.astype(np.float32)


@register_env("MahutKol-PickCube-v1", max_episode_steps=200)
@register_env("PickCubeArm-v1", max_episode_steps=200)
class MahutKolPickCubeEnv(BaseEnv):
    """
    MahutKol Küp Kaldırma Görev Ortamı.
    """

    SUPPORTED_ROBOTS = ["mahut_kol", "robot_arm"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    def __init__(self, *args, robot_uids: str = "mahut_kol", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.35, -0.30, 0.35], [0.12, 0, 0.03])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.40, -0.40, 0.40], [0.10, 0, 0.08])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0558]))

    def _load_scene(self, options):
        build_ground(self.scene)
        self.cube = actors.build_cube(
            self.scene,
            half_size=HALF,
            color=[0.85, 0.2, 0.2, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[YAW_XY[0] + CUBE_R, YAW_XY[1], HALF]),
        )
        # Gerçekçi kavrama için yüksek statik ve dinamik sürtünme (silikon/kauçuk kavrama)
        mat = sapien.physx.PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        for obj in self.cube._objs:
            for shape in obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).collision_shapes:
                shape.set_physical_material(mat)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            r = CUBE_R + (torch.rand(b) * 2 - 1) * CUBE_DR
            bearing = (torch.rand(b) * 2 - 1) * CUBE_ARC
            xyz = torch.stack(
                [
                    YAW_XY[0] + r * torch.cos(bearing),
                    YAW_XY[1] + r * torch.sin(bearing),
                    torch.full((b,), HALF),
                ],
                dim=1,
            )
            yaw = bearing + (torch.rand(b) * 2 - 1) * CUBE_YAW
            quat = torch.stack(
                [torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)],
                dim=1,
            )
            self.cube.set_pose(Pose.create_from_pq(xyz, quat))
            self.agent.reset(torch.tensor(REST, dtype=torch.float32).repeat(b, 1))

    def evaluate(self):
        return dict(success=self.cube.pose.p[:, 2] > HALF + 0.05)

    def _get_obs_extra(self, info):
        if "state" in self.obs_mode:
            return dict(cube_pose=self.cube.pose.raw_pose)
        return dict()
