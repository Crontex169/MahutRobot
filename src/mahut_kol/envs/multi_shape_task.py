"""
MultiShapeRelationalTask: ManiSkill 3 / SAPIEN Environment for SO-100 Robot.

Features:
- Object Pool Pattern: 9 manipulation objects (3 shapes x 3 colors) loaded once.
- Active object selection: 1 Manipulated Object, 1 Reference Object (Target Base),
  and 1-2 optional Distractors.
- Inactive objects parked underground (z = -10.0 m) with isolated physics.
- Collision-free rejection sampling on tabletop workspace (center-to-center >= 0.05 m).
- Spatial Relational Target Pose Calculator:
    FRONT: (x - Delta, y, z)
    BACK:  (x + Delta, y, z)
    LEFT:  (x, y - Delta, z)
    RIGHT: (x, y + Delta, z)
    TOP:   (x, y, z + h)
    where Delta = 0.045 m and h = 0.035 m.
"""

from __future__ import annotations

import math
import random
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
# Constants & Defaults
# ---------------------------------------------------------------------------

RELATIONS = ("FRONT", "BACK", "LEFT", "RIGHT", "TOP")
DEFAULT_RELATIONAL_DELTA = 0.045   # meters
DEFAULT_OBJECT_HEIGHT = 0.035      # meters (35 mm)
TABLETOP_HALF_HEIGHT = 0.0175      # meters (35 mm / 2)
MIN_PAIRWISE_DISTANCE = 0.05       # meters (center-to-center collision-free)

# SO-100 Tabletop Workspace bounds (meters)
WORKSPACE_X_MIN = 0.12
WORKSPACE_X_MAX = 0.22
WORKSPACE_Y_MIN = -0.10
WORKSPACE_Y_MAX = 0.10

# Default ready posture for SO-100
ROBOT_REST_QPOS = [0.0, 0.4, 0.0, 0.0, 0.0, 0.02, 0.02]


# ---------------------------------------------------------------------------
# Spatial Relational Target Pose Calculator
# ---------------------------------------------------------------------------


def compute_relational_target_position(
    ref_pos: Union[Sequence[float], np.ndarray, torch.Tensor],
    relation: str,
    delta: float = DEFAULT_RELATIONAL_DELTA,
    height: float = DEFAULT_OBJECT_HEIGHT,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute relative target position given reference object position (x, y, z).

    Offsets:
    - FRONT: (x - delta, y, z)
    - BACK:  (x + delta, y, z)
    - LEFT:  (x, y - delta, z)
    - RIGHT: (x, y + delta, z)
    - TOP:   (x, y, z + height)
    """
    relation = relation.upper()
    if relation not in RELATIONS:
        raise ValueError(f"Unknown relation '{relation}'. Must be one of {RELATIONS}")

    if isinstance(ref_pos, torch.Tensor):
        out = ref_pos.clone()
        if out.ndim == 1:
            if relation == "FRONT":
                out[0] -= delta
            elif relation == "BACK":
                out[0] += delta
            elif relation == "LEFT":
                out[1] -= delta
            elif relation == "RIGHT":
                out[1] += delta
            elif relation == "TOP":
                out[2] += height
        elif out.ndim == 2:
            if relation == "FRONT":
                out[:, 0] -= delta
            elif relation == "BACK":
                out[:, 0] += delta
            elif relation == "LEFT":
                out[:, 1] -= delta
            elif relation == "RIGHT":
                out[:, 1] += delta
            elif relation == "TOP":
                out[:, 2] += height
        return out

    arr = np.array(ref_pos, dtype=np.float32).copy()
    if arr.ndim == 1:
        if relation == "FRONT":
            arr[0] -= delta
        elif relation == "BACK":
            arr[0] += delta
        elif relation == "LEFT":
            arr[1] -= delta
        elif relation == "RIGHT":
            arr[1] += delta
        elif relation == "TOP":
            arr[2] += height
    elif arr.ndim == 2:
        if relation == "FRONT":
            arr[:, 0] -= delta
        elif relation == "BACK":
            arr[:, 0] += delta
        elif relation == "LEFT":
            arr[:, 1] -= delta
        elif relation == "RIGHT":
            arr[:, 1] += delta
        elif relation == "TOP":
            arr[:, 2] += height
    return arr


def compute_all_relational_target_positions(
    ref_pos: Union[Sequence[float], np.ndarray, torch.Tensor],
    delta: float = DEFAULT_RELATIONAL_DELTA,
    height: float = DEFAULT_OBJECT_HEIGHT,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Compute all 5 spatial target positions for FRONT, BACK, LEFT, RIGHT, TOP."""
    return {
        rel: compute_relational_target_position(ref_pos, rel, delta=delta, height=height)
        for rel in RELATIONS
    }


# ---------------------------------------------------------------------------
# MultiShapeRelationalTask Environment
# ---------------------------------------------------------------------------


@register_env("MahutKol-MultiShapeRelational-v1", max_episode_steps=200)
@register_env("MultiShapeRelationalTask-v0", max_episode_steps=200)
class MultiShapeRelationalTask(BaseEnv):
    """
    Multi-Shape Relational Manipulation Environment for SO-100 Robot.

    In this task:
    - 9 shape-color objects (3 shapes x 3 colors) are managed via an object pool.
    - Each episode chooses:
        1. Manipulated Object (e.g. 'red_cube')
        2. Reference Target Object (e.g. 'green_cylinder')
        3. Optional 1-2 Distractor Objects (e.g. 'blue_triangular_prism')
    - Active objects are spawned collision-free on the tabletop workspace.
    - Inactive objects are placed underground at z = -10.0 m.
    - Relational targets (FRONT, BACK, LEFT, RIGHT, TOP) are computed dynamically.
    """

    SUPPORTED_ROBOTS = ["mahut_kol", "robot_arm"]
    SUPPORTED_REWARD_MODES = ["sparse", "none", "dense"]

    # Tabletop workspace bounds
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
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        """Camera views for agent observation."""
        pose = sapien_utils.look_at(eye=[0.42, -0.35, 0.38], target=[0.16, 0.0, 0.04])
        return [CameraConfig("base_camera", pose, 128, 128, np.deg2rad(55), 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        """High-resolution camera for GUI rendering and diagnostic recording."""
        pose = sapien_utils.look_at(eye=[0.48, -0.42, 0.42], target=[0.15, 0.0, 0.04])
        return CameraConfig("render_camera", pose, 512, 512, np.deg2rad(55), 0.01, 100)

    def _load_agent(self, options: dict):
        """Mount robot onto table reference frame."""
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0558]))

    def _load_scene(self, options: dict):
        """Load table scene and construct 9-object manipulation pool."""
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Build all 9 shape-color objects (pre-parked at z = -10.0)
        self.manipulation_objects: Dict[str, Actor] = build_manipulation_objects(
            self.scene,
            initial_parking_z=DEFAULT_PARKING_Z,
        )

    def _sample_collision_free_positions(
        self,
        b: int,
        num_objects: int,
        max_attempts: int = 200,
    ) -> torch.Tensor:
        """
        Sample non-overlapping 2D (x, y) coordinates for num_objects across b batch items.
        Returns: tensor of shape (b, num_objects, 3) with z = self.table_z.
        """
        x_min, x_max = self.workspace_x_bounds
        y_min, y_max = self.workspace_y_bounds
        min_dist_sq = self.min_pairwise_distance ** 2

        out = torch.zeros((b, num_objects, 3), device=self.device, dtype=torch.float32)
        out[:, :, 2] = self.table_z

        for env_i in range(b):
            placed: List[Tuple[float, float]] = []
            for obj_i in range(num_objects):
                success = False
                for _ in range(max_attempts):
                    cx = random.uniform(x_min, x_max)
                    cy = random.uniform(y_min, y_max)
                    # Check distance against all previously placed objects in this scene
                    collision = False
                    for px, py in placed:
                        if (cx - px) ** 2 + (cy - py) ** 2 < min_dist_sq:
                            collision = True
                            break
                    if not collision:
                        placed.append((cx, cy))
                        out[env_i, obj_i, 0] = cx
                        out[env_i, obj_i, 1] = cy
                        success = True
                        break

                if not success:
                    # Fallback grid placement if random rejection sampling exhausts attempts
                    row = obj_i // 3
                    col = obj_i % 3
                    cx = x_min + col * (x_max - x_min) / 2.0
                    cy = y_min + row * (y_max - y_min) / 2.0
                    out[env_i, obj_i, 0] = cx
                    out[env_i, obj_i, 1] = cy
                    placed.append((cx, cy))

        return out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        Initialize episode:
        - Select manipulated, reference, and distractor objects.
        - Park inactive objects underground at z = -10.0.
        - Place active objects collision-free on tabletop.
        - Reset robot to rest posture.
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            all_keys = list(self.manipulation_objects.keys())

            # 1. Determine active object IDs from options or random choice
            if options and "manipulated_object" in options:
                self.manipulated_object_id = options["manipulated_object"]
            else:
                self.manipulated_object_id = random.choice(all_keys)

            remaining = [k for k in all_keys if k != self.manipulated_object_id]
            if options and "reference_object" in options:
                self.reference_object_id = options["reference_object"]
            else:
                self.reference_object_id = random.choice(remaining)

            remaining2 = [k for k in remaining if k != self.reference_object_id]
            num_distractors = options.get("num_distractors", 1) if options else 1
            num_distractors = max(0, min(num_distractors, len(remaining2)))
            self.distractor_object_ids = random.sample(remaining2, num_distractors)

            active_keys = [self.manipulated_object_id, self.reference_object_id] + self.distractor_object_ids
            inactive_keys = [k for k in all_keys if k not in active_keys]

            # 2. Park inactive objects underground (z = -10.0) and disable collisions
            underground_pose = Pose.create_from_pq(
                p=torch.tensor([0.0, 0.0, DEFAULT_PARKING_Z], device=self.device, dtype=torch.float32).repeat(b, 1),
            )
            for inact_key in inactive_keys:
                disable_actor_collision(self.manipulation_objects[inact_key])
                self.manipulation_objects[inact_key].set_pose(underground_pose)
                if hasattr(self.manipulation_objects[inact_key], "set_linear_velocity"):
                    zero_vel = torch.zeros((b, 3), device=self.device)
                    self.manipulation_objects[inact_key].set_linear_velocity(zero_vel)
                    self.manipulation_objects[inact_key].set_angular_velocity(zero_vel)

            # 3. Position active objects on table without collisions and enable collisions
            num_active = len(active_keys)
            sampled_positions = self._sample_collision_free_positions(b, num_active)

            for i, act_key in enumerate(active_keys):
                enable_actor_collision(self.manipulation_objects[act_key])
                xyz = sampled_positions[:, i, :]
                # Random yaw orientation around z-axis
                yaw = torch.rand(b, device=self.device) * (2 * np.pi)
                quat = torch.stack(
                    [
                        torch.cos(yaw / 2),
                        torch.zeros(b, device=self.device),
                        torch.zeros(b, device=self.device),
                        torch.sin(yaw / 2),
                    ],
                    dim=1,
                )
                pose = Pose.create_from_pq(p=xyz, q=quat)
                self.manipulation_objects[act_key].set_pose(pose)
                if hasattr(self.manipulation_objects[act_key], "set_linear_velocity"):
                    zero_vel = torch.zeros((b, 3), device=self.device)
                    self.manipulation_objects[act_key].set_linear_velocity(zero_vel)
                    self.manipulation_objects[act_key].set_angular_velocity(zero_vel)

            # 4. Relation selection
            if options and "relation" in options:
                self.current_relation = options["relation"]
            else:
                self.current_relation = random.choice(RELATIONS)

            # 5. Reset robot to standard rest posture
            init_qpos = torch.tensor(
                ROBOT_REST_QPOS,
                device=self.device,
                dtype=torch.float32,
            ).repeat(b, 1)
            self.agent.reset(init_qpos)

    # -----------------------------------------------------------------------
    # Relational Helpers on Environment
    # -----------------------------------------------------------------------

    def get_manipulated_object(self) -> Optional[Actor]:
        """Return manipulated Actor instance for the current episode."""
        if self.manipulated_object_id:
            return self.manipulation_objects[self.manipulated_object_id]
        return None

    def get_reference_object(self) -> Optional[Actor]:
        """Return reference Actor instance for the current episode."""
        if self.reference_object_id:
            return self.manipulation_objects[self.reference_object_id]
        return None

    def get_relational_target_position(
        self,
        relation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute the spatial target position for the current episode's reference object.
        Returns: torch.Tensor of shape (b, 3).
        """
        ref_obj = self.get_reference_object()
        if ref_obj is None:
            raise RuntimeError("Reference object not initialized.")
        rel = relation if relation is not None else self.current_relation
        ref_xyz = ref_obj.pose.p  # shape (b, 3)
        return compute_relational_target_position(
            ref_xyz,
            rel,
            delta=self.relational_delta,
            height=self.object_height,
        )

    def get_all_relational_target_positions(self) -> Dict[str, torch.Tensor]:
        """Return dict of all 5 spatial targets for the current reference object."""
        ref_obj = self.get_reference_object()
        if ref_obj is None:
            raise RuntimeError("Reference object not initialized.")
        ref_xyz = ref_obj.pose.p
        return compute_all_relational_target_positions(
            ref_xyz,
            delta=self.relational_delta,
            height=self.object_height,
        )

    # -----------------------------------------------------------------------
    # Evaluation & Observations
    # -----------------------------------------------------------------------

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """
        Evaluate task success:
        Success when manipulated object is placed at target relative position
        within tolerance (radius < 0.035 m).
        """
        manip_obj = self.get_manipulated_object()
        if manip_obj is None:
            return {"success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}

        target_xyz = self.get_relational_target_position()
        manip_xyz = manip_obj.pose.p
        dist = torch.linalg.norm(manip_xyz - target_xyz, dim=-1)
        success = dist < 0.035
        return {
            "success": success,
            "distance_to_target": dist,
        }

    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        """State observation features."""
        obs = {}
        if hasattr(self.agent, "tcp") and self.agent.tcp is not None:
            obs["tcp_pose"] = self.agent.tcp.pose.raw_pose

        manip_obj = self.get_manipulated_object()
        ref_obj = self.get_reference_object()
        if manip_obj is not None:
            obs["manipulated_pose"] = manip_obj.pose.raw_pose
        if ref_obj is not None:
            obs["reference_pose"] = ref_obj.pose.raw_pose
            obs["target_position"] = self.get_relational_target_position()

        return obs
