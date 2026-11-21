"""
Modular 9 Shape-Color Object Factory for SO-100 / ManiSkill 3 Simulation.

Provides factory builders for 9 manipulation objects formed by 3 shapes
(Cube, Cylinder, Triangular Prism) and 3 colors (Red, Green, Blue)
with realistic contact physics (PhysxMaterial friction=2.0, restitution=0.0),
calibrated mass (0.035 kg), and underground pre-parking at z = -10.0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render as render

try:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.actor import Actor
except ImportError:
    ManiSkillScene = Any  # type: ignore
    Actor = Any  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECT_SHAPES = ("cube", "cylinder", "triangular_prism")

OBJECT_COLORS: Dict[str, List[float]] = {
    "red": [0.85, 0.15, 0.15, 1.0],
    "green": [0.15, 0.80, 0.20, 1.0],
    "blue": [0.15, 0.30, 0.85, 1.0],
}

# Geometries (meters) - 35 mm scale
CUBE_HALF_SIZE: List[float] = [0.0175, 0.0175, 0.0175]  # 35 x 35 x 35 mm
CYLINDER_RADIUS: float = 0.0175                         # r = 17.5 mm
CYLINDER_HALF_LENGTH: float = 0.0175                    # h = 35 mm (total length = 2 * half_length)
PRISM_BASE_EDGE: float = 0.035                          # a = 35 mm equilateral base
PRISM_HEIGHT: float = 0.035                             # h = 35 mm

# Physical Defaults
DEFAULT_MASS: float = 0.035                             # 35 grams
DEFAULT_STATIC_FRICTION: float = 2.0
DEFAULT_DYNAMIC_FRICTION: float = 2.0
DEFAULT_RESTITUTION: float = 0.0
DEFAULT_PARKING_Z: float = -10.0
PRE_PARK_POSE: sapien.Pose = sapien.Pose(p=[0.0, 0.0, DEFAULT_PARKING_Z])

# Calibrated densities (kg/m^3) for target mass 0.035 kg:
# - Cube: V = 0.035^3 = 4.2875e-5 m^3 -> rho ≈ 816.33 kg/m^3
# - Cylinder: V = pi * 0.0175^2 * 0.035 ≈ 3.3674e-5 m^3 -> rho ≈ 1039.38 kg/m^3
# - Prism: V = (sqrt(3)/4) * 0.035^3 ≈ 1.8565e-5 m^3 -> rho ≈ 1885.25 kg/m^3
SHAPE_DENSITIES: Dict[str, float] = {
    "cube": 816.33,
    "cylinder": 1039.38,
    "triangular_prism": 1885.25,
}

# Upright orientation for cylinder: SAPIEN cylinder length is along local X-axis.
# A 90 deg rotation around Y aligns the circular base with the tabletop (Z-axis).
CYLINDER_UPRIGHT_ROTATION: List[float] = [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]
CYLINDER_UPRIGHT_POSE: sapien.Pose = sapien.Pose(q=CYLINDER_UPRIGHT_ROTATION)

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------


def get_prism_mesh_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the absolute path to assets/objects/triangular_prism.obj.

    Searches:
    1. Explicit custom_path if provided.
    2. Parent directories walking upward from this module's location.
    3. Current working directory and its parent directories.
    """
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Custom triangular prism mesh not found at: {custom_path}")

    # Search upwards from module file location
    try:
        module_file = Path(__file__).resolve()
        for parent in module_file.parents:
            cand = parent / "assets" / "objects" / "triangular_prism.obj"
            if cand.is_file():
                return cand.resolve()
    except (NameError, ValueError):
        pass

    # Search upwards from current working directory
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        cand = parent / "assets" / "objects" / "triangular_prism.obj"
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        "Triangular prism OBJ asset not found in parent directories or CWD. "
        "Please ensure 'python3 scripts/generate_prism_mesh.py' has been run to generate the mesh."
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def get_entity(actor: Union[Actor, sapien.Entity]) -> sapien.Entity:
    """Extract underlying SAPIEN Entity from either Actor or Entity."""
    if hasattr(actor, "_objs") and len(actor._objs) > 0:
        return actor._objs[0]
    return actor  # type: ignore


def _apply_mass(actor: Union[Actor, sapien.Entity], mass: float) -> None:
    """Set rigid body mass polymorphically across ManiSkill Actor and SAPIEN Entity."""
    if hasattr(actor, "set_mass"):
        actor.set_mass(mass)
    elif hasattr(actor, "mass"):
        actor.mass = mass
    if hasattr(actor, "find_component_by_type"):
        comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if comp is not None:
            comp.set_mass(mass)


def enable_actor_collision(actor: Union[Actor, sapien.Entity]) -> None:
    """Enable standard PhysX collision groups [1, 1, 0, 0] for an actor."""
    if hasattr(actor, "_objs"):
        for obj in actor._objs:
            comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if comp is not None:
                for cs in comp.get_collision_shapes():
                    cs.set_collision_groups([1, 1, 0, 0])
    elif hasattr(actor, "find_component_by_type"):
        comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if comp is not None:
            for cs in comp.get_collision_shapes():
                cs.set_collision_groups([1, 1, 0, 0])


def disable_actor_collision(actor: Union[Actor, sapien.Entity]) -> None:
    """Disable PhysX collision groups [0, 0, 0, 0] for underground parked objects."""
    if hasattr(actor, "_objs"):
        for obj in actor._objs:
            comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if comp is not None:
                for cs in comp.get_collision_shapes():
                    cs.set_collision_groups([0, 0, 0, 0])
    elif hasattr(actor, "find_component_by_type"):
        comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if comp is not None:
            for cs in comp.get_collision_shapes():
                cs.set_collision_groups([0, 0, 0, 0])


def _isolate_collision_groups(actor: Union[Actor, sapien.Entity]) -> None:
    """
    Isolate collision groups to prevent underground parked objects
    from experiencing PhysX ground plane half-space depenetration forces.
    """
    disable_actor_collision(actor)


# ---------------------------------------------------------------------------
# Modular Shape Builders
# ---------------------------------------------------------------------------


def build_cube(
    scene: Union[ManiSkillScene, sapien.Scene],
    color: Sequence[float],
    name: str,
    half_size: Sequence[float] = CUBE_HALF_SIZE,
    material: Optional[physx.PhysxMaterial] = None,
    density: float = SHAPE_DENSITIES["cube"],
    mass: float = DEFAULT_MASS,
    initial_parking_z: float = DEFAULT_PARKING_Z,
    pre_disable_collisions: bool = True,
) -> Union[Actor, sapien.Entity]:
    """Build a cube manipulation object."""
    if material is None:
        material = physx.PhysxMaterial(
            static_friction=DEFAULT_STATIC_FRICTION,
            dynamic_friction=DEFAULT_DYNAMIC_FRICTION,
            restitution=DEFAULT_RESTITUTION,
        )
    builder = scene.create_actor_builder()
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, float(initial_parking_z)])
    builder.add_box_collision(half_size=half_size, material=material, density=density)
    mat_render = render.RenderMaterial()
    mat_render.base_color = list(color)
    builder.add_box_visual(half_size=half_size, material=mat_render)
    actor = builder.build(name=name)
    _apply_mass(actor, mass)
    if pre_disable_collisions:
        _isolate_collision_groups(actor)
    return actor


def build_cylinder(
    scene: Union[ManiSkillScene, sapien.Scene],
    color: Sequence[float],
    name: str,
    radius: float = CYLINDER_RADIUS,
    half_length: float = CYLINDER_HALF_LENGTH,
    material: Optional[physx.PhysxMaterial] = None,
    density: float = SHAPE_DENSITIES["cylinder"],
    mass: float = DEFAULT_MASS,
    initial_parking_z: float = DEFAULT_PARKING_Z,
    pre_disable_collisions: bool = True,
) -> Union[Actor, sapien.Entity]:
    """Build a cylinder manipulation object."""
    if material is None:
        material = physx.PhysxMaterial(
            static_friction=DEFAULT_STATIC_FRICTION,
            dynamic_friction=DEFAULT_DYNAMIC_FRICTION,
            restitution=DEFAULT_RESTITUTION,
        )
    builder = scene.create_actor_builder()
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, float(initial_parking_z)])
    builder.add_cylinder_collision(
        radius=radius,
        half_length=half_length,
        material=material,
        density=density,
        pose=CYLINDER_UPRIGHT_POSE,
    )
    mat_render = render.RenderMaterial()
    mat_render.base_color = list(color)
    builder.add_cylinder_visual(
        radius=radius,
        half_length=half_length,
        material=mat_render,
        pose=CYLINDER_UPRIGHT_POSE,
    )
    actor = builder.build(name=name)
    _apply_mass(actor, mass)
    if pre_disable_collisions:
        _isolate_collision_groups(actor)
    return actor


def build_triangular_prism(
    scene: Union[ManiSkillScene, sapien.Scene],
    color: Sequence[float],
    name: str,
    mesh_path: Optional[Union[str, Path]] = None,
    material: Optional[physx.PhysxMaterial] = None,
    density: float = SHAPE_DENSITIES["triangular_prism"],
    mass: float = DEFAULT_MASS,
    initial_parking_z: float = DEFAULT_PARKING_Z,
    pre_disable_collisions: bool = True,
) -> Union[Actor, sapien.Entity]:
    """Build a triangular prism manipulation object from assets/objects/triangular_prism.obj."""
    mesh_p = str(get_prism_mesh_path(mesh_path))
    if material is None:
        material = physx.PhysxMaterial(
            static_friction=DEFAULT_STATIC_FRICTION,
            dynamic_friction=DEFAULT_DYNAMIC_FRICTION,
            restitution=DEFAULT_RESTITUTION,
        )
    builder = scene.create_actor_builder()
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, float(initial_parking_z)])
    builder.add_convex_collision_from_file(filename=mesh_p, material=material, density=density)
    mat_render = render.RenderMaterial()
    mat_render.base_color = list(color)
    builder.add_visual_from_file(filename=mesh_p, material=mat_render)
    actor = builder.build(name=name)
    _apply_mass(actor, mass)
    if pre_disable_collisions:
        _isolate_collision_groups(actor)
    return actor


# ---------------------------------------------------------------------------
# Factory Entry Point
# ---------------------------------------------------------------------------


def build_manipulation_objects(
    scene: Union[ManiSkillScene, sapien.Scene],
    initial_parking_z: float = DEFAULT_PARKING_Z,
    mesh_path: Optional[Union[str, Path]] = None,
    mass: float = DEFAULT_MASS,
    static_friction: float = DEFAULT_STATIC_FRICTION,
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION,
    restitution: float = DEFAULT_RESTITUTION,
    pre_disable_collisions: bool = True,
) -> Dict[str, Union[Actor, sapien.Entity]]:
    """
    Build 9 manipulation objects (3 shapes x 3 colors) in the given scene.

    The objects are pre-parked at initial_parking_z (default z = -10.0 m)
    with realistic contact friction (default 2.0), exact mass (default 0.035 kg),
    and isolated collision groups.

    Args:
        scene: ManiSkillScene or sapien.Scene instance.
        initial_parking_z: Initial z position for pre-parking (default -10.0 m).
        mesh_path: Optional explicit path to triangular_prism.obj asset.
        mass: Mass in kg to assign to each actor (default 0.035 kg).
        static_friction: Static friction coefficient (default 2.0).
        dynamic_friction: Dynamic friction coefficient (default 2.0).
        restitution: Restitution coefficient (default 0.0).
        pre_disable_collisions: If True, set collision groups to 0 to prevent
            ground-plane depenetration while underground.

    Returns:
        Dict[str, Union[Actor, sapien.Entity]]: Dictionary mapping '{color}_{shape}'
            (e.g. 'red_cube', 'green_cylinder', 'blue_triangular_prism') to built actors.
    """
    mat = physx.PhysxMaterial(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    objects: Dict[str, Union[Actor, sapien.Entity]] = {}
    for color_name, rgba in OBJECT_COLORS.items():
        for shape_name in OBJECT_SHAPES:
            key = f"{color_name}_{shape_name}"
            if shape_name == "cube":
                obj = build_cube(
                    scene=scene,
                    color=rgba,
                    name=key,
                    material=mat,
                    mass=mass,
                    initial_parking_z=initial_parking_z,
                    pre_disable_collisions=pre_disable_collisions,
                )
            elif shape_name == "cylinder":
                obj = build_cylinder(
                    scene=scene,
                    color=rgba,
                    name=key,
                    material=mat,
                    mass=mass,
                    initial_parking_z=initial_parking_z,
                    pre_disable_collisions=pre_disable_collisions,
                )
            elif shape_name == "triangular_prism":
                obj = build_triangular_prism(
                    scene=scene,
                    color=rgba,
                    name=key,
                    mesh_path=mesh_path,
                    material=mat,
                    mass=mass,
                    initial_parking_z=initial_parking_z,
                    pre_disable_collisions=pre_disable_collisions,
                )
            else:
                raise ValueError(f"Unknown shape '{shape_name}'")
            objects[key] = obj

    return objects
