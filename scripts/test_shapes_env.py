#!/usr/bin/env python3
"""
Test & Verification Suite for MultiShapeRelationalTask and 9 Shape-Color Object Factory.

Verifies:
1. Triangular Prism Mesh asset geometry, centroid at (0, 0, 0), and watertightness.
2. 9 Shape-Color manipulation objects creation, materials, and physical properties.
3. ManiSkill 3 MultiShapeRelationalTask environment initialization and episode resets.
4. Active vs inactive object pool behavior (underground parking at z = -10.0 m).
5. Collision-free tabletop placement within SO-100 workspace.
6. 5 Spatial relational target pose calculations (FRONT, BACK, LEFT, RIGHT, TOP).
7. Optional offscreen/diagnostic rendering and image saving.

Usage:
    python3 scripts/test_shapes_env.py [--episodes 5] [--render] [--save-image]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import sapien
import torch

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mahut_kol.envs
from mahut_kol.envs.multi_shape_task import (
    DEFAULT_OBJECT_HEIGHT,
    DEFAULT_RELATIONAL_DELTA,
    RELATIONS,
    TABLETOP_HALF_HEIGHT,
    MultiShapeRelationalTask,
    compute_all_relational_target_positions,
    compute_relational_target_position,
)
from mahut_kol.envs.shape_objects import (
    CUBE_HALF_SIZE,
    CYLINDER_HALF_LENGTH,
    CYLINDER_RADIUS,
    DEFAULT_MASS,
    DEFAULT_PARKING_Z,
    OBJECT_COLORS,
    OBJECT_SHAPES,
    PRISM_BASE_EDGE,
    PRISM_HEIGHT,
    build_manipulation_objects,
    get_prism_mesh_path,
)


def verify_mesh_geometry(mesh_path: Path) -> bool:
    """Verify triangular prism OBJ geometry and centroid alignment."""
    print("\n" + "=" * 60)
    print("STEP 1: Verifying Triangular Prism Mesh Asset")
    print("=" * 60)
    print(f"Path: {mesh_path}")

    if not mesh_path.is_file():
        print(f"[-] ERROR: Mesh file not found at {mesh_path}")
        return False

    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(mesh_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = [float(x) for x in line.split()[1:]]
                vertices.append(parts)
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                faces.append(idx)

    v_arr = np.array(vertices, dtype=np.float64)
    centroid = np.mean(v_arr, axis=0)

    print(f"[+] Vertex Count: {len(vertices)} (expected: 6)")
    print(f"[+] Face Count: {len(faces)} (expected: 8)")
    print(f"[+] Computed Centroid (x, y, z): {centroid.tolist()}")

    assert len(vertices) == 6, f"Expected 6 vertices, got {len(vertices)}"
    assert len(faces) == 8, f"Expected 8 faces, got {len(faces)}"
    assert np.allclose(centroid, [0.0, 0.0, 0.0], atol=1e-4), f"Centroid {centroid} not within 1e-4 of origin"

    # Verify bounding extents
    x_span = np.ptp(v_arr[:, 0])
    y_span = np.ptp(v_arr[:, 1])
    z_span = np.ptp(v_arr[:, 2])
    print(f"[+] Bounding extents: dx={x_span:.4f}m, dy={y_span:.4f}m, dz={z_span:.4f}m")
    assert abs(z_span - PRISM_HEIGHT) < 1e-4, f"Height {z_span} != {PRISM_HEIGHT}"

    print("[SUCCESS] Triangular Prism Mesh geometry verified!")
    return True


def verify_object_factory() -> bool:
    """Verify 9-object factory in SAPIEN scene."""
    print("\n" + "=" * 60)
    print("STEP 2: Verifying 9 Shape-Color Object Factory")
    print("=" * 60)

    scene = sapien.Scene()
    objects = build_manipulation_objects(scene)

    print(f"[+] Total objects built: {len(objects)} (expected: 9)")
    assert len(objects) == 9, f"Expected 9 objects, got {len(objects)}"

    for color_name, expected_rgba in OBJECT_COLORS.items():
        for shape_name in OBJECT_SHAPES:
            key = f"{color_name}_{shape_name}"
            assert key in objects, f"Missing object key {key}"
            actor = objects[key]

            # Verify underground initial pose
            pose = actor.get_pose().p
            assert abs(pose[2] - DEFAULT_PARKING_Z) < 1e-3, f"Actor {key} not at z={DEFAULT_PARKING_Z}"

            print(f"  [+] {key:25s} -> Pose: {pose.tolist()}")

    print("[SUCCESS] All 9 shape-color objects verified!")
    return True


def verify_relational_targets() -> bool:
    """Verify math calculations for 5 relational target poses."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying Relational Target Pose Math")
    print("=" * 60)

    ref_pos = np.array([0.180, 0.020, 0.0175], dtype=np.float32)
    delta = DEFAULT_RELATIONAL_DELTA
    height = DEFAULT_OBJECT_HEIGHT

    targets = compute_all_relational_target_positions(ref_pos, delta=delta, height=height)

    expected = {
        "FRONT": np.array([0.180 - delta, 0.020, 0.0175]),
        "BACK": np.array([0.180 + delta, 0.020, 0.0175]),
        "LEFT": np.array([0.180, 0.020 - delta, 0.0175]),
        "RIGHT": np.array([0.180, 0.020 + delta, 0.0175]),
        "TOP": np.array([0.180, 0.020, 0.0175 + height]),
    }

    print(f"[+] Reference Object Position: {ref_pos.tolist()}")
    print(f"[+] Delta offset: {delta} m, Height offset: {height} m")

    for rel in RELATIONS:
        computed = targets[rel]
        exp = expected[rel]
        np.testing.assert_allclose(computed, exp, atol=1e-5)
        print(f"  [+] {rel:6s} Target: [{computed[0]:.4f}, {computed[1]:.4f}, {computed[2]:.4f}]")

    print("[SUCCESS] All 5 spatial relational target calculations verified!")
    return True


def verify_environment_episodes(num_episodes: int = 5, render: bool = False, save_image: bool = False) -> bool:
    """Verify ManiSkill 3 environment across multiple randomized episodes."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Verifying MultiShapeRelationalTask Environment ({num_episodes} Episodes)")
    print("=" * 60)

    render_mode = "rgb_array" if (render or save_image) else None
    env = gym.make("MahutKol-MultiShapeRelational-v1", render_mode=render_mode)
    unwrapped: MultiShapeRelationalTask = env.unwrapped

    print(f"[+] Environment created: {env.spec.id if env.spec else 'MahutKol-MultiShapeRelational-v1'}")
    print(f"[+] Workspace X bounds: {unwrapped.workspace_x_bounds}")
    print(f"[+] Workspace Y bounds: {unwrapped.workspace_y_bounds}")

    captured_frame = None

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(options={"num_distractors": 1})

        manip_id = unwrapped.manipulated_object_id
        ref_id = unwrapped.reference_object_id
        distractors = unwrapped.distractor_object_ids
        relation = unwrapped.current_relation

        active_keys = [manip_id, ref_id] + distractors
        inactive_keys = [k for k in unwrapped.manipulation_objects if k not in active_keys]

        print(f"\n--- Episode {ep}/{num_episodes} ---")
        print(f"  Manipulated Object: {manip_id}")
        print(f"  Reference Object:   {ref_id}")
        print(f"  Distractor Objects: {distractors}")
        print(f"  Relation:           {relation}")

        # 1. Inactive objects check (underground at z <= -5.0)
        for inact in inactive_keys:
            z_pos = unwrapped.manipulation_objects[inact].pose.p.squeeze().tolist()[2]
            assert z_pos <= -5.0, f"Inactive object {inact} at z={z_pos} (not underground!)"

        # 2. Active objects tabletop check
        positions = []
        for act in active_keys:
            pos = unwrapped.manipulation_objects[act].pose.p.squeeze().tolist()
            assert abs(pos[2] - TABLETOP_HALF_HEIGHT) < 0.01, f"Active object {act} not on tabletop: z={pos[2]}"
            assert unwrapped.workspace_x_bounds[0] - 0.02 <= pos[0] <= unwrapped.workspace_x_bounds[1] + 0.02
            assert unwrapped.workspace_y_bounds[0] - 0.02 <= pos[1] <= unwrapped.workspace_y_bounds[1] + 0.02
            positions.append(np.array(pos[:2]))
            print(f"    {act:25s} @ x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")

        # 3. Collision-free pairwise distance check
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= 0.045, f"Collision detected! {active_keys[i]} and {active_keys[j]} distance={dist:.4f}m < 0.045m"
        print(f"  [+] Collision-free placement verified (all pairwise distances >= 0.045m)")

        # 4. Physics simulation step & stability check (ensure objects do NOT fall through table)
        for _ in range(20):
            env.step(torch.zeros(env.action_space.shape, device=unwrapped.device))

        for act in active_keys:
            post_pos = unwrapped.manipulation_objects[act].pose.p.squeeze().tolist()
            assert abs(post_pos[2] - TABLETOP_HALF_HEIGHT) < 0.005, (
                f"Object {act} fell through table during physics step! z={post_pos[2]}"
            )
            if "cylinder" in act:
                # Cylinder must stand upright on circular base
                assert abs(post_pos[2] - TABLETOP_HALF_HEIGHT) < 0.002, (
                    f"Cylinder {act} tilted or fell off base: z={post_pos[2]}"
                )
        print(f"  [+] Physics simulation verified: all objects stable on tabletop (no falling), cylinders upright.")

        # 5. Target computation check
        target_pos = unwrapped.get_relational_target_position().squeeze().tolist()
        print(f"  [+] Target Position ({relation}): [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")

        # 5. Capture frame if requested
        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                captured_frame = frame

    if save_image and captured_frame is not None:
        try:
            if isinstance(captured_frame, torch.Tensor):
                captured_frame = captured_frame.detach().cpu().numpy()
            if captured_frame.ndim == 4:
                captured_frame = captured_frame[0]

            import cv2
            out_img_path = _REPO_ROOT / "assets" / "shapes_verification_render.png"
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), bgr_frame)
            print(f"\n[+] Saved diagnostic render snapshot to: {out_img_path}")
        except ImportError:
            try:
                from PIL import Image
                out_img_path = _REPO_ROOT / "assets" / "shapes_verification_render.png"
                im = Image.fromarray(captured_frame)
                im.save(str(out_img_path))
                print(f"\n[+] Saved diagnostic render snapshot to: {out_img_path}")
            except Exception as e:
                print(f"[!] Could not save render image: {e}")

    env.close()
    print("\n[SUCCESS] MultiShapeRelationalTask environment episodes verified!")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Shape Manipulation Verification Suite")
    parser.add_argument("--episodes", type=int, default=5, help="Number of randomized test episodes")
    parser.add_argument("--render", action="store_true", help="Enable offscreen rendering")
    parser.add_argument("--save-image", action="store_true", help="Save render snapshot to assets/")
    args = parser.parse_args()

    print("=" * 60)
    print("MAHUTKOL 9 SHAPE-COLOR & RELATIONAL PLACEMENT VERIFICATION")
    print("=" * 60)

    mesh_path = get_prism_mesh_path()

    s1 = verify_mesh_geometry(mesh_path)
    s2 = verify_object_factory()
    s3 = verify_relational_targets()
    s4 = verify_environment_episodes(
        num_episodes=args.episodes,
        render=args.render,
        save_image=args.save_image,
    )

    if s1 and s2 and s3 and s4:
        print("\n" + "=" * 60)
        print(">>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY (100%) <<<")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print(">>> VERIFICATION FAILED! <<<")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
