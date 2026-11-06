# Project: 9 Shape-Color Object Manipulation and Relational Placement Infrastructure

## Architecture
This project develops an infrastructure for the SO-100 robot arm in ManiSkill 3 / SAPIEN simulation, providing 9 distinct shape-color objects (3 shapes: Cube, Cylinder, Triangular Prism; 3 colors: Red, Green, Blue) with realistic contact physics, collision-free tabletop spawning, underground inactive object parking, and a spatial relational target pose calculator.

```
                           +----------------------------------+
                           |   scripts/generate_prism_mesh.py |
                           +-----------------+----------------+
                                             |
                                             v
                           +----------------------------------+
                           | assets/objects/                  |
                           |   triangular_prism.obj           |
                           +-----------------+----------------+
                                             |
                                             v
+-----------------------------+    +----------------------------------+
| Primitive Shapes:           |    | src/mahut_kol/envs/              |
| - Cube (35x35x35 mm)        |--->|   shape_objects.py               |
| - Cylinder (r=17.5, h=35 mm)|    |   - build_manipulation_objects() |
+-----------------------------+    +-----------------+----------------+
                                                     |
                                                     v
                                   +----------------------------------+
                                   | src/mahut_kol/envs/              |
                                   |   multi_shape_task.py            |
                                   |   - MultiShapeRelationalTask     |
                                   |   - Relational Target Calculator |
                                   |   - Collision-free Spawning      |
                                   |   - Inactive Parking (z = -10.0) |
                                   +-----------------+----------------+
                                                     |
                                                     v
                                   +----------------------------------+
                                   | scripts/test_shapes_env.py       |
                                   |   - Automated Verification Suite |
                                   +----------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Triangular Prism Mesh Math | Equilateral base $a=35$ mm, height $h=35$ mm, centroid strictly $(0, 0, 0) \pm 10^{-4}$ m | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Triangular Prism OBJ Export | Watertight convex OBJ file exported to `assets/objects/triangular_prism.obj` | M1 | ORIGINAL_REQUEST §R1 |
| 3 | 9 Shape-Color Object Factory | 3 shapes x 3 colors created via `build_manipulation_objects(scene)` | M2 | ORIGINAL_REQUEST §R2 |
| 4 | Contact Material & Physics | `PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)` and mass $\approx 0.035$ kg | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Color Visual Properties | RGBA colors: Red `[0.85, 0.15, 0.15, 1.0]`, Green `[0.15, 0.80, 0.20, 1.0]`, Blue `[0.15, 0.30, 0.85, 1.0]` | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Relational Target Calculator | Poses for `FRONT`, `BACK`, `LEFT`, `RIGHT` ($\Delta = 0.045$ m) and `TOP` ($+h = 0.035$ m) | M3 | ORIGINAL_REQUEST §R3 |
| 7 | Dynamic Spawning & Collision-Free Placement | Rejection sampling for active objects ($d \ge 0.05$ m) within robot workspace $[0.12, 0.22] \times [-0.10, 0.10]$ | M3 | ORIGINAL_REQUEST §R3 |
| 8 | Inactive Underground Parking | Parking unused objects at $z = -10.0$ with disabled collision filtering to prevent PhysX ground ejection | M3 | ORIGINAL_REQUEST §R3 |
| 9 | SO-100 & MahutKol Support | ManiSkill `MultiShapeRelationalTask` inheriting `BaseEnv`, default `so100`, compatible with `mahut_kol` | M3 | ORIGINAL_REQUEST §R3 |
| 10 | Automated Verification Suite | Test runner covering geometry, physics, spawning, parking, relational poses, and rendering | M4 | ORIGINAL_REQUEST §R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Triangular Prism Mesh Generator | `scripts/generate_prism_mesh.py`, `assets/objects/triangular_prism.obj` | None | DONE |
| M2 | Modular 9 Shape-Color Object Factory | `src/mahut_kol/envs/shape_objects.py` | M1 | IN_PROGRESS |
| M3 | Dynamic Spawning & Relational Placement Environment | `src/mahut_kol/envs/multi_shape_task.py`, `src/mahut_kol/envs/__init__.py` | M2 | PLANNED |
| M4 | Automated Verification & E2E Test Suite | `scripts/test_shapes_env.py`, `TEST_INFRA.md`, `TEST_READY.md` | M1, M2, M3 | PLANNED |

## Interface Contracts

### M1 -> M2
- File: `assets/objects/triangular_prism.obj`
- Format: Standard Wavefront OBJ, 6 vertices, 8 triangular faces with CCW outward normals.
- Centroid: $(0, 0, 0) \pm 10^{-4}$ m (measured: $2.10 \times 10^{-12}$ m).
- Bounding Dimensions: Equilateral base edge $a = 0.035$ m, height $h = 0.035$ m.
- Convex & Watertight: `trimesh.Trimesh.is_convex == True`, `trimesh.Trimesh.is_watertight == True`.

### M2 -> M3
- Module: `src/mahut_kol/envs/shape_objects.py`
- Function: `build_manipulation_objects(scene: sapien.Scene) -> Dict[str, Actor]`
- Keys: 9 keys matching `{color}_{shape}` (`red_cube`, `green_cube`, `blue_cube`, `red_cylinder`, `green_cylinder`, `blue_cylinder`, `red_triangular_prism`, `green_triangular_prism`, `blue_triangular_prism`).
- Physical Properties:
  - Material: `PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)`.
  - Mass: $\approx 0.035$ kg on rigid dynamic component (set explicitly or via shape density).
- Visual Properties:
  - Base colors match exact RGBA specifications (Red `[0.85, 0.15, 0.15, 1.0]`, Green `[0.15, 0.80, 0.20, 1.0]`, Blue `[0.15, 0.30, 0.85, 1.0]`).
- Pre-parking: Builders configured with `initial_pose = sapien.Pose(p=[0, 0, -10.0])` to prevent spawn warnings.

### M3 -> M4 / Users
- Module: `src/mahut_kol/envs/multi_shape_task.py`
- Class: `MultiShapeRelationalTask(BaseEnv)` registered with `@register_env("MultiShapeRelationalTask-v0")`.
- Export: Exported in `src/mahut_kol/envs/__init__.py`.
- Methods / Attributes:
  - `compute_relational_target_poses(ref_pos: Union[np.ndarray, torch.Tensor], delta: float = 0.045, height: float = 0.035) -> Dict[str, Union[np.ndarray, torch.Tensor]]`
    Supporting `'FRONT'`, `'BACK'`, `'LEFT'`, `'RIGHT'`, `'TOP'`.
  - `_initialize_episode(env_idx, options)`: Active objects sampled collision-free ($d \ge 0.05$ m) at $z = 0.0175$ m; inactive objects parked at $z = -10.0$ m with collision groups set to 0.
  - `manipulated_object`, `reference_object`, `distractor_objects` tracked per episode.
  - Headless rendering support via `env.render()`.

## Code Layout
```
/home/deniz/Ortak/MahutKol/
├── assets/
│   └── objects/
│       └── triangular_prism.obj            # M1 output [VERIFIED]
├── scripts/
│   ├── generate_prism_mesh.py             # M1 implementation [VERIFIED]
│   └── test_shapes_env.py                 # M4 implementation
├── src/
│   └── mahut_kol/
│       └── envs/
│           ├── __init__.py                # M3 env registration export
│           ├── shape_objects.py           # M2 implementation
│           └── multi_shape_task.py        # M3 implementation
├── TEST_INFRA.md                          # M4 test specification
├── TEST_READY.md                          # M4 completion signal
└── PROJECT.md                             # Global project blueprint
```
