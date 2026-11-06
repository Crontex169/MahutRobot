# Original User Request

## 2026-11-03T17:32:37Z

Build a modular 9 shape-color object manipulation and relational placement infrastructure for the SO-100 robot arm in ManiSkill 3 / SAPIEN simulation.

Working directory: /home/deniz/Ortak/MahutKol
Integrity mode: development

## Requirements

### R1. Triangular Prism Mesh Generator (`scripts/generate_prism_mesh.py`)
- Generate an equilateral triangular prism 3D mesh with base edge $a = 35\text{ mm}$ and height $h = 35\text{ mm}$.
- Ensure the center of mass / geometric centroid is strictly at $(0, 0, 0)$.
- Save the convex mesh to `assets/objects/triangular_prism.obj` in standard OBJ format.

### R2. Modular 9 Shape-Color Object Factory (`src/mahut_kol/envs/shape_objects.py`)
- Manage 9 distinct object combinations formed by 3 shapes and 3 colors:
  - **Shapes:**
    - Cube: $35 \times 35 \times 35\text{ mm}$ (`half_size = [0.0175, 0.0175, 0.0175]`)
    - Cylinder: radius $r = 17.5\text{ mm}$, height $h = 35\text{ mm}$ (`radius=0.0175, half_length=0.0175`)
    - Triangular Prism: $35\text{ mm}$ equilateral base, $h = 35\text{ mm}$, loaded via convex mesh
  - **Colors (RGBA):**
    - Red: `[0.85, 0.15, 0.15, 1.0]`
    - Green: `[0.15, 0.80, 0.20, 1.0]`
    - Blue: `[0.15, 0.30, 0.85, 1.0]`
- Implement `build_manipulation_objects(scene) -> Dict[str, Actor]` indexing actors by `{color}_{shape}` (e.g. `red_cube`, `green_cylinder`, `blue_triangular_prism`).
- Apply realistic physical material properties (`PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)`) and mass $\approx 0.035\text{ kg}$ to ensure firm gripping by SO-100 fingers.

### R3. Dynamic Spawning & Relational Placement Environment (`src/mahut_kol/envs/multi_shape_task.py`)
- Implement a ManiSkill 3 environment `MultiShapeRelationalTask` inheriting from `BaseEnv`.
- **Object Pool Initialization (`_initialize_episode`):**
  - Select one **Manipulated Object** and one **Reference Object (Target Base)** randomly or by option, plus 1–2 optional distractors.
  - Relocate all inactive/unused objects underground to $z = -10.0$ to remove them from physical interaction.
  - Position active objects in the robot table workspace with random collision-free $x, y$ coordinates (minimum center-to-center distance $\ge 0.05\text{ m}$).
- **Relational Target Pose Calculator:**
  - Given the reference object position $(x, y, z)$ in world coordinates, compute 5 spatial target poses with $\Delta \approx 0.045\text{ m}$:
    - `FRONT`: $(x - \Delta, y, z)$
    - `BACK`: $(x + \Delta, y, z)$
    - `LEFT`: $(x, y - \Delta, z)$
    - `RIGHT`: $(x, y + \Delta, z)$
    - `TOP`: $(x, y, z + h)$

### R4. Automated Verification & Inspection Suite (`scripts/test_shapes_env.py`)
- Provide an automated test script verifying:
  1. Mesh generation and centroid alignment at $(0, 0, 0)$.
  2. Loading of all 9 actors with correct materials and dimensions in ManiSkill.
  3. Episode reset behavior: active objects spawn collision-free on the table, inactive objects are parked at $z = -10.0$.
  4. Accuracy of relative placement target coordinates (`FRONT`, `BACK`, `LEFT`, `RIGHT`, `TOP`).
  5. Optional visual inspection rendering (headless/viewer support).

## Acceptance Criteria

### Mesh & Geometry
- [ ] `assets/objects/triangular_prism.obj` exists and is a valid convex mesh with centroid at $(0, 0, 0) \pm 10^{-4}\text{ m}$.
- [ ] The bounding dimensions of Cube, Cylinder, and Prism all conform to the $35\text{ mm}$ scale.

### Object Creation & Physics
- [ ] `build_manipulation_objects` creates all 9 actors with IDs `{color}_{shape}`.
- [ ] Each object has `PhysxMaterial` with `static_friction=2.0`, `dynamic_friction=2.0`, `restitution=0.0`.
- [ ] Colors accurately reflect the specified RGBA values.

### Environment & Relational Poses
- [ ] `MultiShapeRelationalTask` initializes without error in ManiSkill 3.
- [ ] Inactive objects are placed at $z \le -5.0$ and do not interfere with workspace physics.
- [ ] Active objects are placed without overlap (pairwise distance $> 0.045\text{ m}$).
- [ ] Relational target calculations for `FRONT`, `BACK`, `LEFT`, `RIGHT`, `TOP` match exact mathematical offsets relative to the reference object center.

### Automated Verification
- [ ] `python scripts/test_shapes_env.py` executes end-to-end, passing all assertions with zero errors.
