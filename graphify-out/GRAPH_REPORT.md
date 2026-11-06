# Graph Report - MahutKol  (2026-11-03)

## Corpus Check
- 56 files · ~149,084 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 862 nodes · 1425 edges · 83 communities (36 shown, 47 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Triangular Prism Mesh & Stress
- SAPIEN URDF Visualization
- ManiSkill Agent Core
- Multi-Shape Relational Task
- Demonstration Recording & HDF5
- WebXR Teleoperation Server
- Simulation Visual Scene & Objects
- PushCube Environment & Reward
- LeRobot Policy & Evaluation
- LeRobot Policy & Evaluation
- LeRobot Policy & Evaluation
- Project Documentation & Specs
- PS4 Controller Teleoperation
- Robot Specs & Joint Configs
- Webcam Gesture Teleoperation
- Prism Physics Dynamic Stability
- Webcam Gesture Teleoperation
- Changelog & System History
- Webcam Gesture Teleoperation
- LeRobot Policy & Evaluation
- Original Request Module
- Webcam Gesture Teleoperation
- Webcam Gesture Teleoperation
- LeRobot Policy & Evaluation
- Test Shapes Env Module
- Inverse Kinematics Solver
- PickCube Task Environment
- Multi-Shape Relational Task
- Documentation & Overview
- Task Verification & Testing
- Architecture & Developer Guides
- Donanim Bilgileri Module
- Project Module
- Documentation & Overview
- URDF Kinematics & Mesh Processing
- Teleop Maniskill Ps4 Module
- 2026-10-15-Fizik-Motoru-Ve-Donanim-Duzeltmesi Module
- URDF Kinematics & Mesh Processing
- MahutPS4 Controller Utilities
- Documentation & Overview
- Arm & Gripper Controllers
- Documentation & Overview
- Webcam Gesture Teleoperation
- Documentation & Overview
- ACT & Diffusion Policy Models
- End-to-End Robotics Learning Pipeline
- SO-100 Hardware Specs & Visuals
- Webcam Gesture Teleoperation
- Webcam Gesture Teleoperation
- Evaluate Policy Module
- Multi-Modal Sensors & Dataset
- Task Configuration & Evaluation
- DualShock 4 & VR Teleop Mapping
- Documentation & Overview
- Package Dependencies & Compatibility
- SmolVLA Training Scripts
- Joint Limits Specification
- Workspace & Target Parameters
- Linux Gamepad Kernel Drivers
- Component Responsibility Matrix
- BaseAgent Control Architecture
- MahutKol Top-Level Namespace
- Project Vision & Guidelines
- SO-100 Kinematic Architecture
- MahutKol Project Definition
- Directory & Module Hierarchy

## God Nodes (most connected - your core abstractions)
1. `run_webcam_teleoperation()` - 21 edges
2. `MultiShapeRelationalTask` - 20 edges
3. `TestWebcamTeleopLogic` - 18 edges
4. `MahutRobot` - 18 edges
5. `MahutKolPushCubeEnv` - 17 edges
6. `build_triangular_prism()` - 17 edges
7. `run_teleoperation()` - 16 edges
8. `compute_prism_vertices_and_faces()` - 16 edges
9. `validate_prism_mesh()` - 16 edges
10. `MahutKolMinimalEnv` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Mahut Kol Push Cube Simulation Environment` --conceptually_related_to--> `MahutKolPushCubeEnv`  [INFERRED]
  docs/mahut_kol_push_cube.png → src/mahut_kol/envs/push_cube_task.py
- `Mahut Kol Robotic Arm` --conceptually_related_to--> `MahutRobot`  [INFERRED]
  docs/mahut_kol_push_cube.png → src/mahut_kol/robots/mahut_robot.py
- `MahutKol Robotic Arm Model` --conceptually_related_to--> `MahutRobot`  [INFERRED]
  docs/maniskill_minimal_sim.png → src/mahut_kol/robots/mahut_robot.py
- `MahutKol Minimal Simulation Environment` --conceptually_related_to--> `MahutKolMinimalEnv`  [INFERRED]
  docs/maniskill_minimal_sim.png → src/mahut_kol/envs/minimal_task.py
- `main()` --uses--> `MahutPS4Controller`  [INFERRED]
  scripts/sim_custom_task.py → Teleoperasyonlar/teleop_ps4.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LeRobot Taklit Öğrenme ve ACT Eğitim Yığını** — docs_lerobot_guide_dataset_architecture, docs_lerobot_guide_act_policy, configs_train_lerobot_act_config [EXTRACTED 1.00]
- **MahutKol Robot Öğrenme Uçtan Uca Boru Hattı Mimarisi** — docs_mahutkol_pipeline_architecture_flow [EXTRACTED 1.00]
- **Push Cube Task Scene Interaction** — docs_mahut_kol_push_cube_robotic_arm, docs_mahut_kol_push_cube_cube, docs_mahut_kol_push_cube_target_marker [INFERRED 0.85]
- **Minimal Simulation Scene and Robot Workspace** — docs_maniskill_minimal_sim_environment, docs_maniskill_minimal_sim_robotic_arm, docs_maniskill_minimal_sim_table [INFERRED 0.85]
- **Çok Modlu Teleoperasyon ve Gösterim Verisi Toplama** — docs_gamepad_teleop_guide_mapping, docs_vr_teleoperation_guide_pipeline_integration [INFERRED 0.85]

## Communities (83 total, 47 thin omitted)

### Community 0 - "Triangular Prism Mesh & Stress"
Cohesion: 0.05
Nodes (8): compute_prism_vertices_and_faces(), create_prism_mesh(), main(), parse_args(), save_prism_obj(), validate_prism_mesh(), TestPrismMesh, TestPrismStressAdversarial

### Community 1 - "SAPIEN URDF Visualization"
Cohesion: 0.06
Nodes (15): ManiSkill 3 MahutKol Simülasyon Görseli, ManiSkill 3 MahutKol Hedef Küpe Uzanma Görseli, SAPIEN Robot Modeli Eklem Doğrulama Görseli, check_and_visualize_urdf(), _apply_mass(), build_cube(), build_cylinder(), build_manipulation_objects() (+7 more)

### Community 2 - "ManiSkill Agent Core"
Cohesion: 0.05
Nodes (5): fk(), ik_plane(), res(), MahutRobot, train_policy()

### Community 3 - "Multi-Shape Relational Task"
Cohesion: 0.07
Nodes (4): compute_all_relational_target_positions(), compute_relational_target_position(), MultiShapeRelationalTask, TestRelationalTargetMath

### Community 5 - "WebXR Teleoperation Server"
Cohesion: 0.09
Nodes (5): BackgroundServer, create_app(), main(), run_teleoperation(), TeleopSharedState

### Community 6 - "Simulation Visual Scene & Objects"
Cohesion: 0.09
Nodes (11): Pushable Red Cube Object, Mahut Kol Push Cube Simulation Environment, Blue Parallel Jaw Gripper, Mahut Kol Robotic Arm, Wooden Tabletop Workspace, Goal Bullseye Target Marker, MahutKol Minimal Simulation Environment, Blue Parallel Jaw Gripper (+3 more)

### Community 8 - "LeRobot Policy & Evaluation"
Cohesion: 0.15
Nodes (3): main(), main(), main()

### Community 9 - "LeRobot Policy & Evaluation"
Cohesion: 0.15
Nodes (4): main(), get_model_path(), main(), print_robot_info()

### Community 10 - "LeRobot Policy & Evaluation"
Cohesion: 0.20
Nodes (6): main(), main(), run(), configure_gripper_safety(), main(), main()

### Community 11 - "Project Documentation & Specs"
Cohesion: 0.11
Nodes (17): 1. MahutKol Özel Modelini Simülasyonda PS4 / Gamepad ile Kontrol Etme:, 1. 🎯 Projenin Amacı ve Vizyonu, 2. 🦾 Robot Modeli ve Donanım Spesifikasyonu, 2. Özel Görev (PushCube) Ortamını Çalıştırma:, 3. MuJoCo İnteraktif Sahne ve Robot Görüntüleyici:, 3. 🔄 Uçtan Uca Boru Hattı (Workflow Architecture), 4. 📁 Klasör ve Dosya Hiyerarşisi, 4. URDF Doğrulama ve Eklem Limitleri Testi: (+9 more)

### Community 13 - "Robot Specs & Joint Configs"
Cohesion: 0.16
Nodes (15): MahutKol Arm PD Joint Position Controller, MahutKol 5-DOF Arm Joint Configuration, MahutKol Default Starting Joint Positions, MahutKol End-Effector TCP Link Reference, MahutKol Gripper PD Joint Position Controller, MahutKol Gripper Finger Joints Configuration, MahutKol Robot Specification Configuration, MahutKol Project Changelog and Activity Log (+7 more)

### Community 14 - "Webcam Gesture Teleoperation"
Cohesion: 0.13
Nodes (3): compute_gripper_value(), map_camera_deltas_to_robot(), TestWebcamTeleopLogic

### Community 16 - "Webcam Gesture Teleoperation"
Cohesion: 0.15
Nodes (3): apply_ema(), apply_step_limiter(), compute_pinch_ratio()

### Community 17 - "Changelog & System History"
Cohesion: 0.13
Nodes (14): 10. ⚡ Donanım Spesifikasyonu (FT5330M & ESP32-S3) ve Fizik Motoru Dengeleme, 11. 🛡️ Küp Tutma Sırasında Kolun Kayması (Drifting) ve Küpün Düşmesi Sorununun Çözümü, 12. 🔄 Model Yapılandırmasının Özel Modele (MahutKol 5-DOF + Kıskaç) Tam Geçirilmesi, 13. 🎮 Teleoperasyon Kodlarının `Teleoperasyonlar/` Klasöründe Toplanması, 1. 📚 Proje İncelemesi ve Dokümantasyon Kurulumu, 2. 🎮 PS4 Kolu (DualShock 4) ve Teleoperasyon Entegrasyonu, 3. 🤖 SO-100 (SO-ARM100) Robot Entegrasyonu, 4. 🕹️ İnteraktif 3D Simülasyon Araçları Geliştirildi (+6 more)

### Community 18 - "Webcam Gesture Teleoperation"
Cohesion: 0.19
Nodes (3): main(), run_webcam_teleoperation(), ThreadedCamera

### Community 20 - "Original Request Module"
Cohesion: 0.15
Nodes (12): 2026-11-03T17:32:37Z, Acceptance Criteria, Automated Verification, Environment & Relational Poses, Mesh & Geometry, Object Creation & Physics, Original User Request, R1. Triangular Prism Mesh Generator (`scripts/generate_prism_mesh.py`) (+4 more)

### Community 21 - "Webcam Gesture Teleoperation"
Cohesion: 0.15
Nodes (12): 1. Gecikme ve Kontrol İyileştirmeleri (Neler Değişti?), 2. Kurulum ve Gereksinimler, 3. Çalıştırma Komutları, 4. Kontrol Mantığı ve Kullanım, 5. Kısayol Tuşları Tablosu, 6. Birim Testleri (Unit Tests), A. Debriyaj (Clutch), B. Robot Eksen Haritalaması (+4 more)

### Community 23 - "LeRobot Policy & Evaluation"
Cohesion: 0.20
Nodes (3): get_model_path(), main(), print_robot_info()

### Community 24 - "Test Shapes Env Module"
Cohesion: 0.23
Nodes (5): main(), verify_environment_episodes(), verify_mesh_geometry(), verify_object_factory(), verify_relational_targets()

### Community 28 - "Documentation & Overview"
Cohesion: 0.18
Nodes (10): 1. Adım: Telefonu USB ile PC'ye Bağlayın ve Portu Köprüleyin, 1. Mimarisi ve Çalışma Mantığı, 2. Adım: Teleoperasyon Betiğini Başlatın, 2. Gereksinimler ve Kurulum, 3. Adım: Telefonda Tarayıcıyı Açın, 3. Sistemi Çalıştırma Adımları, 4. Adım: Teleoperasyona Başlayın, 4. Yapılandırma ve Eksen Ayarları (`CONFIG`) (+2 more)

### Community 30 - "Architecture & Developer Guides"
Cohesion: 0.24
Nodes (8): CAD to URDF Dönüştürme ve Hazırlık Kılavuzu, PS4 Gamepad Teleoperasyon Rehberi, Hugging Face LeRobot Mimari Kılavuzu, ManiSkill & LeRobot Entegrasyon Hattı Kılavuzu, ManiSkill 3 Geliştirici ve Mimari Kılavuzu, Dokümantasyon Merkezi Ana Dizini, VR Teleoperasyon Kılavuzu (Meta Quest), VR Meta Quest Teleoperasyon Şeması

### Community 31 - "Donanim Bilgileri Module"
Cohesion: 0.22
Nodes (9): 1. 🦾 Aktüatörler: Feetech FT5330M 35Kg Digital Servo, 2. 🧠 Ana Kontrol Kartı (MCU): ESP32-S3-N16R8, 3. 🦀 Kıskaç Mekaniği (Gripper-v2): Kramayer ve Pinyon Hesabı, 4. 📐 Fiziksel Montaj ve Koordinat Referansı, Dişli Parametreleri:, 🔌 MahutKol Fiziksel Donanım Spesifikasyonları ve Sim-to-Real Kılavuzu, Simülasyondaki Karşılığı:, Teknik Veriler: (+1 more)

### Community 32 - "Project Module"
Cohesion: 0.20
Nodes (9): Architecture, Code Layout, Feature Inventory, Interface Contracts, M1 -> M2, M2 -> M3, M3 -> M4 / Users, Milestones (+1 more)

### Community 33 - "Documentation & Overview"
Cohesion: 0.20
Nodes (9): 1. Gamepad / PS4 Kolunu Test Etme, 2. Simülasyonda Robotu Teleoperasyon ile Sürme (Varsayılan: MahutKol-PickCube-v1), 3. Teleoperasyon ile İnsan Gösterimi (Demonstration) Toplama, 📁 Dosyalar ve Açıklamaları, 🚀 Hızlı Kullanım Komutları, Klavye (Yedek Kontrol), 🕹️ Kontrol Tuş Haritası (PS4 & Klavye), PS4 Kolu (DualShock 4 / Genel Gamepad) (+1 more)

### Community 34 - "URDF Kinematics & Mesh Processing"
Cohesion: 0.31
Nodes (4): get_link_name(), main(), process_body(), quat_to_rpy()

### Community 36 - "2026-10-15-Fizik-Motoru-Ve-Donanim-Duzeltmesi Module"
Cohesion: 0.25
Nodes (7): Fizik Motoru Dengeleme ve Donanım Entegrasyonu Uygulama Planı, Global Constraints, Review Focus, Task 1: Donanım Spesifikasyonu Dokümantasyonu, Task 2: Robot Kontrolcü ve Konfigürasyon Parametrelerinin Düzeltilmesi, Task 3: URDF İç Çarpışmalarının ve Scripted Actions Boyutunun Düzeltilmesi, Task 4: Birim Testlerin ve Simülasyon Doğrulamasının Çalıştırılması

### Community 37 - "URDF Kinematics & Mesh Processing"
Cohesion: 0.46
Nodes (4): holds(), main(), set_payload(), worst_case()

### Community 39 - "Documentation & Overview"
Cohesion: 0.29
Nodes (6): Çalıştırma, Fare ve Klavye Kontrolleri, Gereksinimler, İçerik, Seçenekler, SO-100 (SO-ARM100) MuJoCo Simülasyonu

### Community 41 - "Documentation & Overview"
Cohesion: 0.33
Nodes (5): Changelog, License, Overview, Standard Open Arm-100 5DOF - Version 1.3 Description (MJCF), URDF → MJCF derivation steps

### Community 43 - "Documentation & Overview"
Cohesion: 0.40
Nodes (4): 🚀 İş Akışı Mimarisi (Pipeline), 🛠️ Kurulum Adımları (Geliştirici Notu), MahutKol: ManiSkill & LeRobot Tabanlı Robot Öğrenme Şablonu, 📁 Proje Klasör Yapısı

### Community 44 - "ACT & Diffusion Policy Models"
Cohesion: 0.50
Nodes (4): LeRobot ACT Eğitim Konfigürasyonu, ACT Model ve Optimizasyon Hiperparametreleri, Action Chunking with Transformers (ACT) Politikası, Diffusion Policy Modeli

### Community 45 - "End-to-End Robotics Learning Pipeline"
Cohesion: 0.50
Nodes (4): Uçtan Uca Robotik Öğrenme Veri Hattı Akışı, VR Teleoperasyonun MahutKol Pipeline Entegrasyonu, Uçtan Uca Boru Hattı Mimarisi, İş Akışı Mimarisi (Pipeline)

### Community 46 - "SO-100 Hardware Specs & Visuals"
Cohesion: 0.50
Nodes (3): [2024-11-25], [2025-06-09], Changelog – Standard Open Arm-100 5DOF - Version 1.3 Description

### Community 50 - "Multi-Modal Sensors & Dataset"
Cohesion: 0.67
Nodes (3): Görev Kamera Sensör Ayarları (cam_top & cam_wrist), LeRobotDataset Çok Modlu Veri Seti Mimarisi, ManiSkill Gözlem Modları (State, RGB, RGBD, PointCloud)

### Community 51 - "Task Configuration & Evaluation"
Cohesion: 0.67
Nodes (3): Pick & Place Görev Konfigürasyonu, Simülasyonda Kapalı Çevrim Değerlendirme Döngüsü, ManiSkill Görev İnşası ve Başarı Değerlendirmesi

### Community 52 - "DualShock 4 & VR Teleop Mapping"
Cohesion: 0.67
Nodes (3): DualShock 4 6-DoF Eksen ve Buton Haritalaması, ALVR & WebXR ile 6-DoF El Takibi ve Stereoskopik Akış, PS4 DualShock 4 Teleoperasyon Şeması

## Knowledge Gaps
- **122 isolated node(s):** `mahut_kol`, `PS4 Gamepad Teleoperasyon Rehberi`, `Wooden Tabletop Workspace`, `Blue Parallel Jaw Gripper`, `Wooden Tabletop Workspace` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 472 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **47 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MahutKolPushCubeEnv` connect `PushCube Environment & Reward` to `ManiSkill Agent Core`, `Simulation Visual Scene & Objects`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **What connects `mahut_kol`, `PS4 Gamepad Teleoperasyon Rehberi`, `Wooden Tabletop Workspace` to the rest of the system?**
  _122 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Triangular Prism Mesh & Stress` be split into smaller, more focused modules?**
  _Cohesion score 0.050203527815468114 - nodes in this community are weakly interconnected._
- **Why does `MultiShapeRelationalTask` connect `Multi-Shape Relational Task` to `Test Shapes Env Module`, `LeRobot Policy & Evaluation`, `ManiSkill Agent Core`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `SAPIEN URDF Visualization` be split into smaller, more focused modules?**
  _Cohesion score 0.06340326340326341 - nodes in this community are weakly interconnected._
- **Why does `MahutRobot` connect `ManiSkill Agent Core` to `Simulation Visual Scene & Objects`, `PushCube Environment & Reward`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Should `ManiSkill Agent Core` be split into smaller, more focused modules?**
  _Cohesion score 0.05191256830601093 - nodes in this community are weakly interconnected._