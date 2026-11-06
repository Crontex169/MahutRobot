# Graph Report - MahutKol  (2026-11-06)

## Corpus Check
- 14 files · ~155,148 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1009 nodes · 1691 edges · 98 communities (36 shown, 62 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 79 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- shape_objects.py
- MultiShapeRelationalTask
- audio_recorder.py
- TeleopController
- webxr_teleop.py
- MahutKolMinimalEnv
- MahutKol Aktivite Günlüğü ve Değişiklik Kaydı
- MahutKolPushCubeEnv
- TestCommandParser
- DummyRecorder
- argparse
- os
- 6. 🚀 Hızlı Başlangıç ve Komut Kılavuzu
- MahutPS4Controller
- MahutRobot
- mujoco
- STTEngine
- TestPrismMesh
- speech/__init__.py
- TestPrismPhysicsStress
- test_webcam_teleop.py
- TestWebcamTeleopLogic
- VoiceCommander
- run_webcam_teleoperation
- CommandParser
- Teleoperasyonlar/teleop_maniskill_ps4.py
- TestPrismStressAdversarial
- Acceptance Criteria
- SO-100 Robot Kolu İçin Web Kamerası (MediaPipe Hands) Teleoperasyon Modülü
- pick_cube_task.py
- generate_prism_mesh.py
- validate_prism_mesh
- SO100IKSolver
- MahutKolPickCubeEnv
- TestMultiShapeRelationalTaskEnv
- main.py
- compute_prism_vertices_and_faces
- SO-100 Robot Kolu WebXR (ARCore) Telefon Teleoperasyon Sistemi
- TestMahutKolTasks
- Dokümantasyon Merkezi Ana Dizini
- 🔌 MahutKol Fiziksel Donanım Spesifikasyonları ve Sim-to-Real Kılavuzu
- torch
- Project: 9 Shape-Color Object Manipulation and Relational Placement Infrastructure
- view_so100.py
- 🚀 Hızlı Kullanım Komutları
- Review Focus
- test_voice_command.py
- TestMahutPS4Controller
- MahutKol Robot Specification Configuration
- SO-100 (SO-ARM100) MuJoCo Simülasyonu
- controllers.py
- Standard Open Arm-100 5DOF - Version 1.3 Description (MJCF)
- WebcamTeleopHUD
- typing
- train_policy.py
- LeRobot ACT Eğitim Konfigürasyonu
- Teleoperasyonlar/test_ps4.py
- Changelog – Standard Open Arm-100 5DOF - Version 1.3 Description
- apply_step_limiter
- compute_hand_orientation
- gripper_to_jaw_angle
- Görev Kamera Sensör Ayarları (cam_top & cam_wrist)
- ManiSkill Görev İnşası ve Başarı Değerlendirmesi
- DualShock 4 6-DoF Eksen ve Buton Haritalaması
- Uçtan Uca Robotik Öğrenme Veri Hattı Akışı
- 📚 MahutKol Dokümantasyon Merkezi
- train_smolvla.sh
- Eklem Sınırları ve Tipleri Spesifikasyonu
- Çalışma Masası ve Hedef Nesne Parametreleri
- Linux Çekirdeği Gamepad Sürücü Yapısı (hid-playstation)
- Bileşen ve Kütüphane Sorumluluk Matrisi
- ManiSkill BaseAgent ve Kontrolcü Mimarisi
- mahut_kol
- MahutKol Proje Rehberi ve Vizyonu
- SO-100 (SO-ARM100) 6-DOF Robot Mimarisi

## God Nodes (most connected - your core abstractions)
1. `run_webcam_teleoperation()` - 21 edges
2. `MultiShapeRelationalTask` - 20 edges
3. `TestWebcamTeleopLogic` - 18 edges
4. `MahutRobot` - 18 edges
5. `MahutKolPushCubeEnv` - 17 edges
6. `build_triangular_prism()` - 17 edges
7. `VoiceCommander` - 17 edges
8. `MahutKolMinimalEnv` - 16 edges
9. `compute_prism_vertices_and_faces()` - 16 edges
10. `validate_prism_mesh()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Mahut Kol Robotic Arm` --conceptually_related_to--> `MahutRobot`  [INFERRED]
  docs/mahut_kol_push_cube.png → src/mahut_kol/robots/mahut_robot.py
- `MahutKol Robotic Arm Model` --conceptually_related_to--> `MahutRobot`  [INFERRED]
  docs/maniskill_minimal_sim.png → src/mahut_kol/robots/mahut_robot.py
- `Mahut Kol Push Cube Simulation Environment` --conceptually_related_to--> `MahutKolPushCubeEnv`  [INFERRED]
  docs/mahut_kol_push_cube.png → src/mahut_kol/envs/push_cube_task.py
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
- **Türkçe Sesli Komut ve Çözümleme Mimarisi** — docs_superpowers_plans_2026_11_03_turkish_speech_command_parser_audio_recorder, docs_superpowers_plans_2026_11_03_turkish_speech_command_parser_stt_engine, docs_superpowers_plans_2026_11_03_turkish_speech_command_parser_command_parser, docs_superpowers_plans_2026_11_03_turkish_speech_command_parser_voice_commander [EXTRACTED 1.00]
- **Robot Kontrol, Dengeleme ve Sim-to-Real Stabilizasyonu** — yapilanlar_custom_5dof_robot, yapilanlar_gripper_tuning, yapilanlar_drift_and_drop_fix, yapilanlar_hardware_specs [INFERRED 0.85]
- **MahutKol Veri Toplama, Simülasyon ve Eğitim Akışı** — readme_pipeline_architecture, yapilanlar_maniskill_env_integration, yapilanlar_ps4_teleoperation, requirements_robot_learning_stack [INFERRED 0.85]

## Communities (98 total, 62 thin omitted)

### Community 0 - "shape_objects.py"
Cohesion: 0.07
Nodes (14): ManiSkill 3 MahutKol Simülasyon Görseli, ManiSkill 3 MahutKol Hedef Küpe Uzanma Görseli, SAPIEN Robot Modeli Eklem Doğrulama Görseli, _apply_mass(), build_cube(), build_cylinder(), build_manipulation_objects(), build_triangular_prism() (+6 more)

### Community 1 - "MultiShapeRelationalTask"
Cohesion: 0.05
Nodes (9): main(), verify_environment_episodes(), verify_mesh_geometry(), verify_object_factory(), verify_relational_targets(), compute_all_relational_target_positions(), compute_relational_target_position(), MultiShapeRelationalTask (+1 more)

### Community 2 - "audio_recorder.py"
Cohesion: 0.07
Nodes (6): AudioRecorder, audio_callback(), audio_callback(), compute_rms(), _ensure_portaudio_loaded(), TestAudioRecorder

### Community 4 - "webxr_teleop.py"
Cohesion: 0.10
Nodes (5): BackgroundServer, create_app(), main(), run_teleoperation(), TeleopSharedState

### Community 5 - "MahutKolMinimalEnv"
Cohesion: 0.09
Nodes (11): Pushable Red Cube Object, Mahut Kol Push Cube Simulation Environment, Blue Parallel Jaw Gripper, Mahut Kol Robotic Arm, Wooden Tabletop Workspace, Goal Bullseye Target Marker, MahutKol Minimal Simulation Environment, Blue Parallel Jaw Gripper (+3 more)

### Community 6 - "MahutKol Aktivite Günlüğü ve Değişiklik Kaydı"
Cohesion: 0.10
Nodes (20): AudioRecorder (Push-to-Talk ve VAD), CommandParser ve Türkçe Morfolojik Normalizasyon, Türkçe Sesli Komut ve Robot Komut Çözücü Planı, VoiceCommander Orkestrasyonu, Adım Adım Geliştirme Sırası, MahutKol Robot Öğrenme Şablonu, İş Akışı Mimarisi (Pipeline), Yerel Türkçe Sesli Komut Modülü (+12 more)

### Community 9 - "DummyRecorder"
Cohesion: 0.11
Nodes (3): DummyRecorder, DummySTT, TestVoiceCommander

### Community 10 - "argparse"
Cohesion: 0.23
Nodes (6): main(), main(), main(), run(), main(), main()

### Community 12 - "6. 🚀 Hızlı Başlangıç ve Komut Kılavuzu"
Cohesion: 0.11
Nodes (17): 1. MahutKol Özel Modelini Simülasyonda PS4 / Gamepad ile Kontrol Etme:, 1. 🎯 Projenin Amacı ve Vizyonu, 2. 🦾 Robot Modeli ve Donanım Spesifikasyonu, 2. Özel Görev (PushCube) Ortamını Çalıştırma:, 3. MuJoCo İnteraktif Sahne ve Robot Görüntüleyici:, 3. 🔄 Uçtan Uca Boru Hattı (Workflow Architecture), 4. 📁 Klasör ve Dosya Hiyerarşisi, 4. URDF Doğrulama ve Eklem Limitleri Testi: (+9 more)

### Community 15 - "mujoco"
Cohesion: 0.18
Nodes (8): get_link_name(), main(), process_body(), quat_to_rpy(), holds(), main(), set_payload(), worst_case()

### Community 21 - "test_webcam_teleop.py"
Cohesion: 0.15
Nodes (3): apply_ema(), clamp_to_workspace(), compute_pinch_ratio()

### Community 22 - "TestWebcamTeleopLogic"
Cohesion: 0.15
Nodes (3): compute_gripper_value(), map_camera_deltas_to_robot(), TestWebcamTeleopLogic

### Community 25 - "run_webcam_teleoperation"
Cohesion: 0.19
Nodes (3): main(), run_webcam_teleoperation(), ThreadedCamera

### Community 27 - "Teleoperasyonlar/teleop_maniskill_ps4.py"
Cohesion: 0.20
Nodes (3): configure_gripper_safety(), main(), PS4TeleopController

### Community 29 - "Acceptance Criteria"
Cohesion: 0.15
Nodes (12): 2026-11-03T17:32:37Z, Acceptance Criteria, Automated Verification, Environment & Relational Poses, Mesh & Geometry, Object Creation & Physics, Original User Request, R1. Triangular Prism Mesh Generator (`scripts/generate_prism_mesh.py`) (+4 more)

### Community 30 - "SO-100 Robot Kolu İçin Web Kamerası (MediaPipe Hands) Teleoperasyon Modülü"
Cohesion: 0.15
Nodes (12): 1. Gecikme ve Kontrol İyileştirmeleri (Neler Değişti?), 2. Kurulum ve Gereksinimler, 3. Çalıştırma Komutları, 4. Kontrol Mantığı ve Kullanım, 5. Kısayol Tuşları Tablosu, 6. Birim Testleri (Unit Tests), A. Debriyaj (Clutch), B. Robot Eksen Haritalaması (+4 more)

### Community 31 - "pick_cube_task.py"
Cohesion: 0.24
Nodes (5): fk(), ik(), ik_plane(), res(), scripted_actions()

### Community 32 - "generate_prism_mesh.py"
Cohesion: 0.30
Nodes (3): create_prism_mesh(), main(), parse_args()

### Community 37 - "main.py"
Cohesion: 0.22
Nodes (3): get_model_path(), main(), print_robot_info()

### Community 39 - "SO-100 Robot Kolu WebXR (ARCore) Telefon Teleoperasyon Sistemi"
Cohesion: 0.18
Nodes (10): 1. Adım: Telefonu USB ile PC'ye Bağlayın ve Portu Köprüleyin, 1. Mimarisi ve Çalışma Mantığı, 2. Adım: Teleoperasyon Betiğini Başlatın, 2. Gereksinimler ve Kurulum, 3. Adım: Telefonda Tarayıcıyı Açın, 3. Sistemi Çalıştırma Adımları, 4. Adım: Teleoperasyona Başlayın, 4. Yapılandırma ve Eksen Ayarları (`CONFIG`) (+2 more)

### Community 41 - "Dokümantasyon Merkezi Ana Dizini"
Cohesion: 0.24
Nodes (8): CAD to URDF Dönüştürme ve Hazırlık Kılavuzu, PS4 Gamepad Teleoperasyon Rehberi, Hugging Face LeRobot Mimari Kılavuzu, ManiSkill & LeRobot Entegrasyon Hattı Kılavuzu, ManiSkill 3 Geliştirici ve Mimari Kılavuzu, Dokümantasyon Merkezi Ana Dizini, VR Teleoperasyon Kılavuzu (Meta Quest), VR Meta Quest Teleoperasyon Şeması

### Community 43 - "🔌 MahutKol Fiziksel Donanım Spesifikasyonları ve Sim-to-Real Kılavuzu"
Cohesion: 0.22
Nodes (9): 1. 🦾 Aktüatörler: Feetech FT5330M 35Kg Digital Servo, 2. 🧠 Ana Kontrol Kartı (MCU): ESP32-S3-N16R8, 3. 🦀 Kıskaç Mekaniği (Gripper-v2): Kramayer ve Pinyon Hesabı, 4. 📐 Fiziksel Montaj ve Koordinat Referansı, Dişli Parametreleri:, 🔌 MahutKol Fiziksel Donanım Spesifikasyonları ve Sim-to-Real Kılavuzu, Simülasyondaki Karşılığı:, Teknik Veriler: (+1 more)

### Community 45 - "Project: 9 Shape-Color Object Manipulation and Relational Placement Infrastructure"
Cohesion: 0.20
Nodes (9): Architecture, Code Layout, Feature Inventory, Interface Contracts, M1 -> M2, M2 -> M3, M3 -> M4 / Users, Milestones (+1 more)

### Community 46 - "view_so100.py"
Cohesion: 0.24
Nodes (3): get_model_path(), main(), print_robot_info()

### Community 47 - "🚀 Hızlı Kullanım Komutları"
Cohesion: 0.20
Nodes (9): 1. Gamepad / PS4 Kolunu Test Etme, 2. Simülasyonda Robotu Teleoperasyon ile Sürme (Varsayılan: MahutKol-PickCube-v1), 3. Teleoperasyon ile İnsan Gösterimi (Demonstration) Toplama, 📁 Dosyalar ve Açıklamaları, 🚀 Hızlı Kullanım Komutları, Klavye (Yedek Kontrol), 🕹️ Kontrol Tuş Haritası (PS4 & Klavye), PS4 Kolu (DualShock 4 / Genel Gamepad) (+1 more)

### Community 49 - "Review Focus"
Cohesion: 0.25
Nodes (7): Fizik Motoru Dengeleme ve Donanım Entegrasyonu Uygulama Planı, Global Constraints, Review Focus, Task 1: Donanım Spesifikasyonu Dokümantasyonu, Task 2: Robot Kontrolcü ve Konfigürasyon Parametrelerinin Düzeltilmesi, Task 3: URDF İç Çarpışmalarının ve Scripted Actions Boyutunun Düzeltilmesi, Task 4: Birim Testlerin ve Simülasyon Doğrulamasının Çalıştırılması

### Community 50 - "test_voice_command.py"
Cohesion: 0.39
Nodes (5): Colors, main(), parse_args(), print_banner(), print_command_summary()

### Community 52 - "MahutKol Robot Specification Configuration"
Cohesion: 0.33
Nodes (7): MahutKol Arm PD Joint Position Controller, MahutKol 5-DOF Arm Joint Configuration, MahutKol Default Starting Joint Positions, MahutKol End-Effector TCP Link Reference, MahutKol Gripper PD Joint Position Controller, MahutKol Gripper Finger Joints Configuration, MahutKol Robot Specification Configuration

### Community 54 - "SO-100 (SO-ARM100) MuJoCo Simülasyonu"
Cohesion: 0.29
Nodes (6): Çalıştırma, Fare ve Klavye Kontrolleri, Gereksinimler, İçerik, Seçenekler, SO-100 (SO-ARM100) MuJoCo Simülasyonu

### Community 56 - "Standard Open Arm-100 5DOF - Version 1.3 Description (MJCF)"
Cohesion: 0.33
Nodes (5): Changelog, License, Overview, Standard Open Arm-100 5DOF - Version 1.3 Description (MJCF), URDF → MJCF derivation steps

### Community 60 - "LeRobot ACT Eğitim Konfigürasyonu"
Cohesion: 0.50
Nodes (4): LeRobot ACT Eğitim Konfigürasyonu, ACT Model ve Optimizasyon Hiperparametreleri, Action Chunking with Transformers (ACT) Politikası, Diffusion Policy Modeli

### Community 62 - "Changelog – Standard Open Arm-100 5DOF - Version 1.3 Description"
Cohesion: 0.50
Nodes (3): [2024-11-25], [2025-06-09], Changelog – Standard Open Arm-100 5DOF - Version 1.3 Description

### Community 66 - "Görev Kamera Sensör Ayarları (cam_top & cam_wrist)"
Cohesion: 0.67
Nodes (3): Görev Kamera Sensör Ayarları (cam_top & cam_wrist), LeRobotDataset Çok Modlu Veri Seti Mimarisi, ManiSkill Gözlem Modları (State, RGB, RGBD, PointCloud)

### Community 67 - "ManiSkill Görev İnşası ve Başarı Değerlendirmesi"
Cohesion: 0.67
Nodes (3): Pick & Place Görev Konfigürasyonu, Simülasyonda Kapalı Çevrim Değerlendirme Döngüsü, ManiSkill Görev İnşası ve Başarı Değerlendirmesi

### Community 68 - "DualShock 4 6-DoF Eksen ve Buton Haritalaması"
Cohesion: 0.67
Nodes (3): DualShock 4 6-DoF Eksen ve Buton Haritalaması, ALVR & WebXR ile 6-DoF El Takibi ve Stereoskopik Akış, PS4 DualShock 4 Teleoperasyon Şeması

### Community 69 - "Uçtan Uca Robotik Öğrenme Veri Hattı Akışı"
Cohesion: 0.67
Nodes (3): Uçtan Uca Robotik Öğrenme Veri Hattı Akışı, VR Teleoperasyonun MahutKol Pipeline Entegrasyonu, Uçtan Uca Boru Hattı Mimarisi

## Knowledge Gaps
- **108 isolated node(s):** `mahut_kol`, `ManiSkill 3 MahutKol Hedef Küpe Uzanma Görseli`, `1. MahutKol Özel Modelini Simülasyonda PS4 / Gamepad ile Kontrol Etme:`, `1. 🎯 Projenin Amacı ve Vizyonu`, `2. 🦾 Robot Modeli ve Donanım Spesifikasyonu` (+103 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 533 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **62 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MahutKolPushCubeEnv` connect `MahutKolPushCubeEnv` to `multi_shape_task.py`, `torch`, `MahutKolMinimalEnv`, `MahutRobot`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **What connects `mahut_kol`, `ManiSkill 3 MahutKol Hedef Küpe Uzanma Görseli`, `1. MahutKol Özel Modelini Simülasyonda PS4 / Gamepad ile Kontrol Etme:` to the rest of the system?**
  _108 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `shape_objects.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06874669487043893 - nodes in this community are weakly interconnected._
- **Why does `MultiShapeRelationalTask` connect `MultiShapeRelationalTask` to `argparse`, `torch`, `multi_shape_task.py`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `MultiShapeRelationalTask` be split into smaller, more focused modules?**
  _Cohesion score 0.05387205387205387 - nodes in this community are weakly interconnected._
- **Why does `MahutKolMinimalEnv` connect `MahutKolMinimalEnv` to `torch`, `multi_shape_task.py`, `MahutRobot`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Should `audio_recorder.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06882591093117409 - nodes in this community are weakly interconnected._