# 🤖 ManiSkill 3 Geliştirici ve Mimari Kılavuzu

**ManiSkill (SAPIEN & PhysX tabanlı)**, GPU hızlandırmalı robot manipülasyon simülasyonu ve robot öğrenme çatısıdır. Bu belge, resmi dokümantasyon ve API referansları taranarak projemiz için derlenmiştir.

* **Resmi Dokümantasyon:** [https://maniskill.readthedocs.io/](https://maniskill.readthedocs.io/)
* **Kaynak Kod (GitHub):** [https://github.com/haosulab/ManiSkill](https://github.com/haosulab/ManiSkill)
* **Özel Robot Eğitimi (Custom Robots):** [ManiSkill Custom Robots Tutorial](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html)
* **Görev Oluşturma (Task Building):** [ManiSkill Task Building Tutorial](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/task_building.html)

---

## 1. Temel Kavramlar & Fizik Motoru Mimarisi

* **SAPIEN & PhysX:** Arka planda NVIDIA PhysX motoru çalışır. Çoklu sahne paralelleştirmesini GPU üzerinde PyTorch tensörleri ile doğrudan paylaşır.
* **Actor vs. Articulation:**
  * `Actor`: Tekil serbest cisimler (örn: masa, küp, engel). Kinematik veya dinamik olabilir.
  * `Articulation`: Eklem ve link hiyerarşisine sahip gövdeler (robotlar, kıskaçlar, menteşeli dolaplar). Robotumuz `Articulation` olarak yüklenir.
* **Pose & Dönüşümler:** Pozisyonlar `[x, y, z]` metre cinsinden, rotasyonlar ise kuaterniyon `[w, x, y, z]` veya `[x, y, z, w]` formatında temsil edilir (SAPIEN kuaterniyon varsayılanı: `w, x, y, z`).

---

## 2. Özel Robot Tanımlama (`BaseAgent`)

Kendi robotunuzu (`MahutKol`) ManiSkill ortamına eklemek için `BaseAgent` sınıfı genişletilir ve `@register_agent` ile sisteme kaydedilir.

```python
from typing import Dict, List
import sapien
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

@register_agent()
class MahutRobot(BaseAgent):
    uid = "mahut_kol"
    urdf_path = "assets/urdf/mahut_kol.urdf"

    # Kol ve Gripper Eklem İsimleri (URDF ile birebir eşleşmeli)
    arm_joint_names: List[str] = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    gripper_joint_names: List[str] = ["finger_joint_left", "finger_joint_right"]
    ee_link_name: str = "link_gripper_center"

    @property
    def _controller_configs(self) -> Dict:
        """Robotun kontrolcüleri ve eylem uzayı (action space) tanımlanır."""
        return {
            "arm": PDJointPosControllerConfig(
                joint_names=self.arm_joint_names,
                lower=None,
                upper=None,
                stiffness=1000.0,  # Kp
                damping=100.0,     # Kd
                force_limit=100.0,
                normalize_action=True,
            ),
            "gripper": PDJointPosControllerConfig(
                joint_names=self.gripper_joint_names,
                lower=0.0,
                upper=0.04,
                stiffness=500.0,
                damping=50.0,
                force_limit=40.0,
                normalize_action=True,
            ),
        }

    @property
    def _sensor_configs(self) -> List[CameraConfig]:
        """Robota takılı (Eye-in-Hand / Bilek) kameralar."""
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=1.047,  # ~60 derece (radyan)
                mount=self.ee_link_name,
            )
        ]
```

### Desteklenen Kontrolcü Tipleri:
* `PDJointPosControllerConfig`: Hedef eklem açılarını doğrudan PD torklarına çevirir (En yaygın ve stabil yöntem).
* `PDJointVelControllerConfig`: Hedef eklem hızları.
* `PDEEPoseControllerConfig` / `PDEEPosControllerConfig`: Ters kinematik (IK) kullanarak End-Effector'ü hedef Kartezyen poza sürer.

---

## 3. Görev ve Ortam Tasarımı (`BaseEnv` / Task Building)

Görev ortamı tanımlanırken `BaseEnv` miras alınır ve `@register_env` dekoratörü kullanılır:

```python
import torch
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

@register_env("MahutKol-PickCube-v0", max_episode_steps=200)
class MahutPickCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["mahut_kol"]
    agent: MahutRobot

    def _load_scene(self, options: dict):
        """Statik ve dinamik aktörleri oluşturur."""
        # 1. Zemin
        self.scene.add_ground(altitude=0.0)
        
        # 2. Taşınacak Küp (Dinamik Aktör)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0, 1])
        self.cube = builder.build(name="cube")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Bölüm başı sıfırlama ve rassallık (domain randomization)."""
        b = len(env_idx)
        # Küpü rastgele (x, y) koordinatlarında masaya yerleştir
        xy = torch.rand((b, 2), device=self.device) * 0.2 + torch.tensor([0.2, -0.1], device=self.device)
        self.cube.set_pose(sapien.Pose(p=torch.cat([xy, torch.tensor([[0.02]], device=self.device)], dim=-1)))

    def evaluate(self):
        """Başarı kriteri (success condition)."""
        cube_pos = self.cube.pose.p
        goal_pos = torch.tensor([0.25, 0.20, 0.10], device=self.device)
        dist = torch.norm(cube_pos - goal_pos, dim=-1)
        success = dist < 0.03
        return {"success": success, "dist_to_goal": dist}
```

---

## 4. Gözlem Modları (Observation Modes)

ManiSkill 3'te `gym.make(env_id, obs_mode=...)` parametresi ile farklı gözlem türleri alınabilir:
* `"state"`: Robot qpos, qvel ve aktörlerin dünya koordinatları.
* `"rgb"`: Sahne ve bilek kameralarından sadece RGB tensörleri (`H, W, 3`).
* `"rgbd"`: RGB + Derinlik (Depth) haritaları (LeRobot için ideal veri kaynağı).
* `"pointcloud"`: 3D nokta bulutu tensörleri.

---

## 5. Kritik İpuçları & Sık Karşılaşılan Hatalar

1. **URDF Sabitleme (`fix_root_link`):** Robot kolunun gövdesi dünyada serbest düşmemeli, zemine/masaya sabitlenmelidir (`fix_root_link=True`).
2. **Atalet Değerleri:** Sıfır atalet (`ixx=0, iyy=0`) PhysX simülasyonunu anında patlatır veya NaN hatası verir.
3. **GPU Bellek Yönetimi:** ManiSkill çoklu paralel ortam (`num_envs=64, 128...`) destekler; kamera gözlemleri açıkken VRAM kullanımını kontrol altında tutmak için çözünürlüğü 128x128 veya 224x224 seçmek en iyisidir.
