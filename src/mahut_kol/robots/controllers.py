"""
Özel Kontrolcü Tanımları ve Yardımcı Araçlar.

ManiSkill varsayılan olarak PDJointPos, PDEEPos gibi kontrolcüler sunar.
Bu modülde:
1. Kol için Ters Kinematik (IK) tabanlı delta pozisyon kontrolcüleri,
2. Özel tutucu (gripper) senkronizasyon kontrolcüleri,
3. Tork limit ve hız yumuşatma (smoothing / filtering) filtreleri
yapılandırılabilir.
"""

from typing import Dict, Any


def build_arm_controller_config(config_dict: Dict[str, Any]):
    """
    YAML dosyasından okunan parametrelerle kol kontrolcüsü yapılandırması üretir.

    Parametreler:
    - config_dict: YAML dosyasındaki 'controllers.arm' sözlüğü

    TODO:
    - Kontrolcü tipine göre ('pd_joint_pos' vs 'pd_ee_delta_pose') ManiSkill nesnesi döndürün.
    """
    raise NotImplementedError("TODO: build_arm_controller_config fonksiyonunu doldurun.")


def build_gripper_controller_config(config_dict: Dict[str, Any]):
    """
    YAML dosyasından okunan parametrelerle tutucu kontrolcüsü yapılandırması üretir.

    TODO:
    - Tutucunun parmak eklemlerini simetrik veya bağımsız sürecek kontrolcüyü döndürün.
    """
    raise NotImplementedError("TODO: build_gripper_controller_config fonksiyonunu doldurun.")
