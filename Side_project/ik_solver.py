#!/usr/bin/env python3
"""
SO-100 (5-DoF + Gripper) Öncelikli Damped Least Squares (DLS) Ters Kinematik (IK) Çözücü

SO-100 robot kolu 5 aktif serbestlik derecesine (DoF) ve 1 adet kıskaç (gripper)
eklemine sahiptir:
  - Joint 0: Rotation    (Taban Yaw / Gövde dönüşü)
  - Joint 1: Pitch       (Omuz Eğimi)
  - Joint 2: Elbow       (Dirsek Eğimi)
  - Joint 3: Wrist_Pitch (Bilek Eğimi)
  - Joint 4: Wrist_Roll  (Bilek Dönüşü / Eksenel Roll)
  - Joint 5: Jaw         (Gripper / Kıskaç)

5-DoF Kilitlenmesini Önleyen Nullspace (Boşuzay) Projeksiyon Yaklaşımı:
  1. Birincil Görev (Primary Task): Uç nokta (End-Effector) 3B konumu (X, Y, Z).
     Damped Least Squares ile J_pos_pinv hesaplanır ve pozisyon hatası sıfırlanır.
  2. Nullspace Projeksiyonu: N = I - J_pos_pinv * J_pos matrisi oluşturulur.
  3. İkincil Görev (Secondary Task): Bilek eğimi (Pitch) ve bilek dönüşü (Roll),
     pozisyonu HİÇBİR ŞEKİLDE bozmadan nullspace içinde takip edilir.
  4. Yanal sapma (Yaw) serbest bırakılır; taban dönüşü pozisyona göre otomatik ayarlanır.
"""

from typing import Optional, Tuple
import numpy as np
import mujoco


class SO100IKSolver:
    """MuJoCo SO-100 robotu için Nullspace-Projected Damped Least Squares IK Çözücü."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "ee_site",
        body_name: str = "Fixed_Jaw",
        damping_pos: float = 1e-3,
        step_size: float = 0.8,
        w_pitch: float = 0.5,
        w_roll: float = 0.8,
    ):
        self.model = model
        self.damping_pos = damping_pos
        self.step_size = step_size
        self.w_pitch = w_pitch
        self.w_roll = w_roll

        # Eklemler ve limitler (sadece kol eklemleri 0..4, kıskaç hariç)
        self.num_arm_joints = 5
        self.q_min = model.jnt_range[: self.num_arm_joints, 0].copy()
        self.q_max = model.jnt_range[: self.num_arm_joints, 1].copy()

        # Uç nokta tespiti (önce site, yoksa body)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' modelde bulunamadı!")

        # Dahili hesaplama için geçici MjData kopyası
        self._ik_data = mujoco.MjData(model)

    def get_ee_pose(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """Uç noktanın güncel dünya koordinatlarındaki pozisyon ve rotasyon matrisini döner."""
        if self.site_id >= 0:
            pos = data.site_xpos[self.site_id].copy()
            rot = data.site_xmat[self.site_id].reshape(3, 3).copy()
        else:
            pos = data.xpos[self.body_id].copy()
            rot = data.xmat[self.body_id].reshape(3, 3).copy()
        return pos, rot

    def solve(
        self,
        target_pos: np.ndarray,
        target_pitch: Optional[float] = None,
        target_roll: Optional[float] = None,
        q_init: Optional[np.ndarray] = None,
        n_iters: int = 5,
        tolerance: float = 1e-4,
    ) -> Tuple[np.ndarray, bool]:
        """Nullspace Projeksiyonlu DLS ile hedef pozisyon ve ikincil açılara göre eklem açılarını hesaplar.

        Args:
            target_pos: Hedef 3B konum [x, y, z] (metre)
            target_pitch: Hedef bilek eğimi açısı (radyan, isteğe bağlı)
            target_roll: Hedef bilek eksenel dönüş açısı (radyan, isteğe bağlı)
            q_init: Başlangıç eklem açıları (5 elemanlı dizi).
            n_iters: Maksimum DLS iterasyon sayısı (varsayılan: 5).
            tolerance: Pozisyon hata yakınsama eşiği (metre).

        Returns:
            Tuple[np.ndarray, bool]: (Hesaplanan 5 eklem açısı, Başarılı yakınsama durumu)
        """
        if q_init is not None:
            q = np.array(q_init[: self.num_arm_joints], dtype=np.float64, copy=True)
        else:
            q = self._ik_data.qpos[: self.num_arm_joints].copy()

        q = np.clip(q, self.q_min, self.q_max)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        eye_task = (self.damping_pos ** 2) * np.eye(3)
        eye_joints = np.eye(self.num_arm_joints)

        converged = False

        for _ in range(n_iters):
            self._ik_data.qpos[: self.num_arm_joints] = q
            mujoco.mj_forward(self.model, self._ik_data)

            p_cur, _ = self.get_ee_pose(self._ik_data)
            e_pos = target_pos - p_cur

            if np.linalg.norm(e_pos) < tolerance:
                converged = True
                break

            # Jacobian
            if self.site_id >= 0:
                mujoco.mj_jacSite(self.model, self._ik_data, jacp, jacr, self.site_id)
            else:
                mujoco.mj_jacBody(self.model, self._ik_data, jacp, jacr, self.body_id)

            J_p = jacp[:, : self.num_arm_joints]

            # 1. Birincil Görev: Pozisyon Damped Pseudo-Inverse
            # J_pinv = J^T * (J * J^T + lambda^2 * I)^(-1)
            J_p_pinv = J_p.T @ np.linalg.inv(J_p @ J_p.T + eye_task)
            dq_pos = J_p_pinv @ e_pos

            # 2. Nullspace Projeksiyon Matrisi: N = I - J_pinv * J
            N = eye_joints - J_p_pinv @ J_p

            # 3. İkincil Görev: Bilek Pitch ve Roll Açılarını Nullspace İçinde Takip Et
            dq_sec = np.zeros(self.num_arm_joints)
            if target_pitch is not None:
                # Joint 3: Wrist_Pitch
                dq_sec[3] = self.w_pitch * (target_pitch - q[3])
            if target_roll is not None:
                # Joint 4: Wrist_Roll
                dq_sec[4] = self.w_roll * (target_roll - q[4])

            # Toplam Adım: dq = dq_pos + N * dq_sec
            dq = dq_pos + N @ dq_sec

            # Eklem açılarını güncelle ve sınırla
            q = np.clip(q + dq * self.step_size, self.q_min, self.q_max)

        return q, converged
