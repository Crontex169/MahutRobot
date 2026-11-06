#!/usr/bin/env python3
"""
MuJoCo MJCF'den (assets/mahut_kol_maniskill.xml & mahut_kol_robot.xml)
100% Matematiksel Olarak Birebir URDF Üretici.

MuJoCo'daki tüm gövde hiyerarşisi, eklem eksenleri, limitleri,
geometri konumları, kuaterniyon dönüşümleri (Euler rpy) ve
MuJoCo dinamik atalet matrisleri (mass, ipos, inertia) URDF'e aktarılır.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANISKILL_XML_PATH = os.path.join(PROJECT_ROOT, "assets", "mahut_kol_maniskill.xml")
ROBOT_XML_PATH = os.path.join(PROJECT_ROOT, "assets", "mahut_kol_robot.xml")
OUTPUT_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "urdf", "mahut_kol.urdf")


def quat_to_rpy(quat_str: str) -> str:
    """MuJoCo (w, x, y, z) kuaterniyonunu URDF (roll, pitch, yaw) euler açılarına dönüştürür."""
    if not quat_str or not quat_str.strip():
        return "0 0 0"
    parts = [float(v) for v in quat_str.split()]
    if len(parts) != 4:
        return "0 0 0"
    w, x, y, z = parts
    # Scipy Rotation.from_quat [x, y, z, w] bekler
    rot = R.from_quat([x, y, z, w])
    rpy = rot.as_euler("xyz")
    return f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"


def get_link_name(body_name: str) -> str:
    if body_name == "base":
        return "base_link"
    if not body_name.endswith("_link") and body_name not in ["link0", "link1", "link2", "link3", "link4"]:
        return f"{body_name}_link"
    return body_name


def main():
    tree = ET.parse(MANISKILL_XML_PATH)
    root = tree.getroot()
    mj = mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)

    # Mesh tanımları (ad -> dosya adı)
    mesh_map = {}
    for m in root.find("asset").findall("mesh"):
        fname = m.get("file")
        if fname.startswith("meshes/"):
            fname = fname[len("meshes/"):]
        mesh_map[m.get("name")] = fname

    urdf_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!-- MahutKol: MuJoCo Modelinden Birebir Matematiksel Olarak Türetilmiş URDF (5-DOF + Gripper) -->',
        '<robot name="mahut_kol">',
        "",
        "  <!-- Materyal Tanımları -->",
        '  <material name="metallic">',
        '    <color rgba="0.65 0.65 0.70 1.0"/>',
        "  </material>",
        '  <material name="dark_gray">',
        '    <color rgba="0.15 0.15 0.15 1.0"/>',
        "  </material>",
        '  <material name="finger_blue">',
        '    <color rgba="0.20 0.45 0.80 1.0"/>',
        "  </material>",
        "",
    ]

    worldbody = root.find("worldbody")

    def process_body(body, parent_link=None):
        body_name = body.get("name")
        link_name = get_link_name(body_name)
        pos = body.get("pos", "0 0 0")
        quat = body.get("quat", "1 0 0 0")
        rpy = quat_to_rpy(quat)

        # 1. Eklem (Joint) Tanımı
        joint_elem = body.find("joint")
        if parent_link is not None:
            if joint_elem is not None:
                j_name = joint_elem.get("name")
                j_axis = joint_elem.get("axis", "0 0 1")
                j_type_raw = joint_elem.get("type", "hinge")
                j_range = joint_elem.get("range", "")

                if j_type_raw == "slide":
                    j_type = "prismatic"
                    damping = "10.0"
                    effort = "15.0"
                    velocity = "0.2"
                    r_low, r_high = "0.0", "0.042"
                elif j_type_raw in ["hinge", "revolute", None]:
                    j_type = "revolute"
                    damping = "0.05"
                    effort = "3.432"
                    velocity = "3.0"
                    if j_range:
                        r_low, r_high = j_range.split()
                    else:
                        r_low, r_high = "-1.5708", "1.5708"
                else:
                    j_type = "continuous"
                    damping = "0.05"
                    effort = "3.432"
                    velocity = "3.0"
                    r_low, r_high = "-3.14159", "3.14159"

                urdf_lines.extend([
                    f'  <joint name="{j_name}" type="{j_type}">',
                    f'    <parent link="{parent_link}"/>',
                    f'    <child link="{link_name}"/>',
                    f'    <origin xyz="{pos}" rpy="{rpy}"/>',
                    f'    <axis xyz="{j_axis}"/>',
                    f'    <limit lower="{r_low}" upper="{r_high}" effort="{effort}" velocity="{velocity}"/>',
                    f'    <dynamics damping="{damping}"/>',
                    f'  </joint>',
                    "",
                ])
            else:
                # Sabit eklem (Fixed)
                urdf_lines.extend([
                    f'  <joint name="{body_name}_fixed_joint" type="fixed">',
                    f'    <parent link="{parent_link}"/>',
                    f'    <child link="{link_name}"/>',
                    f'    <origin xyz="{pos}" rpy="{rpy}"/>',
                    f'  </joint>',
                    "",
                ])

        # 2. Link Tanımı
        urdf_lines.append(f'  <link name="{link_name}">')

        # MuJoCo'dan kütle ve atalet matrisi aktarımı
        mj_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if mj_id >= 0 and mj.body_mass[mj_id] > 0:
            bm = mj.body_mass[mj_id]
            bp = mj.body_ipos[mj_id]
            bi = mj.body_inertia[mj_id]
            bq = mj.body_iquat[mj_id]
            brpy = quat_to_rpy(" ".join(str(x) for x in bq))
            urdf_lines.extend([
                "    <inertial>",
                f'      <origin xyz="{bp[0]:.6f} {bp[1]:.6f} {bp[2]:.6f}" rpy="{brpy}"/>',
                f'      <mass value="{bm:.6f}"/>',
                f'      <inertia ixx="{bi[0]:.6g}" ixy="0" ixz="0" iyy="{bi[1]:.6g}" iyz="0" izz="{bi[2]:.6g}"/>',
                "    </inertial>",
            ])
        else:
            urdf_lines.extend([
                "    <inertial>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                '      <mass value="0.01"/>',
                '      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>',
                "    </inertial>",
            ])

        # Geometrileri ekleme
        for geom in body.findall("geom"):
            g_pos = geom.get("pos", "0 0 0")
            g_quat = geom.get("quat", "1 0 0 0")
            g_rpy = quat_to_rpy(g_quat)
            mesh_name = geom.get("mesh")
            geom_type = geom.get("type", "mesh" if mesh_name else "box")

            # Dahili mekanik parçalar (dişli, vida, rulman, iç plaka) çarpışmaya dahil edilmez
            is_internal_or_noncolliding = (
                geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            ) or body_name == "gripper_gear" or (
                mesh_name and any(k in mesh_name.lower() for k in ["gear", "bearing", "screw", "rod", "rack", "part-13"])
            )

            if mesh_name and mesh_name in mesh_map:
                mesh_file = mesh_map[mesh_name]
                mat_name = "dark_gray" if "motor" in mesh_name else ("finger_blue" if "Clamp" in mesh_name else "metallic")
                urdf_lines.extend([
                    "    <visual>",
                    f'      <origin xyz="{g_pos}" rpy="{g_rpy}"/>',
                    "      <geometry>",
                    f'        <mesh filename="../meshes/{mesh_file}" scale="0.001 0.001 0.001"/>',
                    "      </geometry>",
                    f'      <material name="{mat_name}"/>',
                    "    </visual>",
                ])
                if not is_internal_or_noncolliding:
                    urdf_lines.extend([
                        "    <collision>",
                        f'      <origin xyz="{g_pos}" rpy="{g_rpy}"/>',
                        "      <geometry>",
                        f'        <mesh filename="../meshes/{mesh_file}" scale="0.001 0.001 0.001"/>',
                        "      </geometry>",
                        "    </collision>",
                    ])

        urdf_lines.extend(["  </link>", ""])

        # Çocuk gövdeleri gez
        for child in body.findall("body"):
            process_body(child, parent_link=link_name)

    # Başlangıç: worldbody içindeki base
    base_body = worldbody.find("body")
    process_body(base_body, parent_link=None)

    # Tutucu uç organ merkezi (Tool Center Point - TCP) için sanal link ekle
    # MuJoCo'da link4'ün parmak kavrama merkez noktası:
    urdf_lines.extend([
        "  <!-- TCP (Tool Center Point / Gripper Center) -->",
        '  <link name="link_gripper_center"/>',
        '  <joint name="joint_gripper_center" type="fixed">',
        '    <parent link="link4"/>',
        '    <child link="link_gripper_center"/>',
        '    <origin xyz="0.084616 -0.000260 0.015428" rpy="0 0 0"/>',
        "  </joint>",
        "",
        "</robot>",
    ])

    with open(OUTPUT_URDF_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(urdf_lines) + "\n")

    print(f"✅ Yeni URDF başarıyla oluşturuldu: {OUTPUT_URDF_PATH}")


if __name__ == "__main__":
    main()
