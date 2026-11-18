#!/usr/bin/env python3
"""
MultiShapeRelationalTask kameralarını render eder ve yön tanımlarını görsel olarak doğrular.

Front (sabit) kamera görüntüsüne referans nesnenin 5 yön hedefi (FRONT/BACK/LEFT/RIGHT/TOP)
kamera iç/dış parametreleriyle izdüşürülüp etiketlenir; yanına bilek kamerası eklenir.
Beklenen: RIGHT etiketi referansın görüntüde SAĞINDA, FRONT robota doğru (görüntüde yukarı).
Ek (G5): çalışma alanı köşeleri ve kol TCP'si front görüntüsünün içinde mi; scripted politika
kavrama anına kadar oynatılıp bilek kamerasında manipüle nesne görünüyor mu (nesne maskesi).

Kullanım:
    python3 scripts/render_task_cameras.py --out docs/vla_kameralar.png
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mahut_kol.envs  # noqa: F401,E402
import mahut_kol.robots  # noqa: F401,E402
from mahut_kol.envs.multi_shape_task import (  # noqa: E402
    RELATIONS,
    TABLETOP_HALF_HEIGHT,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
)
from mahut_kol.envs.scripted_policy import MultiShapeScriptedPolicy  # noqa: E402


def project(K, E, p):
    pc = E[:3, :3] @ p + E[:3, 3]
    uv = K @ pc
    return uv[:2] / uv[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/vla_kameralar.png")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()

    env = gym.make(
        "MahutKol-MultiShapeRelational-v1",
        obs_mode="rgb+segmentation",
        control_mode="pd_joint_pos",
        render_backend="cpu",
        render_mode="rgb_array",
    )
    obs, _ = env.reset(
        seed=args.seed,
        options=dict(manipulated_object="blue_cube", reference_object="red_cylinder", relation="RIGHT"),
    )
    u = env.unwrapped
    s = args.scale
    front = Image.fromarray(obs["sensor_data"]["front"]["rgb"][0].numpy()).resize((224 * s, 224 * s), Image.LANCZOS)
    wrist = Image.fromarray(obs["sensor_data"]["wrist"]["rgb"][0].numpy()).resize((224 * s, 224 * s), Image.LANCZOS)
    Kc = obs["sensor_param"]["front"]["intrinsic_cv"][0].numpy()
    Ec = obs["sensor_param"]["front"]["extrinsic_cv"][0].numpy()
    d = ImageDraw.Draw(front)
    ref = u.get_reference_object().pose.p[0].numpy()
    r0 = project(Kc, Ec, ref) * s
    d.ellipse([r0[0] - 5, r0[1] - 5, r0[0] + 5, r0[1] + 5], outline=(255, 255, 255), width=2)
    d.text((r0[0] + 6, r0[1] - 6), "REF", fill=(255, 255, 255))
    targets = u.get_all_relational_target_positions()
    for rel in RELATIONS:
        uv = project(Kc, Ec, targets[rel][0].numpy()) * s
        d.ellipse([uv[0] - 4, uv[1] - 4, uv[0] + 4, uv[1] + 4], fill=(255, 255, 0))
        d.text((uv[0] + 6, uv[1] + 2), rel, fill=(255, 255, 0))
    # Doğrulama: RIGHT hedefi referansın sağında mı?
    ur = project(Kc, Ec, targets["RIGHT"][0].numpy())[0]
    ul = project(Kc, Ec, targets["LEFT"][0].numpy())[0]
    vf = project(Kc, Ec, targets["FRONT"][0].numpy())[1]
    vb = project(Kc, Ec, targets["BACK"][0].numpy())[1]
    ok = ur > ul and vf < vb
    # Çalışma alanı köşeleri (masa ve nesne üstü) ve TCP front görüntüsünde mi
    corners = [(x, y, z) for x in (WORKSPACE_X_MIN, WORKSPACE_X_MAX) for y in (WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
               for z in (0.0, 2 * TABLETOP_HALF_HEIGHT)]
    inside = lambda uv: 0 <= uv[0] < 224 and 0 <= uv[1] < 224  # noqa: E731
    ws_in = all(inside(project(Kc, Ec, np.array(c))) for c in corners)
    tcp_in = inside(project(Kc, Ec, u.agent.tcp.pose.p[0].numpy()))
    for c in corners:
        uv = project(Kc, Ec, np.array(c)) * s
        d.rectangle([uv[0] - 3, uv[1] - 3, uv[0] + 3, uv[1] + 3], outline=(0, 255, 255))
    ok = ok and ws_in and tcp_in
    d.text((8, 8), "front (sabit kamera)", fill=(255, 255, 255))
    ImageDraw.Draw(wrist).text((8, 8), "wrist (bilek kamerasi)", fill=(255, 255, 255))
    # Kavrama anı: scripted politika "close" fazının sonuna kadar
    pol = MultiShapeScriptedPolicy(u)
    pol.reset(u.agent.robot.get_qpos()[0].numpy())
    grasp_seen = None
    if pol.plan_error is None:
        for _ in range(pol.phase_end("close")):
            obs, *_ = env.step(torch.as_tensor(pol.act(u.agent.robot.get_qpos()[0].numpy())))
        seg = obs["sensor_data"]["wrist"]["segmentation"][0, ..., 0].numpy()
        frac = float((seg == u.get_manipulated_object().per_scene_id[0].item()).mean())
        grasp_seen = frac
        ok = ok and frac > 0.01
    wrist_g = Image.fromarray(obs["sensor_data"]["wrist"]["rgb"][0].numpy()).resize((224 * s, 224 * s), Image.LANCZOS)
    front_g = Image.fromarray(obs["sensor_data"]["front"]["rgb"][0].numpy()).resize((224 * s, 224 * s), Image.LANCZOS)
    ImageDraw.Draw(wrist_g).text((8, 8), "wrist - kavrama ani", fill=(255, 255, 255))
    ImageDraw.Draw(front_g).text((8, 8), "front - kavrama ani", fill=(255, 255, 255))
    canvas = Image.new("RGB", (224 * s * 2, 224 * s * 2))
    canvas.paste(front, (0, 0))
    canvas.paste(wrist, (224 * s, 0))
    canvas.paste(front_g, (0, 224 * s))
    canvas.paste(wrist_g, (224 * s, 224 * s))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    human = env.render()[0].numpy()
    base, ext = os.path.splitext(args.out)
    Image.fromarray(human).save(base + "_genel" + ext)
    print(f"Kaydedildi: {args.out}, {base}_genel{ext}")
    print(f"RIGHT görüntüde LEFT'in sağında: {ur > ul}; FRONT görüntüde BACK'in üstünde (robota doğru): {vf < vb}")
    print(f"Çalışma alanı köşeleri front görüntüsünde: {ws_in}; TCP front görüntüsünde: {tcp_in}")
    print(f"Kavrama anında manipüle nesne bilek görüntüsünün oranı: "
          f"{'plan yok: ' + str(pol.plan_error) if grasp_seen is None else f'{grasp_seen:.1%}'}")
    print("SONUÇ:", "GEÇTİ" if ok else "KALDI")
    env.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
