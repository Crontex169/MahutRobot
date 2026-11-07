#!/usr/bin/env python3
"""
MultiShapeRelationalTask kameralarını render eder ve yön tanımlarını görsel olarak doğrular.

Front (sabit) kamera görüntüsüne referans nesnenin 5 yön hedefi (FRONT/BACK/LEFT/RIGHT/TOP)
kamera iç/dış parametreleriyle izdüşürülüp etiketlenir; yanına bilek kamerası eklenir.
Beklenen: RIGHT etiketi referansın görüntüde SAĞINDA, FRONT robota doğru (görüntüde yukarı).

Kullanım:
    python3 scripts/render_task_cameras.py --out docs/vla_kameralar.png
"""

import argparse
import os
import sys

import gymnasium as gym
from PIL import Image, ImageDraw

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mahut_kol.envs  # noqa: F401,E402
import mahut_kol.robots  # noqa: F401,E402
from mahut_kol.envs.multi_shape_task import RELATIONS  # noqa: E402


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
        obs_mode="rgb",
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
    d.text((8, 8), "front (sabit kamera)", fill=(255, 255, 255))
    ImageDraw.Draw(wrist).text((8, 8), "wrist (bilek kamerasi)", fill=(255, 255, 255))
    canvas = Image.new("RGB", (224 * s * 2, 224 * s))
    canvas.paste(front, (0, 0))
    canvas.paste(wrist, (224 * s, 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    human = env.render()[0].numpy()
    base, ext = os.path.splitext(args.out)
    Image.fromarray(human).save(base + "_genel" + ext)
    print(f"Kaydedildi: {args.out}, {base}_genel{ext}")
    print(f"RIGHT görüntüde LEFT'in sağında: {ur > ul}; FRONT görüntüde BACK'in üstünde (robota doğru): {vf < vb}")
    env.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
