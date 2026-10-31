"""Render a robot recorded by ``record_states_stair_step.py`` with MuJoCo's OpenGL renderer (no RT cores needed).

Kinematic replay, nothing is simulated here: every frame puts the recorded root pose and joint positions (the
four-bar's passive joints and the wheel angles included) on the MuJoCo model of v01_2_2 generated from the
training USD (``cosim-anaylze/tools/build_flamingo_light_v01_2_2.py``; only its visual meshes are used), standing on
the env's own stair tile, rebuilt from the terrain generator's seed like Isaac Lab's
``inverted_pyramid_stairs_terrain``. The camera follows the robot from behind and above (``--cam_offset``).

``--pick best`` films the env that climbed the most risers, ``--pick fail`` one that did not climb 3 consecutive
risers (the most common end among those, its median climb), ``--env`` a given one.

    MUJOCO_GL=egl python scripts/co_rl/render_states_mujoco.py --states <dir>/states_L9.npz --pick best \\
        --robot_xml <assets>/xml/flamingo_light_v01_2_2.xml --out <dir>/L9_best.mp4

Needs mujoco, numpy, pillow and imageio-ffmpeg.
"""

import argparse
import json
import os
import re
from collections import Counter

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--states", required=True)
ap.add_argument("--robot_xml", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pick", choices=("best", "fail"), default="best")
ap.add_argument("--env", type=int, default=None)
ap.add_argument("--title", default=None, help="first overlay line (default: the checkpoint's run and iteration)")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--hold", type=float, default=1.0, help="seconds to hold the last frame")
ap.add_argument("--cam_offset", type=float, nargs=3, default=(-1.3, -0.9, 0.9),
                help="camera position relative to the robot's CoM [m], world axes (the robots climb toward +x)")
ap.add_argument("--cam_target", type=float, nargs=3, default=(0.35, 0.0, -0.05),
                help="point the camera looks at, relative to the robot's CoM [m]")
a = ap.parse_args()

S = np.load(a.states, allow_pickle=False)
T = json.loads(str(S["terrain_json"]))
risers = S["risers"]
ends = [str(e) for e in S["end"]]
if a.env is not None:
    env = a.env
elif a.pick == "best":
    env = int(np.lexsort((S["end_step"], -risers))[0])  # most risers, then the earliest finisher
else:
    fails = [j for j in range(len(risers)) if risers[j] < 2.9]
    if not fails:
        raise SystemExit(f"{a.states}: every env climbed >= 3 risers, nothing to film with --pick fail")
    common = Counter(ends[j] for j in fails).most_common(1)[0][0]
    group = sorted((j for j in fails if ends[j] == common), key=lambda j: risers[j])
    env = group[len(group) // 2]
row, col = int(S["row"][env]), int(S["col"][env])


def tile_geometry(row: int, col: int) -> tuple[float, float]:
    """Riser and tread of tile (row, col): TerrainGenerator draws (row + U) / rows per tile, columns outer."""
    rng = np.random.default_rng(T["seed"])
    lo, hi = T["difficulty_range"]
    diff = np.zeros((T["num_cols"], T["num_rows"]))
    for c in range(T["num_cols"]):
        for r in range(T["num_rows"]):
            diff[c, r] = lo + (hi - lo) * (r + rng.uniform()) / T["num_rows"]
    d = min(diff[col, row] * T["difficulty_scale"], 1.0)
    (h0, h1), (w0, w1) = T["step_height_range"], T["step_width_range"]
    return h0 + d * (h1 - h0), w0 + d * (w1 - w0)


def stair_boxes(h: float, w: float) -> list[tuple[tuple, tuple, int]]:
    """(centre, half size, step index) of inverted_pyramid_stairs_terrain's boxes, pit floor at z = 0."""
    tile, border = T["size"][0], T["border_width"]
    size = tile - 2 * border
    n = int((tile - 2 * border - T["platform_width"]) // (2 * w) + 1)
    total = (n + 1) * h
    out = []
    for k in range(n):
        bz, bh, off, sx = -total / 2 - (k + 1) * h / 2 + total, total - (k + 1) * h, (k + 0.5) * w, size - 2 * k * w
        for y in (size / 2 - off, -size / 2 + off):
            out.append(((0.0, y, bz), (sx / 2, w / 2, bh / 2), n - k))
        for x in (size / 2 - off, -size / 2 + off):
            out.append(((x, 0.0, bz), (w / 2, (sx - 2 * w) / 2, bh / 2), n - k))
    mid = size - 2 * n * w
    out.append(((0.0, 0.0, -h / 2), (mid / 2, mid / 2, h / 2), 0))
    for s in (1, -1):
        out.append(((0.0, s * (tile / 2 - border / 2), total - h / 2), (tile / 2, border / 2, h / 2), n + 1))
        out.append(((s * (tile / 2 - border / 2), 0.0, total - h / 2), (border / 2, tile / 2 - border, h / 2), n + 1))
    return out


h, w = tile_geometry(row, col)
xml = open(a.robot_xml).read()
mesh_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.robot_xml))), "mesh")
xml = xml.replace('file="../mesh/', f'file="{mesh_dir}/')
xml = re.sub(r'<geom name="[^"]*_collision"[^>]*/>', "", xml)  # visual meshes only
xml = xml.replace('rgba="0.75 0.78 0.82 1"', 'rgba="0.30 0.40 0.58 1"', 1)  # the visual class: stand out on white treads
xml = xml.replace("<compiler", f'<visual><global offwidth="{a.width}" offheight="{a.height}" azimuth="90" elevation="-15"/>'
                  '<quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55"/></visual>\n'
                  "  <compiler", 1)
shades = ["0.80 0.80 0.78 1", "0.62 0.66 0.72 1"]
boxes = "\n".join(f'<geom type="box" pos="{c[0]} {c[1]} {c[2]}" size="{s[0]} {s[1]} {s[2]}" rgba="{shades[k % 2]}" '
                  'contype="0" conaffinity="0" />' for c, s, k in stair_boxes(h, w))
start = xml.index('<geom name="ground"')
xml = (xml[:start] + '<light directional="true" pos="2 -3 5" dir="-0.3 0.5 -1" castshadow="true" diffuse="0.6 0.6 0.6"/>\n'
       + boxes + xml[xml.index("/>", start) + 2:])
xml = xml.replace("</asset>", '<texture type="skybox" builtin="gradient" rgb1="0.75 0.85 0.95" rgb2="0.95 0.97 1" '
                  'width="512" height="512"/>\n  </asset>', 1)
# camera: a fixed offset from the robot's subtree CoM with a world-fixed orientation, behind (toward the pit centre,
# which is always lower: a lateral camera ends up inside the higher rings of the inverted pyramid) and above
off, tgt = np.array(a.cam_offset), np.array(a.cam_target)
view = (tgt - off) / np.linalg.norm(tgt - off)
right = np.cross(view, [0.0, 0.0, 1.0])
right /= np.linalg.norm(right)
up = np.cross(right, view)
cam = (f'<camera name="view" mode="trackcom" pos="{off[0]} {off[1]} {off[2]}" '
       f'xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} {up[0]:.4f} {up[1]:.4f} {up[2]:.4f}"/>')
xml = re.sub(r'(<body name="base_link"[^>]*>)', r"\1" + cam, xml, count=1)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
qadr = {m.joint(i).name: m.jnt_qposadr[i] for i in range(m.njnt)}
names = [str(x) for x in S["joint_names"]]
pos, quat, jp, climb = S["pos"][:, env], S["quat"][:, env], S["joint_pos"][:, env], S["climb"][:, env]
last = int(S["end_step"][env])
dt = float(S["step_dt"])
fps = int(round(1.0 / dt))

renderer = mujoco.Renderer(m, a.height, a.width)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except OSError:
    font = small = ImageFont.load_default()
ck = str(S["checkpoint"])
title = a.title or f"{os.path.basename(os.path.dirname(ck))} / {os.path.basename(ck)}"
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
writer = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
for k in range(last + 1):
    d.qpos[:3], d.qpos[3:7] = pos[k], quat[k]
    for i, n in enumerate(names):
        if n in qadr:
            d.qpos[qadr[n]] = jp[k, i]
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera="view")
    img = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(img)
    draw.text((24, 18), title, fill=(20, 20, 20), font=font)
    draw.text((24, 54), f"riser {h * 100:.1f} cm, tread {w * 100:.0f} cm (row {row}, col {col}), spawn yaw "
              f"{float(S['yaw'][env]):+.2f} rad", fill=(20, 20, 20), font=small)
    draw.text((24, 82), f"t {k * dt:5.2f} s   risers climbed {int(np.floor(climb[k] + 0.1))}", fill=(20, 20, 20), font=small)
    if k == last:
        draw.text((24, 110), f"end: {ends[env]} at {last * dt:.2f} s, {risers[env]:.1f} risers", fill=(150, 20, 20), font=small)
    frame = np.asarray(img)
    for _ in range(1 if k < last else max(1, int(a.hold * fps))):
        writer.append_data(frame)
writer.close()
renderer.close()
print(json.dumps({"out": a.out, "env": env, "row": row, "col": col, "riser_m": round(h, 4), "tread_m": round(w, 4),
                  "risers": round(float(risers[env]), 2), "end": ends[env], "seconds": round(last * dt, 2)}), flush=True)
os._exit(0)  # skip the EGL context teardown noise at interpreter exit
