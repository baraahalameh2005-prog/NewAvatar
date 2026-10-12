"""
Face comparison renders of existing GLBs (experimental; read-only on the GLBs).

Renders front and side face close-ups of the production-fit and adult-fit F1 and M2
bodies from var/adult_experiment/20261011_172629/, all at the same scale and with a
crop of the same size anchored to each head. Reuses render_front from
eye_texture_check.py; the side view rotates a copy of the vertices in memory.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\face_compare_render.py
Writes only var/face_compare/<timestamp>/.
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))
sys.path.insert(0, str(TOOLS))

import trimesh
from PIL import Image, ImageDraw, ImageFont

from app import config
from eye_texture_check import render_front

SOURCE = config.MODULE_ROOT / "var" / "adult_experiment" / "20261011_172629"
OUT = config.MODULE_ROOT / "var" / "face_compare" / time.strftime("%Y%m%d_%H%M%S")
MODELS = [
    ("F1 PRODUCTION", SOURCE / "production" / "F1_162.6cm_50kg.glb"),
    ("F1 ADULT FIT", SOURCE / "adult" / "F1_162.6cm_50kg.glb"),
    ("M2 PRODUCTION", SOURCE / "production" / "M2_175cm_72kg.glb"),
    ("M2 ADULT FIT", SOURCE / "adult" / "M2_175cm_72kg.glb"),
]
PX_PER_M = 2600
HALF_WIDTH = 0.13      # m, around the head centre
BELOW_TOP = 0.30       # m, from the top of the head down
SKIN = (200, 153, 126)


def views(vertices):
    """Front (camera on +z) and side (camera on the body's left, +x) vertex copies."""
    side = np.stack([-vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=1)
    return {"front": vertices, "side": side}


def crop_for(vertices):
    top = float(vertices[:, 1].max())
    head = vertices[vertices[:, 1] > top - 0.25]
    cx = float((head[:, 0].min() + head[:, 0].max()) / 2)
    return ((cx - HALF_WIDTH, top - BELOW_TOP), (cx + HALF_WIDTH, top + 0.02))


def labelled_row(images, labels):
    font = ImageFont.truetype("arial.ttf", 22)
    w = sum(i.size[0] for i in images) + 12 * (len(images) - 1)
    h = max(i.size[1] for i in images) + 40
    sheet = Image.new("RGB", (w, h), (30, 30, 34))
    d = ImageDraw.Draw(sheet)
    x = 0
    for img, label in zip(images, labels):
        sheet.paste(img, (x, 40))
        d.text((x + 8, 9), label, fill=(240, 240, 240), font=font)
        x += img.size[0] + 12
    return sheet


def main():
    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for _, p in MODELS}

    renders = {}
    for label, path in MODELS:
        mesh = trimesh.load(path, force="mesh", process=False)
        faces = np.asarray(mesh.faces)
        for view, v in views(np.asarray(mesh.vertices)).items():
            renders[(label, view)] = render_front(v, faces, None, None, crop_for(v), PX_PER_M, flat_rgb=SKIN)

    OUT.mkdir(parents=True)
    for view in ("front", "side"):
        for person in ("F1", "M2"):
            labels = [f"{person} PRODUCTION", f"{person} ADULT FIT"]
            labelled_row([renders[(l, view)] for l in labels], [f"{l} ({view})" for l in labels]).save(
                OUT / f"{person}_{view}.png")
        labels = [l for l, _ in MODELS]
        labelled_row([renders[(l, view)] for l in labels], [f"{l} ({view})" for l in labels]).save(
            OUT / f"all_{view}.png")

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for _, p in MODELS}
    print("Source GLBs unchanged:", before == after)
    print("Saved", OUT)
    for f in sorted(OUT.iterdir()):
        print(" ", f.name, Image.open(f).size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
