"""
Eye-texture diagnostic (experimental; does not change production code or the API).

The production GLB has one mesh, one primitive and one flat "skin" material, so the
two eyeballs (separate 72-vertex pieces of the mesh) render skin-coloured. This
prototype keeps exactly the same vertices, faces and vertex order and only adds:
  - TEXCOORD_0 for every vertex, and
  - one small generated base-colour image: a solid skin block (body and tongue
    point into it) and a procedural eye (sclera, iris, pupil) that the eyeballs
    are projected onto along each eye's forward direction.
The skin colour moves from the material factor into the image (factor = 1).

Eye proportions follow Anny/MPFB's procedural eye settings
(data/mpfb2/settings/eye_settings.default.json) in spirit; colours are fixed here.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\eye_texture_check.py
Writes only var/eye_check/<timestamp>/.
"""

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import fit_to_measurements as ftm  # applies the module cache settings first
from app import config

import trimesh
import anny
from anny.face_segmentation import get_face_segmentation_mask
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from generate_avatar import DEFAULT_SKIN_TONE, build, export_avatar, parse_skin_tone

CASES = {c["case"]: c for c in json.loads((BACKEND / "tools" / "baseline_cases.json").read_text())["cases"]}
BODIES = ["F1_162.6cm_50kg", "M2_175cm_72kg"]
REFERENCE_GLB = config.MODULE_ROOT / "var" / "measurement_check" / "after"
OUT_ROOT = config.MODULE_ROOT / "var" / "eye_check"

TEX_W, TEX_H = 256, 128           # left half: skin block, right half: eye
SKIN_UV = (0.25, 0.5)
EYE_UV_CENTRE = (0.75, 0.5)
EYE_UV_HALF = (0.25, 0.5)         # the right half is a 128 x 128 square
IRIS_RADIUS = 0.45                # fraction of the eyeball's projected radius
PUPIL_RADIUS = 0.33               # fraction of the iris radius (MPFB PupilSize 0.3)
SCLERA = (236, 232, 226)
IRIS = (92, 58, 34)               # brown
IRIS_EDGE = (46, 30, 20)          # darker limbal ring
PUPIL = (14, 12, 12)


# ------------------------------------------------------------------ geometry

def export_frame(vertices):
    """Same axis change and floor shift as generate_avatar.export_avatar."""
    v = np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)
    v[:, 1] -= v[:, 1].min()
    return v


def eyeballs(faces, n_vertices):
    """The two 72-vertex connected pieces, as {"L": idx, "R": idx} (L = +x)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_vertices, n_vertices))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    pieces = [np.where(labels == k)[0] for k in np.where(sizes == 72)[0]]
    if len(pieces) != 2:
        raise RuntimeError(f"expected two 72-vertex eyeballs, found {len(pieces)}")
    return pieces


def eye_uv(vertices, faces, pieces, front_masks):
    """Planar projection of each eyeball along its forward axis into the eye square."""
    uv = np.tile(np.array(SKIN_UV), (len(vertices), 1))
    info = {}
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    for idx in pieces:
        side = "L" if vertices[idx, 0].mean() > 0 else "R"
        front = front_masks[side]
        # forward axis: area-weighted mean normal of the eye_front faces of this eye
        axis = (mesh.face_normals[front] * mesh.area_faces[front, None]).sum(axis=0)
        axis /= np.linalg.norm(axis)
        lo, hi = vertices[idx].min(axis=0), vertices[idx].max(axis=0)
        centre = (lo + hi) / 2
        radius = float(np.linalg.norm(vertices[idx] - centre, axis=1).max())
        right = np.cross([0.0, 1.0, 0.0], axis)
        right /= np.linalg.norm(right)
        up = np.cross(axis, right)
        d = vertices[idx] - centre
        x, y, z = d @ right, d @ up, d @ axis
        local = np.stack([x, y], axis=1) / radius
        local[z < 0] = [0.95, 0.0]  # back hemisphere: sclera only
        uv[idx, 0] = EYE_UV_CENTRE[0] + EYE_UV_HALF[0] * local[:, 0]
        uv[idx, 1] = EYE_UV_CENTRE[1] + EYE_UV_HALF[1] * local[:, 1]
        info[side] = {"forward_axis": axis.tolist(), "centre_m": centre.tolist(), "radius_mm": radius * 1000,
                      "angle_from_straight_ahead_deg": float(np.degrees(np.arccos(np.clip(axis[2], -1, 1))))}
    return uv, info


def eye_texture(skin_tone):
    """RGB image: left half skin colour (sRGB, as entered), right half a procedural eye."""
    skin = tuple(parse_skin_tone(skin_tone)[:3])
    img = np.zeros((TEX_H, TEX_W, 3), dtype=np.uint8)
    img[:, : TEX_W // 2] = skin
    half = TEX_W // 2
    yy, xx = np.mgrid[0:TEX_H, 0:half]
    r = np.hypot((xx + 0.5 - half / 2) / (half / 2), (yy + 0.5 - TEX_H / 2) / (TEX_H / 2))
    eye = np.empty((TEX_H, half, 3))
    eye[:] = SCLERA
    t = np.clip((r - IRIS_RADIUS * 0.82) / (IRIS_RADIUS * 0.18), 0, 1)[..., None]
    iris = (1 - t) * np.array(IRIS) + t * np.array(IRIS_EDGE)
    eye[r <= IRIS_RADIUS] = iris[r <= IRIS_RADIUS]
    eye[r <= IRIS_RADIUS * PUPIL_RADIUS] = PUPIL
    img[:, half:] = eye.astype(np.uint8)
    return Image.fromarray(img, "RGB")


def export_textured(vertices_anny, faces, uv, image, path):
    """Same as export_avatar (axes, floor, smooth normals, single mesh) plus UVs and the image."""
    v = export_frame(vertices_anny)
    material = trimesh.visual.material.PBRMaterial(
        name="skin", baseColorTexture=image, baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.65)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
                           process=False)
    mesh.vertex_normals
    mesh.export(path, file_type="glb", include_normals=True)
    return mesh


# ------------------------------------------------------------------ GLB inspection

def glb_parts(path):
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    binary = data[20 + n + 8:]

    def accessor_bytes(i):
        acc = gltf["accessors"][i]
        view = gltf["bufferViews"][acc["bufferView"]]
        comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
        size = {5126: 4, 5125: 4, 5123: 2, 5121: 1}[acc["componentType"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return binary[start:start + acc["count"] * comps * size], acc

    prims = [p for m in gltf["meshes"] for p in m["primitives"]]
    out = {"gltf": gltf, "header_ok": data[:4] == b"glTF" and struct.unpack("<I", data[4:8])[0] == 2
           and struct.unpack("<I", data[8:12])[0] == len(data), "meshes": len(gltf["meshes"]), "primitives": len(prims)}
    p = prims[0]
    for key in ("POSITION", "NORMAL", "TEXCOORD_0"):
        if key in p["attributes"]:
            out[key] = accessor_bytes(p["attributes"][key])
    out["indices"] = accessor_bytes(p["indices"])
    return out


# ------------------------------------------------------------------ textured rendering (z-buffer, orthographic front)

def render_front(vertices, faces, uv, texture, crop, px_per_m, flat_rgb=None):
    """Per-pixel textured front view (camera on +z). flat_rgb renders an untextured mesh."""
    tex = None if texture is None else np.asarray(texture, dtype=float)
    lo, hi = np.array(crop[0]), np.array(crop[1])
    w, h = ((hi - lo) * px_per_m).astype(int)
    img = np.full((h, w, 3), (60.0, 60.0, 64.0))
    zbuf = np.full((h, w), -np.inf)
    light = np.array([0.25, 0.35, 1.0]) / np.linalg.norm([0.25, 0.35, 1.0])
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    px = (vertices[:, :2] - lo) * px_per_m
    px[:, 1] = h - px[:, 1]
    keep = (normals[:, 2] > 0) & (px[faces][:, :, 0].max(1) >= 0) & (px[faces][:, :, 0].min(1) < w) \
           & (px[faces][:, :, 1].max(1) >= 0) & (px[faces][:, :, 1].min(1) < h)
    for f in np.where(keep)[0]:
        a, b, c = px[faces[f]]
        x0, x1 = int(max(np.floor(min(a[0], b[0], c[0])), 0)), int(min(np.ceil(max(a[0], b[0], c[0])), w - 1))
        y0, y1 = int(max(np.floor(min(a[1], b[1], c[1])), 0)), int(min(np.ceil(max(a[1], b[1], c[1])), h - 1))
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        l0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / den
        l1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not inside.any():
            continue
        zs = vertices[faces[f], 2]
        z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]
        yi, xi = np.where(inside)
        zi = z[inside]
        ys, xs = yi + y0, xi + x0
        closer = zi > zbuf[ys, xs]
        if not closer.any():
            continue
        ys, xs, zi = ys[closer], xs[closer], zi[closer]
        shade = 0.3 + 0.7 * max(float(normals[f] @ light), 0.0)
        if flat_rgb is not None:
            col = np.tile(np.array(flat_rgb, dtype=float), (len(ys), 1))
        else:
            uvs = uv[faces[f]]
            w0, w1, w2 = l0[inside][closer], l1[inside][closer], l2[inside][closer]
            u = w0 * uvs[0, 0] + w1 * uvs[1, 0] + w2 * uvs[2, 0]
            v = w0 * uvs[0, 1] + w1 * uvs[1, 1] + w2 * uvs[2, 1]
            tx = np.clip((u * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
            ty = np.clip(((1 - v) * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
            col = tex[ty, tx]
        img[ys, xs] = col * shade
        zbuf[ys, xs] = zi
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def side_by_side(images, labels, path):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype("arial.ttf", 20)
    w = sum(i.size[0] for i in images) + 12 * (len(images) - 1)
    h = max(i.size[1] for i in images) + 36
    sheet = Image.new("RGB", (w, h), (30, 30, 34))
    d = ImageDraw.Draw(sheet)
    x = 0
    for img, label in zip(images, labels):
        sheet.paste(img, (x, 36))
        d.text((x + 8, 8), label, fill=(240, 240, 240), font=font)
        x += img.size[0] + 12
    sheet.save(path)


# ------------------------------------------------------------------ main

def main():
    out_dir = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    if out_dir.exists():
        sys.exit(f"Refusing to overwrite {out_dir}")

    model = anny.Anny()
    faces = model.faces.detach().cpu().numpy()
    front_masks = {s: get_face_segmentation_mask(model, [f"eye_front.{s}"]).cpu().numpy() for s in ("L", "R")}
    out_dir.mkdir(parents=True)
    results = {"settings": {"texture_px": [TEX_W, TEX_H], "iris_radius": IRIS_RADIUS, "pupil_radius": PUPIL_RADIUS,
                            "sclera": SCLERA, "iris": IRIS, "pupil": PUPIL, "skin_tone": DEFAULT_SKIN_TONE},
               "bodies": {}}
    image = eye_texture(DEFAULT_SKIN_TONE)
    image.save(out_dir / "eye_texture.png")

    for name in BODIES:
        p = CASES[name]["parameters"]
        output = build(model=model, gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"],
                       age=p["age"], muscle=p["muscle"], proportions=p["proportions"])
        rest = output["rest_vertices"][0].detach().cpu().numpy()
        v = export_frame(rest)
        pieces = eyeballs(faces, len(v))
        uv, info = eye_uv(v, faces, pieces, front_masks)

        base_glb = out_dir / f"{name}_baseline.glb"
        tex_glb = out_dir / f"{name}_eyes.glb"
        export_avatar(output, model, base_glb)          # unchanged production exporter
        export_textured(rest, faces, uv, image, tex_glb)

        a, b = glb_parts(base_glb), glb_parts(tex_glb)
        ref = glb_parts(REFERENCE_GLB / f"{name}.glb")
        mat = b["gltf"]["materials"][0]["pbrMetallicRoughness"]
        checks = {
            "positions_bytes_identical": a["POSITION"][0] == b["POSITION"][0],
            "normals_bytes_identical": a["NORMAL"][0] == b["NORMAL"][0],
            "indices_bytes_identical": a["indices"][0] == b["indices"][0],
            "positions_equal_reference_glb": a["POSITION"][0] == ref["POSITION"][0],
            "vertex_count": [a["POSITION"][1]["count"], b["POSITION"][1]["count"]],
            "index_count": [a["indices"][1]["count"], b["indices"][1]["count"]],
            "meshes_primitives": [[a["meshes"], a["primitives"]], [b["meshes"], b["primitives"]]],
            "textured_has_texcoord": "TEXCOORD_0" in b,
            "texcoord_count": b["TEXCOORD_0"][1]["count"] if "TEXCOORD_0" in b else None,
            "images": len(b["gltf"].get("images", [])),
            "base_color_texture": "baseColorTexture" in mat,
            "base_color_factor": mat.get("baseColorFactor"),
            "header_ok": [a["header_ok"], b["header_ok"]],
            "static_mesh": "skins" not in b["gltf"],
            "bytes": [base_glb.stat().st_size, tex_glb.stat().st_size],
            "eyeball_vertices": [len(x) for x in pieces],
        }
        results["bodies"][name] = {"eyes": info, "checks": checks}
        print(f"{name}: {json.dumps(checks)}", flush=True)
        print(f"  eyes: " + "; ".join(f"{s} axis {i['angle_from_straight_ahead_deg']:.1f} deg from straight ahead, "
                                      f"radius {i['radius_mm']:.1f} mm" for s, i in info.items()), flush=True)

        # renders: baseline (flat skin colour) vs textured, full body and face close-up
        skin_lin = np.array(parse_skin_tone(DEFAULT_SKIN_TONE)[:3], dtype=float)
        hgt = float(v[:, 1].max())
        head_top = hgt
        eye_y = float(v[np.concatenate(pieces), 1].mean())
        body_crop = ((-0.55, -0.02), (0.55, hgt + 0.03))
        face_crop = ((-0.11, eye_y - 0.12), (0.11, head_top + 0.01))
        eye_crop = ((-0.055, eye_y - 0.025), (0.055, eye_y + 0.025))
        for tag, crop, scale in (("body", body_crop, 380), ("face", face_crop, 3200), ("eyes", eye_crop, 7000)):
            base_img = render_front(v, faces, uv, None, crop, scale, flat_rgb=skin_lin)
            tex_img = render_front(v, faces, uv, image, crop, scale)
            side_by_side([base_img, tex_img], [f"{name} BASELINE (current export)", f"{name} EYE TEXTURE"],
                         out_dir / f"{name}_{tag}.png")

    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("Saved", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
