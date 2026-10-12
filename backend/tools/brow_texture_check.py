"""
Female eyebrow prototype (experimental; approved 2026-10-12). Does not change production
code, the API, the frontend or any existing GLB.

Base: F1 with the approved state of the face prototype - eye texture, female lip tint
(lip_texture_check.LIP_SETTINGS["female"]) and mouth-scale-horiz-incr = -0.35 (63-label
cached model). Anny has no eyebrow geometry, mask or texture, so the brows are painted:

  - A brow "patch" (front-facing skin faces in a band above the eyes) is split into a
    second primitive of the same mesh. Both primitives use the SAME vertex accessors
    (POSITION, NORMAL, TEXCOORD_0), so vertices, normals and their order are unchanged;
    only the triangle list is split.
  - The patch has its own material (same PBR values) whose base colour texture is a brow
    image read through TEXCOORD_1, a frontal planar projection of the patch. The image is
    the exact skin colour with hair-like strokes; the strokes stay well inside the patch,
    so the patch edge matches the surrounding skin.

Brow placement is anatomical, relative to the eye centres (the eyebrows-* shape targets
move the whole forehead and cannot locate the brows).

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\brow_texture_check.py [--out DIR]
Default output: var/brow_check/<timestamp>/.
"""

import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import lip_texture_check as ltc
from lip_texture_check import etc
from eye_texture_check import ftm, config, anny, get_face_segmentation_mask, Image

from PIL import ImageDraw, ImageFilter
from adult_fit_experiment import REF, ControlledModel
from generate_avatar import DEFAULT_SKIN_TONE, build, parse_skin_tone
from measurements import BodyMeasurements

CASE = "F1_162.6cm_50kg"
MOUTH_WIDTH = -0.35
REFERENCE = config.MODULE_ROOT / "var" / "lip_shape_check" / "20261012_011832" / "F1_162.6cm_50kg_mouthwidth_-0.35.glb"

# patch (metres, relative to the mean eye centre; x symmetric)
PATCH_HALF_W = 0.075
PATCH_Y = (0.004, 0.046)
PATCH_MIN_NORMAL_Z = 0.15
# brow image: tile covers the patch box; resolution
TILE_PX = (1024, 288)
SUPERSAMPLE = 3

# brow design in mm, relative to the eye centre height and the face midline (one side; mirrored)
BROW_INNER = (12.0, 15.0)     # (x from midline, y above eye centre)
BROW_PEAK = (36.0, 20.0)
BROW_TAIL = (53.0, 15.5)
THICKNESS = (4.2, 3.4, 1.0)   # inner, peak, tail (mm)
HAIRS = 1000
HAIR_LEN = (2.2, 3.6)         # mm
BROW_RGB = np.array([68, 47, 35], dtype=float)   # dark brown (sRGB)
SEED = 7
# refinement 2026-10-12 (untidy first version): hairs clipped to a feathered outline,
# rounded soft inner end, shorter and flatter hairs at the head
CLIP_SCALE = 0.55             # outline half-width = thickness x this
CLIP_FEATHER_MM = 0.35
BASE_SCALE = 0.42
BASE_FILL = 80
BASE_BLUR_MM = 0.5
HEAD_T = 0.18                 # first 18 % of the brow length is the "head"
HEAD_ANGLE_DEG = (25.0, 12.0) # lift above the arch at the inner end -> end of the head
HEAD_LEN_FACTOR = 0.6


# ------------------------------------------------------------------ brow image

def brow_curve(t):
    """Quadratic-Bezier-like arch through inner -> peak -> tail; t in [0, 1]. Returns points (mm) and tangent."""
    p0, p1, p2 = (np.array(BROW_INNER), np.array(BROW_PEAK), np.array(BROW_TAIL))
    # control point so the curve passes through the peak at t = 0.5
    c = 2 * p1 - 0.5 * (p0 + p2)
    t = np.asarray(t)[..., None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2
    tan = 2 * (1 - t) * (c - p0) + 2 * t * (p2 - c)
    return pts, tan / np.linalg.norm(tan, axis=-1, keepdims=True)


def thickness(t):
    t = np.asarray(t)
    return np.where(t < 0.5, THICKNESS[0] + (THICKNESS[1] - THICKNESS[0]) * (t / 0.5),
                    THICKNESS[1] + (THICKNESS[2] - THICKNESS[1]) * (np.clip(t - 0.5, 0, None) / 0.5) ** 1.3)


def brow_image(skin_tone, tile_mm):
    """RGB tile: exact skin colour with hair strokes. tile_mm = (x0, x1, y0, y1) in mm (eye-centre frame)."""
    skin = np.array(parse_skin_tone(skin_tone)[:3], dtype=float)
    w, h = TILE_PX[0] * SUPERSAMPLE, TILE_PX[1] * SUPERSAMPLE
    x0, x1, y0, y1 = tile_mm
    sx, sy = w / (x1 - x0), h / (y1 - y0)

    def to_px(p):
        return ((p[..., 0] - x0) * sx, (y1 - p[..., 1]) * sy)

    def outline(scale):
        """Brow outline polygon (mm, one side, +x) with a rounded inner end; half-width = thickness x scale."""
        ts = np.linspace(0, 1, 200)
        pts, tan = brow_curve(ts)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        half = thickness(ts)[:, None] * scale
        upper, lower = pts + nrm * half, pts - nrm * half
        phi = np.linspace(0, np.pi, 24)[:, None]      # lower edge -> inner side -> upper edge
        cap = pts[0] + half[0] * (-nrm[0] * np.cos(phi) - tan[0] * np.sin(phi))
        return np.concatenate([upper, lower[::-1], cap])

    def fill(poly_mm, side, value, blur_mm):
        img = Image.new("L", (w, h), 0)
        p = poly_mm * [side, 1]
        px, py = to_px(p)
        ImageDraw.Draw(img).polygon(list(zip(px, py)), fill=value)
        return np.asarray(img.filter(ImageFilter.GaussianBlur(blur_mm * sx)), dtype=float)

    base_cov = np.zeros((h, w))      # soft fill so the brow reads as a whole
    clip = np.zeros((h, w))          # feathered outline that the hairs are kept inside
    hairs = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(hairs)
    rng = np.random.default_rng(SEED)
    for side in (1, -1):
        base_cov = np.maximum(base_cov, fill(outline(BASE_SCALE), side, BASE_FILL, BASE_BLUR_MM))
        clip = np.maximum(clip, fill(outline(CLIP_SCALE), side, 255, CLIP_FEATHER_MM))
        # hairs: denser toward the inner/body part; short, gently lifted at the head; along the arch elsewhere
        t = rng.beta(1.3, 1.6, HAIRS)
        pts, tan = brow_curve(t)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        off = (rng.random(HAIRS) - 0.5) * thickness(t) * 0.8
        start = pts + nrm * off[:, None]
        base_dir = np.arctan2(tan[:, 1], tan[:, 0])
        head = t < HEAD_T
        lift = np.radians(np.where(head, HEAD_ANGLE_DEG[0] - (HEAD_ANGLE_DEG[0] - HEAD_ANGLE_DEG[1]) * t / HEAD_T, 7.0))
        ang = base_dir + lift + rng.normal(0, 0.10, HAIRS)
        length = rng.uniform(*HAIR_LEN, HAIRS) * np.where(head, HEAD_LEN_FACTOR, np.where(t > 0.8, 0.75, 1.0))
        end = start + np.stack([np.cos(ang), np.sin(ang)], 1) * length[:, None]
        sx0, sy0 = to_px(start * [side, 1])
        ex0, ey0 = to_px(end * [side, 1])
        alpha = rng.uniform(150, 235, HAIRS)
        width = max(1, int(round(0.14 * sx)))
        for i in range(HAIRS):
            draw.line([(sx0[i], sy0[i]), (ex0[i], ey0[i])], fill=int(alpha[i]), width=width)
    hair_cov = np.asarray(hairs, dtype=float) * clip / 255.0
    full = Image.fromarray(np.clip(np.maximum(base_cov, hair_cov), 0, 255).astype(np.uint8))
    cov = np.asarray(full.resize(TILE_PX, Image.LANCZOS), dtype=float)[..., None] / 255.0
    img = skin * (1 - cov) + BROW_RGB * cov
    return Image.fromarray(np.clip(img.round(), 0, 255).astype(np.uint8), "RGB"), skin


# ------------------------------------------------------------------ GLB: add the brow primitive

def _pad(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def add_brow_primitive(src, dst, faces, brow_mask, uv1_gltf, png_bytes):
    data = Path(src).read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    binary = bytearray(data[20 + n + 8:20 + n + 8 + struct.unpack("<I", data[20 + n:24 + n])[0]])
    prim = gltf["meshes"][0]["primitives"][0]

    # the existing index buffer must be exactly the model's faces (process=False export)
    acc = gltf["accessors"][prim["indices"]]
    view = gltf["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    dtype = {5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
    existing = np.frombuffer(bytes(binary[start:start + acc["count"] * np.dtype(dtype).itemsize]), dtype=dtype)
    if not np.array_equal(existing.reshape(-1, 3), faces):
        raise RuntimeError("exported index buffer differs from the model faces")

    def add_view(blob, target=None):
        nonlocal binary
        binary = bytearray(_pad(bytes(binary)))
        off = len(binary)
        binary += blob
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(blob)}
        if target:
            view["target"] = target
        gltf["bufferViews"].append(view)
        return len(gltf["bufferViews"]) - 1

    uv = uv1_gltf.astype(np.float32)
    v_uv = add_view(uv.tobytes(), 34962)
    gltf["accessors"].append({"bufferView": v_uv, "componentType": 5126, "count": len(uv), "type": "VEC2",
                              "min": uv.min(0).tolist(), "max": uv.max(0).tolist()})
    a_uv = len(gltf["accessors"]) - 1
    acc_ids = []
    for part in (faces[~brow_mask], faces[brow_mask]):
        flat = part.astype(np.uint32).ravel()
        vi = add_view(flat.tobytes(), 34963)
        gltf["accessors"].append({"bufferView": vi, "componentType": 5125, "count": len(flat), "type": "SCALAR",
                                  "min": [int(flat.min())], "max": [int(flat.max())]})
        acc_ids.append(len(gltf["accessors"]) - 1)
    v_img = add_view(png_bytes)
    gltf.setdefault("images", []).append({"bufferView": v_img, "mimeType": "image/png", "name": "brows"})
    tex = {"source": len(gltf["images"]) - 1}
    if gltf.get("samplers"):
        tex["sampler"] = 0
    gltf.setdefault("textures", []).append(tex)
    mat = json.loads(json.dumps(gltf["materials"][prim["material"]]))
    mat["name"] = "skin_brows"
    mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(gltf["textures"]) - 1, "texCoord": 1}
    gltf["materials"].append(mat)

    attrs = dict(prim["attributes"], TEXCOORD_1=a_uv)
    old_indices = prim["indices"]
    gltf["meshes"][0]["primitives"] = [
        dict(prim, attributes=attrs, indices=acc_ids[0]),
        {"attributes": attrs, "indices": acc_ids[1], "material": len(gltf["materials"]) - 1, "mode": prim.get("mode", 4)},
    ]
    # drop the now-unused original index accessor's role (keep buffer data; mark nothing else)
    gltf["buffers"][0]["byteLength"] = len(_pad(bytes(binary)))
    js = _pad(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    bn = _pad(bytes(binary))
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(bn))
    out += struct.pack("<I", len(js)) + b"JSON" + js + struct.pack("<I", len(bn)) + b"BIN\x00" + bn
    Path(dst).write_bytes(out)
    return old_indices


def read_attr(path, name, prim_index=0):
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    binary = data[20 + n + 8:]
    prim = gltf["meshes"][0]["primitives"][prim_index]
    idx = prim["indices"] if name == "indices" else prim["attributes"][name]
    acc = gltf["accessors"][idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(binary[start:start + acc["count"] * comps * np.dtype(dt).itemsize], dtype=dt)
    return arr.reshape(acc["count"], comps) if comps > 1 else arr, gltf


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.MODULE_ROOT / "var" / "brow_check" / time.strftime("%Y%m%d_%H%M%S")))
    out_dir = Path(ap.parse_args().out)
    if out_dir.exists():
        sys.exit(f"Refusing to overwrite {out_dir}")

    p = etc.CASES[CASE]["parameters"]
    kw = dict(gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"], age=p["age"],
              muscle=p["muscle"], proportions=p["proportions"])
    model = anny.Anny(local_changes=REF["local_changes_list"])
    proxy = ControlledModel(model)
    proxy.controls = {l: 0.0 for l in REF["local_changes_list"]}
    proxy.controls["mouth-scale-horiz-incr"] = MOUTH_WIDTH
    output = build(model=proxy, **kw)
    out_dir.mkdir(parents=True)
    log_lines = []

    def log(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        log_lines.append(line)
        (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("Output:", out_dir)
    rest = output["rest_vertices"][0].detach().cpu().numpy()
    v = etc.export_frame(rest)
    faces = model.faces.detach().cpu().numpy()

    # same textures / TEXCOORD_0 as the approved lip-shape state
    settings = ltc.LIP_SETTINGS["female"]
    tex0 = ltc.lip_eye_texture(DEFAULT_SKIN_TONE, settings)
    front = {s: get_face_segmentation_mask(model, [f"eye_front.{s}"]).cpu().numpy() for s in ("L", "R")}
    eyes = etc.eyeballs(faces, len(v))
    eye_idx = np.concatenate(eyes)
    uv0, _ = etc.eye_uv(v, faces, eyes, front)
    raw = ltc.lip_weights(model, faces, len(v))
    raw[eye_idx] = 0.0
    wts = ltc.shape_weights(raw, settings)
    non_eye = np.setdiff1d(np.arange(len(v)), eye_idx)
    uv0[non_eye, 0] = ltc.strip_u(wts[non_eye])
    uv0[non_eye, 1] = 0.5

    # brow patch
    eye_c = v[eye_idx].mean(0)
    rel = v - [0.0, eye_c[1], 0.0]
    import trimesh
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    in_box = (np.abs(rel[:, 0]) <= PATCH_HALF_W) & (rel[:, 1] >= PATCH_Y[0]) & (rel[:, 1] <= PATCH_Y[1])
    in_box[eye_idx] = False
    in_box &= raw == 0
    brow_mask = in_box[faces].all(1) & (mesh.face_normals[:, 2] > PATCH_MIN_NORMAL_Z)
    patch_v = np.unique(faces[brow_mask])
    log(f"eye centre y {eye_c[1]:.4f} m; brow patch: {int(brow_mask.sum())} faces, {len(patch_v)} vertices")

    # TEXCOORD_1: frontal projection of the patch box into the tile (our v = up)
    x0, x1 = -PATCH_HALF_W, PATCH_HALF_W
    y0, y1 = PATCH_Y
    uv1 = np.zeros((len(v), 2))
    uv1[:, 0] = (rel[:, 0] - x0) / (x1 - x0)
    uv1[:, 1] = (rel[:, 1] - y0) / (y1 - y0)
    uv1 = np.clip(uv1, 0, 1)
    tile_mm = (x0 * 1000, x1 * 1000, y0 * 1000, y1 * 1000)
    brow_img, skin = brow_image(DEFAULT_SKIN_TONE, tile_mm)
    brow_img.save(out_dir / "brow_texture.png")
    buf = io.BytesIO()
    brow_img.save(buf, format="PNG")

    # seam check: texels under every patch-boundary vertex must be the exact skin colour
    edges = np.concatenate([faces[brow_mask][:, [0, 1]], faces[brow_mask][:, [1, 2]], faces[brow_mask][:, [2, 0]]])
    other = np.unique(faces[~brow_mask])
    boundary = np.intersect1d(patch_v, other)
    arr = np.asarray(brow_img, dtype=float)
    bx = np.clip((uv1[boundary, 0] * TILE_PX[0]).astype(int), 0, TILE_PX[0] - 1)
    by = np.clip(((1 - uv1[boundary, 1]) * TILE_PX[1]).astype(int), 0, TILE_PX[1] - 1)
    seam_err = float(np.abs(arr[by, bx] - skin).max())
    # also a 4-texel neighbourhood (texture filtering)
    seam_err_nb = 0.0
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            seam_err_nb = max(seam_err_nb, float(np.abs(arr[np.clip(by + dy, 0, TILE_PX[1] - 1),
                                                           np.clip(bx + dx, 0, TILE_PX[0] - 1)] - skin).max()))
    log(f"patch boundary vertices {len(boundary)}; brow texture vs skin at boundary: max {seam_err:.1f} "
        f"(+-2 texels {seam_err_nb:.1f}) levels")

    base_glb = out_dir / f"{CASE}_mouthwidth-0.35_base.glb"
    brow_glb = out_dir / f"{CASE}_brows.glb"
    etc.export_textured(rest, faces, uv0, tex0, base_glb)
    # check TEXCOORD_0 convention written by trimesh (glTF v = 1 - ours) to write TEXCOORD_1 the same way
    t0, _ = read_attr(base_glb, "TEXCOORD_0")
    flip_ok = bool(np.allclose(t0[:, 1], 1 - uv0[:, 1], atol=1e-6) and np.allclose(t0[:, 0], uv0[:, 0], atol=1e-6))
    log(f"trimesh TEXCOORD_0 uses v_gltf = 1 - v: {flip_ok}")
    if not flip_ok:
        log("STOP: unexpected TEXCOORD_0 convention")
        return 3
    uv1_gltf = np.stack([uv1[:, 0], 1 - uv1[:, 1]], 1)
    add_brow_primitive(base_glb, brow_glb, faces, brow_mask, uv1_gltf, buf.getvalue())

    # ---- verification
    checks = {}
    ref = etc.glb_parts(REFERENCE)
    b = etc.glb_parts(base_glb)
    for key in ("POSITION", "NORMAL", "TEXCOORD_0", "indices"):
        checks[f"base_{key}_identical_to_mouthwidth-0.35_preview"] = b[key][0] == ref[key][0]
    pos, gl = read_attr(brow_glb, "POSITION")
    nor, _ = read_attr(brow_glb, "NORMAL")
    tc0, _ = read_attr(brow_glb, "TEXCOORD_0")
    i0, _ = read_attr(brow_glb, "indices", 0)
    i1, _ = read_attr(brow_glb, "indices", 1)
    checks["positions_identical"] = pos.tobytes() == b["POSITION"][0]
    checks["normals_identical"] = nor.tobytes() == b["NORMAL"][0]
    checks["texcoord0_identical"] = tc0.tobytes() == b["TEXCOORD_0"][0]
    both = np.concatenate([i0.reshape(-1, 3), i1.reshape(-1, 3)])
    checks["faces_split_exactly"] = (len(both) == len(faces)
                                     and np.array_equal(np.sort(both.view([("", both.dtype)] * 3), axis=0),
                                                        np.sort(faces.astype(np.uint32).view([("", np.uint32)] * 3), axis=0)))
    checks["primitives"] = len(gl["meshes"][0]["primitives"])
    checks["meshes"] = len(gl["meshes"])
    checks["static_mesh"] = "skins" not in gl
    data = brow_glb.read_bytes()
    checks["header_ok"] = data[:4] == b"glTF" and struct.unpack("<I", data[4:8])[0] == 2 \
        and struct.unpack("<I", data[8:12])[0] == len(data)
    loaded = trimesh.load(brow_glb, force="mesh", process=False)
    checks["trimesh_reload_vertices"] = len(loaded.vertices)
    checks["eye_texture_half_unchanged"] = bool(
        (np.asarray(tex0)[:, etc.TEX_W // 2:] == np.asarray(etc.eye_texture(DEFAULT_SKIN_TONE))[:, etc.TEX_W // 2:]).all())
    checks["brow_faces_have_no_lip_or_eye_vertices"] = bool((raw[patch_v] == 0).all() and not np.isin(patch_v, eye_idx).any())
    meas = BodyMeasurements(model)
    plain = anny.Anny()
    prod = BodyMeasurements(plain).measure(build(model=plain, **kw))
    m = meas.measure(output)
    checks["height_chest_waist_hip_unchanged_vs_production"] = all(
        m[k] == prod[k] for k in ("height_m", "chest_m", "waist_m", "hip_m"))
    log("checks:", json.dumps(checks))

    # ---- renders (per-face texture choice via a combined atlas)
    t0img = np.asarray(tex0.resize((TILE_PX[0], TILE_PX[0] // 2), Image.NEAREST))
    atlas = np.concatenate([t0img, np.asarray(brow_img)], 0)
    H = atlas.shape[0]
    top_h, bot_h = t0img.shape[0], TILE_PX[1]
    V2 = v[faces].reshape(-1, 3)
    F2 = np.arange(len(V2)).reshape(-1, 3)
    uvA = uv0[faces].reshape(-1, 2).copy()
    uvA[:, 1] = 1 - (1 - uvA[:, 1]) * top_h / H
    uvB = uv1[faces].reshape(-1, 2).copy()
    uvB[:, 1] = uvB[:, 1] * bot_h / H
    corner_brow = np.repeat(brow_mask, 3)
    uv_r = np.where(corner_brow[:, None], uvB, uvA)
    uv_base = uvA
    atlas_img = Image.fromarray(atlas.astype(np.uint8))
    eye_y = float(eye_c[1])
    top = float(v[:, 1].max())
    crops = {"face": (((-0.11, eye_y - 0.13), (0.11, top + 0.01)), 2600),
             "brows": (((-0.075, eye_y - 0.02), (0.075, eye_y + 0.05)), 5200)}
    for tag, (crop, scale) in crops.items():
        etc.side_by_side([etc.render_front(V2, F2, uv_base, atlas_img, crop, scale),
                          etc.render_front(V2, F2, uv_r, atlas_img, crop, scale)],
                         ["F1 CURRENT (eyes, lips, mouth -0.35)", "F1 WITH EYEBROWS"], out_dir / f"{CASE}_{tag}.png")
    side = np.stack([-V2[:, 2], V2[:, 1], V2[:, 0]], 1)
    zc = float(np.median(side[corner_brow, 0]))
    etc.side_by_side([etc.render_front(side, F2, uv_r, atlas_img, ((zc - 0.06, eye_y - 0.05), (zc + 0.03, eye_y + 0.06)), 4000)],
                     ["F1 WITH EYEBROWS (side)"], out_dir / f"{CASE}_brows_side.png")
    # debug: patch faces highlighted
    dbg_uv = np.where(corner_brow[:, None], np.array([[0.0, 0.0]]), uv_base)
    dbg_atlas = atlas.copy()
    dbg_atlas[-2:, :2] = [40, 160, 255]
    etc.side_by_side([etc.render_front(V2, F2, dbg_uv, Image.fromarray(dbg_atlas.astype(np.uint8)), crops["face"][0], 2600)],
                     ["brow patch (blue)"], out_dir / f"{CASE}_patch_debug.png")

    (out_dir / "results.json").write_text(json.dumps({
        "case": CASE, "mouth_width": MOUTH_WIDTH, "lip_settings": settings, "patch": {
            "half_width_m": PATCH_HALF_W, "y_m": PATCH_Y, "faces": int(brow_mask.sum()), "vertices": int(len(patch_v)),
            "boundary_vertices": int(len(boundary)), "seam_max_levels": seam_err, "seam_max_levels_nb": seam_err_nb},
        "brow": {"inner": BROW_INNER, "peak": BROW_PEAK, "tail": BROW_TAIL, "thickness_mm": THICKNESS,
                 "hairs_per_side": HAIRS, "rgb": BROW_RGB.tolist(), "seed": SEED},
        "checks": checks}, indent=1, default=float))
    log("Saved", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
