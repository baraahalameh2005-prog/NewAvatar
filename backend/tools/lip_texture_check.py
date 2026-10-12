"""
Lip-colour diagnostic (experimental; does not change production code or the API).

Extends the eye-texture prototype (tools/eye_texture_check.py): same single mesh,
same vertices / faces / order, same eyes. The plain skin block of the generated
texture becomes a skin-to-lip colour strip; each non-eye vertex points into the
strip according to its lip weight, read from Anny's MPFB lip mask
(data/mpfb2/textures/mpfb_lips.jpg) through the model's own texture coordinates.
The lip colour is derived from the selected skin tone (darker and warmer).

Before the colour is judged, the mask's placement is checked against the mouth
(position relative to the mouth cavity, symmetry) and shown in a debug render.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\lip_texture_check.py
Writes only var/lip_check/<timestamp>/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import eye_texture_check as etc  # applies the module cache settings first
from eye_texture_check import ftm, config, anny, get_face_segmentation_mask, trimesh, Image

from generate_avatar import DEFAULT_SKIN_TONE, build, export_avatar, parse_skin_tone

LIP_MASK = Path(anny.__file__).resolve().parent / "data" / "mpfb2" / "textures" / "mpfb_lips.jpg"
OUT = config.MODULE_ROOT / "var" / "lip_check" / time.strftime("%Y%m%d_%H%M%S")
BODIES = etc.BODIES

STRIP_FIRST, STRIP_LAST = 2, 125      # texture columns of the skin (weight 0) and lip (weight 1) ends
PREVIOUS_RUN = config.MODULE_ROOT / "var" / "lip_check" / "20261012_005011"

# Per-sex lip settings. factor: lip = skin x factor (sRGB); strength: blend at full weight;
# edge: smoothstep (low, high) applied to the mask weight (None = mask weight as is).
# "male" keeps the first run's values exactly; "female" is the stronger, tighter tint (approved 2026-10-12).
LIP_SETTINGS = {
    "male": {"factor": (0.88, 0.74, 0.74), "strength": 0.85, "edge": None},
    "female": {"factor": (0.80, 0.58, 0.60), "strength": 1.0, "edge": (0.35, 0.75)},
}


def lip_colour(skin_tone, settings):
    skin = np.array(parse_skin_tone(skin_tone)[:3], dtype=float)
    return skin, skin * np.array(settings["factor"])


def lip_eye_texture(skin_tone, settings, debug=False):
    """Eye texture with the left half replaced by a skin -> lip strip (debug: white -> red)."""
    img = np.asarray(etc.eye_texture(skin_tone)).copy()
    skin, lip = lip_colour(skin_tone, settings)
    if debug:
        skin, lip = np.array([235.0, 235.0, 235.0]), np.array([230.0, 20.0, 20.0])
        strength = 1.0
    else:
        strength = settings["strength"]
    x = np.arange(etc.TEX_W // 2)
    t = np.clip((x - STRIP_FIRST) / (STRIP_LAST - STRIP_FIRST), 0, 1)[:, None] * strength
    img[:, : etc.TEX_W // 2] = (skin * (1 - t) + lip * t).round().astype(np.uint8)[None]
    return Image.fromarray(img, "RGB")


def shape_weights(weights, settings):
    """Optional smoothstep on the mask weight: removes the faint halo and saturates the lip core."""
    if settings["edge"] is None:
        return weights
    lo, hi = settings["edge"]
    t = np.clip((weights - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def lip_weights(model, faces, n_vertices):
    """Per-vertex lip weight: the mask sampled at each vertex's texture coordinate (max over its corners)."""
    mask = np.asarray(Image.open(LIP_MASK).convert("L"), dtype=float) / 255.0
    tc = model.texture_coordinates.detach().cpu().numpy()
    fti = model.face_texture_coordinate_indices.detach().cpu().numpy()
    h, w = mask.shape
    weights = np.zeros(n_vertices)
    for k in range(3):
        uv = tc[fti[:, k]]
        col = np.clip(np.round(uv[:, 0] * w).astype(int), 0, w - 1)
        row = np.clip(np.round((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
        np.maximum.at(weights, faces[:, k], mask[row, col])
    return weights


def strip_u(weights):
    return (STRIP_FIRST + weights * (STRIP_LAST - STRIP_FIRST) + 0.5) / etc.TEX_W


def alignment(vertices, faces, weights, mouth_cavity_faces, eyes):
    lips = np.where(weights > 0.5)[0]
    cavity = np.unique(faces[mouth_cavity_faces])
    eye_y = float(vertices[np.concatenate(eyes), 1].mean())
    lc, cc = vertices[lips].mean(axis=0), vertices[cavity].mean(axis=0)
    mirrored = vertices[lips] * np.array([-1, 1, 1])
    from scipy.spatial import cKDTree
    sym = float(cKDTree(vertices[lips]).query(mirrored)[0].max()) if len(lips) else None
    return {
        "lip_vertices_w_gt_0.5": int(len(lips)),
        "lip_vertices_w_gt_0": int((weights > 0).sum()),
        "lip_centroid_m": lc.tolist(), "mouth_cavity_centroid_m": cc.tolist(),
        "lip_to_cavity_centroid_mm": float(np.linalg.norm(lc - cc) * 1000),
        "lip_x_range_mm": [float(vertices[lips, 0].min() * 1000), float(vertices[lips, 0].max() * 1000)],
        "lip_below_eyes_mm": float((eye_y - lc[1]) * 1000),
        "lip_front_of_cavity_mm": float((lc[2] - cc[2]) * 1000),
        "lip_mirror_max_mm": None if sym is None else sym * 1000,
        "lip_vertices_far_from_mouth": int((np.linalg.norm(vertices[lips] - cc, axis=1) > 0.04).sum()),
    }


def main():
    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")
    model = anny.Anny()
    faces = model.faces.detach().cpu().numpy()
    front = {s: get_face_segmentation_mask(model, [f"eye_front.{s}"]).cpu().numpy() for s in ("L", "R")}
    cavity_faces = get_face_segmentation_mask(model, ["mouth_cavity"]).cpu().numpy()
    OUT.mkdir(parents=True)

    eye_tex = etc.eye_texture(DEFAULT_SKIN_TONE)
    results = {"skin_tone": DEFAULT_SKIN_TONE, "settings": LIP_SETTINGS, "previous_run": str(PREVIOUS_RUN), "bodies": {}}

    for name in BODIES:
        p = etc.CASES[name]["parameters"]
        sex = p["gender"]
        settings = LIP_SETTINGS[sex]
        tex = lip_eye_texture(DEFAULT_SKIN_TONE, settings)
        dbg = lip_eye_texture(DEFAULT_SKIN_TONE, settings, debug=True)
        tex.save(OUT / f"lip_eye_texture_{sex}.png")
        eye_half_unchanged = bool((np.asarray(tex)[:, etc.TEX_W // 2:] == np.asarray(eye_tex)[:, etc.TEX_W // 2:]).all())
        skin, lip = lip_colour(DEFAULT_SKIN_TONE, settings)
        print(f"\n{name} ({sex}): skin {skin.tolist()} -> lip {lip.round().tolist()} (strength {settings['strength']}, "
              f"edge {settings['edge']}); eye half of texture unchanged: {eye_half_unchanged}")

        output = build(model=model, gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"],
                       age=p["age"], muscle=p["muscle"], proportions=p["proportions"])
        rest = output["rest_vertices"][0].detach().cpu().numpy()
        v = etc.export_frame(rest)
        eyes = etc.eyeballs(faces, len(v))
        uv_eyes, info = etc.eye_uv(v, faces, eyes, front)
        raw_weights = lip_weights(model, faces, len(v))
        eye_idx = np.concatenate(eyes)
        raw_weights[eye_idx] = 0.0
        weights = shape_weights(raw_weights, settings)
        uv = uv_eyes.copy()
        non_eye = np.setdiff1d(np.arange(len(v)), eye_idx)
        uv[non_eye, 0] = strip_u(weights[non_eye])
        uv[non_eye, 1] = 0.5
        align = alignment(v, faces, raw_weights, cavity_faces, eyes)
        align["tinted_vertices_after_edge"] = int((weights > 0).sum())
        align["tinted_vertices_raw_mask"] = int((raw_weights > 0).sum())

        base_glb, eyes_glb, lips_glb = (OUT / f"{name}_{t}.glb" for t in ("baseline", "eyes", "eyes_lips"))
        export_avatar(output, model, base_glb)
        etc.export_textured(rest, faces, uv_eyes, eye_tex, eyes_glb)
        etc.export_textured(rest, faces, uv, tex, lips_glb)

        a, b, c = etc.glb_parts(base_glb), etc.glb_parts(eyes_glb), etc.glb_parts(lips_glb)
        checks = {k: a[k][0] == c[k][0] for k in ("POSITION", "NORMAL", "indices")}
        prev_glb = PREVIOUS_RUN / f"{name}_eyes_lips.glb"
        prev = etc.glb_parts(prev_glb)
        checks.update({
            "eye_texture_half_unchanged": eye_half_unchanged,
            "geometry_equal_previous_run": all(prev[k][0] == c[k][0] for k in ("POSITION", "NORMAL", "indices")),
            "glb_bytes_identical_to_previous_run": prev_glb.read_bytes() == lips_glb.read_bytes(),
            "eye_uv_unchanged_vs_eye_prototype": bool(np.array_equal(uv[eye_idx], uv_eyes[eye_idx])),
            "meshes_primitives": [c["meshes"], c["primitives"]], "header_ok": c["header_ok"],
            "texcoord_count": c["TEXCOORD_0"][1]["count"], "vertex_count": c["POSITION"][1]["count"],
            "static_mesh": "skins" not in c["gltf"], "bytes": lips_glb.stat().st_size,
        })
        results["bodies"][name] = {"alignment": align, "checks": checks}
        print(f"\n{name}\n  alignment: {json.dumps(align)}\n  checks: {json.dumps(checks)}", flush=True)

        # renders: debug mask, then baseline / eyes / eyes+lips (face and mouth close-ups)
        eye_y = float(v[eye_idx, 1].mean())
        top = float(v[:, 1].max())
        mouth_y = align["lip_centroid_m"][1]
        face_crop = ((-0.11, eye_y - 0.13), (0.11, top + 0.01))
        mouth_crop = ((-0.045, mouth_y - 0.03), (0.045, mouth_y + 0.025))
        etc.side_by_side([etc.render_front(v, faces, uv, dbg, face_crop, 2600),
                          etc.render_front(v, faces, uv, dbg, mouth_crop, 7000)],
                         [f"{name} LIP MASK (red = weight)", "mouth close-up"], OUT / f"{name}_lip_mask_debug.png")
        flat = np.array(parse_skin_tone(DEFAULT_SKIN_TONE)[:3], dtype=float)
        for tag, crop, scale in (("face", face_crop, 2600), ("mouth", mouth_crop, 7000)):
            etc.side_by_side([etc.render_front(v, faces, uv, None, crop, scale, flat_rgb=flat),
                              etc.render_front(v, faces, uv_eyes, eye_tex, crop, scale),
                              etc.render_front(v, faces, uv, tex, crop, scale)],
                             [f"{name} BASELINE", "EYES (prototype)", "EYES + LIPS"], OUT / f"{name}_{tag}.png")
        if sex == "female":
            # previous-run look (first-run settings = the "male" values) vs the new female settings
            old = LIP_SETTINGS["male"]
            uv_old = uv_eyes.copy()
            uv_old[non_eye, 0] = strip_u(raw_weights[non_eye])
            uv_old[non_eye, 1] = 0.5
            old_tex = lip_eye_texture(DEFAULT_SKIN_TONE, old)
            for tag, crop, scale in (("face", face_crop, 2600), ("mouth", mouth_crop, 7000)):
                etc.side_by_side([etc.render_front(v, faces, uv_old, old_tex, crop, scale),
                                  etc.render_front(v, faces, uv, tex, crop, scale)],
                                 [f"{name} PREVIOUS LIP TINT", "NEW FEMALE LIP TINT"], OUT / f"{name}_old_vs_new_{tag}.png")

    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nSaved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
