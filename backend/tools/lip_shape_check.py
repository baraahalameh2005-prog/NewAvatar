"""
Female lip-shape diagnostic (experimental; approved 2026-10-12). Does not change
production code, the API, the frontend or any existing GLB.

F1 only. Uses the cached 63-label controlled model (the face-sweep model) and sets
mouth-upperlip-volume-incr and mouth-lowerlip-volume-incr together to -0.3, -0.5 and
-0.8 (negative = decrease); every other control stays 0. The approved female lip
tint (lip_texture_check.LIP_SETTINGS["female"]) and the eye texture are applied
unchanged; the tint follows the vertices, so it moves with the lips.

Order of checks:
  1. Zero check: the 63-label model with all controls 0 must reproduce the production
     model's F1 rest vertices within 1e-9 m; otherwise stop.
  2. Per value: movement of lip vertices, of the non-lip mouth area and of every
     other vertex; faces/indices, eyes, eye texture half and measurements unchanged.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\lip_shape_check.py
Other controls / values (same checks):
    ..\\.venv\\Scripts\\python.exe -B tools\\lip_shape_check.py --labels mouth-scale-horiz-incr --values=-0.2,-0.35,-0.5 --tag mouthwidth
Defaults (no options) repeat the lip-volume run of 2026-10-12.
Writes only var/lip_shape_check/<timestamp>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import lip_texture_check as ltc
from lip_texture_check import etc
from eye_texture_check import ftm, config, anny, get_face_segmentation_mask

from adult_fit_experiment import REF, ControlledModel
from generate_avatar import DEFAULT_SKIN_TONE, build, parse_skin_tone
from measurements import BodyMeasurements

CASE = "F1_162.6cm_50kg"
LABELS = ["mouth-upperlip-volume-incr", "mouth-lowerlip-volume-incr"]
VALUES = [-0.3, -0.5, -0.8]
ZERO_TOLERANCE_M = 1e-9
MOUTH_RADIUS_M = 0.035   # "mouth area" = within this distance of the lip centroid
OUT = config.MODULE_ROOT / "var" / "lip_shape_check" / time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--values", default=",".join(str(x) for x in VALUES))
    ap.add_argument("--tag", default="lips")
    args = ap.parse_args()
    labels_ctl = args.labels.split(",")
    values = [float(x) for x in args.values.split(",")]
    tag_name = args.tag

    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")
    p = etc.CASES[CASE]["parameters"]
    kw = dict(gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"], age=p["age"],
              muscle=p["muscle"], proportions=p["proportions"])
    if p["gender"] != "female":
        sys.exit("F1 is expected to be female")

    plain = anny.Anny()
    model = anny.Anny(local_changes=REF["local_changes_list"])
    for label in labels_ctl:
        if label not in model.local_change_labels:
            sys.exit(f"{label} not available in the cached 63-label model")
    OUT.mkdir(parents=True)
    log("Output:", OUT)

    faces = model.faces.detach().cpu().numpy()
    if not np.array_equal(faces, plain.faces.detach().cpu().numpy()):
        log("STOP: face topology of the 63-label model differs from the production model")
        return 3

    proxy = ControlledModel(model)
    proxy.controls = {label: 0.0 for label in REF["local_changes_list"]}
    zero_out = build(model=proxy, **kw)
    prod_out = build(model=plain, **kw)
    zero_diff = float((zero_out["rest_vertices"] - prod_out["rest_vertices"]).abs().max())
    log(f"Zero check: 63-label model (all controls 0) vs production rest vertices: max {zero_diff:.3e} m")
    if zero_diff > ZERO_TOLERANCE_M:
        log(f"STOP: zero check failed (> {ZERO_TOLERANCE_M} m)")
        return 3

    meas_plain = BodyMeasurements(plain)
    meas = BodyMeasurements(model)
    prod_meas = {k: v for k, v in meas_plain.measure(prod_out).items() if k != "hip_z"}

    # textures, UVs, lip weights: identical to the approved female setup
    settings = ltc.LIP_SETTINGS["female"]
    eye_tex = etc.eye_texture(DEFAULT_SKIN_TONE)
    tex = ltc.lip_eye_texture(DEFAULT_SKIN_TONE, settings)
    rest0 = zero_out["rest_vertices"][0].detach().cpu().numpy()
    v0 = etc.export_frame(rest0)
    front = {s: get_face_segmentation_mask(model, [f"eye_front.{s}"]).cpu().numpy() for s in ("L", "R")}
    eyes = etc.eyeballs(faces, len(v0))
    eye_idx = np.concatenate(eyes)
    uv_eyes, _ = etc.eye_uv(v0, faces, eyes, front)
    raw = ltc.lip_weights(model, faces, len(v0))
    raw[eye_idx] = 0.0
    weights = ltc.shape_weights(raw, settings)
    non_eye = np.setdiff1d(np.arange(len(v0)), eye_idx)
    uv = uv_eyes.copy()
    uv[non_eye, 0] = ltc.strip_u(weights[non_eye])
    uv[non_eye, 1] = 0.5
    eye_half_ok = bool((np.asarray(tex)[:, etc.TEX_W // 2:] == np.asarray(eye_tex)[:, etc.TEX_W // 2:]).all())

    lips = np.where(raw > 0)[0]
    lip_centre = v0[lips].mean(axis=0)
    near = np.linalg.norm(v0 - lip_centre, axis=1) < MOUTH_RADIUS_M
    mouth_area = np.where(near & (raw == 0))[0]
    elsewhere = np.where(~near & (raw == 0))[0]
    log(f"Regions: lip vertices {len(lips)}, mouth area (non-lip, < {MOUTH_RADIUS_M * 100:.1f} cm) {len(mouth_area)}, "
        f"elsewhere {len(elsewhere)}; eye texture half unchanged {eye_half_ok}")

    results = {"case": CASE, "labels": labels_ctl, "values": values, "zero_check_max_m": zero_diff,
               "lip_settings": settings, "eye_texture_half_unchanged": eye_half_ok, "runs": {}}

    glbs = {}
    base = OUT / f"{CASE}_{tag_name}_0.glb"
    etc.export_textured(rest0, faces, uv, tex, base)
    glbs[0.0] = (base, v0)
    base_parts = etc.glb_parts(base)

    core = np.where(raw > 0.5)[0]
    width0 = float(np.ptp(v0[core, 0]) * 1000)
    mouth_set = np.zeros(len(v0), dtype=bool)
    mouth_set[lips] = True
    mouth_set[mouth_area] = True
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    medges = edges[mouth_set[edges].all(axis=1)]
    len0 = np.linalg.norm(v0[medges[:, 0]] - v0[medges[:, 1]], axis=1)
    log(f"Lip width (span of lip vertices with weight > 0.5) at 0: {width0:.2f} mm; mouth-region edges {len(medges)}")
    results["lip_width_0_mm"] = width0

    for value in values:
        proxy.controls = {label: 0.0 for label in REF["local_changes_list"]}
        for label in labels_ctl:
            proxy.controls[label] = value
        out = build(model=proxy, **kw)
        rest = out["rest_vertices"][0].detach().cpu().numpy()
        v = etc.export_frame(rest)
        d = np.linalg.norm(v - v0, axis=1) * 1000

        glb = OUT / f"{CASE}_{tag_name}_{value:+.2f}.glb"
        etc.export_textured(rest, faces, uv, tex, glb)
        parts = etc.glb_parts(glb)
        m = {k: v_ for k, v_ in meas.measure(out).items() if k != "hip_z"}
        changed = {k: (prod_meas[k], m.get(k)) for k in prod_meas if m.get(k) != prod_meas[k]}
        run = {
            "lip_max_move_mm": float(d[lips].max()), "lip_mean_move_mm": float(d[lips].mean()),
            "mouth_area_max_move_mm": float(d[mouth_area].max()),
            "elsewhere_max_move_mm": float(d[elsewhere].max()),
            "eyeball_max_move_mm": float(d[eye_idx].max()),
            "moved_vertices_gt_0.01mm": int((d > 0.01).sum()),
            "indices_identical": parts["indices"][0] == base_parts["indices"][0],
            "texcoord_identical": parts["TEXCOORD_0"][0] == base_parts["TEXCOORD_0"][0],
            "vertex_count": parts["POSITION"][1]["count"], "meshes_primitives": [parts["meshes"], parts["primitives"]],
            "header_ok": parts["header_ok"], "static_mesh": "skins" not in parts["gltf"],
            "measurements_changed": changed,
        }
        width = float(np.ptp(v[core, 0]) * 1000)
        stretch = np.linalg.norm(v[medges[:, 0]] - v[medges[:, 1]], axis=1) / len0
        run["lip_width_mm"] = width
        run["lip_width_change_mm"] = width - width0
        run["lip_width_change_pct"] = (width / width0 - 1) * 100
        run["mouth_edge_stretch"] = {"min": float(stretch.min()), "max": float(stretch.max()),
                                     "p01": float(np.percentile(stretch, 1)), "p99": float(np.percentile(stretch, 99)),
                                     "outside_0.8_1.25": int(((stretch < 0.8) | (stretch > 1.25)).sum())}
        far = elsewhere[d[elsewhere] > 0.01]
        run["elsewhere_moved_count"] = int(len(far))
        if len(far):
            run["elsewhere_moved_height_range_m"] = [float(v0[far, 1].min()), float(v0[far, 1].max())]
        results["runs"][f"{value:+.2f}"] = run
        glbs[value] = (glb, v)
        log(f"value {value:+.2f}: lip width {run['lip_width_mm']:.2f} mm ({run['lip_width_change_mm']:+.2f} mm, "
            f"{run['lip_width_change_pct']:+.1f}%) | edge stretch {run['mouth_edge_stretch']['min']:.3f}-{run['mouth_edge_stretch']['max']:.3f} "
            f"(outside 0.8-1.25: {run['mouth_edge_stretch']['outside_0.8_1.25']}) | lips max {run['lip_max_move_mm']:.2f} mm (mean {run['lip_mean_move_mm']:.2f}) | "
            f"mouth area max {run['mouth_area_max_move_mm']:.2f} mm | elsewhere max {run['elsewhere_max_move_mm']:.4f} mm "
            f"({run['elsewhere_moved_count']} > 0.01 mm) | eyes {run['eyeball_max_move_mm']:.4f} mm | "
            f"indices same {run['indices_identical']} | uv same {run['texcoord_identical']} | "
            f"measurements changed {list(changed) or 'none'}")

    # renders: front face, mouth close-up, side profile of the mouth
    eye_y = float(v0[eye_idx, 1].mean())
    top = float(v0[:, 1].max())
    my = float(lip_centre[1])
    crops = {"face": (((-0.11, eye_y - 0.13), (0.11, top + 0.01)), 2600),
             "mouth": (((-0.045, my - 0.03), (0.045, my + 0.025)), 7000)}
    order = [0.0] + values
    labels = ["CURRENT (0)"] + [f"{tag_name.upper()} {x:+.2f}" for x in values]
    for tag, (crop, scale) in crops.items():
        etc.side_by_side([etc.render_front(glbs[k][1], faces, uv, tex, crop, scale) for k in order], labels,
                         OUT / f"{CASE}_{tag_name}_{tag}.png")
    side_imgs = []
    for k in order:
        vv = glbs[k][1]
        side = np.stack([-vv[:, 2], vv[:, 1], vv[:, 0]], axis=1)
        z0 = float(side[lips, 0].mean())
        side_imgs.append(etc.render_front(side, faces, uv, tex, ((z0 - 0.05, my - 0.04), (z0 + 0.03, my + 0.03)), 6000))
    etc.side_by_side(side_imgs, labels, OUT / f"{CASE}_{tag_name}_mouth_side.png")

    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log("Saved", OUT / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
