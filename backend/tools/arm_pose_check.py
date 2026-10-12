"""
Arm-pose diagnostic (experimental; does not change production code or the API).

Poses a copy of a fitted body by rotating each upper arm (and everything below it)
about its shoulder joint, mirrored left/right, then exports the posed vertices
through the unchanged production export_avatar. Fitting and measurements are
untouched: they use rest_vertices / rest_bone_poses, which posing does not change.

Posing uses Anny's "world" pose parameterization with the rest bone poses as the
starting point. The model's default ("local-ref") zero pose is NOT the rest pose
(measured: up to 7.8 cm vertex difference), so it is not used.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\arm_pose_check.py --sweep
        in-memory angle sweep, prints metrics, writes nothing
    ..\\.venv\\Scripts\\python.exe -B tools\\arm_pose_check.py --export F2=a,b M1=c,d
        writes var/pose_check/<timestamp>/ (baseline + candidate GLBs, renders, results)

Angles are degrees of extra adduction (arm moved toward the body) at the shoulder.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import fit_to_measurements as ftm  # applies the module cache settings first
from app import config

import torch
import trimesh
import anny
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from generate_avatar import build, export_avatar, make_phenotype
from measurements import BodyMeasurements

CASES = {c["case"]: c for c in json.loads((BACKEND / "tools" / "baseline_cases.json").read_text())["cases"]}
BODIES = {"F2": "F2_165cm_60kg", "M1": "M1_170cm_60kg"}
REFERENCE_GLB = config.MODULE_ROOT / "var" / "measurement_check" / "after"
REFERENCE_MEASUREMENTS = config.MODULE_ROOT / "var" / "measurement_check" / "post_stage1_validation.json"
OUT_ROOT = config.MODULE_ROOT / "var" / "pose_check"
SWEEP_DEGREES = list(range(0, 37, 3))
PENETRATION_OFFSET_M = 0.002


# ------------------------------------------------------------------ model helpers

class Rig:
    def __init__(self, model):
        self.model = model
        self.labels = list(model.bone_labels)
        self.parents = [int(p) for p in model.bone_parents]
        weights = model.vertex_bone_weights.detach().cpu().numpy()
        indices = model.vertex_bone_indices.detach().cpu().numpy()
        n_bones = len(self.labels)
        self.weight_matrix = np.zeros((weights.shape[0], n_bones))
        np.add.at(self.weight_matrix, (np.arange(weights.shape[0])[:, None], indices), weights)

    def index(self, label):
        return self.labels.index(label)

    def subtree(self, label):
        root = self.index(label)
        out = {root}
        changed = True
        while changed:
            changed = False
            for i, p in enumerate(self.parents):
                if p in out and i not in out:
                    out.add(i)
                    changed = True
        return sorted(out)

    def weight_on(self, bones):
        return self.weight_matrix[:, sorted(bones)].sum(axis=1)


def rotation_about(axis_angle_deg, pivot):
    """4x4 rotation by angle about the Anny +y (front-back) axis through pivot."""
    a = np.radians(axis_angle_deg)
    c, s = np.cos(a), np.sin(a)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pivot - r @ pivot
    return m


def posed_output(model, rig, phenotype, rest_bone_poses, degrees):
    """Anny forward pass with mirrored upper-arm rotations (world parameterization)."""
    poses = rest_bone_poses.clone()
    for side, sign in (("L", 1.0), ("R", -1.0)):
        joint = rig.index(f"upperarm01.{side}")
        pivot = rest_bone_poses[0, joint, :3, 3].detach().cpu().numpy()
        m = torch.as_tensor(rotation_about(sign * degrees, pivot), dtype=poses.dtype)
        for j in rig.subtree(f"upperarm01.{side}"):
            poses[0, j] = m @ rest_bone_poses[0, j]
    return model(phenotype_kwargs=phenotype, pose_parameters=poses, pose_parameterization="world")


def to_export_frame(vertices):
    """Same axis change and floor shift as export_avatar (Y-up, feet at y = 0)."""
    v = np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)
    v[:, 1] -= v[:, 1].min()
    return v


# ------------------------------------------------------------------ geometry metrics

class Regions:
    def __init__(self, rig, faces, rest_export):
        arm_l = rig.subtree("upperarm01.L")
        arm_r = rig.subtree("upperarm01.R")
        lower_l = rig.subtree("upperarm02.L")
        lower_r = rig.subtree("upperarm02.R")
        shoulder = [rig.index(n) for n in ("clavicle.L", "shoulder01.L", "clavicle.R", "shoulder01.R")]
        w_arm = rig.weight_on(arm_l + arm_r + shoulder)
        self.body = np.where(w_arm == 0)[0]                         # torso, legs, head
        self.arm = {"L": np.where(rig.weight_on(lower_l) >= 0.5)[0],  # mid upper arm to hand
                    "R": np.where(rig.weight_on(lower_r) >= 0.5)[0]}
        self.all_arm = np.where(rig.weight_on(arm_l + arm_r) > 0)[0]
        # shoulder / armpit region: any weight on clavicle, shoulder01 or upperarm01
        sh_bones = shoulder + [rig.index("upperarm01.L"), rig.index("upperarm01.R")]
        region = rig.weight_on(sh_bones) > 0
        self.region_faces = np.where(region[faces].all(axis=1))[0]
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        edges = np.unique(edges, axis=0)
        self.region_edges = edges[region[edges].all(axis=1)]
        mirrored = rest_export * np.array([-1, 1, 1])
        dist, self.mirror = cKDTree(rest_export).query(mirrored)
        self.mirror_rest_error = float(dist.max())


def slice_profile(mesh, height):
    rows = []
    for frac in np.arange(0.40, 0.80, 0.025):
        sec = mesh.section(plane_origin=[0, frac * height, 0], plane_normal=[0, 1, 0])
        if sec is None:
            continue
        loops = [np.asarray(p)[:, [0, 2]] for p in sec.discrete if len(p) >= 3]
        spans = [(lp[:, 0].min(), lp[:, 0].max()) for lp in loops]
        centre = [i for i, (a, b) in enumerate(spans) if a < 0 < b]
        if not centre:
            rows.append({"frac": round(frac, 3), "loops": len(loops), "torso": False})
            continue
        t = max(centre, key=lambda i: spans[i][1] - spans[i][0])
        tree = cKDTree(loops[t])
        gaps = {"L": None, "R": None}
        for i, lp in enumerate(loops):
            if i == t or abs(lp[:, 0].mean()) < 0.03:
                continue
            side = "L" if lp[:, 0].mean() > 0 else "R"
            g = float(tree.query(lp)[0].min())
            gaps[side] = g if gaps[side] is None else min(gaps[side], g)
        rows.append({"frac": round(frac, 3), "loops": len(loops), "torso": True,
                     "gap_L_cm": None if gaps["L"] is None else gaps["L"] * 100,
                     "gap_R_cm": None if gaps["R"] is None else gaps["R"] * 100})
    return rows


def frontal_angle(bone_poses, rig, a, b):
    """Angle from vertical (deg) of the a->b bone-origin vector, projected on the frontal plane."""
    p = bone_poses[0].detach().cpu().numpy()
    out = {}
    for side in ("L", "R"):
        d = p[rig.index(f"{b}.{side}"), :3, 3] - p[rig.index(f"{a}.{side}"), :3, 3]
        out[side] = float(np.degrees(np.arctan2(abs(d[0]), -d[2])))
    return out


def metrics(model, rig, regions, faces, rest_export, output, posed_vertices_anny, bone_poses, render_mesh_check=True):
    v = to_export_frame(posed_vertices_anny)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    height = float(v[:, 1].max())

    body_tree = cKDTree(v[regions.body])
    clearance = {}
    for side in ("L", "R"):
        d, idx = body_tree.query(v[regions.arm[side]])
        k = int(np.argmin(d))
        clearance[side] = {"min_cm": float(d[k] * 100),
                           "at_height_frac": float(v[regions.arm[side][k], 1] / height),
                           "arm_vertices_within_1cm": int((d < 0.01).sum())}

    # mirror symmetry using the rest-pose vertex correspondence
    sym = float(np.abs(v - v[regions.mirror] * np.array([-1, 1, 1])).max())

    # deformation in the shoulder / armpit region relative to the rest pose
    e = regions.region_edges
    rest_len = np.linalg.norm(rest_export[e[:, 0]] - rest_export[e[:, 1]], axis=1)
    pose_len = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    stretch = pose_len / rest_len
    rest_mesh = trimesh.Trimesh(vertices=rest_export, faces=faces, process=False)
    rf = regions.region_faces
    area_ratio = mesh.area_faces[rf] / rest_mesh.area_faces[rf]
    # folds: adjacent region faces meeting at a sharp angle (> 100 deg) - count posed vs rest
    def folds(m):
        adj = m.face_adjacency
        keep = np.isin(adj, rf).all(axis=1)
        return int((m.face_adjacency_angles[keep] > np.radians(100)).sum())

    result = {
        "height_m": height,
        "upper_arm_angle_deg": frontal_angle(bone_poses, rig, "upperarm01", "lowerarm01"),
        "shoulder_to_wrist_angle_deg": frontal_angle(bone_poses, rig, "upperarm01", "wrist"),
        "clearance": clearance,
        "slices": slice_profile(mesh, height),
        "mirror_max_error_mm": sym * 1000,
        "region_edge_stretch": {"min": float(stretch.min()), "p01": float(np.percentile(stretch, 1)),
                                "p99": float(np.percentile(stretch, 99)), "max": float(stretch.max()),
                                "outside_0.7_1.3": int(((stretch < 0.7) | (stretch > 1.3)).sum()),
                                "edges": int(len(stretch))},
        "region_face_area_ratio": {"min": float(area_ratio.min()), "max": float(area_ratio.max())},
        "region_sharp_folds_gt100deg": folds(mesh),
        "region_sharp_folds_rest": folds(rest_mesh),
        "volume_change_pct": float((mesh.volume / rest_mesh.volume - 1) * 100),
    }

    if render_mesh_check:
        # Penetration proxy: arm/hand vertices pushed 2 mm outward along their normal that end up
        # inside the closed body surface. Calibrated against the rest pose (reported separately).
        pts = v[regions.all_arm] + mesh.vertex_normals[regions.all_arm] * PENETRATION_OFFSET_M
        inside = mesh.contains(pts)
        result["arm_points_inside_body"] = int(inside.sum())
        result["arm_points_inside_heights"] = sorted({round(float(y / height), 2) for y in pts[inside, 1]})
    return result, mesh


# ------------------------------------------------------------------ rendering (numpy + PIL, orthographic)

def render(meshes, view, crop=None, px_per_m=420, label_list=None):
    """Painter's-algorithm flat-shaded orthographic render of several meshes side by side."""
    light = np.array([0.3, 0.4, 1.0])
    light /= np.linalg.norm(light)
    panels = []
    for mesh in meshes:
        v = mesh.vertices
        if view == "front":
            uv, depth = v[:, [0, 1]] * [1, 1], v[:, 2]
        elif view == "back":
            uv, depth = v[:, [0, 1]] * [-1, 1], -v[:, 2]
        else:  # side: viewed from the body's left (+x)
            uv, depth = np.stack([-v[:, 2], v[:, 1]], axis=1), v[:, 0]
        normals = mesh.face_normals
        nview = {"front": normals, "back": normals * [-1, 1, -1],
                 "side": np.stack([-normals[:, 2], normals[:, 1], normals[:, 0]], axis=1)}[view]
        shade = 0.25 + 0.75 * np.clip(nview @ light, 0, 1)
        if crop is None:
            lo, hi = uv.min(axis=0) - 0.05, uv.max(axis=0) + 0.05
        else:
            lo, hi = np.array(crop[0]), np.array(crop[1])
        size = ((hi - lo) * px_per_m).astype(int)
        img = Image.new("RGB", tuple(size), (60, 60, 64))
        draw = ImageDraw.Draw(img)
        f = mesh.faces
        order = np.argsort(depth[f].mean(axis=1))
        pix = (uv - lo) * px_per_m
        pix[:, 1] = size[1] - pix[:, 1]
        visible = nview[:, 2] > 0
        for i in order:
            if not visible[i]:
                continue
            c = int(205 * shade[i])
            draw.polygon([tuple(pix[k]) for k in f[i]], fill=(c, int(c * 0.93), int(c * 0.86)))
        panels.append(img)
    h = max(p.size[1] for p in panels)
    sheet = Image.new("RGB", (sum(p.size[0] for p in panels) + 10 * (len(panels) - 1), h + 22), (40, 40, 44))
    x = 0
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(panels):
        sheet.paste(p, (x, 22))
        if label_list:
            d.text((x + 6, 4), label_list[i], fill=(235, 235, 235))
        x += p.size[0] + 10
    return sheet


# ------------------------------------------------------------------ main

def load_body(model, key):
    case = CASES[BODIES[key]]
    p = case["parameters"]
    gender = ftm.gender_value(p["gender"])
    kwargs = dict(height=p["height"], weight=p["weight"], age=p["age"], muscle=p["muscle"], proportions=p["proportions"])
    output = build(model=model, gender=gender, **kwargs)
    return case, output, make_phenotype(model, gender, **kwargs)


def summarize(name, deg, m):
    c = m["clearance"]
    return (f"{name:3} {deg:5.1f} deg | upper-arm angle {m['upper_arm_angle_deg']['L']:5.1f}/{m['upper_arm_angle_deg']['R']:5.1f}"
            f" | min clearance L {c['L']['min_cm']:5.2f} cm @ {c['L']['at_height_frac']:.2f}H, R {c['R']['min_cm']:5.2f}"
            f" | <1cm verts {c['L']['arm_vertices_within_1cm']}/{c['R']['arm_vertices_within_1cm']}"
            f" | stretch {m['region_edge_stretch']['min']:.2f}-{m['region_edge_stretch']['max']:.2f}"
            f" (out {m['region_edge_stretch']['outside_0.7_1.3']})"
            f" | folds {m['region_sharp_folds_gt100deg']} (rest {m['region_sharp_folds_rest']})"
            f" | sym {m['mirror_max_error_mm']:.3f} mm | vol {m['volume_change_pct']:+.2f}%"
            + (f" | inside {m['arm_points_inside_body']} {m['arm_points_inside_heights']}" if "arm_points_inside_body" in m else ""))


def main(argv):
    model = anny.Anny()
    rig = Rig(model)
    meas = BodyMeasurements(model)
    faces = model.faces.detach().cpu().numpy()
    ref_meas = {r["case"]: r["measurements"] for r in json.loads(REFERENCE_MEASUREMENTS.read_text())}

    if argv[0] == "--sweep":
        for key in BODIES:
            case, output, phenotype = load_body(model, key)
            rest_anny = output["rest_vertices"][0].detach().cpu().numpy()
            rest_export = to_export_frame(rest_anny)
            regions = Regions(rig, faces, rest_export)
            for deg in SWEEP_DEGREES:
                out = posed_output(model, rig, phenotype, output["rest_bone_poses"], deg)
                m, _ = metrics(model, rig, regions, faces, rest_export, output,
                               out["vertices"][0].detach().cpu().numpy(), out["bone_poses"], render_mesh_check=(deg % 6 == 0))
                print(summarize(key, deg, m), flush=True)
        return 0

    if argv[0] != "--export":
        sys.exit(__doc__)

    candidates = {}
    for item in argv[1:]:
        key, values = item.split("=")
        candidates[key] = [float(x) for x in values.split(",")]

    out_dir = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    if out_dir.exists():
        sys.exit(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    log_lines = []

    def log(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        log_lines.append(line)
        (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("Output:", out_dir)
    results = {"rotation": "upperarm01.L/.R subtrees about the shoulder joint, Anny +y axis, mirrored; world parameterization",
               "bodies": {}}

    for key, degrees in candidates.items():
        case, output, phenotype = load_body(model, key)
        rest_anny = output["rest_vertices"][0].detach().cpu().numpy()
        rest_export = to_export_frame(rest_anny)
        regions = Regions(rig, faces, rest_export)
        body = {"case": case["case"], "parameters": case["parameters"], "candidates": {}}

        # --- baseline checks
        ref = trimesh.load(REFERENCE_GLB / f"{case['case']}.glb", force="mesh", process=False)
        body["baseline_vs_reference_glb_max_m"] = float(np.abs(np.asarray(ref.vertices) - rest_export).max())
        zero = posed_output(model, rig, phenotype, output["rest_bone_poses"], 0.0)
        body["zero_pose_vs_rest_max_m"] = float((zero["vertices"] - output["rest_vertices"]).abs().max())
        default_pose = model(phenotype_kwargs=phenotype)
        body["default_local_ref_zero_pose_vs_rest_max_m"] = float(
            (default_pose["vertices"] - output["rest_vertices"]).abs().max())
        rest_meas = {k: v for k, v in meas.measure(output).items() if k != "hip_z"}
        body["rest_measurements_equal_reference"] = rest_meas == ref_meas[case["case"]]
        body["mirror_correspondence_rest_error_m"] = regions.mirror_rest_error
        log(f"\n== {key} ({case['case']})")
        log(f"baseline export vs reference GLB ({REFERENCE_GLB.name}/): max {body['baseline_vs_reference_glb_max_m']:.3e} m")
        log(f"world-parameterized zero rotation vs rest_vertices: max {body['zero_pose_vs_rest_max_m']:.3e} m")
        log(f"default (local-ref) zero pose vs rest_vertices: max {body['default_local_ref_zero_pose_vs_rest_max_m']:.4f} m (not used)")
        log(f"rest measurements equal recorded post_stage1_validation values: {body['rest_measurements_equal_reference']}")

        meshes, labels = [], []
        for deg in [0.0] + degrees:
            name = "baseline" if deg == 0 else f"adduct{deg:g}"
            out = posed_output(model, rig, phenotype, output["rest_bone_poses"], deg) if deg else zero
            posed = out["vertices"][0].detach().cpu().numpy()
            m, mesh = metrics(model, rig, regions, faces, rest_export, output, posed, out["bone_poses"])
            posed_meas = {k: v for k, v in meas.measure(out).items() if k != "hip_z"}
            m["measurements_equal_rest"] = posed_meas == rest_meas

            glb = out_dir / f"{key}_{name}.glb"
            if glb.exists():
                sys.exit(f"Refusing to overwrite {glb}")
            # Production exporter, unchanged; it reads output["rest_vertices"], so pass the posed ones there.
            export_avatar({"rest_vertices": out["vertices"]}, model, glb)
            content = glb.read_bytes()
            length = int.from_bytes(content[12:16], "little")
            gltf = json.loads(content[20:20 + length])
            loaded = trimesh.load(glb, force="mesh", process=False)
            m["glb"] = {"file": glb.name, "glTF_v2": content[:4] == b"glTF" and int.from_bytes(content[4:8], "little") == 2,
                        "length_ok": int.from_bytes(content[8:12], "little") == len(content),
                        "static_mesh": "skins" not in gltf and "animations" not in gltf,
                        "vertices": len(loaded.vertices), "faces": len(loaded.faces),
                        "max_diff_vs_posed_m": float(np.abs(np.asarray(loaded.vertices) - to_export_frame(posed)).max())}
            body["candidates"][name] = m
            log(summarize(key, deg, m) + f" | measurements unchanged {m['measurements_equal_rest']}"
                f" | glb ok {all(m['glb'][k] for k in ('glTF_v2', 'length_ok', 'static_mesh'))}")
            meshes.append(mesh)
            labels.append(f"{key} {name}")

        h = float(rest_export[:, 1].max())
        render(meshes, "front", label_list=labels).save(out_dir / f"{key}_front.png")
        render(meshes, "back", label_list=labels).save(out_dir / f"{key}_back.png")
        render(meshes, "side", label_list=labels).save(out_dir / f"{key}_side.png")
        crop = ((0.0, 0.55 * h), (0.42, 0.86 * h))  # body's left half, armpit to shoulder
        render(meshes, "front", crop=crop, px_per_m=900, label_list=labels).save(out_dir / f"{key}_armpit_front.png")
        crop_back = ((-0.42, 0.55 * h), (0.0, 0.86 * h))
        render(meshes, "back", crop=crop_back, px_per_m=900, label_list=labels).save(out_dir / f"{key}_armpit_back.png")
        results["bodies"][key] = body

    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log("\nSaved", out_dir / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
