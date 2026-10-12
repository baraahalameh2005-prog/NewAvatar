"""
Face phase, step 1: diagnostic sweep of Anny face controls (approved 2026-10-11).
Diagnostic only: no fitting, no change to the API, production fit or defaults.

Model: the experiment's 63-label controlled model (3 body controls + 30 face
labels), whose cache entry already exists in the module, so this run writes
nothing outside var/face_sweep/. The 30 face labels form 25 controls (left and
right labels are set together). Body controls stay at 0.

For F1 and M2 (recorded production parameters) each control is evaluated at
-1, 0 and +1 and compared with the same body at 0 for every control.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\face_control_sweep.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
BACKEND = TOOLS.parent
sys.path.insert(0, str(BACKEND))

import fit_to_measurements as ftm  # applies the module cache settings first
from app import config

for _key in ("ANNY_CACHE_DIR", "WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
    if not config.is_inside(os.environ[_key], config.MODULE_ROOT):
        sys.exit(f"Refusing to run: {_key}={os.environ[_key]} is outside the module.")

import trimesh
import anny
from anny.face_segmentation import get_face_segmentation_mask
from generate_avatar import build, export_avatar
from measurements import BodyMeasurements

REF = json.loads((TOOLS / "stage5_ac67_reference.json").read_text())
BASE = {c["case"]: c for c in json.loads((TOOLS / "baseline_cases.json").read_text())["cases"]}
CASES = ["F1_162.6cm_50kg", "M2_175cm_72kg"]
VALUES = (-1.0, 1.0)

ANNY_ROOT = Path(anny.__file__).resolve().parent / "data" / "mpfb2" / "3dobjs" / "base.obj"
OUT = config.MODULE_ROOT / "var" / "face_sweep" / time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def face_controls():
    """The 30 face labels of the 63-label list, grouped into 25 symmetric controls."""
    body = set(REF["controls"])
    face = [x for x in REF["local_changes_list"][-30:] if x not in body]
    controls = {}
    for label in face:
        name = label[2:] if label[:2] in ("l-", "r-") else label
        controls.setdefault(name, []).append(label)
    return controls


def base_obj_groups(names):
    groups, g = {n: set() for n in names}, None
    for line in open(ANNY_ROOT):
        if line.startswith("g "):
            g = line.split()[1]
        elif line.startswith("f ") and g in groups:
            groups[g].update(int(t.split("/")[0]) - 1 for t in line.split()[1:])
    return groups


class Geometry:
    """Fixed index sets on the model's own vertices and faces."""

    def __init__(self, model):
        base_index = model.base_mesh_vertex_indices.detach().cpu().numpy()
        groups = base_obj_groups(["helper-l-eye", "helper-r-eye", "helper-tongue"])
        to_model = {int(b): i for i, b in enumerate(base_index)}
        self.eye = {side: np.array(sorted(to_model[b] for b in groups[g] if b in to_model))
                    for side, g in (("L", "helper-l-eye"), ("R", "helper-r-eye"))}
        tongue = {to_model[b] for b in groups["helper-tongue"] if b in to_model}
        self.faces = model.faces.detach().cpu().numpy()
        excluded = set(self.eye["L"]) | set(self.eye["R"]) | tongue
        self.skin_faces = self.faces[~np.isin(self.faces, list(excluded)).any(axis=1)]
        self.eye_all = np.concatenate([self.eye["L"], self.eye["R"]])
        self.bone = {n: i for i, n in enumerate(model.bone_labels)}


def joint_z(output, geometry, name):
    return float(output["rest_bone_poses"][0, geometry.bone[name], 2, 3])


def eye_inside_fraction(vertices, geometry):
    skin = trimesh.Trimesh(vertices=vertices, faces=geometry.skin_faces, process=False)
    if not skin.is_watertight:
        return None, "skin surface not watertight; inside test unreliable"
    inside = skin.contains(vertices[geometry.eye_all])
    return float(inside.mean()), None


def degenerate_faces(vertices, faces):
    v = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    return int(np.count_nonzero(area < 1e-12))


def evaluate(model, meas, case, settings):
    p = case["parameters"]
    proxy = _Proxy(model, settings)
    output = build(model=proxy, gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"],
                   age=p["age"], muscle=p["muscle"], proportions=p["proportions"])
    return output, meas.measure(output)


class _Proxy:
    def __init__(self, model, settings):
        self._model = model
        self._settings = settings

    def __call__(self, phenotype_kwargs=None, **kwargs):
        return self._model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=dict(self._settings), **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def main():
    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    log("Output:", OUT)

    t = time.perf_counter()
    model = anny.Anny(local_changes=REF["local_changes_list"])
    meas = BodyMeasurements(model)
    log(f"63-label model loaded in {time.perf_counter() - t:.1f} s")
    geometry = Geometry(model)
    controls = face_controls()
    log(f"{len(controls)} controls: {list(controls)}")
    log(f"eyeball vertices L/R: {len(geometry.eye['L'])}/{len(geometry.eye['R'])}")

    # ---- UV check (model data; the exported GLB itself has no UVs).
    uv = {"exported_glb_has_uv": False,
          "note": "export_avatar builds the mesh without texture coordinates"}
    if model.texture_coordinates is not None and model.face_texture_coordinate_indices is not None:
        tc = model.texture_coordinates.detach().cpu().numpy()
        fti = model.face_texture_coordinate_indices.detach().cpu().numpy()
        uv["model_has_uv"] = True
        uv["uv_face_count_equals_faces"] = len(fti) == len(geometry.faces)
        eye_face = np.isin(geometry.faces, geometry.eye_all).all(axis=1)
        for label in ("eye_front.L", "eye_back.L", "eye_front.R", "eye_back.R"):
            mask = get_face_segmentation_mask(model, [label]).cpu().numpy()
            uv[f"{label}_faces"] = int(mask.sum())
            uv[f"{label}_faces_on_eyeball"] = int((mask & eye_face).sum())
        coords = tc[fti[eye_face]].reshape(-1, 2)
        uv["eyeball_uv_range"] = [coords.min(0).tolist(), coords.max(0).tolist()]
        uv["eyeball_faces"] = int(eye_face.sum())
    else:
        uv["model_has_uv"] = False
    log("UV check:", json.dumps(uv))

    results = {"controls": controls, "values": [-1.0, 0.0, 1.0], "uv": uv, "cases": {}}
    zero = {label: 0.0 for label in REF["local_changes_list"]}

    for name in CASES:
        case = BASE[name]
        out0, res0 = evaluate(model, meas, case, zero)
        v0 = out0["rest_vertices"][0].detach().cpu().numpy()
        neck_z = joint_z(out0, geometry, "neck01")
        head_z = joint_z(out0, geometry, "head")
        frac0, why0 = eye_inside_fraction(v0, geometry)
        e0 = {s: v0[idx].mean(0) for s, idx in geometry.eye.items()}
        near_eye = {s: np.where((np.linalg.norm(v0 - e0[s], axis=1) < 0.03)
                                & ~np.isin(np.arange(len(v0)), geometry.eye_all))[0] for s in geometry.eye}
        below_neck = v0[:, 2] < neck_z
        below_head = v0[:, 2] < head_z
        edges = geometry.skin_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        head_edges = edges[(v0[edges[:, 0], 2] > head_z) & (v0[edges[:, 1], 2] > head_z)]
        len0 = np.linalg.norm(v0[head_edges[:, 0]] - v0[head_edges[:, 1]], axis=1)

        glb0 = OUT / name / "reference_0.glb"
        glb0.parent.mkdir(parents=True)
        export_avatar(out0, model, glb0)

        case_rows = {"reference": {"eye_inside_fraction": frac0, "eye_inside_note": why0,
                                   "neck01_z": neck_z, "head_joint_z": head_z,
                                   "vertices_below_neck01": int(below_neck.sum()),
                                   "vertices_below_head_joint": int(below_head.sum()),
                                   "degenerate_faces": degenerate_faces(v0, geometry.faces),
                                   "glb": str(glb0)},
                     "controls": {}}
        log(f"\n===== {name}: eyeball inside skin at 0 = {frac0}; neck01 z {neck_z:.3f}, head joint z {head_z:.3f}")

        for control, labels in controls.items():
            rows = {}
            for value in VALUES:
                settings = dict(zero)
                for label in labels:
                    settings[label] = value
                out, res = evaluate(model, meas, case, settings)
                v = out["rest_vertices"][0].detach().cpu().numpy()
                d = np.linalg.norm(v - v0, axis=1)
                frac, why = eye_inside_fraction(v, geometry)

                eye_motion = {}
                for s, idx in geometry.eye.items():
                    disp = v[idx] - v0[idx]
                    eye_motion[s] = {
                        "eyeball_max_move_mm": float(np.linalg.norm(disp, axis=1).max() * 1000),
                        "eyeball_non_rigid_mm": float(np.linalg.norm(disp - disp.mean(0), axis=1).max() * 1000),
                        "skin_near_eye_max_move_mm": float(d[near_eye[s]].max() * 1000) if len(near_eye[s]) else None,
                    }

                ratio = np.linalg.norm(v[head_edges[:, 0]] - v[head_edges[:, 1]], axis=1) / np.maximum(len0, 1e-12)
                deltas = {}
                for key, scale, unit in (("height_m", 1000, "mm"), ("mass_kg", 1, "kg"), ("waist_m", 1000, "mm"),
                                         ("chest_m", 1000, "mm"), ("hip_m", 1000, "mm")):
                    a, b = res.get(key), res0.get(key)
                    deltas[f"{key[:-2] if key.endswith('_m') else key} ({unit})"] = (
                        None if a is None or b is None else float((a - b) * scale))

                glb = OUT / name / f"{control}_{'minus1' if value < 0 else 'plus1'}.glb"
                export_avatar(out, model, glb)

                rows[str(value)] = {
                    "labels": labels,
                    "max_move_below_neck01_mm": float(d[below_neck].max() * 1000),
                    "max_move_below_head_joint_mm": float(d[below_head].max() * 1000),
                    "max_move_anywhere_mm": float(d.max() * 1000),
                    "measurement_deltas": deltas,
                    "eye_inside_fraction": frac, "eye_inside_note": why,
                    "eye_motion": eye_motion,
                    "nan_vertices": int(np.isnan(v).any(axis=1).sum()),
                    "degenerate_faces": degenerate_faces(v, geometry.faces),
                    "head_edge_stretch_min_max": [float(ratio.min()), float(ratio.max())],
                    "glb": str(glb),
                }
            case_rows["controls"][control] = rows
            r1, rm = rows["1.0"], rows["-1.0"]
            log(f"{control:24} below-neck {max(r1['max_move_below_neck01_mm'], rm['max_move_below_neck01_mm']):8.3f} mm | "
                f"below-head-joint {max(r1['max_move_below_head_joint_mm'], rm['max_move_below_head_joint_mm']):7.3f} mm | "
                f"height {rm['measurement_deltas']['height (mm)']:+.2f}/{r1['measurement_deltas']['height (mm)']:+.2f} mm | "
                f"eye inside {rm['eye_inside_fraction']}/{r1['eye_inside_fraction']} | "
                f"eyeball move L {rm['eye_motion']['L']['eyeball_max_move_mm']:.1f}/{r1['eye_motion']['L']['eyeball_max_move_mm']:.1f} mm "
                f"non-rigid {max(rm['eye_motion']['L']['eyeball_non_rigid_mm'], r1['eye_motion']['L']['eyeball_non_rigid_mm']):.2f} mm")

        results["cases"][name] = case_rows
        (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))

    log("\nSaved", OUT / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
