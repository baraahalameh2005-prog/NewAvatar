"""
Verification for the integrated appearance pipeline (appearance.py + api.py).

  --presets  Recorded bodies (baseline_cases.json; no fitting): plain vs face-preset body,
             exported with export_appearance_avatar. Checks that only head vertices move,
             measurements stay equal, GLB structure / seam / textures; renders faces.
  --api      The real /generate-avatar endpoint in-process (FastAPI TestClient, real fit),
             for the appearance test matrix plus invalid-value and old-request checks.
             GLBs go to the run folder (AVATAR_OUTPUT_DIR), never to var/generated.

Renders are made from the exported GLB files themselves (both primitives, their
textures and texture-coordinate sets), i.e. from what the browser receives.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\appearance_check.py --presets
    ..\\.venv\\Scripts\\python.exe -B tools\\appearance_check.py --api
Writes only var/appearance_check/<timestamp>_<mode>/.
"""

import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
BACKEND = TOOLS.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(TOOLS))

MODE = sys.argv[1] if len(sys.argv) > 1 else "--presets"

# The output folder must be set before app.config is imported (it reads the environment once).
OUT = BACKEND.parent / "var" / "appearance_check" / f"{time.strftime('%Y%m%d_%H%M%S')}_{MODE.strip('-')}"
if OUT.exists():
    sys.exit(f"Refusing to overwrite {OUT}")
if MODE == "--api":
    os.environ["AVATAR_OUTPUT_DIR"] = str(OUT / "generated")

from app import config  # noqa: E402

if OUT.parents[2] != config.MODULE_ROOT:
    sys.exit("Unexpected module root")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CASES = {c["case"]: c for c in json.loads((TOOLS / "baseline_cases.json").read_text())["cases"]}
LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ GLB reading and rendering

def read_glb(path):
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    binary = data[20 + n + 8:]

    def accessor(i):
        acc = gltf["accessors"][i]
        view = gltf["bufferViews"][acc["bufferView"]]
        comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        arr = np.frombuffer(binary[start:start + acc["count"] * comps * np.dtype(dt).itemsize], dtype=dt)
        return arr.reshape(acc["count"], comps) if comps > 1 else arr

    def image(i):
        img = gltf["images"][i]
        view = gltf["bufferViews"][img["bufferView"]]
        blob = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
        return Image.open(io.BytesIO(blob)).convert("RGB")

    prims = []
    for p in gltf["meshes"][0]["primitives"]:
        mat = gltf["materials"][p["material"]]
        bct = mat["pbrMetallicRoughness"].get("baseColorTexture")
        prims.append({
            "indices": accessor(p["indices"]).reshape(-1, 3),
            "positions": accessor(p["attributes"]["POSITION"]),
            "uv": accessor(p["attributes"][f"TEXCOORD_{bct.get('texCoord', 0)}"]) if bct else None,
            "image": image(gltf["textures"][bct["index"]]["source"]) if bct else None,
            "factor": mat["pbrMetallicRoughness"].get("baseColorFactor"),
            "material": mat.get("name"),
            "attributes": p["attributes"],
        })
    return {"gltf": gltf, "bytes": data, "prims": prims}


def render_glb(glb, view, crop, px_per_m):
    """Per-pixel textured orthographic render of every primitive (z-buffer)."""
    from eye_texture_check import render_front

    images = [p["image"] for p in glb["prims"]]
    width = max(i.size[0] for i in images)
    rows, offsets = [], []
    for img in images:
        im = np.asarray(img.resize((width, max(1, round(img.size[1] * width / img.size[0]))), Image.NEAREST))
        offsets.append(sum(r.shape[0] for r in rows))
        rows.append(im)
    atlas = np.concatenate(rows, 0)
    H = atlas.shape[0]
    V, U = [], []
    for p, off, im in zip(glb["prims"], offsets, rows):
        tri = p["indices"]
        V.append(p["positions"][tri].reshape(-1, 3))
        uv = p["uv"][tri].reshape(-1, 2).astype(float)          # glTF: v down
        row = off + uv[:, 1] * im.shape[0]
        U.append(np.stack([uv[:, 0], 1 - row / H], 1))
    v = np.concatenate(V).astype(float)
    uv = np.concatenate(U)
    if view == "side":
        v = np.stack([-v[:, 2], v[:, 1], v[:, 0]], 1)
    f = np.arange(len(v)).reshape(-1, 3)
    return render_front(v, f, uv, Image.fromarray(atlas.astype(np.uint8)), crop, px_per_m)


def sheet(images, labels, path, cols=None):
    font = ImageFont.truetype("arial.ttf", 18)
    cols = cols or len(images)
    w = max(i.size[0] for i in images)
    h = max(i.size[1] for i in images)
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + 10), rows * (h + 34)), (30, 30, 34))
    d = ImageDraw.Draw(out)
    for k, (img, label) in enumerate(zip(images, labels)):
        x, y = (k % cols) * (w + 10), (k // cols) * (h + 34)
        out.paste(img, (x, y + 30))
        d.text((x + 6, y + 6), label, fill=(240, 240, 240), font=font)
    out.save(path)


def face_crops(positions, eyes_y):
    top = float(positions[:, 1].max())
    return {"face": (((-0.11, eyes_y - 0.13), (0.11, top + 0.01)), 2600),
            "eyes": (((-0.075, eyes_y - 0.02), (0.075, eyes_y + 0.05)), 5200)}


def glb_checks(glb, expected_vertices):
    g = glb["gltf"]
    prims = glb["prims"]
    data = glb["bytes"]
    shared = len({json.dumps(p["attributes"], sort_keys=True) for p in prims}) == 1
    tri = np.concatenate([p["indices"] for p in prims])
    return {
        "header_ok": data[:4] == b"glTF" and struct.unpack("<I", data[4:8])[0] == 2
                     and struct.unpack("<I", data[8:12])[0] == len(data),
        "meshes": len(g["meshes"]), "primitives": len(prims), "static": "skins" not in g and "animations" not in g,
        "vertex_count": int(len(prims[0]["positions"])), "vertex_count_ok": len(prims[0]["positions"]) == expected_vertices,
        "attributes_shared": shared, "triangles": int(len(tri)),
        "materials": [p["material"] for p in prims],
        "factors_white": all(p["factor"] in (None, [1.0, 1.0, 1.0, 1.0]) for p in prims),
    }


# ------------------------------------------------------------------ presets mode

def presets_mode():
    import anny
    import appearance as ap
    from generate_avatar import DEFAULT_SKIN_TONE, build
    import fit_to_measurements as ftm
    from measurements import BodyMeasurements

    plain = anny.Anny()
    face = ap.FaceModel()
    meas = BodyMeasurements(plain)
    OUT.mkdir(parents=True)
    log("Output:", OUT)
    zero = face.controls({})
    results = {}
    renders = {"face": [], "eyes": [], "side": []}
    labels = []
    for name in ("F1_162.6cm_50kg", "F2_165cm_60kg", "M1_170cm_60kg", "M2_175cm_72kg"):
        p = CASES[name]["parameters"]
        kw = dict(gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"], age=p["age"],
                  muscle=p["muscle"], proportions=p["proportions"])
        out_plain = build(model=plain, **kw)
        out_zero = build(model=zero, **kw)
        appearance = ap.appearance_for(p["gender"], DEFAULT_SKIN_TONE, ap.DEFAULT_EYE_COLOR)
        out_face = build(model=face.controls(appearance.face_controls), **kw)
        rp = out_plain["rest_vertices"][0].detach().cpu().numpy()
        rf = out_face["rest_vertices"][0].detach().cpu().numpy()
        zero_diff = float(np.abs(out_zero["rest_vertices"][0].detach().cpu().numpy() - rp).max())
        d = np.linalg.norm(rf - rp, axis=1)
        neck_z = float(out_plain["rest_bone_poses"][0, list(plain.bone_labels).index("neck01"), 2, 3])
        below_neck = rp[:, 2] < neck_z
        m_plain = {k: v for k, v in meas.measure(out_plain).items() if k != "hip_z"}
        m_face = {k: v for k, v in meas.measure(out_face).items() if k != "hip_z"}
        main = ("height_m", "chest_m", "waist_m", "hip_m")
        glb_path = OUT / f"{name}_appearance.glb"
        lay = ap.export_appearance_avatar(out_face, face, glb_path, appearance)
        glb = read_glb(glb_path)
        checks = glb_checks(glb, len(rp))
        pos = glb["prims"][0]["positions"]
        checks["positions_equal_face_body"] = bool(np.abs(pos - ap.export_frame(rf)).max() < 1e-6)
        # seam: brow texture under band-edge vertices equals the skin colour
        bi = np.unique(face.faces[lay.brow_faces])
        bo = np.unique(face.faces[~lay.brow_faces])
        edge = np.intersect1d(bi, bo)
        tex = np.asarray(ap.brow_texture(appearance.skin_tone, appearance.gender), dtype=float)
        skin = np.array(ap.parse_skin_tone(appearance.skin_tone)[:3], dtype=float)
        tx = np.clip((lay.uv1[edge, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
        ty = np.clip(((1 - lay.uv1[edge, 1]) * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
        checks["brow_seam_max_levels"] = float(np.abs(tex[ty, tx] - skin).max())
        checks["brow_faces"] = int(lay.brow_faces.sum())
        checks["brow_scale"] = lay.scale
        results[name] = {
            "zero_controls_vs_plain_max_m": zero_diff,
            "face_preset": appearance.face_controls,
            "max_move_mm": float(d.max() * 1000),
            "max_move_below_neck_mm": float(d[below_neck].max() * 1000),
            "main_measurements_equal": all(m_plain[k] == m_face[k] for k in main),
            "mass_change_g": (m_face["mass_kg"] - m_plain["mass_kg"]) * 1000,
            "checks": checks,
        }
        log(f"{name}: zero-control diff {zero_diff:.1e} m | preset max move {d.max() * 1000:.2f} mm, below neck "
            f"{d[below_neck].max() * 1000:.4f} mm | height/chest/waist/hip equal {results[name]['main_measurements_equal']} "
            f"| mass {results[name]['mass_change_g']:+.2f} g | {json.dumps(checks)}")
        eyes_y = float(pos[face.eye_index, 1].mean())
        crops = face_crops(pos, eyes_y)
        for tag in ("face", "eyes"):
            crop, scale = crops[tag]
            renders[tag].append(render_glb(glb, "front", crop, scale))
        zc = float(np.median(-pos[face.eye_index, 2]))
        renders["side"].append(render_glb(glb, "side", ((zc - 0.13, eyes_y - 0.13), (zc + 0.05, eyes_y + 0.11)), 2600))
        labels.append(f"{name} ({p['gender']})")
    for tag, imgs in renders.items():
        sheet(imgs, labels, OUT / f"presets_{tag}.png")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log("Saved", OUT)


# ------------------------------------------------------------------ api mode

API_MATRIX = [
    ("1_female_brown_default", {"gender": "female", "height": 170, "weight": 62, "waist": 72, "chest": 90, "hip": 98,
                                "eye_color": "brown"}),
    ("2_female_blue", {"gender": "female", "height": 170, "weight": 62, "waist": 72, "chest": 90, "hip": 98,
                       "eye_color": "blue"}),
    ("3_female_dark_skin", {"gender": "female", "height": 170, "weight": 62, "waist": 72, "chest": 90, "hip": 98,
                            "eye_color": "brown", "skin_tone": "#6b4430"}),
    ("4_male_default", {"gender": "male", "height": 170, "weight": 68, "waist": 82, "chest": 96, "hip": 96}),
    ("5_male_green", {"gender": "male", "height": 170, "weight": 68, "waist": 82, "chest": 96, "hip": 96,
                      "eye_color": "green"}),
    ("6_male_light_skin_gray", {"gender": "male", "height": 182, "weight": 85, "waist": 90, "chest": 104, "hip": 102,
                                "eye_color": "gray", "skin_tone": "#e2bfa3"}),
]


def api_mode():
    from fastapi.testclient import TestClient

    import api
    import appearance as ap

    OUT.mkdir(parents=True)
    log("Output:", OUT, "| API output dir:", api.OUTPUT_DIR)
    if Path(api.OUTPUT_DIR).resolve() != (OUT / "generated").resolve():
        log("STOP: API output directory is not the run folder")
        return 3
    results = {}
    with TestClient(api.app) as client:
        r = client.get("/avatar-options").json()
        results["avatar_options_eye_color"] = r.get("eye_color")
        log("avatar-options eye_color:", json.dumps(r.get("eye_color")))
        # invalid appearance values -> 422, no file
        for label, body in (("invalid_eye", {"eye_color": "purple"}), ("invalid_skin", {"skin_tone": "pink"})):
            before = set(os.listdir(api.OUTPUT_DIR)) if Path(api.OUTPUT_DIR).exists() else set()
            resp = client.post("/generate-avatar", json={"height": 165, "weight": 60, "waist": 72, "chest": 90,
                                                          "hip": 98, **body})
            after = set(os.listdir(api.OUTPUT_DIR)) if Path(api.OUTPUT_DIR).exists() else set()
            results[label] = {"status": resp.status_code, "detail": resp.json().get("detail"), "new_files": sorted(after - before)}
            log(f"{label}: {resp.status_code} new files {sorted(after - before)}")
        # old-style request (only the original fields, no appearance fields)
        renders, labels = {"face": [], "eyes": []}, []
        matrix = [("0_old_request_fields_only", {"height": 162.62, "weight": 49.99, "waist": 71.8, "chest": 86.6,
                                                  "hip": 92.15})] + API_MATRIX
        for name, body in matrix:
            t = time.perf_counter()
            resp = client.post("/generate-avatar", json=body)
            wall = time.perf_counter() - t
            data = resp.json()
            row = {"request": body, "status": resp.status_code, "wall_s": wall}
            if resp.status_code != 200:
                row["error"] = data
                results[name] = row
                log(f"{name}: FAILED {resp.status_code} {data}")
                continue
            served = client.get(data["avatar_url"])
            path = Path(api.OUTPUT_DIR) / f"{data['avatar_id']}.glb"
            glb = read_glb(path)
            checks = glb_checks(glb, 13718)
            checks["served_bytes_equal_file"] = served.status_code == 200 and served.content == path.read_bytes()
            app = data.get("appearance", {})
            base_img = np.asarray(glb["prims"][0]["image"], dtype=float)
            skin = np.array(ap.parse_skin_tone(app.get("skin_tone", "#c8997e"))[:3], dtype=float)
            iris = np.array(ap.EYE_COLORS[app.get("eye_color", "brown")][0], dtype=float)
            checks["skin_texel_matches_request"] = bool(np.abs(base_img[64, 2] - skin).max() <= 1)
            checks["iris_texel_matches_request"] = bool(np.abs(base_img[64, 192 + 20] - iris).max() <= 1)
            lip_end = base_img[64, ap.STRIP_LAST]
            checks["lip_strip_tinted"] = bool(np.abs(lip_end - skin).max() > 5)
            checks["male_no_lip_tint"] = None if app.get("gender_style") != "male" else bool(np.abs(lip_end - skin).max() <= 1)
            row.update({"avatar_id": data["avatar_id"], "appearance": app, "fit_time_s": data.get("fit_time_seconds"),
                        "warnings": data.get("warnings"), "accuracy_pct": {k: v.get("percentage_error") for k, v in data["accuracy"].items()},
                        "checks": checks})
            results[name] = row
            log(f"{name}: {resp.status_code} in {wall:.1f} s (fit {data.get('fit_time_seconds'):.1f} s) | appearance "
                f"{json.dumps({k: app.get(k) for k in ('gender_style', 'eye_color', 'skin_tone')})} | {json.dumps(checks)}")
            pos = glb["prims"][0]["positions"].astype(float)
            tri = np.concatenate([p["indices"] for p in glb["prims"]]).astype(np.int64)
            eyes_y = float(pos[np.concatenate(ap._eyeballs(tri, len(pos))), 1].mean())
            crops = face_crops(pos, eyes_y)
            for tag in ("face", "eyes"):
                crop, scale = crops[tag]
                renders[tag].append(render_glb(glb, "front", crop, scale))
            labels.append(name)
        for tag, imgs in renders.items():
            sheet(imgs, labels, OUT / f"api_{tag}.png", cols=4)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log("Saved", OUT)


if __name__ == "__main__":
    if MODE == "--presets":
        sys.exit(presets_mode())
    if MODE == "--api":
        sys.exit(api_mode())
    sys.exit(__doc__)
