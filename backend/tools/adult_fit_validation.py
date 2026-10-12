"""
Stage 1 validation of the adult-fit candidate (approved 2026-10-11).
Diagnostic only: does not change the production fit, API or defaults.

Candidate:
  model  anny.Anny(local_changes=<the 3 control labels>)
  fit    adult_fit_experiment.exp_fit (age >= 0.67, controls within +-0.6)
  order  S_a; if S_a stopped as "good enough", use it. Otherwise run the
         production fit, then S_b from its solution (age raised to 0.67),
         and keep the start with the lowest residual norm.

K1 (hard stop): the 3-label model must reproduce the 63-label model used in
the experiment within 1e-9 m on every vertex, for the recorded S_a and S_b
parameters of all six cases. If it does not, the six-case run is not started.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\adult_fit_validation.py

Writes only to var/adult_validation/<timestamp>/ and the module cache.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

# Imports the experiment module; it applies the module cache settings and
# refuses to run if a cache points outside the module.
import adult_fit_experiment as afe
from adult_fit_experiment import ftm, config, anny, BodyMeasurements, build, export_avatar

import trimesh

K1_TOLERANCE_M = 1e-9
RUNTIME_LIMIT_S = 60.0
EXPERIMENT = config.MODULE_ROOT / "var" / "adult_experiment" / "20261011_172629" / "results.json"
OUT = config.MODULE_ROOT / "var" / "adult_validation" / time.strftime("%Y%m%d_%H%M%S")

# Written before the run (validation plan): expected path, time and gate result,
# derived from the experiment's recorded times. Not measurements.
PREDICTIONS = {
    "F1_162.6cm_50kg": ("S_a good-enough", 14, "pass"),
    "F2_165cm_60kg": ("S_a -> production -> S_b, S_b chosen", 40, "pass"),
    "F3_160cm_75kg": ("S_a -> production -> S_b, S_a chosen", 59.6, "fail (weight)"),
    "M1_170cm_60kg": ("S_a -> production -> S_b, S_a chosen", 59.5, "fail (hip)"),
    "M2_175cm_72kg": ("S_a good-enough", 18, "pass"),
    "M3_182cm_88kg": ("S_a good-enough", 30, "pass"),
}

LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def save(data):
    (OUT / "results.json").write_text(json.dumps(data, indent=1, default=float))


def build_controlled(model, gender, rec):
    proxy = afe.ControlledModel(model)
    proxy.controls = {c: rec["controls_exact"][c.split("-")[1]] for c in afe.CONTROLS}
    return build(model=proxy, gender=gender, height=rec["height_param"], weight=rec["weight_param"],
                 age=rec["age"], muscle=rec["muscle"], proportions=rec["proportions"])


def k1_check(model63, meas63, model3, meas3):
    experiment = json.loads(EXPERIMENT.read_text())["cases"]
    rows = []

    for case in afe.BASE:
        name = case["case"]
        gender = ftm.gender_value(case["parameters"]["gender"])

        for start in ("S_a", "S_b"):
            rec = experiment[name]["starts"][start]
            out63 = build_controlled(model63, gender, rec)
            out3 = build_controlled(model3, gender, rec)

            v63 = out63["rest_vertices"][0].detach().cpu().numpy()
            v3 = out3["rest_vertices"][0].detach().cpu().numpy()
            vmax = float(np.abs(v63 - v3).max())

            m63, m3 = meas63.measure(out63), meas3.measure(out3)
            mdiff = {k: abs(m63[k] - m3[k]) for k in ("height_m", "mass_kg", "waist_m", "chest_m", "hip_m")}

            g63 = OUT / "k1" / "model63" / f"{name}_{start}.glb"
            g3 = OUT / "k1" / "model3" / f"{name}_{start}.glb"
            g63.parent.mkdir(parents=True, exist_ok=True)
            g3.parent.mkdir(parents=True, exist_ok=True)
            export_avatar(out63, model63, g63)
            export_avatar(out3, model3, g3)
            h63, h3 = afe.sha256(g63), afe.sha256(g3)

            row = {
                "case": name, "start": start,
                "max_vertex_diff_m": vmax, "measurement_diffs": mdiff,
                "glb_identical": h63 == h3,
                # Sanity: the 63-label rebuild reproduces the experiment's S_a GLB.
                "model63_equals_experiment_S_a_glb": (h63 == rec["glb_sha256"]) if start == "S_a" else None,
                "pass": vmax <= K1_TOLERANCE_M,
            }
            rows.append(row)
            log(f"K1 {name} {start}: max vertex diff {vmax:.3e} m, measurement diffs "
                f"{ {k: f'{v:.2e}' for k, v in mdiff.items()} }, GLB identical {row['glb_identical']}"
                + (f", 63-label = experiment S_a GLB {row['model63_equals_experiment_S_a_glb']}" if start == "S_a" else ""))

    return rows


def check_glb(path):
    content = path.read_bytes()
    scene = trimesh.load(path)
    meshes = list(scene.geometry.values())
    return {
        "header_glTF_v2": content[:4] == b"glTF" and int.from_bytes(content[4:8], "little") == 2
                          and int.from_bytes(content[8:12], "little") == len(content),
        "meshes": len(meshes),
        "vertices": sum(len(m.vertices) for m in meshes),
        "faces": sum(len(m.faces) for m in meshes),
        "min_y": float(scene.bounds[0][1]),
        "sha256": afe.sha256(path),
    }


def run_case(case, model3, meas3, plain, meas_plain):
    name, inp = case["case"], case["input"]
    t_case = time.perf_counter()
    stages = {}
    starts = {}
    entries = {}

    best, acc, st, sc, bounds = afe.exp_fit(model3, meas3, inp, (afe.AGE_MIN, 0.5, 0.5))
    starts["S_a"] = afe.describe(best, acc, st, sc, bounds, inp)
    entries["S_a"] = best
    stages["S_a_s"] = st["seconds"]
    path = "S_a good-enough"

    production = None
    if sc["status"] != "stopped early":
        t = time.perf_counter()
        fitted = ftm.fit_body(model=plain, measurements=meas_plain, gender=inp["gender"], height_cm=inp["height_cm"],
                              weight_kg=inp["weight_kg"], waist_cm=inp["waist_cm"], chest_cm=inp["chest_cm"],
                              hip_cm=inp["hip_cm"])
        stages["production_fit_s"] = time.perf_counter() - t
        p = fitted["parameters"]
        production = {k: p[k] for k in ("age", "muscle", "proportions", "height", "weight")}
        production["equals_recorded_baseline"] = all(
            abs(p[k] - case["parameters"][k]) <= 1e-9 for k in ("age", "muscle", "proportions", "height", "weight"))

        x0 = (max(p["age"], afe.AGE_MIN), p["muscle"], p["proportions"])
        best, acc, st, sc, bounds = afe.exp_fit(model3, meas3, inp, x0)
        starts["S_b"] = afe.describe(best, acc, st, sc, bounds, inp)
        starts["S_b"]["x0_shape"] = list(x0)
        entries["S_b"] = best
        stages["S_b_s"] = st["seconds"]

        chosen = min(starts, key=lambda s: starts[s]["residual_norm"])
        path = f"S_a -> production -> S_b, {chosen} chosen"
    else:
        chosen = "S_a"

    total = time.perf_counter() - t_case

    glb = OUT / "glb" / f"{name}.glb"
    glb.parent.mkdir(parents=True, exist_ok=True)
    export_avatar(entries[chosen]["output"], model3, glb)

    return {
        "case": name, "input": inp, "path": path, "chosen": chosen,
        "stages_s": stages, "total_s": total,
        "starts": starts, "production_fit": production,
        "glb": check_glb(glb),
        "selected_result_has_all_measurements": all(
            entries[chosen]["result"].get(k) is not None for k in ("height_m", "mass_kg", "waist_m", "chest_m", "hip_m")),
    }


def gates(rows):
    prod_circ, cand_circ = [], []
    out = {}

    for row in rows:
        case = next(c for c in afe.BASE if c["case"] == row["case"])
        recorded = case["measurements"]
        prod = afe.errors_pct(case["input"], recorded)
        sel = row["starts"][row["chosen"]]
        err = sel["errors"]

        acc = {k: err[k]["pct"] is not None and err[k]["pct"] <= max(prod[k]["pct"] + 1.0, 2.0)
               for k in ("weight", "waist", "chest", "hip")}
        acc["height"] = err["height"]["pct"] is not None and err["height"]["pct"] <= 0.01
        prod_circ += [prod[k]["pct"] for k in ("waist", "chest", "hip")]
        cand_circ += [err[k]["pct"] for k in ("waist", "chest", "hip")]
        g = row["glb"]

        out[row["case"]] = {
            "G1_child_share_zero": sel["child_share"] == 0.0,
            "accuracy_per_target": acc, "accuracy_pass": all(acc.values()),
            "runtime_s": row["total_s"], "runtime_pass": row["total_s"] <= RUNTIME_LIMIT_S,
            "robust_pass": row["selected_result_has_all_measurements"] and g["header_glTF_v2"] and g["meshes"] == 1
                           and g["vertices"] == 13718 and abs(g["min_y"]) < 1e-9,
            "production_errors": prod,
            "prediction": PREDICTIONS[row["case"]],
        }

    out["_overall"] = {"mean_circumference_production": float(np.mean(prod_circ)),
                       "mean_circumference_candidate": float(np.mean(cand_circ)),
                       "pass": float(np.mean(cand_circ)) <= float(np.mean(prod_circ))}
    return out


def main():
    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    log("Output:", OUT)

    t = time.perf_counter()
    model63 = anny.Anny(local_changes=afe.REF["local_changes_list"])
    meas63 = BodyMeasurements(model63)
    model3 = anny.Anny(local_changes=afe.CONTROLS)
    meas3 = BodyMeasurements(model3)
    log(f"Models loaded (63-label and 3-label) in {time.perf_counter() - t:.1f} s; 3-label local changes: {model3.local_change_labels}")

    k1 = k1_check(model63, meas63, model3, meas3)
    data = {"k1": k1, "k1_pass": all(r["pass"] for r in k1)}
    save(data)

    if not data["k1_pass"]:
        log(f"\nK1 FAIL: 3-label model differs from the 63-label model by more than {K1_TOLERANCE_M} m. "
            "Stopping before the six-case validation.")
        return 3
    log("\nK1 PASS: 3-label model equivalent within tolerance for all 12 parameter sets.")
    del model63, meas63

    t = time.perf_counter()
    plain = anny.Anny()
    meas_plain = BodyMeasurements(plain)
    log(f"Production model loaded in {time.perf_counter() - t:.1f} s")

    rows = []
    for case in afe.BASE:
        log(f"\n===== {case['case']} =====")
        row = run_case(case, model3, meas3, plain, meas_plain)
        rows.append(row)
        data["cases"] = rows
        save(data)
        sel = row["starts"][row["chosen"]]
        log(f"{case['case']}: {row['path']} | total {row['total_s']:.1f} s {row['stages_s']} | "
            + " ".join(f"{k} {v['abs']:+.2f} ({v['pct']:.2f}%)" for k, v in sel["errors"].items())
            + f" | age {sel['age']:.3f}")

    data["gates"] = gates(rows)
    save(data)
    log("\nSaved", OUT / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
