"""
Diagnostic adult-fit experiment (approved 2026-10-11). Does not change the
production fit, measurements, export or API.

AC67 setup from the experiment's Stage 5 (age >= 0.67 plus Anny's bust, waist
and hips circumference controls within +-0.6), run from three starting points:
  S_a  Stage 5 start (age 0.67, muscle 0.5, proportions 0.5)  -> K0 control
  S_b  production solution, age raised to >= 0.67
  S_c  age 0.85, muscle 0.5, proportions 0.5
All controls start at 0. K0 (S_a reproduces the recorded Stage 5 AC67
parameters within 1e-6) is a hard stop: S_b and S_c are not run if it fails.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\adult_fit_experiment.py

Writes only to var/adult_experiment/<timestamp>/ and the module cache.
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import fit_to_measurements as ftm  # applies the module cache settings first
from app import config

for _key in ("ANNY_CACHE_DIR", "WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
    if not config.is_inside(os.environ[_key], config.MODULE_ROOT):
        sys.exit(f"Refusing to run: {_key}={os.environ[_key]} is outside the module.")

from scipy.optimize import least_squares

import anny
from generate_avatar import build, export_avatar
from measurements import BodyMeasurements

REF = json.loads((TOOLS / "stage5_ac67_reference.json").read_text())
BASE = json.loads((TOOLS / "baseline_cases.json").read_text())["cases"]

CONTROLS = REF["controls"]
CONTROL_LIMIT = REF["control_limit"]
CONTROL_OFFSET = REF["control_offset"]
AGE_MIN = REF["age_min"]
K0_TOLERANCE = 1e-6
CHILD_ANCHOR, YOUNG_ANCHOR = 1 / 3, 2 / 3  # Anny age anchors: linspace(-1/3, 1, 5)

OUT = config.MODULE_ROOT / "var" / "adult_experiment" / time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ControlledModel:
    """Same as the Stage 2 proxy: passes the current controls as local changes."""

    def __init__(self, model):
        self._model = model
        self.controls = {}

    def __call__(self, phenotype_kwargs=None, **kwargs):
        return self._model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=dict(self.controls), **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class _GoodEnough(Exception):
    pass


def exp_fit(model, measurements, inp, x0_shape):
    """Stage 5 exp_fit_s with controls on and age_min 0.67; x0_shape = (age, muscle, proportions)."""

    start_time = time.perf_counter()
    proxy = ControlledModel(model)
    targets = np.array([inp["waist_cm"] / 100, inp["chest_cm"] / 100, inp["hip_cm"] / 100, inp["height_cm"] / 100, inp["weight_kg"]])
    n = len(CONTROLS)
    x0 = np.array(list(x0_shape) + [CONTROL_OFFSET] * n)
    lower = np.array([AGE_MIN, ftm.PARAM_MIN, ftm.PARAM_MIN] + [CONTROL_OFFSET - CONTROL_LIMIT] * n)
    upper = np.array([ftm.AGE_MAX, ftm.PARAM_MAX, ftm.PARAM_MAX] + [CONTROL_OFFSET + CONTROL_LIMIT] * n)
    stats = {"anny_evaluations": 0, "shape_evaluations": 0}
    state = {"start": (0.5, 0.5), "cache": {}, "best": None}

    def run(params):
        key = tuple(float(v) for v in params)
        if key in state["cache"]:
            return state["cache"][key]
        stats["shape_evaluations"] += 1
        controls = {c: float(v) - CONTROL_OFFSET for c, v in zip(CONTROLS, params[3:])}
        proxy.controls = controls
        output, result, h, w = ftm.evaluate_shape(proxy, measurements, ftm.GENDER_VALUES[inp["gender"]], targets[3], targets[4],
                                                  params[0], params[1], params[2], start=state["start"], stats=stats)
        state["start"] = (h, w)
        vals = [result.get(k) for k in ("waist_m", "chest_m", "hip_m", "height_m", "mass_kg")]
        entry = dict(params=np.array(params), controls=controls, output=output, result=result, height=h, weight=w, residual=None)
        if all(ftm._finite(v) for v in vals):
            entry["residual"] = (np.array(vals, dtype=float) - targets) / targets
            if state["best"] is None or np.linalg.norm(entry["residual"]) < np.linalg.norm(state["best"]["residual"]):
                state["best"] = entry
        state["cache"][key] = entry
        return entry

    def objective(params):
        e = run(params)
        if e["residual"] is None:
            return np.full(len(targets), 10.0)
        if np.max(np.abs(e["residual"])) <= ftm.GOOD_ENOUGH_RELATIVE_ERROR:
            raise _GoodEnough()
        return e["residual"]

    scipy_info = {}
    try:
        res = least_squares(objective, x0, bounds=(lower, upper), method="trf", diff_step=ftm.DIFF_STEP,
                            xtol=1e-4, ftol=1e-4, gtol=1e-6, max_nfev=30, verbose=0)
        scipy_info = {"status": int(res.status), "message": str(res.message), "nfev": int(res.nfev),
                      "njev": None if res.njev is None else int(res.njev)}
    except _GoodEnough:
        scipy_info = {"status": "stopped early", "message": "all targets within 0.2 % (GoodEnough)",
                      "nfev": None, "njev": None}
    best = state["best"] or run(x0)
    stats["seconds"] = time.perf_counter() - start_time
    accuracy = ftm.accuracy_report({"height": inp["height_cm"], "weight": inp["weight_kg"], "waist": inp["waist_cm"],
                                    "chest": inp["chest_cm"], "hip": inp["hip_cm"]}, best["result"])
    return best, accuracy, stats, scipy_info, (lower, upper)


def child_share(age):
    return float(min(1.0, max(0.0, (YOUNG_ANCHOR - age) / (YOUNG_ANCHOR - CHILD_ANCHOR))))


def ratios(model, output, result, neck03):
    vertices = output["rest_vertices"][0].detach().cpu().numpy()
    height = result["height_m"]
    neck_z = float(output["rest_bone_poses"][0, neck03, 2, 3])
    return {
        "head_share": float((vertices[:, 2].max() - neck_z) / height),
        "shoulder_joint_per_height": result["shoulder_width_m"] / height,
        "leg_length_per_height": result["leg_length_m"] / height,
    }


def errors_pct(inp, result):
    pairs = {"height": (inp["height_cm"], result.get("height_m"), 100), "weight": (inp["weight_kg"], result.get("mass_kg"), 1),
             "waist": (inp["waist_cm"], result.get("waist_m"), 100), "chest": (inp["chest_cm"], result.get("chest_m"), 100),
             "hip": (inp["hip_cm"], result.get("hip_m"), 100)}
    out = {}
    for name, (req, actual, scale) in pairs.items():
        if actual is None:
            out[name] = {"requested": req, "actual": None, "abs": None, "pct": None}
        else:
            a = actual * scale
            out[name] = {"requested": req, "actual": a, "abs": a - req, "pct": abs(a - req) / abs(req) * 100}
    return out


def at_bounds(params, lower, upper):
    names = ["age", "muscle", "proportions"] + [c.split("-")[1] for c in CONTROLS]
    hits = []
    for name, v, lo, hi in zip(names, params, lower, upper):
        if abs(v - lo) <= 1e-4:
            hits.append(f"{name}=lower")
        elif abs(v - hi) <= 1e-4:
            hits.append(f"{name}=upper")
    return hits


def describe(best, acc, stats, scipy_info, bounds, inp):
    p = best["params"]
    return {
        "age": float(p[0]), "muscle": float(p[1]), "proportions": float(p[2]),
        "height_param": float(best["height"]), "weight_param": float(best["weight"]),
        "controls_exact": {c.split("-")[1]: float(v) for c, v in best["controls"].items()},
        "residual_norm": float(np.linalg.norm(best["residual"])) if best["residual"] is not None else None,
        "errors": errors_pct(inp, best["result"]),
        "at_bounds": at_bounds(p, *bounds) + (["weight_param=upper"] if best["weight"] >= ftm.PARAM_MAX - 1e-6 else [])
                     + (["weight_param=lower"] if best["weight"] <= ftm.PARAM_MIN + 1e-6 else []),
        "child_share": child_share(float(p[0])),
        "age_floor_binding": abs(float(p[0]) - AGE_MIN) <= 1e-6,
        "seconds": stats["seconds"], "shape_evals": stats["shape_evaluations"], "anny_evals": stats["anny_evaluations"],
        "scipy": scipy_info,
    }


def main():
    OUT.mkdir(parents=True)
    log("Output:", OUT)
    for key in ("ANNY_CACHE_DIR", "WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
        log(f"{key} = {os.environ[key]}")

    t = time.perf_counter()
    model = anny.Anny(local_changes=REF["local_changes_list"])
    meas = BodyMeasurements(model)
    log(f"Controlled model ({len(REF['local_changes_list'])} local-change labels) loaded in {time.perf_counter() - t:.1f} s")
    neck03 = {n: i for i, n in enumerate(model.bone_labels)}["neck03"]
    ac67 = {r["case"]: r for r in REF["ac67"]}

    results = {c["case"]: {"input": c["input"], "starts": {}} for c in BASE}

    # ---- Phase 1: S_a for all cases, then K0.
    k0 = {}
    for case in BASE:
        name, inp = case["case"], case["input"]
        log(f"\n[S_a] {name}")
        best, acc, st, sc, bounds = exp_fit(model, meas, inp, (AGE_MIN, 0.5, 0.5))
        d = describe(best, acc, st, sc, bounds, inp)
        glb = OUT / "starts" / "S_a" / f"{name}.glb"
        glb.parent.mkdir(parents=True, exist_ok=True)
        export_avatar(best["output"], model, glb)
        d["glb_sha256"] = sha256(glb)
        results[name]["starts"]["S_a"] = d
        results[name]["_best_entries"] = {"S_a": best}

        r = ac67[name]
        diffs = {k: abs(d[k] - r[k]) for k in ("age", "muscle", "proportions", "height_param", "weight_param")}
        diffs.update({f"control_{k}": abs(d["controls_exact"][k] - v) for k, v in r["controls_exact"].items()})
        k0[name] = {"max_param_diff": max(diffs.values()), "pass": max(diffs.values()) <= K0_TOLERANCE,
                    "glb_matches_stage5": d["glb_sha256"] == r["glb_sha256"], "diffs": diffs}
        log(f"  {d['seconds']:.1f} s, scipy {sc}, K0 max diff {k0[name]['max_param_diff']:.2e}, "
            f"GLB = Stage 5: {k0[name]['glb_matches_stage5']}")

    summary = {"k0": k0}
    if not all(v["pass"] for v in k0.values()):
        log("\nK0 FAIL: S_a does not reproduce Stage 5 AC67 within", K0_TOLERANCE, "-> stopping (no S_b / S_c).")
        save(results, summary)
        return 3
    log("\nK0 PASS for all 6 cases.")

    # ---- Phase 2: S_b and S_c.
    for case in BASE:
        name, inp, prod = case["case"], case["input"], case["parameters"]
        for label, x0 in (("S_b", (max(prod["age"], AGE_MIN), prod["muscle"], prod["proportions"])),
                          ("S_c", (0.85, 0.5, 0.5))):
            log(f"\n[{label}] {name} x0 shape {tuple(round(v, 4) for v in x0)}")
            best, acc, st, sc, bounds = exp_fit(model, meas, inp, x0)
            d = describe(best, acc, st, sc, bounds, inp)
            d["x0_shape"] = list(x0)
            results[name]["starts"][label] = d
            results[name]["_best_entries"][label] = best
            log(f"  {d['seconds']:.1f} s, scipy {sc}, residual {d['residual_norm']:.5f}")

    # ---- Phase 3: select best start, export, ratios, references.
    for case in BASE:
        name, inp, prod = case["case"], case["input"], case["parameters"]
        entry = results[name]
        starts = entry["starts"]
        chosen = min(starts, key=lambda s: starts[s]["residual_norm"])
        entry["selected_start"] = chosen
        best = entry.pop("_best_entries")[chosen]

        adult_glb = OUT / "adult" / f"{name}.glb"
        adult_glb.parent.mkdir(parents=True, exist_ok=True)
        export_avatar(best["output"], model, adult_glb)
        entry["adult_glb_sha256"] = sha256(adult_glb)

        prod_src = config.MODULE_ROOT / "var" / "measurement_check" / "after" / f"{name}.glb"
        prod_glb = OUT / "production" / f"{name}.glb"
        prod_glb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prod_src, prod_glb)
        entry["production_glb_matches_baseline"] = sha256(prod_glb) == case["glb_sha256"]

        # Ratios: selected adult body, production body (rebuilt from its recorded parameters
        # with this model and zero controls), and the Anny adult reference bodies.
        entry["adult_ratios"] = ratios(model, best["output"], best["result"], neck03)
        g = ftm.gender_value(prod["gender"])
        prod_out = build(model=model, gender=g, height=prod["height"], weight=prod["weight"], age=prod["age"],
                         muscle=prod["muscle"], proportions=prod["proportions"])
        prod_res = meas.measure(prod_out)
        entry["production_ratios"] = ratios(model, prod_out, prod_res, neck03)
        entry["production_errors"] = errors_pct(inp, prod_res)
        entry["production_age"] = prod["age"]
        entry["production_child_share"] = child_share(prod["age"])

        refs = []
        for age in (0.77, 1.0):
            for weight in (0.25, 0.5, 0.75):
                for muscle in (0.25, 0.5, 0.75):
                    h = ftm.solve_height(model, meas, g, age, weight, muscle, 0.5, inp["height_cm"] / 100)
                    out = build(model=model, gender=g, height=h, weight=weight, age=age, muscle=muscle, proportions=0.5)
                    refs.append(ratios(model, out, meas.measure(out), neck03))
        entry["reference_range"] = {k: [min(r[k] for r in refs), max(r[k] for r in refs)] for k in refs[0]}
        entry["reference_count"] = len(refs)
        log(f"\n{name}: selected {chosen}; ratios adult {entry['adult_ratios']} range {entry['reference_range']}")

    summary.update(gates(results))
    save(results, summary)
    return 0


def gates(results):
    prod_circ, adult_circ = [], []
    per_case = {}
    for name, e in results.items():
        a = e["starts"][e["selected_start"]]["errors"]
        p = e["production_errors"]
        acc_ok = {}
        for k in ("waist", "chest", "hip", "weight"):
            limit = max(p[k]["pct"] + 1.0, 2.0)
            acc_ok[k] = a[k]["pct"] is not None and a[k]["pct"] <= limit
        acc_ok["height"] = a["height"]["pct"] is not None and a["height"]["pct"] <= 0.01
        prod_circ += [p[k]["pct"] for k in ("waist", "chest", "hip")]
        adult_circ += [a[k]["pct"] for k in ("waist", "chest", "hip")]
        total_seconds = sum(s["seconds"] for s in e["starts"].values())
        sel = e["starts"][e["selected_start"]]
        g2 = {k: e["reference_range"][k][0] <= v <= e["reference_range"][k][1] for k, v in e["adult_ratios"].items()}
        g2_prod = {k: e["reference_range"][k][0] <= v <= e["reference_range"][k][1] for k, v in e["production_ratios"].items()}
        per_case[name] = {
            "G1_child_share_zero": sel["child_share"] == 0.0,
            "age_floor_binding": sel["age_floor_binding"],
            "accuracy_per_target": acc_ok, "accuracy_pass": all(acc_ok.values()),
            "time_all_starts_s": total_seconds, "time_pass": total_seconds <= 60,
            "G2_adult_in_range": g2, "G2_production_in_range": g2_prod,
        }
    return {
        "gates_per_case": per_case,
        "mean_circumference_error": {"production": float(np.mean(prod_circ)), "adult_selected": float(np.mean(adult_circ)),
                                     "pass": float(np.mean(adult_circ)) <= float(np.mean(prod_circ))},
    }


def save(results, summary):
    clean = {n: {k: v for k, v in e.items() if not k.startswith("_")} for n, e in results.items()}
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "cases": clean}, indent=1, default=float))
    log("Saved", OUT / "results.json")


if __name__ == "__main__":
    sys.exit(main())
