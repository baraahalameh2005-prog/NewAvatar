"""
Adult-fit time-budget diagnostic (Option A; approved 2026-10-12). Diagnostic only:
no change to the production fit, API or defaults.

Same candidate and order as the Stage 1 validation (tools/adult_fit_validation.py):
S_a first; if it is not "good enough", the production fit, then S_b from its
solution, keeping the start with the lowest residual. Added: a per-case deadline.
  - The adult-fit optimizations (S_a, S_b) stop at the deadline and return the
    best result found so far (checked before every body evaluation, so the
    overshoot is at most one evaluation).
  - The production fit cannot be interrupted without changing production code; it
    is skipped when less than PRODUCTION_FIT_RESERVE_S remain.

Cases: F2, F3, M1 only (the three that exceeded 60 s in Stage 1).

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\adult_fit_budget_test.py
Writes only var/adult_budget/<timestamp>/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import adult_fit_experiment as afe  # applies and checks the module cache settings
from adult_fit_experiment import ftm, config, anny, BodyMeasurements, export_avatar

from scipy.optimize import least_squares

DEADLINE_S = 55.0
RUNTIME_LIMIT_S = 60.0
PRODUCTION_FIT_RESERVE_S = 25.0
CASES = ["F2_165cm_60kg", "F3_160cm_75kg", "M1_170cm_60kg"]
STAGE1 = config.MODULE_ROOT / "var" / "adult_validation" / "20261011_190649" / "results.json"
OUT = config.MODULE_ROOT / "var" / "adult_budget" / time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


class _Deadline(Exception):
    pass


def exp_fit_deadline(model, measurements, inp, x0_shape, deadline):
    """adult_fit_experiment.exp_fit with a wall-clock deadline (same bounds, solver and stop rule)."""
    start_time = time.perf_counter()
    proxy = afe.ControlledModel(model)
    targets = np.array([inp["waist_cm"] / 100, inp["chest_cm"] / 100, inp["hip_cm"] / 100,
                        inp["height_cm"] / 100, inp["weight_kg"]])
    n = len(afe.CONTROLS)
    x0 = np.array(list(x0_shape) + [afe.CONTROL_OFFSET] * n)
    lower = np.array([afe.AGE_MIN, ftm.PARAM_MIN, ftm.PARAM_MIN] + [afe.CONTROL_OFFSET - afe.CONTROL_LIMIT] * n)
    upper = np.array([ftm.AGE_MAX, ftm.PARAM_MAX, ftm.PARAM_MAX] + [afe.CONTROL_OFFSET + afe.CONTROL_LIMIT] * n)
    stats = {"anny_evaluations": 0, "shape_evaluations": 0}
    state = {"start": (0.5, 0.5), "cache": {}, "best": None}

    def run(params):
        key = tuple(float(v) for v in params)
        if key in state["cache"]:
            return state["cache"][key]
        if time.perf_counter() > deadline and state["best"] is not None:
            raise _Deadline()
        stats["shape_evaluations"] += 1
        controls = {c: float(v) - afe.CONTROL_OFFSET for c, v in zip(afe.CONTROLS, params[3:])}
        proxy.controls = controls
        output, result, h, w = ftm.evaluate_shape(proxy, measurements, ftm.GENDER_VALUES[inp["gender"]], targets[3],
                                                  targets[4], params[0], params[1], params[2],
                                                  start=state["start"], stats=stats)
        state["start"] = (h, w)
        vals = [result.get(k) for k in ("waist_m", "chest_m", "hip_m", "height_m", "mass_kg")]
        entry = dict(params=np.array(params), controls=controls, output=output, result=result, height=h, weight=w,
                     residual=None)
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
            raise afe._GoodEnough()
        return e["residual"]

    try:
        res = least_squares(objective, x0, bounds=(lower, upper), method="trf", diff_step=ftm.DIFF_STEP,
                            xtol=1e-4, ftol=1e-4, gtol=1e-6, max_nfev=30, verbose=0)
        scipy_info = {"status": int(res.status), "message": str(res.message), "nfev": int(res.nfev),
                      "njev": None if res.njev is None else int(res.njev)}
    except afe._GoodEnough:
        scipy_info = {"status": "stopped early", "message": "all targets within 0.2 % (GoodEnough)",
                      "nfev": None, "njev": None}
    except _Deadline:
        scipy_info = {"status": "deadline", "message": f"stopped at the {DEADLINE_S:.0f} s case deadline",
                      "nfev": None, "njev": None}
    best = state["best"] or run(x0)
    stats["seconds"] = time.perf_counter() - start_time
    accuracy = ftm.accuracy_report({"height": inp["height_cm"], "weight": inp["weight_kg"], "waist": inp["waist_cm"],
                                    "chest": inp["chest_cm"], "hip": inp["hip_cm"]}, best["result"])
    return best, accuracy, stats, scipy_info, (lower, upper)


def run_case(case, model3, meas3, plain, meas_plain, out_dir):
    name, inp = case["case"], case["input"]
    t0 = time.perf_counter()
    deadline = t0 + DEADLINE_S
    stages, starts, entries, notes = {}, {}, {}, []

    best, acc, st, sc, bounds = exp_fit_deadline(model3, meas3, inp, (afe.AGE_MIN, 0.5, 0.5), deadline)
    starts["S_a"] = afe.describe(best, acc, st, sc, bounds, inp)
    entries["S_a"] = best
    stages["S_a_s"] = st["seconds"]
    path = f"S_a {sc['status']}"
    chosen = "S_a"

    if sc["status"] != "stopped early":
        remaining = deadline - time.perf_counter()
        if remaining < PRODUCTION_FIT_RESERVE_S:
            notes.append(f"production fit and S_b skipped: {remaining:.1f} s left (< {PRODUCTION_FIT_RESERVE_S:.0f} s reserve)")
            path += " -> fallback skipped"
        else:
            t = time.perf_counter()
            fitted = ftm.fit_body(model=plain, measurements=meas_plain, gender=inp["gender"],
                                  height_cm=inp["height_cm"], weight_kg=inp["weight_kg"], waist_cm=inp["waist_cm"],
                                  chest_cm=inp["chest_cm"], hip_cm=inp["hip_cm"])
            stages["production_fit_s"] = time.perf_counter() - t
            p = fitted["parameters"]
            x0 = (max(p["age"], afe.AGE_MIN), p["muscle"], p["proportions"])
            best, acc, st, sc, bounds = exp_fit_deadline(model3, meas3, inp, x0, deadline)
            starts["S_b"] = afe.describe(best, acc, st, sc, bounds, inp)
            starts["S_b"]["x0_shape"] = list(x0)
            entries["S_b"] = best
            stages["S_b_s"] = st["seconds"]
            chosen = min(starts, key=lambda s: starts[s]["residual_norm"])
            path += f" -> production -> S_b {sc['status']}, {chosen} chosen"

    total = time.perf_counter() - t0
    glb = out_dir / "glb" / f"{name}.glb"
    glb.parent.mkdir(parents=True, exist_ok=True)
    export_avatar(entries[chosen]["output"], model3, glb)
    return {"case": name, "input": inp, "path": path, "chosen": chosen, "stages_s": stages, "total_s": total,
            "notes": notes, "starts": starts, "glb": str(glb), "glb_sha256": afe.sha256(glb)}


def gate(row, case):
    prod = afe.errors_pct(case["input"], case["measurements"])
    sel = row["starts"][row["chosen"]]
    err = sel["errors"]
    acc = {k: err[k]["pct"] is not None and err[k]["pct"] <= max(prod[k]["pct"] + 1.0, 2.0)
           for k in ("weight", "waist", "chest", "hip")}
    acc["height"] = err["height"]["pct"] is not None and err["height"]["pct"] <= 0.01
    return {"child_share_zero": sel["child_share"] == 0.0, "accuracy_per_target": acc,
            "accuracy_pass": all(acc.values()), "runtime_pass": row["total_s"] <= RUNTIME_LIMIT_S,
            "production_errors_pct": {k: v["pct"] for k, v in prod.items()}}


def main():
    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    log("Output:", OUT)
    log(f"Deadline {DEADLINE_S:.0f} s per case; production-fit reserve {PRODUCTION_FIT_RESERVE_S:.0f} s; cases {CASES}")

    stage1 = {r["case"]: r for r in json.loads(STAGE1.read_text())["cases"]}
    base = {c["case"]: c for c in afe.BASE}

    t = time.perf_counter()
    model3 = anny.Anny(local_changes=afe.CONTROLS)
    meas3 = BodyMeasurements(model3)
    plain = anny.Anny()
    meas_plain = BodyMeasurements(plain)
    log(f"Models loaded in {time.perf_counter() - t:.1f} s")

    rows = []
    for name in CASES:
        log(f"\n===== {name} =====")
        row = run_case(base[name], model3, meas3, plain, meas_plain, OUT)
        row["gate"] = gate(row, base[name])
        s1 = stage1[name]
        s1_err = s1["starts"][s1["chosen"]]["errors"]
        row["stage1"] = {"path": s1["path"], "total_s": s1["total_s"], "stages_s": s1["stages_s"],
                         "errors_pct": {k: v["pct"] for k, v in s1_err.items()}}
        rows.append(row)
        (OUT / "results.json").write_text(json.dumps({"cases": rows}, indent=1, default=float))

        sel = row["starts"][row["chosen"]]
        log(f"{name}: {row['path']} | total {row['total_s']:.1f} s "
            f"{ {k: round(v, 1) for k, v in row['stages_s'].items()} } {row['notes']}")
        log("  now:     " + " ".join(f"{k} {v['abs']:+.2f} ({v['pct']:.2f}%)" for k, v in sel["errors"].items())
            + f" | age {sel['age']:.3f}")
        log("  stage 1: " + " ".join(f"{k} {v:.2f}%" for k, v in row["stage1"]["errors_pct"].items())
            + f" | total {s1['total_s']:.1f} s ({s1['path']})")
        g = row["gate"]
        log(f"  gates: runtime {'PASS' if g['runtime_pass'] else 'FAIL'}, accuracy {'PASS' if g['accuracy_pass'] else 'FAIL'}"
            f" {g['accuracy_per_target']}, child share 0 {g['child_share_zero']}")

    log("\nSaved", OUT / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
