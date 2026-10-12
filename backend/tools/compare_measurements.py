"""
Rebuild the six baseline bodies from their recorded fitted parameters (one
model evaluation each, no fitting), measure them, export and hash the GLB.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\compare_measurements.py <label>

Writes var/measurement_check/<label>.json and the GLBs to
var/measurement_check/<label>/. Compare two labels with:
    ..\\.venv\\Scripts\\python.exe -B tools\\compare_measurements.py --compare <before> <after>
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app import config

OUT = config.MODULE_ROOT / "var" / "measurement_check"
CASES = json.loads((Path(__file__).with_name("baseline_cases.json")).read_text())["cases"]


def run(label):
    import fit_to_measurements  # applies the module cache settings first

    for key in ("ANNY_CACHE_DIR", "WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
        if not config.is_inside(os.environ[key], config.MODULE_ROOT):
            sys.exit(f"Refusing to run: {key} is outside the module.")

    import anny
    from generate_avatar import build, export_avatar
    from measurements import BodyMeasurements

    folder = OUT / label
    folder.mkdir(parents=True)

    model = anny.Anny()
    meas = BodyMeasurements(model)
    results = []

    for case in CASES:
        p = case["parameters"]
        output = build(
            model=model,
            gender=fit_to_measurements.gender_value(p["gender"]),
            height=p["height"],
            weight=p["weight"],
            age=p["age"],
            muscle=p["muscle"],
            proportions=p["proportions"],
        )

        timings = []
        for _ in range(3):
            t = time.perf_counter()
            result = meas.measure(output)
            timings.append(time.perf_counter() - t)

        glb = folder / f"{case['case']}.glb"
        export_avatar(output, model, glb)
        sha = hashlib.sha256(glb.read_bytes()).hexdigest()

        results.append({
            "case": case["case"],
            "glb_sha256": sha,
            "glb_matches_baseline": sha == case["glb_sha256"],
            "measure_seconds": min(timings),
            "measurements": {k: v for k, v in result.items() if k != "hip_z"},
        })
        print(f"{case['case']:16} glb {'MATCH' if sha == case['glb_sha256'] else 'MISMATCH'}"
              f"  measure {min(timings):.3f} s", flush=True)

    (OUT / f"{label}.json").write_text(json.dumps(results, indent=1, default=float))
    return 0 if all(r["glb_matches_baseline"] for r in results) else 1


def compare(before_label, after_label):
    before = {r["case"]: r for r in json.loads((OUT / f"{before_label}.json").read_text())}
    after = {r["case"]: r for r in json.loads((OUT / f"{after_label}.json").read_text())}

    changed_keys = set()
    for name, a in after.items():
        b = before[name]
        for key in sorted(set(a["measurements"]) | set(b["measurements"])):
            if a["measurements"].get(key) != b["measurements"].get(key):
                changed_keys.add(key)

    print("Keys whose value changed in any case:", sorted(changed_keys))

    cols = sorted(changed_keys)
    for name in after:
        b, a = before[name]["measurements"], after[name]["measurements"]
        cells = []
        for key in cols:
            fb = "None" if b.get(key) is None else f"{b[key] * 100:.1f}"
            fa = "None" if a.get(key) is None else f"{a[key] * 100:.1f}"
            cells.append(f"{key[:-2]} {fb}->{fa}")
        print(f"{name:16} " + " | ".join(cells))
        print(f"{'':16} glb before {before[name]['glb_matches_baseline']} after {after[name]['glb_matches_baseline']}"
              f"; measure {before[name]['measure_seconds']:.3f} -> {after[name]['measure_seconds']:.3f} s")

    return changed_keys


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(run(sys.argv[1]))
