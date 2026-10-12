"""
Reproduce the six production baseline avatars in the module and compare
their GLB SHA-256 hashes with the recorded baseline.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\reproduce_baseline.py

Writes only to var/baseline_check/<timestamp>/ (GLBs, results.json) plus the
module's own caches under var/cache/.

Source of the inputs and expected hashes (verified 2026-10-11 against the
stored GLB files): D:\\BarahaAvatar_Experiment\\baseline\\baseline_results.json,
produced by baseline\\capture_baseline.py with fit_to_measurements.fit_body and
generate_avatar.export_avatar (default skin tone), the same code as here.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Applies the module cache settings before torch and anny are imported.
import fit_to_measurements as ftm
from app import config

for key in ("ANNY_CACHE_DIR", "WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
    if not config.is_inside(os.environ[key], config.MODULE_ROOT):
        sys.exit(f"Refusing to run: {key}={os.environ[key]} is outside the module.")

import anny
import trimesh
from generate_avatar import DEFAULT_SKIN_TONE, export_avatar
from measurements import BodyMeasurements


CASES = [
    ("F1_162.6cm_50kg", dict(gender="female", height_cm=162.62, weight_kg=49.99, waist_cm=71.80, chest_cm=86.60, hip_cm=92.15),
     "759f27fa5109541b007dda9ec9202c9b9ea8af57c0236c31d464cf5f6a154321"),
    ("F2_165cm_60kg", dict(gender="female", height_cm=165.0, weight_kg=60.0, waist_cm=74.0, chest_cm=90.0, hip_cm=98.0),
     "e8d40d769c801db16282ae8674b9b3539b7fc69ac0753f7d7f5be7e085fecba4"),
    ("F3_160cm_75kg", dict(gender="female", height_cm=160.0, weight_kg=75.0, waist_cm=88.0, chest_cm=104.0, hip_cm=112.0),
     "c2048b450acd87fd60e50a4f422e96004f1223927344ec9cf2e869ccdc9597a9"),
    ("M1_170cm_60kg", dict(gender="male", height_cm=170.0, weight_kg=60.0, waist_cm=74.0, chest_cm=88.0, hip_cm=90.0),
     "874379f4574576c64eebb1bf899e4776564cad9f2cd7b33419339fa283c69a9d"),
    ("M2_175cm_72kg", dict(gender="male", height_cm=175.0, weight_kg=72.0, waist_cm=82.0, chest_cm=96.0, hip_cm=98.0),
     "bb8247a0e6064551ea8ea6c1a488b293828b41057ef98a67285d01c03907a433"),
    ("M3_182cm_88kg", dict(gender="male", height_cm=182.0, weight_kg=88.0, waist_cm=92.0, chest_cm=106.0, hip_cm=104.0),
     "ae61d69ee400ee52430a15d023b2cb23822d32ddcc0e182afd27b4e74f5c9233"),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = config.MODULE_ROOT / "var" / "baseline_check" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)

    print("Output:", out)
    for key in ("ANNY_CACHE_DIR", "WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
        print(f"{key} = {os.environ[key]}")
    print("anny cache path used by Anny:", anny.paths.get_anny_cache_path())

    t = time.perf_counter()
    model = anny.Anny()
    meas = BodyMeasurements(model)
    load_seconds = time.perf_counter() - t
    print(f"Model load: {load_seconds:.1f} s", flush=True)

    results = []

    for name, inp, expected in CASES:
        print(f"\n===== {name} =====", flush=True)
        glb = out / f"{name}.glb"

        t = time.perf_counter()
        fitted = ftm.fit_body(model=model, measurements=meas, **inp)
        fit_seconds = time.perf_counter() - t

        t = time.perf_counter()
        export_avatar(fitted["output"], model, glb)
        export_seconds = time.perf_counter() - t

        actual = sha256(glb)
        mesh = list(trimesh.load(glb).geometry.values())[0]

        results.append({
            "case": name,
            "input": inp,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
            "fit_seconds": fit_seconds,
            "export_seconds": export_seconds,
            "glb": str(glb),
            "glb_bytes": glb.stat().st_size,
            "glb_vertices": len(mesh.vertices),
            "parameters": fitted["parameters"],
        })

        print(f"{name}: {'MATCH' if actual == expected else 'MISMATCH'} "
              f"fit {fit_seconds:.1f} s, export {export_seconds:.2f} s, {actual}", flush=True)

    summary = {
        "model_load_seconds": load_seconds,
        "skin_tone": DEFAULT_SKIN_TONE,
        "all_match": all(r["match"] for r in results),
        "cases": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=float))

    print("\n===== SUMMARY =====")
    for r in results:
        print(f"{r['case']:16} {'MATCH   ' if r['match'] else 'MISMATCH'} "
              f"fit {r['fit_seconds']:5.1f} s  expected {r['expected_sha256'][:16]}  actual {r['actual_sha256'][:16]}")
    print("ALL MATCH:", summary["all_match"])

    return 0 if summary["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
