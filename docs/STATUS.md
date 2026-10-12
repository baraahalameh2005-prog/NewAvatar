# Module status log

Only what has actually been done and verified is listed here.

## Step 1: structure and safe configuration (2026-10-11)

Created:
- `backend/app/config.py`: host 127.0.0.1, port 8002; output folder `var/generated`;
  Anny cache `var/cache/anny`; Warp cache `var/cache/warp`; each can be overridden by an
  environment variable (`AVATAR_HOST`, `AVATAR_PORT`, `AVATAR_OUTPUT_DIR`, `ANNY_CACHE_DIR`,
  `WARP_CACHE_PATH`), but any folder inside a protected project is refused. The shared
  `generated_avatar.glb` copy of the production API is turned off.
- `backend/tests/test_config.py`: 11 standard-library tests.
- `README.md`, `.gitignore`, `frontend/README.md` (placeholder), this file.
- Git repository initialised; nothing committed yet.

Verified:
- `python -B -m unittest tests.test_config` (system Python 3.12.6, run from `backend/`):
  11 tests, OK.
- Protected paths refused, including case and slash variants and environment overrides.
- `D:\BarahaAvatarModule` is not mistaken for `D:\BarahaAvatar` (name-prefix check).
- No `var/` folder is created by loading the configuration.

Not done yet: no backend code copied, no dependencies installed, no Anny loaded.

## Step 2: production backend copied unchanged (2026-10-11)

Source: `D:\BarahaAvatar` working tree at HEAD `45816ad` plus its uncommitted and untracked
changes. Files copied byte for byte; the source was only read.

| Source | Destination | SHA-256 (source = destination) |
|---|---|---|
| `api.py` | `backend/api.py` | `a9f31675b8a5176c39212982974eeb762da6fcfb985be3dde0034a4e993b2979` |
| `fit_to_measurements.py` | `backend/fit_to_measurements.py` | `7da4eac27aad786fd89368bf2cce03c90142391a85f0c35245bfe4941dbb3799` |
| `generate_avatar.py` | `backend/generate_avatar.py` | `c4486002b9c2b432db9d52f10b793084a739b2522c703e4ee548bdfff3de4702` |
| `measurements.py` | `backend/measurements.py` | `22984b482b19e4cc96c159f158aa6a690a1f1e69d59d55762659b2dcbfb32d4` |
| `tests/test_avatar.py` | `backend/tests/test_avatar.py` | `56d092836d3d2d65c0563f51a9907a970e238ee3208f19c504fbd67d2a7f3cc6` |
| `API.md` | `docs/API.md` | `8dbd96663b6fb8db8ed4c6305241ef6a2ac7ea478b8bf3477f780001f530aa58` |

Verified: every destination hash and size equals the source; the source hashes were the same
before and after copying and equal the verified backup in `D:\Avatar_Backups\BarahaAvatar`.

Layout note: the Python files sit directly in `backend/`, not `backend/app/`, because they
import each other by plain module name (`import fit_to_measurements`) and
`tests/test_avatar.py` puts its parent folder on `sys.path`. Moving them into `app/` would
need code changes; that is a later, separate step.

At that point `api.py` did not use `app/config.py` yet (fixed in step 4).

## Step 3: module environment (2026-10-11)

- `.venv` created with system Python 3.12.6 at `D:\BarahaAvatarModule\.venv` (1.31 GB, not in Git).
- `backend/requirements.txt`: the 11 direct dependencies (anny, torch, numpy, scipy, trimesh,
  rtree, fastapi, uvicorn, pydantic, starlette, httpx).
- `backend/constraints-production.txt`: 152 exact versions read from the production
  `.venv`'s `*.dist-info` folder names (production Python was not run).
- Installed with `pip install -r requirements.txt -c constraints-production.txt
  --extra-index-url https://download.pytorch.org/whl/cpu`, pip cache off, pip temp files in
  `var/tmp`. Exit code 0.

Verified:
- 39 packages installed, 0 version differences from production; `pip check`: no broken
  requirements. Key versions: anny 0.6.1, torch 2.14.1+cpu, numpy 2.5.3, fastapi 0.142.2,
  scipy 1.18.1, trimesh 5.1.0.
- The installed `anny` package (1,599 files, including all model data) is byte-identical to
  production except `anny/examples/interactive_demo.py`, which was edited locally in
  production on 2026-10-02 (`delete=False` on two temporary files). Not used by the backend.
- pip is 24.2 here (production 26.2.1); not pinned, not used at runtime.
- Production-only packages not installed (Jupyter, Gradio, matplotlib, pandas, py_soma_x
  and others): checked that neither the backend nor Anny's normal code path imports them.

## Step 4: backend connected to the module configuration (2026-10-11)

Changed (reviewed diff, then three approved fixes):
- `backend/api.py`: output folder from `config.py` (`var/generated`); the shared
  `generated_avatar.glb` copy is off (`WRITE_LEGACY_AVATAR = False`) and
  `legacy_avatar_url` is `null`; `python api.py` starts uvicorn on the configured host/port.
- `backend/fit_to_measurements.py`, `generate_avatar.py`, `measurements.py`: call
  `apply_environment(settings)` before importing torch and anny.
- `backend/app/config.py`: also sets `TORCHINDUCTOR_CACHE_DIR` (`var/cache/torchinductor`).
  Reason: the installed torch resolves this folder in
  `torch/_inductor/runtime/cache_dir_utils.py` and creates it on `import torch`
  (via `torch._dynamo.package`); without the setting it goes to the user's TEMP folder.
- `backend/tests/test_avatar.py`: expects no legacy file; legacy copy still tested when turned
  on; new tests for the module's output folder and the Anny, Warp and PyTorch cache folders;
  the static-file test now serves fixture files from a temporary folder (each blocked file
  exists in it, so a 404 proves the filter works).
- `backend/tests/test_config.py`: the "creates no folders" test uses an empty stand-in module
  root; the PyTorch setting is covered.
- `docs/API.md`: port 8002, module run/test commands, `legacy_avatar_url: null`,
  static-file note.

First test run: 55 tests, 4 failures, all from test assumptions (an existing `var/` folder;
production files that were not copied). Fixed as above.

Verified (final run, `AVATAR_SKIP_MODEL_TESTS=1`, from `backend/`):
- `python -B -m unittest tests.test_config tests.test_avatar`: **56 tests, OK, 9 skipped**
  (the real-model tests).
- Settings: output `var/generated`, caches `var/cache/anny`, `var/cache/warp`,
  `var/cache/torchinductor`, all inside the module; host 127.0.0.1, port 8002.
- After the run: only `var/cache/torchinductor` (empty) was created; the shared
  `C:\Users\ACTC\.cache\anny`, `...\NVIDIA\warp` and `%TEMP%\torchinductor_ACTC` were not
  changed; protected projects unchanged.

Not done: the API server has not been started, the Anny model has not been loaded, no avatar
has been generated in the module, no experimental fix has been ported.

## Step 5: production baseline reproduced in the module (2026-10-11)

Inputs and expected hashes: `D:\BarahaAvatar_Experiment\baseline\baseline_results.json`
(read only). Checked before running: the recorded hashes equal the stored baseline GLBs, and
the baseline was made with the same `fit_to_measurements.py` and `measurements.py` (identical
bytes) and `generate_avatar.py` (identical apart from CRLF/LF line endings) as the module.

New file: `backend/tools/reproduce_baseline.py` (fit_body, then export_avatar with the default
skin tone, the same path as the baseline capture; refuses to run if any cache points outside
the module). Run from `backend/` with TEMP in `var/tmp`. Exit code 0, wall time 279 s.

| Case | Expected = actual SHA-256 | Fit time |
|---|---|---|
| F1_162.6cm_50kg | `759f27fa5109541b007dda9ec9202c9b9ea8af57c0236c31d464cf5f6a154321` | 13.8 s |
| F2_165cm_60kg | `e8d40d769c801db16282ae8674b9b3539b7fc69ac0753f7d7f5be7e085fecba4` | 14.3 s |
| F3_160cm_75kg | `c2048b450acd87fd60e50a4f422e96004f1223927344ec9cf2e869ccdc9597a9` | 11.7 s |
| M1_170cm_60kg | `874379f4574576c64eebb1bf899e4776564cad9f2cd7b33419339fa283c69a9d` | 10.0 s |
| M2_175cm_72kg | `bb8247a0e6064551ea8ea6c1a488b293828b41057ef98a67285d01c03907a433` | 8.1 s |
| M3_182cm_88kg | `ae61d69ee400ee52430a15d023b2cb23822d32ddcc0e182afd27b4e74f5c9233` | 14.9 s |

All 6 match; re-hashed independently from the files on disk.

- Outputs: `var/baseline_check/20261011_165253/` (6 GLBs, `results.json`); log
  `var/baseline_check/run_log.txt`; snapshots `snapshot_before.txt` / `snapshot_after.txt`.
- First model load (empty module cache): **199.7 s**. It built `var/cache/anny/v12/` with 2 files,
  741 MB (`build_model_data_93dd57e1…` 215,931,168 B and `load_data_5c4a3ed7…` 561,253,600 B).
  Warp wrote 3 small files to `var/cache/warp`; `var/cache/torchinductor` stayed empty.
  Load time with the cache already built: not measured yet.
- Only log warning: Warp found no NVIDIA CUDA driver (CPU only, as in production).
- Before/after snapshots identical: shared Anny/Warp/torchinductor caches, the 3 protected Git
  states, the original GLB, the 6 experiment baseline GLBs and the protected file trees.

## Step 6: API server start test (2026-10-11)

New file: `backend/tools/server_smoke_test.py`. It refuses to start if port 8002 is in use,
starts the documented command `..\.venv\Scripts\python.exe api.py` (cwd `backend/`) in its own
process group, waits for `/health`, calls read-only endpoints, then stops the server with
Ctrl+Break. It never calls `/generate-avatar`.

Run: `..\.venv\Scripts\python.exe -B tools\server_smoke_test.py` from `backend/`, TEMP in
`var/tmp`. Port 8002 was free beforehand (no TCP entries).

Results:
- Startup: ready after **10.8 s**, including loading Anny from the module cache built in step 5
  (first load had taken 199.7 s). Server log: "Application startup complete", "Uvicorn running
  on http://127.0.0.1:8002". Warp kernel cache: `D:\BarahaAvatarModule\var\cache\warp\1.17.0`.
- `GET /health` → 200 `{"status": "ok", "model_loaded": true}`
- `GET /avatar-options` → 200, `api_version` 1; height 100–230 cm, weight 25–250 kg, waist
  40–200 cm, chest 50–200 cm, hip 50–200 cm (all required); gender `female`/`male`, default
  `female`, case-insensitive; skin_tone `#RRGGBB`, default `#c8997e`.
- `GET /` → 200 `{"message": "Avatar API is running"}`
- `GET /models/generated/not-a-real-id.glb` → 404 (no avatar generated; `var/generated` not
  created).
- Shutdown: Ctrl+Break → log shows "Shutting down", "Application shutdown complete",
  "Finished server process". Exit code 3: after a graceful shutdown uvicorn re-raises the
  captured signal (`uvicorn/server.py`, `signal.raise_signal`), and the Windows default action
  for Ctrl+Break ends the process with code 3. No server process left; port 8002 free.
- The PID in the log (24856) differs from the started PID (16004) because the venv's
  `python.exe` on Windows is a launcher that starts the base interpreter as a child.
- Only warning: no NVIDIA CUDA driver (CPU only).
- The documented command runs without `-B`, so `backend/__pycache__` and
  `backend/app/__pycache__` now exist (inside the module, ignored by Git).

Integrity: before/after snapshots identical (shared Anny/Warp/torchinductor caches, the 3
protected Git states, original GLB, the 6 experiment baseline GLBs, protected file trees,
module cache size). Files: `var/server_check/20261011_170050/` (`server_log.txt`,
`responses.json`), `var/server_check/snapshot_before.txt` / `snapshot_after.txt`.

## Milestone 1: end-to-end API generation (2026-10-11)

`tools/server_smoke_test.py --generate` (option added): starts `api.py`, sends an invalid
request (height 50) and baseline case F1, checks the result, stops the server with Ctrl+Break.
All 14 checks passed: 422 on `body.height` with no file written; 200 with all response fields,
`legacy_avatar_url` null; one new file `var/generated/d4f0f413c1a446f9a7687740480e5255.glb`;
URL returns 200 with identical bytes; GLB header glTF v2, one mesh, 13,718 vertices, 27,420
faces, feet at y = 0; SHA-256 = baseline F1 `759f27fa…`. Server ready in 6.6 s, fit 11.5 s.
Integrity snapshots identical. Files: `var/server_check/20261011_170940/`,
`var/server_check/m1_snapshot_before.txt` / `m1_snapshot_after.txt`.

## Milestone 2: fitting quality (read-only, 2026-10-11)

Source: the six baseline records (same code, hash-identical output). The fit uses only height,
weight, waist, chest and hip; neck, upper arm, thigh, calf and shoulder are reported only.
- Fitted targets: height within 0.01 %; worst errors M1 waist +6.0 cm (8.1 %), chest +6.4 cm
  (7.2 %); F3 weight −4.7 kg (6.3 %), waist +6.2 cm (7.0 %), chest −4.7 cm (4.6 %); M2 waist
  +4.0 cm (4.8 %). Fitted age 0.50–0.66 for every case.
- Reported values that are wrong: neck 89–141 cm for M1–M3 (also F2 49 cm); upper arm
  137–153 cm for F3, M2, M3 (armpit slice). Thigh/calf left and right differ by up to 2.6 cm on
  a symmetric body (slice placement).
- Shoulder width is a joint-to-joint estimate (30.6–34.8 cm); no validated definition exists.

## Milestone 3: measurement-reading fixes ported (2026-10-11)

Ported (approved): mid-upper-arm slice (experiment Stage 1, commit `7896af5`), neck slice along
neck01 → head and per-side thigh/calf slices (experiment Stage 7, uncommitted).
Provenance: the Stage 7 validation log records the tested `measurements.py` as SHA-256
`8c9afba9…`, which is the experiment's current file; its parent `43a885ce…` is `7896af5`. The
experiment's tests changed after that run, so they were reviewed and run here, not assumed.

Changed: `backend/measurements.py` (now identical to the validated file apart from the module's
config header), `backend/tests/test_avatar.py` (2 polygon unit tests, 3 real-model tests).
Not ported: the experiment's `api.py` guard and anything else. Fitting, model and export are
unchanged.

New: `backend/tools/compare_measurements.py` (rebuilds the 6 bodies from recorded parameters,
no fitting) and `backend/tools/baseline_cases.json` (recorded inputs, parameters, hashes).

Results:
- Tests: `tests.test_config tests.test_avatar` with the real-model tests: 61 tests, OK.
- All 6 GLB hashes unchanged before and after (equal to the baseline).
- Changed values (cm, before → after): neck F1 32.0→30.1, F2 49.3→31.2, F3 42.3→33.9,
  M1 89.0→38.2, M2 117.0→39.6, M3 141.1→44.7; upper arm 28.5→25.7, 33.7→31.0, 153.4→30.6,
  33.9→30.2, 138.5→34.7, 137.4→41.9; thigh left now equals right (45.2, 49.9, 56.4, 48.1,
  51.9, 59.1); calf left = right (33.4, 35.7, 39.6, 35.3, 38.8, 42.0). Matches the experiment's
  validated values. No other measurement changed.
- `measure()` time per body 0.044–0.131 s before, 0.055–0.117 s after.
- Integrity snapshots identical. Files: `var/measurement_check/` (before/after JSON, GLBs,
  snapshots).

## Adult-fit diagnostic experiment (2026-10-11, diagnostic only)

Stage 5 AC67 setup (age ≥ 0.67, bust/waist/hips controls ±0.6, 63-label controlled model) from
three starts: S_a Stage 5 start, S_b production solution with age raised to 0.67, S_c age 0.85.
New files: `backend/tools/adult_fit_experiment.py`, `backend/tools/stage5_ac67_reference.json`.
Outputs: `var/adult_experiment/20261011_172629/` (run_log, results.json, `production/` and
`adult/` GLBs side by side, `starts/S_a/`). Wall time 482 s.

- K0 PASS: S_a reproduced the recorded Stage 5 AC67 parameters exactly (max difference 0) and
  byte-identical GLBs for all 6 cases.
- Child-shape share 0 in every selected fit; the 0.67 age floor binds only for F3.
- Accuracy gate (each target ≤ max(production + 1 point, 2 %)): PASS F1, F2, M2, M3; FAIL F3
  (weight −6.94 kg, 9.3 %, production 6.3 %), FAIL M1 (hip +5.20 cm, 5.8 %, production 3.9 %).
- Mean circumference error, all 18 targets: production 2.99 %, selected adult 1.23 %. Mean of
  each case's worst circumference error: production 4.33 %, adult 1.75 %.
- Time gate (all three starts ≤ 60 s): PASS F1 (39 s), F2 (47 s); FAIL F3 86 s, M1 99 s,
  M2 105 s, M3 65 s. The selected start alone took 13–40 s.
- G2 (ratios inside the sampled Anny adult reference range, 18 bodies per case): adult fits
  3/3 only for M3; head share above the range for F1, M1, M2; shoulder-joint and leg-length
  ratios below the range for all three female cases. Production: 0–2 of 3.
- Isolation: production source files unchanged; 61 tests OK; 6 baseline GLBs unchanged;
  new cache file `var/cache/anny/v12/build_model_data_cb432537…` (258.7 MB), existing cache
  entries unchanged; integrity snapshots identical.

## Next

Waiting for approval; the production fit is unchanged.
