# Avatar Module Handoff — Baraah

Prepared 2026-10-12 for Saja and Nagham. Repository: <https://github.com/baraahalameh2005-prog/NewAvatar>
(branch `main`, first commit `7e177e3`).

This report describes the avatar module as it exists in the repository. Every claim is labelled with
its evidence. Status words used throughout:

| Label | Meaning |
|---|---|
| **Implemented** | The code exists in the repository. |
| **Tested** | Covered by a documented test run or an observed run (date given). |
| **Experimental** | A prototype under `backend/tools/`; **not** part of the API. |
| **Planned** | Proposed, not built. |
| **Unverified** | Not confirmed by any available evidence. |

Nothing in this report was re-run while writing it. Earlier results are quoted with their source.

---

## 1. Project overview

The Virtual Fitting Platform lets customers of Palestinian clothing shops see clothing on a 3D body
that matches their measurements. The **avatar module** is the part that turns a customer's body
measurements into that 3D body:

1. The customer enters gender, height, weight, chest, waist and hip, and optionally skin tone and eye
   colour.
2. The backend fits the **Anny** parametric body model (NAVER LABS) to those measurements.
3. The backend exports the fitted body as a **GLB** file (a single-file 3D model format that three.js and
   Blender can load) and returns its URL plus the measurements of the generated body.
4. A frontend loads the GLB and displays it. Clothing display and fit evaluation are built on top of
   this by other team members.

Anny is the chosen avatar technology for the project and is not being replaced.

## 2. Baraah's responsibilities

| Area | Owner |
|---|---|
| Python / FastAPI backend of the avatar module | **Baraah** |
| Anny integration and measurement-based body fitting | **Baraah** |
| Body measurement extraction from the generated mesh | **Baraah** |
| Appearance customisation (skin tone, eye colour, face preset, eyebrows, lip tint) | **Baraah** |
| GLB export and the avatar API contract | **Baraah** |
| Avatar-related integration support for the frontend | **Baraah** |
| Shop product upload, product data, garment measurements, 3D clothing models and their integration | Saja |
| Clothing fit evaluation, size recommendation, fit-result explanations | Nagham |

The React app in `frontend/` is Baraah's **reference viewer** for the avatar API. It is not the
clothing frontend.

## 3. Implemented functionality

### 3.1 Avatar-generation flow — Implemented, Tested

Source: `backend/api.py` (`generate_avatar`, lines 424–551).

1. The request is validated (Pydantic model `AvatarRequest`). Invalid input returns HTTP **422** and
   no file is written.
2. `fit_body` (`backend/fit_to_measurements.py`) fits Anny to the measurements.
3. `appearance_for` (`backend/appearance.py`) selects the gender face preset, skin tone and eye colour.
4. The fitted parameters are evaluated once more with the face model plus the gender preset
   (`build`), which moves head vertices only.
5. `export_appearance_avatar` writes the GLB to the output folder under a new random name.
6. The response returns the GLB URL, fitted parameters, appearance, measurements and an accuracy
   report.

Generations are processed **one at a time** (a lock in `api.py`). A second request waits.

### 3.2 FastAPI endpoints — Implemented, Tested

Source: `backend/api.py`. Default address `http://127.0.0.1:8002` (`backend/app/config.py`;
override with `AVATAR_HOST` / `AVATAR_PORT`). CORS allows any origin (`allow_origins=["*"]`).

| Method | Path | Returns |
|---|---|---|
| GET | `/` | `{"message": "Avatar API is running"}` |
| GET | `/health` | `{"status": "ok", "model_loaded": true\|false}` |
| GET | `/avatar-options` | Accepted inputs, ranges, defaults, eye colours, automatic appearance features |
| POST | `/generate-avatar` | Generates one avatar (see below) |
| GET | `/models/generated/<avatar_id>.glb` | The generated GLB file |

`/models/<name>.glb` and `/test/<name>.html` are also mounted, but the module's `backend/` folder
contains no such files, so they return 404 (`docs/API.md`).

### 3.3 Measurement inputs — Implemented, Tested

`POST /generate-avatar`, JSON body (source: `MEASUREMENT_LIMITS` and `AvatarRequest` in `api.py`):

| Field | Type | Required | Unit / allowed values | Default |
|---|---|---|---|---|
| `height` | number | yes | 100–230 **cm** | — |
| `weight` | number | yes | 25–250 **kg** | — |
| `waist` | number | yes | 40–200 **cm** (circumference) | — |
| `chest` | number | yes | 50–200 **cm** (circumference) | — |
| `hip` | number | yes | 50–200 **cm** (circumference) | — |
| `gender` | string | no | `female` or `male`, case-insensitive | `female` |
| `skin_tone` | string | no | hex colour `#RRGGBB` (`RRGGBB` without `#` is also accepted) | `#c8997e` |
| `eye_color` | string | no | `brown`, `hazel`, `green`, `blue`, `gray`, case-insensitive | `brown` |

NaN and Infinity are rejected. Fields not listed are ignored.

**How the fit uses them** (`backend/fit_to_measurements.py`): height and weight are solved directly
through Anny's `height` and `weight` parameters. Waist, chest and hip are reached **indirectly** by
optimising Anny's `age` (limited to 0.5–0.999), `muscle` and `proportions`. All other Anny phenotype
values stay at 0.5. The fit stops early when every target is within 0.2 %.

### 3.4 Appearance customisation — Implemented; tested in code, not yet visually approved by Baraah

Source: `backend/appearance.py`, `backend/generate_avatar.py`, `GET /avatar-options`.

| Feature | How it works | Status |
|---|---|---|
| Skin tone | Any hex colour. The frontend offers 5 presets (`#e2bfa3`, `#c8997e`, `#a8754f`, `#8a5a3c`, `#6b4430`) plus a custom picker (`frontend/src/App.tsx`) | Implemented, tested |
| Eye colour | `brown`, `hazel`, `green`, `blue`, `gray`, painted into the eye texture | Implemented, tested |
| Gender face preset | Female: mouth width −0.35, cheek volume +0.25, chin width −0.2, nose width −0.15. Male: chin width +0.4, chin bones +0.35, chin prominence +0.25, nose width +0.15. Head vertices only | Implemented, tested |
| Eyebrows | Painted texture on a separate brow band; separate female and male designs | Implemented, tested. The female design was visually approved earlier on a prototype only |
| Lip tint | Female only | Implemented, tested |
| Hair | — | **Not implemented** (Anny provides no hair assets) |

The appearance actually applied is echoed back in the response field `appearance`.

### 3.5 GLB output — Implemented, Tested

Source: `backend/appearance.py` (`export_appearance_avatar`), `docs/API.md`, and a structural check of
generated avatar `fc209005…` on 2026-10-12.

- Every request gets a **new file** named after a random `uuid4` (`<avatar_id>.glb`). No shared file is
  overwritten (`legacy_avatar_url` is always `null`).
- Saved in `var/generated/` by default (`AVATAR_OUTPUT_DIR` overrides it). Files are **not** deleted
  automatically.
- Format: glTF 2.0 binary. One static mesh, **Y up, metres, feet at y = 0, facing +Z**. **No skeleton and
  no animations.** Two primitives that share the same 13,718 vertices: `skin` (body) and `skin_brows`
  (eyebrow band, using `TEXCOORD_1`). Two embedded PNG textures, clamp-to-edge sampler.
- Observed file size: about 1.2 MB (1,238,672 bytes for `fc209005…`).
- Body pose: Anny's **rest pose**. The arms are about 40° from the body and the fingers are spread
  (see section 8).

### 3.6 Response — Implemented, Tested

| Field | Content |
|---|---|
| `message` | `"Avatar generated successfully"` |
| `avatar_url` | **Relative** path, e.g. `/models/generated/<avatar_id>.glb` |
| `avatar_id` | 32 hex characters |
| `legacy_avatar_url` | Always `null` |
| `parameters` | Fitted Anny values (`gender`, `height`, `weight`, `age`, `muscle`, `proportions`) |
| `appearance` | `gender_style`, `skin_tone`, `eye_color`, `face_controls`, `features`, `not_available` |
| `measurements` | Generated body, in **metres** (`*_m`) and **kilograms** (`mass_kg`) |
| `failed_measurements` | Measurements that could not be read on this body |
| `accuracy` | Requested vs actual for height, weight, waist, chest and hip, in **cm / kg** |
| `warnings` | Targets outside the fitting tolerance. **Not an error**: the avatar is still valid |
| `fit_time_seconds` | Fitting time |

Errors: **422** with `{"detail": [...]}` for invalid input. An unexpected backend failure returns a
plain **500** (there is no custom error format for it).

### 3.7 Frontend integration (reference viewer) — Implemented, partly tested

Source: `frontend/src/api.ts`, `frontend/src/App.tsx`, `frontend/src/AvatarViewer.tsx`. React 18.3.1,
three.js 0.169.0, Vite 5.4.10 on `127.0.0.1:5173`.

- The base URL is `VITE_API_URL`, falling back to `http://127.0.0.1:8002`.
- The app reads `/avatar-options` for ranges and eye colours, checks `/health`, and posts the form to
  `/generate-avatar` with a 120 s timeout.
- It builds the full GLB URL from `avatar_url` and loads it with three.js `GLTFLoader`.
- Responses to older requests are ignored, so a stale avatar is never shown.
- **Testing:** browser runs used headless Edge on 2026-10-12 (`var/appearance_check/20261012_023914_browser/`,
  4 scenarios, 2 female and 2 male). The flow has **not yet been reviewed by Baraah in her own browser**.

## 4. Architecture and data flow

```
Customer form (frontend)
   │  POST /generate-avatar  {gender, height, weight, waist, chest, hip, skin_tone, eye_color}
   ▼
backend/api.py ── validate (422 on bad input)
   │
   ├─ fit_to_measurements.fit_body ─ Anny plain model
   │      solve height & weight; optimise age, muscle, proportions for waist/chest/hip
   │      measurements.BodyMeasurements measures the fitted mesh
   │
   ├─ appearance.appearance_for ─ gender preset, skin tone, eye colour
   ├─ generate_avatar.build ─ Anny face model + preset (head vertices only)
   └─ appearance.export_appearance_avatar ─ var/generated/<avatar_id>.glb
   ▼
JSON response {avatar_url, measurements, accuracy, warnings, appearance, ...}
   ▼
Frontend: base URL + avatar_url → GLTFLoader → replaces the previous avatar in the scene
```

At startup the backend loads two Anny models (plain and face) and needs roughly 1.5 GB of RAM (observed
on 2026-10-12: 1,550 MB committed). Model caches live in `var/cache/` and are not in Git.

## 5. How Saja's clothing work connects

**Clothing integration is not implemented in this repository.** There are no clothing assets, product
data or garment endpoints here. The intended connection points are:

| Connection point | What the avatar module provides | What still needs agreement |
|---|---|---|
| Body model | A per-customer GLB at `avatar_url` | Whether clothing tools accept the current GLB (2 primitives, textures, no skeleton). **Unverified** |
| Coordinate frame | Y up, metres, feet at y = 0, facing +Z | Garment models must use the same scale and orientation |
| Pose | Static rest pose (arms about 40° out) | **The pose may change** (section 8). Garments should not be finalised for the arms until the pose is agreed |
| Body measurements | `measurements` and `accuracy` in the response | How product sizes map to body measurements (shared with Nagham) |
| Product data | — (not part of this module) | Owned by Saja |

Because the GLB has no skeleton, clothing cannot be rigged to the avatar. It has to be fitted to the
static mesh, or a rig has to be requested and designed together.

## 6. How Nagham's fit logic connects

**Fit decisions, size recommendations and their explanations belong to Nagham's component.** The avatar
module only supplies body data. It does not evaluate garments.

### 6.1 Body data available per avatar (response field `measurements`)

| Key | Meaning | Kind | Unit |
|---|---|---|---|
| `height_m` | Body height | length | **m** |
| `mass_kg` | Body mass | — | kg |
| `bmi` | Body-mass index | — | — |
| `chest_m`, `waist_m`, `hip_m` | Chest, waist, hip | **circumference** (around the body) | **m** |
| `chest_width_m`, `waist_width_m`, `hip_width_m` | Same levels | **width** (left-to-right extent) | **m** |
| `neck_m`, `upper_arm_m`, `forearm_m`, `thigh_m`, `calf_m` (+ `_left` / `_right` for thigh and calf) | Limb and neck | circumference | **m** |
| `shoulder_width_m`, `arm_length_m`, `leg_length_m` | Joint-centre distances | length estimate, **not a tape measurement** | **m** |

### 6.2 Units and definitions to respect

- **Request** values are in **cm and kg**. **`measurements`** values are in **metres and kg**. **`accuracy`**
  values are in **cm and kg**. Multiply `*_m` by 100 to compare with cm garment charts.
- **Circumference ≠ width.** A garment chart's "chest 50 cm" is often a *flat width* (half the
  circumference). Compare like with like.
- **How the body is measured** (`backend/measurements.py`):
  - Waist uses Anny's anthropometry waist loop.
  - The chest plane is 0.145 × height above the waist and the hip plane is 0.10 × height below it.
  - Each circumference is the perimeter of the mesh cross-section at that plane, **not a convex
    "tape" hull**. On concave areas it can differ from a tape measurement. The size of this difference
    is **unverified**.
- `shoulder_width_m` runs between the shoulder joints and is **narrower** than a tape-measured shoulder
  width.
- The generated body can differ from the request (see `accuracy` and section 8). For fit logic, decide
  whether to use the **requested** measurements, the **generated** ones, or both. The avatar module does
  not make this decision.

## 7. Tests and verification

**None of the results below were re-run while writing this report.**

| What | Result | When / source |
|---|---|---|
| Unit and integration tests (`backend/tests/`: `test_avatar.py` 56, `test_config.py` 11, `test_appearance.py` 10, `test_arm_pose.py` 5, `test_eye_texture.py` 3 test functions) | Reported 85 OK | 2026-10-12, `docs/PROJECT_HANDOFF_REPORT.md` §0 |
| Production baseline reproduced (6 GLB hashes identical) | Pass | 2026-10-11, `docs/STATUS.md` step 5 |
| API server checks (health, options, 422, unique GLB, served bytes = file) | 14/14 pass | 2026-10-11, `docs/STATUS.md` step 6 |
| End-to-end with appearance (6-case matrix, invalid values rejected) | All checks true | `var/appearance_check/20261012_022231_api/` (not in Git) |
| Headless-Edge browser runs (2 female, 2 male; new avatar replaces the old one) | Appearance matched the selection in 4/4 | `var/appearance_check/20261012_023914_browser/` (not in Git) |
| One female generation via the API (165 cm, 60 kg, 72/90/98 cm, `#8a5a3c`, `green`) | HTTP 200 in 14 s; valid 1,238,672-byte GLB; appearance echoed correctly; waist/chest/hip/weight 1.7–1.9 % off | Observed 2026-10-12 (this handoff session) |
| Offline renders of that GLB (front, side, back, neck, arms, hands) | No seam or gap at the neck; pose and shape issues visible (section 8) | 2026-10-12, temporary scratchpad (not in the repository) |

**Not tested or not verified:**

- A male avatar through the current API in this session.
- The UI flow in Baraah's own browser.
- The GLB with any clothing tool.
- A re-run of the test suite after the most recent changes.

## 8. Known limitations and open questions

### Measurement accuracy (evidenced)
- Height is matched within 0.01 %. Waist, chest, hip and weight can be off: 1.7–1.9 % in the
  2026-10-12 female case, and up to about 8 % in the earlier 6-case baseline (male M1 waist 8.1 %,
  chest 7.2 %; female F3 weight −4.7 kg). Sources: `docs/PROJECT_HANDOFF_REPORT.md` §4, `docs/STATUS.md`
  milestone 2.
- When the fit cannot reach a target it does not fail. It returns the closest body and lists the miss
  in `warnings`.
- The face preset changes the head by at most 5.4 mm after the fit. `measurements` describe the body
  before the preset (`docs/API.md`).
- **The avatar is a statistical body model fitted to five numbers. It is not a physically accurate scan
  of the customer.**

### Visual appearance (evidenced by renders, 2026-10-12)
- **Adult look:** fitted ages of 0.50–0.66 blend in child proportions (24–50 %). The 2026-10-12 female
  case had age 0.539. Female bodies can look young or androgynous, with a weak waist-to-hip shape;
  breasts are the main visible gender cue.
- **Neck and shoulders:** a thick neck and a steep shoulder slope. **No seam or gap** in the mesh.
- **Arms and hands:** the rest pose has the arms about 40° out and the hands about 22 cm in front of the
  torso, with a bent wrist and spread fingers.
- Female eyebrows are subtle on very dark skin. No hair.

### Experimental, not in the API
- **Adult fit** (age ≥ 0.67 + bust/waist/hip controls): looks more adult. Fails accuracy for F3 weight and
  M1 hip, and some bodies take 55–68 s (`docs/PROJECT_HANDOFF_REPORT.md` §4). It would change the frozen
  fitting algorithm.
- **Arm pose** (export-only upper-arm rotation, `backend/tools/arm_pose_check.py`): not integrated, not
  approved.
- **Relaxed hands:** planned only.

### Fit estimation
- Not part of this module (Nagham). The measurement definitions in section 6 apply.

### Open questions
1. The final avatar pose for clothing (current rest pose, lowered arms, hand position).
2. Whether the clothing pipeline needs a single primitive, a skeleton, or other GLB changes.
3. The mapping between garment size charts and body measurements (requested vs generated; circumference
   vs width).
4. The production base URL and port.
5. Cleanup and privacy of generated files: they are never deleted, and anyone with a URL can load one.

## 9. File map

| Path | Purpose |
|---|---|
| `backend/api.py` | FastAPI app: endpoints, validation, generation flow, static GLB serving, CORS |
| `backend/fit_to_measurements.py` | Measurement fitting (frozen algorithm) |
| `backend/measurements.py` | Measuring circumferences, widths and lengths on the fitted mesh |
| `backend/generate_avatar.py` | Anny phenotype build, skin colour parsing, base GLB export |
| `backend/appearance.py` | Face presets, eye colours, eye/lip/brow textures, appearance GLB export |
| `backend/app/config.py` | Host and port, output and cache folders, protection against writing into other projects |
| `backend/data/face_model_labels.json` | The 63 Anny face-control labels the face model loads |
| `backend/tests/` | Automated tests (section 7) |
| `backend/tools/` | Experiment and check scripts (adult fit, arm pose, face, eye, lip, brow, baseline). **Not used by the API** |
| `backend/requirements.txt`, `backend/constraints-production.txt` | Python dependencies and pinned versions |
| `frontend/src/api.ts` | Reference API client: types, timeouts, error handling |
| `frontend/src/App.tsx` | Reference form, appearance selection, result display |
| `frontend/src/AvatarViewer.tsx` | Reference three.js GLB viewer |
| `docs/API.md` | API contract (some status notes are dated) |
| `docs/PROJECT_HANDOFF_REPORT.md` | Detailed status and evidence log |
| `docs/STATUS.md` | Step-by-step build log (current up to the adult-fit experiment) |

Not in Git: `.venv/`, `var/` (generated avatars, caches, experiment outputs), `node_modules/`, and the
temporary face-preview GLBs in `frontend/public/face_preview_20261011_235711/`.

## 10. Handoff instructions

**Running the avatar API locally (Windows):**

```powershell
cd backend
..\.venv\Scripts\python.exe -B api.py      # http://127.0.0.1:8002
cd ..\frontend
npm.cmd run dev                            # reference viewer, http://127.0.0.1:5173
```

A fresh clone needs a Python 3.12 virtual environment set up from `backend/requirements.txt` with
`backend/constraints-production.txt`, and `npm install` in `frontend/`. The first start builds the Anny
caches in `var/cache/` (about 1.2 GB on disk). Setting up a fresh clone is **unverified**: it has not
been tested from the repository.

**For Saja:**
1. Call `POST /generate-avatar` and load `base URL + avatar_url`. Always use the URL from the latest
   response.
2. Test one generated GLB in your clothing tools and report what works and what doesn't.
3. Tell Baraah which pose and which GLB properties your garments need.

**For Nagham:**
1. Use `measurements` (metres) and `accuracy` (cm) from the response. Keep circumference and width
   apart.
2. Decide whether fit uses the requested or the generated measurements, and document that choice.

**For both:**
- Treat `backend/` as Baraah's component. Please don't edit the fitting or export code directly.
- If something in the avatar or the API blocks your work, write down the request, the response or GLB
  ID, and what you expected, and raise it with Baraah. Most issues can be solved inside the existing
  Anny-based implementation (for example a pose change, an extra response field, or a GLB option)
  without rewriting it.
- API changes are agreed together before they are made.

## 11. Contribution summary

Baraah has, with evidence in this repository and its documentation:

- Set up the avatar module as a separate workspace, with a FastAPI backend reproducing the original
  production backend (6 baseline GLBs identical) and protections against writing into other projects.
- Built the measurement-based Anny fitting pipeline (height and weight solved directly; waist, chest and
  hip fitted indirectly), with an accuracy report and warnings in every response.
- Added measurement-reading fixes (neck, upper arm, per-side thigh and calf).
- Implemented the avatar API: health, options, generation with validation, and unique per-request GLB
  files with CORS for frontend use.
- Implemented appearance customisation: any skin tone, five eye colours, gender face presets, female
  and male eyebrows, and female lip tint.
- Built a reference React/three.js viewer that shows the complete flow.
- Wrote automated tests (85 reported passing on 2026-10-12) and documented the API, status and evidence.
- Ran experiments on a more adult body fit, arm pose and face features. These are documented as
  experimental and are not part of the API.

Still open on Baraah's side:
- Her own visual approval of the integrated appearance.
- An agreed pose and arm/hand improvement.
- A decision on the adult/feminine body shape.
- Clothing-tool compatibility.

The overall graduation project is **not** complete. This module is one part of it.
