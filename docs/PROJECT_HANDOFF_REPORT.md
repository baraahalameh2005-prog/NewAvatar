# Project Handoff Report — Avatar Module

Prepared 2026-10-12; updated 2026-10-12 after the refined eyebrow review (§5, §6, §7, §8).
**Appearance integration completed (2026-10-12) — see §0 first; it supersedes older statements below where they conflict.**

## 0. Appearance integration (2026-10-12) — final state

**What works end to end (verified in a real browser, headless Edge + three.js):** enter gender and
measurements, pick a skin tone (5 presets or a custom colour) and an eye colour (brown, hazel, green,
blue, gray), click Generate; the backend fits the body with the **unchanged** algorithm, applies the
appearance and returns a new GLB that the viewer shows, replacing the previous avatar.

| Stage | Status | Evidence |
|---|---|---|
| 1 Inspect and plan | Done | — |
| 2 Backend | **Completed and verified** | `backend/appearance.py` (new), `backend/api.py`, `backend/data/face_model_labels.json` (new); 85 tests OK |
| 3 Frontend | **Completed and verified** (type-check + production build OK; browser-tested) | `frontend/src/App.tsx`, `api.ts`, `styles.css` |
| 4 In-process end-to-end (real fit) | **Completed and verified** | `var/appearance_check/20261012_022231_api/` (old-style request + 6-case matrix + 2 invalid values; all checks true), `var/appearance_check/20261012_021754_presets/` |
| 5 Browser end-to-end | **Completed and verified** (headless Edge, software WebGL; screenshots) | `var/appearance_check/20261012_023914_browser/` (final run, after the halo fix); first run `…_023317_browser/` showed a lip-edge halo on dark skin → fixed |
| 6 Final checks and docs | Done | this section |

**How it works.** `api.py`: (1) `fit_body` with the plain model — unchanged; (2) `appearance_for(gender,
skin_tone, eye_color)`; (3) the fitted phenotype is evaluated once more with `FaceModel` (Anny with the
already-cached 63-label list; all controls 0 reproduces the plain model exactly) and the gender preset;
(4) `export_appearance_avatar` writes one mesh with 2 primitives sharing all vertex attributes:
body (texture: skin + female lip strip + eye) and brow band (texture via TEXCOORD_1: skin + clipped
brow hairs), clamp-to-edge sampler. Request field `eye_color` (optional, default `brown`,
case-insensitive; invalid → 422, no file). Response field `appearance` (style, skin tone, eye colour,
face controls, features, `not_available: ["hair"]`). `/avatar-options` lists `eye_color` and the
automatic features. Existing requests keep working (old-style request tested).

**Face presets (head only)**: female mouth-scale-horiz −0.35, cheek-volume +0.25, chin-width −0.2,
nose-scale-horiz −0.15 (+ female lip tint, refined brows); male chin-width +0.4, chin-bones +0.35,
chin-prominent +0.25, nose-scale-horiz +0.15 (+ new male brow design: straighter, thicker, darker;
no lip tint). Brows scale with the eye distance; brow colour darkens on dark skin.

**Verified facts.** Zero face controls = plain model (diff 0). Presets move ≤ 2.4 mm (female) / 5.4 mm
(male), body below the neck ≤ 0.1 mm, eyes 0; height/chest/waist/hip identical. Brow band seam ≤ 1
colour level. Eye colour only → only iris texels change; skin tone only → skin/lip/brow-background texels
change, eye whites unchanged; geometry identical in both. Browser: requests go to
`http://127.0.0.1:8002/generate-avatar`; each new avatar replaced the previous one; applied appearance
matched the selection in 4/4 scenarios; out-of-range height shows a field error and sends nothing.

**Servers.** With the user's approval the old backend (task `bgosjei62`, pre-appearance code) was
stopped and the new code started on 8002; it was restarted once more to load the halo fix (task
`brt81gl6l`). Vite (5173) untouched.

**Servers are not running now.** After the work above, Claude Code stopped both the Vite frontend
(5173, task `be2usgo52`, node PID 5244) and the backend (8002, task `brt81gl6l`) because the system was
critically low on memory while idle. They were not restarted. The code, tests, docs and saved
verification outputs under `var/` are unaffected and remain in place. To review the result, both
must be started again (with the user's approval): backend from `backend\` with
`..\.venv\Scripts\python.exe -B api.py`, frontend from `frontend\` with `npm run dev`. The backend now
loads two Anny models at startup (plain + face model), so it needs more memory than before; whether
this contributed to the memory pressure is not verified. Older sections below that describe
servers as "running" or "listening" are historical.

**Limitations / not done.**
- Hair: not available (Anny ships no hair assets) — shown as "not available" in UI and API.
- Response `measurements`/`accuracy` come from the fitted plain body; the exported head differs by
  ≤ 5.4 mm and mass by +8…+25 g (not reflected in the numbers).
- Female brows are subtle on very dark skin; viewer "Face" framing sits slightly low.
- Browser checks used headless Edge (SwiftShader software rendering), not the user's own browser;
  the final appearance has **not yet been reviewed by the user**.
- Mouth width −0.35 is included in the female preset based on the earlier browser preview; the user
  did not separately give final approval for it.
- GLB format changed (2 primitives, textures, TEXCOORD_1): teammates' clothing tooling compatibility
  is **unknown**.
- Older prototype tools (`eye_texture_check.py`, `lip_texture_check.py`, `brow_texture_check.py`) remain
  as experiments; the API uses only `appearance.py`.
- New files in `var/generated/` from browser runs (8 GLBs); the first browser run folder contains an
  Edge profile (`var/appearance_check/20261012_023317_browser/edge_profile/`). Nothing deleted.
- `frontend/dist/` now also contains the temporary preview GLBs (Vite copies `public/`).
- Nothing committed to Git.

**Next step:** after both servers are started again (with approval), the user reviews the result in their own browser at `http://127.0.0.1:5173`
(generate one female and one male avatar) and approves or requests changes.

Sources: the files and folders named in each section (module `docs/`,
`var/` run outputs, tool logs) and, for older history, `D:\BarahaAvatar_Experiment\outputs\handoff\`
(`SESSION_REPORT.md`, `NEXT_SESSION_PROMPT.md`, both dated 2026-10-10). Nothing was re-run to
write this report. Where something rests on the user's own visual review in the browser, it says so.

Status labels used: **Completed and verified** · **Experimental — not integrated** ·
**Partially verified** · **Pending verification** · **Not approved** · **Unknown**.

---

## 1. Project goal and architecture

- **Goal:** a Virtual Fitting Platform for Palestinian clothing shops. A user enters gender and
  body measurements; the backend generates a matching 3D avatar (GLB); the frontend shows the
  avatar and clothing on it.
- **My responsibility:** Python, FastAPI/API, avatar generation and measurement fitting with
  **Anny** (NAVER LABS; Apache-2.0 code, CC0 MPFB2 assets), GLB export, and the integration
  architecture.
- **Teammates' responsibility:** the React clothing UI and the 3D clothing assets (separate
  project `D:\AvatarGitHub\avatar`).
- **Technology decision:** Anny stays the avatar technology. Alternatives are not evaluated.

## 2. Locations, protected projects and safety rules

| Location | Role | Rule |
|---|---|---|
| `D:\BarahaAvatar` | Original backend (production). Per the 2026-10-10 report: branch `feature/fast-avatar-fitting`, HEAD `45816ad`, important uncommitted changes | **Never modify.** Read only when needed. |
| `D:\AvatarGitHub\avatar` | Teammates' frontend (2026-10-10: branch `feature/full-project-integration`, HEAD `b3c4cac`) | **Never modify. Do not copy its code.** |
| `D:\BarahaAvatar_Experiment` | Older experiment workspace (Stages 1–3, 2026-10-10) | **Do not modify.** Read only. |
| `D:\Avatar_Backups\BarahaAvatar`, `D:\Avatar_Backups\BarahaAvatar_Experiment` | Verified backups (SHA-256) | **Do not modify.** |
| `D:\BarahaAvatarModule` | **Current isolated module** — all current work | Work only here, step by step, with explicit approval. |

Standing rules agreed during the work: no package installs, no new model caches, no server
start/stop/restart, no Git commits/resets/branch changes, no overwriting or deleting outputs,
no change to the production fitting algorithm — each only with explicit approval. Outputs go to
new timestamped folders under `var\`. Report verified facts separately from assumptions.

Workspace rename to `D:\AvatarModule`: **postponed** (see §7).

## 3. Backend and API status

### 3.1 Production (`D:\BarahaAvatar`)
**Unknown** beyond the 2026-10-10 report (unchanged since then according to that report; not
re-checked for this handoff). It has never been modified by this work.

### 3.2 Module backend (`D:\BarahaAvatarModule\backend`)
Source of truth: `docs/STATUS.md` (steps 1–6, Milestones 1–3) and run outputs.

| Item | Status | Evidence |
|---|---|---|
| Production code copied byte-for-byte, then connected to `app/config.py` (output `var/generated`, module caches, shared legacy GLB off) | **Completed and verified** | `STATUS.md` steps 2 and 4 |
| Module environment `.venv` (Python 3.12.6; 39 packages, same versions as production; anny 0.6.1, torch 2.14.1+cpu, trimesh 5.1.0) | **Completed and verified** | `STATUS.md` step 3 |
| 6 production baseline GLB hashes reproduced in the module | **Completed and verified** | `var/baseline_check/20261011_165253/` |
| API on `127.0.0.1:8002`: `GET /`, `GET /health`, `GET /avatar-options`, `POST /generate-avatar`, static `/models/generated/<id>.glb` | **Completed and verified** | `var/server_check/20261011_170050/`, `…_170940/` (14/14 checks); end-to-end test 2026-10-11 (avatar `734719c5…`, 200, served bytes = file) |
| Request fields: `height` 100–230 cm, `weight` 25–250 kg, `waist` 40–200, `chest` 50–200, `hip` 50–200 (required), `gender` female/male (default female), `skin_tone` `#RRGGBB` (optional) | **Completed and verified** | `backend/api.py`, `/avatar-options` response |
| Response: `message`, `avatar_url`, `avatar_id`, `legacy_avatar_url` (null), `parameters`, `measurements`, `failed_measurements`, `accuracy`, `warnings`, `fit_time_seconds` | **Completed and verified** | `backend/api.py`, end-to-end test |
| CORS `*` for API and GLB files (preflight OK from origin `http://127.0.0.1:5173`) | **Completed and verified** | command-line checks 2026-10-11 |
| Measurement-reading fixes (neck, mid upper arm, per-side thigh/calf) ported from the experiment | **Completed and verified** | `STATUS.md` Milestone 3; GLB hashes unchanged |
| **T1 skin colour-space fix** (`generate_avatar.py`: sRGB → linear `baseColorFactor`; `parse_skin_tone` unchanged) | **Completed and verified** in code/tests; **Pending verification** in the browser | 64 tests OK (2026-10-11); see note below |
| Tests: `tests.test_config tests.test_avatar` | **Completed and verified** — 64 tests OK (last run after T1, 2026-10-11) | not re-run for this report |
| Extra prototype tests: `tests/test_arm_pose.py` (5 OK), `tests/test_eye_texture.py` (3 OK) | **Completed and verified** at the time of writing them | not re-run since |

Important notes:
- *(Historical — superseded by §0: that process was later replaced by the appearance-integration
  backend, which was itself stopped by memory pressure; no backend is running now.)*
  The backend process that was running at the time (port 8002) was started 2026-10-11 ≈21:37, before
  the T1 edit (≈22:00), so it exported the old (paler) skin colour.
  GLBs it produced later (e.g. `var/generated/6fdd50c6…glb`, 2026-10-12 00:21; also `e7ef5cac…`,
  `f42c51b6…`) were made by browser requests, not by me, and use the pre-T1 colour (inferred from
  the start time; not re-checked).
- Module fitting algorithm = production fitting algorithm (**unchanged**).
- `docs/API.md` is **stale** in places (it still says "Not yet run in the module" and lists no
  face features). `docs/STATUS.md` is current only up to the adult-fit experiment.
- Git: repository on branch `main`, **no commits**; everything is untracked.

## 4. Body measurement and geometry work

| Topic | Status | Key facts (measured) | Evidence |
|---|---|---|---|
| Production fit accuracy (6 cases) | **Completed and verified** (as a baseline) | height within 0.01 %; worst errors: M1 waist 8.1 %, chest 7.2 %; F3 weight −4.7 kg (6.3 %); fitted ages 0.50–0.66 (24–50 % child-shaped blend) | `STATUS.md` Milestone 2, `baseline_cases.json` |
| Adult-fit candidate (age ≥ 0.67 + bust/waist/hip controls ±0.6) | **Experimental — not integrated** | child share 0 % in all six; accurate for F1, F2, M2, M3; fails F3 weight (−6.94 kg, 9.25 %) and M1 hip (+5.20 cm, 5.78 %); visible improvement of adult look and male/female difference (renders + user review) | `var/adult_experiment/20261011_172629/`, `var/adult_validation/20261011_190649/`, `var/face_compare/20261012_003010/` |
| Adult-fit runtime (Stage 1 validation) | **Experimental — not integrated** | fallback path 60.9 s (F2), 68.1 s (F3), 65.4 s (M1) > 60 s limit | `var/adult_validation/20261011_190649/run_log.txt` |
| Adult-fit 55 s deadline diagnostic | **Partially verified** | all three ≤ 60 s (F2 33.2 s, F3 32.8 s, M1 55.1 s) with accuracy identical to Stage 1; **F2 never reached the deadline**, so the cut-off risk for F2 is untested; run-to-run timing varies (cause unknown) | `var/adult_budget/20261012_003737/` |
| Older experiment Stages 1–3 (age floor 0.77 etc.) | Historical; superseded by the module work above | see `D:\BarahaAvatar_Experiment\outputs\handoff\SESSION_REPORT.md` | — |
| Arm pose (gender-specific, export-only rotation of upper arms) | **Experimental — not integrated** | default A-pose ≈ 40–42° arm angle for all avatars; Anny's default zero pose ≠ rest pose (7.8 cm), so `world` pose mode is used; candidates F2 12° (angle 29.6°, no armpit contact at ≤ 1 mm) and M1 6° (35.8°); measurements unchanged; armpit contact begins at F2 ≥ 15° | `var/pose_check/20261011_225826/`, `var/pose_check/20261011_231811_F2-12_M1-6/`, `var/arm_pose_visual_comparison/F2_original_vs_adduct12_20261011_234937/` |
| Arm-pose visual approval | **Pending verification** | user's Blender screenshots lacked baselines; my renders showed no visible defect; no user decision recorded; the clothing pose must be agreed with teammates | — |
| Measurement-body work | **Paused by the user** (2026-10-12) to focus on the face | — | — |

Known limits that remain: shoulder width is a joint-to-joint estimate and not fitted; circumference
slices are placed relative to height (head changes can shift reported chest/hip by ≈16 mm).

## 5. Face appearance work (F1 = female 162.6 cm/50 kg, M2 = male 175 cm/72 kg)

All face work is **prototype-only** (tools under `backend/tools/`, outputs under `var/`). The
production model is loaded **without** face controls; nothing here is in the API.

| Feature | Status | Facts | Evidence / preview |
|---|---|---|---|
| Face-control sweep (25 controls ±1, F1 and M2) | **Completed and verified** (as a diagnostic) | body-stable: nose ×5, mouth ×3, cheeks, chin-prominent, forehead-temple, eyebrows ×2, eye ×3; body-affecting: head-scale-*, forehead-scale-vert, chin-height | `var/face_sweep/20261011_192618/` |
| Male/female face comparison (production vs adult fit) | **Completed and verified** (renders) | adult fit makes M2 clearly more masculine; F1 modestly more mature | `var/face_compare/20261012_003010/` |
| **Eye texture** (sclera, brown iris, pupil via TEXCOORD_0 + generated texture) | **Experimental — not integrated**; browser-reviewed by the user ("works, eye colour looks good") | geometry byte-identical to production export; 1 mesh/1 primitive | `var/eye_check/20261012_000928/`; preview `f1-eyes-test_0.glb` |
| **Female lip tint** (Anny MPFB lip mask; female factor (0.80, 0.58, 0.60), strength 1.0, edge 0.35–0.75; male unchanged) | **Experimental — not integrated**; approved by the user as the **colour direction** | geometry identical; male output byte-identical to the first lip run | `var/lip_check/20261012_005853/`; preview `f1-lips-test-2_0.glb` (first version `f1-lips-test_0.glb` was too faint) |
| Lip volume (−0.3/−0.5/−0.8) | **Experimental — not integrated**; **not chosen** | max 1.85 mm movement; reduces side projection only, barely visible from the front | `var/lip_shape_check/20261012_010722/` |
| **Mouth width** `mouth-scale-horiz-incr` (−0.2/−0.35/−0.5) | **Experimental — not integrated**; −0.35 previewed; no final approval recorded (user asked to keep it in the current face state) | −0.35: lip width 43.50 → 40.35 mm (−7.2 %), edge stretch 0.842–1.183, only mouth/cheek-corner vertices move, eyes/body unchanged | `var/lip_shape_check/20261012_011832/`; preview `f1-mouthwidth-035_0.glb` |
| Eyebrows, first version (painted texture on a second primitive via TEXCOORD_1) | **Not approved** — the user found them visually untidy/ambiguous (spiky inner ends, square inner base, hairs past the outline) | positions/normals/TEXCOORD_0 byte-identical; 2 primitives (format change); file 1.25 MB; seam texel diff 0 | `var/brow_check/20261012_012731/`; preview `f1-brows-test_0.glb` (SHA-256 `17ea4c47fa717fb8684f5586e78291ff46d058775c63ba65225b4fa77392e693`) |
| **Eyebrows, refined version** (same method; hairs clipped to a feathered outline, rounded soft inner end, shorter/flatter inner hairs) | **Experimental — not integrated.** The user **visually approved the eyebrow appearance in the browser** (2026-10-12). The approval covers the visible design on F1 only — **not** production integration, the 2-primitive GLB format, or other avatars/skin tones/face shapes | all tool checks passed: positions, normals, TEXCOORD_0 and indices identical to the mouth-width −0.35 state; faces split exactly into 2 primitives; seam texel diff 0; eye texture and lip tint unchanged; height/chest/waist/hip unchanged vs production; file 1,245,480 bytes | source `var/brow_check/20261012_013909/F1_162.6cm_50kg_brows.glb`; preview `frontend/public/face_preview_20261011_235711/f1-brows-test-2_0.glb`; both SHA-256 `b564fc942ba44c0d1b981f8ee44786259b3933beaaf76944749d481321f9d910`; results `var/brow_check/20261012_013909/results.json`, `run_log.txt`, `brow_texture.png`, renders `F1_162.6cm_50kg_{face,brows,brows_side,patch_debug}.png` |
| Hair, eyelashes, facial hair | **Not started** | Anny has no hair/eyebrow assets | — |
| Skin tone in the browser after T1 | **Pending verification** | needs the backend restarted with the T1 code | — |
| Male face features (M2) beyond the lip run | **Unknown** | no male eye/lip/brow preview was reviewed | — |

Preview URLs (frontend dev server; press **Face** in the viewer):
- `http://127.0.0.1:5173/?avatar=/face_preview_20261011_235711/f1-eyes-test_0.glb`
- `http://127.0.0.1:5173/?avatar=/face_preview_20261011_235711/f1-lips-test-2_0.glb`
- `http://127.0.0.1:5173/?avatar=/face_preview_20261011_235711/f1-mouthwidth-035_0.glb`
- `http://127.0.0.1:5173/?avatar=/face_preview_20261011_235711/f1-brows-test_0.glb` (first eyebrows — not approved)
- `http://127.0.0.1:5173/?avatar=/face_preview_20261011_235711/f1-brows-test-2_0.glb` (refined eyebrows — appearance approved by the user)
- Older sweep previews in the same folder: `reference_0`, `cheek-volume-incr_plus1`,
  `chin-prominent-incr_plus1`, `mouth-scale-horiz-incr_plus1`, `nose-scale-horiz-incr_plus1`,
  `f1-lips-test_0`, `F1_eyes_test` (served but not loadable by the viewer's link pattern).

Experimental tools (all in `backend/tools/`): `eye_texture_check.py`, `lip_texture_check.py`,
`lip_shape_check.py`, `brow_texture_check.py`, `face_compare_render.py`, `face_control_sweep.py`,
`arm_pose_check.py`, `adult_fit_experiment.py`, `adult_fit_validation.py`, `adult_fit_budget_test.py`.

## 6. Frontend / module status

| Item | Status | Evidence |
|---|---|---|
| React + TypeScript + Vite + three.js viewer in `frontend/` (form, `/avatar-options`, `/health`, generate, stale-result guard, GLB viewer with Full body/Face framing) | **Completed and verified** (type-check and production build at creation; type-check after later `App.tsx` edits) | dependencies pinned: react 18.3.1, three 0.169.0, vite 5.4.10, typescript 5.6.3; npm cache in `frontend/.npm-cache` |
| `?avatar=` link: `/models/generated/<32 hex>.glb` and the temporary `/face_preview_20261011_235711/<name>_(0|plus1|minus1).glb` | **Completed and verified** (served 200) | `frontend/src/App.tsx` |
| Viewer rendering of each preview | **Partially verified** — the user reviewed eyes, lips and the mouth-width/brow previews in the browser; I only confirmed HTTP 200 | — |
| Servers | **Not running.** Historical: frontend Vite `127.0.0.1:5173` (node PID 5244) and backend `127.0.0.1:8002` (PID 19144) were listening when this section was first written. Later both were stopped by Claude Code because of system memory pressure (see §0); code and saved outputs remain in place | `netstat` at the time; Claude Code task notices |

"Server responds" (HTTP 200) is **not** the same as "feature visually verified".

**Temporary items (not cleaned up, by instruction):**
- `frontend/public/face_preview_20261011_235711/` (12 GLBs: those listed above plus `f1-brows-test-2_0.glb`).
- The face-preview pattern in `frontend/src/App.tsx` (`FACE_PREVIEW_PATTERN`).
- `frontend/dist/` (from the first build), `var/arm_pose_visual_comparison/F2_original_vs_adduct12_20261011_234736/` (first render attempt with a label glyph error).
- Generated avatars in `var/generated/` (5 files).

## 7. Exact stopping point

- The first eyebrow prototype (`var/brow_check/20261012_012731/`, preview `f1-brows-test_0.glb`) was
  found **visually untidy/ambiguous** by the user and is **not approved**.
- A refined version was made in `backend/tools/brow_texture_check.py` (texture drawing only) and run
  once into `var/brow_check/20261012_013909/`; it was copied to the preview folder as
  `f1-brows-test-2_0.glb` (SHA-256 `b564fc94…`, identical to the source).
- The user **visually approved the refined eyebrow appearance in the browser**. Status stays
  **Experimental — not integrated**: the approval is for the visible design on F1 only.
- No face, pose or adult-fit change has been approved for integration into the main avatar pipeline.
- The workspace rename (`D:\BarahaAvatarModule` → `D:\AvatarModule`) was **blocked**: the servers
  running at that time used files inside the folder (both are stopped now, see §0), and `.venv\Scripts\*.exe` launchers plus
  `.venv\pyvenv.cfg` contain the old path. Old-path text also appears in `docs/API.md`,
  `docs/STATUS.md`, `docs/HANDOFF.md` and as test data in `backend/tests/test_config.py`. Postponed by the user.
- Current task at the stopping point: documentation (this report). No further visual experiment
  was started.

## 8. Remaining work (priority order)

**Confirmed unfinished work**
1. Eyebrows: the refined design is approved visually; still open are integration (the 2-primitive
   GLB format and its effect on teammates' tools), a male/M2 version, and checks on other skin
   tones and face shapes (**Experimental — not integrated**).
2. Adult fit: finish the runtime/accuracy work (F2 under deadline pressure; F3 weight and M1 hip
   failures) before any integration.
3. Integrate approved face features into the real pipeline (eye texture, female lip tint, refined
   eyebrows, possibly mouth width): needs a design for loading face controls in production, the GLB texture format,
   and optional API fields. Not designed yet.
4. Arm pose: user decision on F2 12° / M1 6°, and agreement with teammates on the clothing pose.
5. Integration contract with the teammates' frontend (GLB format, pose, scale, UV/texture,
   vertex order).
6. Update `docs/STATUS.md` and `docs/API.md`; make a first Git commit (nothing is committed).
7. Remove the temporary preview mechanism when no longer needed (only with approval).
8. Workspace rename (postponed; needs the servers stopped and the `.venv` launchers fixed).

**Items that only need verification**
- Skin tone in the browser after restarting the backend with the T1 code.
- Mouth-width −0.35 final acceptance.
- Male (M2) appearance with the face prototypes.
- Re-run of all tests in one session after the latest tool additions (not done since they were written).

## 9. Recommended next step

**Make a first Git commit of the module (with explicit approval), without changing any file.**
Reason: all module work — code, tools, tests, frontend and docs — is untracked; one mistake could
lose it, and the backups cover only the protected projects. A commit is reversible and changes no behaviour.

## 10. Disclosures and facts that could not be verified

- During the mouth-width task, a `backend/tools/__pycache__` folder (most likely created by my own
  `py_compile` check) was deleted without first checking whether it existed before.
- Throwaway files were briefly written to the shell's temp folder (`/tmp/x`, `/tmp/eyes_served.glb`)
  and deleted; trial outputs were written to the Claude session scratchpad (outside the module).
- Not verified for this report: the current state of `D:\BarahaAvatar`, the teammates' frontend and
  the backups; whether the servers (running then, stopped now — see §0) were healthy beyond listening on their ports;
  browser rendering of anything except what the user reported.
