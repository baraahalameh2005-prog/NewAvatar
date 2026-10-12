# Handoff: Baraha avatar module (as of 2026-10-11)

## 1. Project goal
Palestinian Virtual Fitting Platform.
- **You own:** the Python/FastAPI backend, Anny avatar generation and fitting, and the React integration of the avatar module.
- **Your teammates own:** the clothing UI and the 3D clothing assets.
- **Technology:** Anny (NAVER Labs). No technology migration.

## 2. Protected paths (read-only, never modify)
- `D:\BarahaAvatar` (original backend; its important work is uncommitted)
- `D:\AvatarGitHub\avatar` (teammates' frontend)
- `D:\BarahaAvatar_Experiment` (experiment workspace)
- Verified backups: `D:\Avatar_Backups\BarahaAvatar` and `D:\Avatar_Backups\BarahaAvatar_Experiment`

## 3. Workspace
- **Location:** `D:\BarahaAvatarModule`, Git branch `main`. **Nothing is committed**; every file is untracked.
- **Environment:** `.venv` with Python 3.12.6, the same package versions as production.
- **Caches:** module-local in `var\cache\` (Anny, Warp, torchinductor).
- **Server:** port 8002.
- **Status log:** `docs\STATUS.md` is current only up to the adult-fit experiment. Stage 1 validation, face discovery and the face sweep are **not** recorded there, because those tasks didn't allow editing docs.

## 4. Completed work (with evidence)
- **Production backend in the module:** copied byte for byte and connected to `app\config.py` (outputs in `var\generated`, module caches, the shared legacy GLB turned off). The 6 production baseline GLB hashes are reproduced exactly.
- **API:** starts on 127.0.0.1:8002. Checks passed: `/health`, `/avatar-options`, a 422 on invalid input, a valid F1 request → unique GLB (glTF v2, 13,718 vertices), and the served bytes equal the file. All 14 checks passed.
- **Measurement fixes:** neck, upper arm and per-side thigh/calf ported from the reviewed experiment. Values are plausible and the GLBs are unchanged. 61 tests pass.
- **Body baseline v1 is frozen.** Known limits:
  - circumference errors up to 8 % (M1);
  - F3's weight is out of reach (−4.7 kg);
  - shoulder width is a joint-to-joint estimate;
  - fitted bodies are 24–50 % child-shaped (age 0.50–0.66).
- **Adult-fit approach (diagnostic only, not in production):**
  - fully adult and accurate for F1, F2, M2 and M3;
  - fails accuracy for F3 (weight) and M1 (hip);
  - Stage 1 validation: the fallback path runs 61–68 s, over the 60 s limit.
  - **Still an essential task before handover.**
- **Face discovery:** 129 face/head controls can be loaded. The named head-shape presets can't be loaded. The eyeballs are separate pieces, and the model has eye UVs, but the exported GLB has none.
- **Face sweep:** 25 controls × ±1 on F1 and M2, results in `var\face_sweep\20261011_192618\` (`results.json`, `run_log.txt`, 102 GLBs).
  - **Body-stable:** nose ×5, mouth ×3, chin-prominent, cheek-bones, cheek-volume, forehead-temple, eyebrows ×2, eye ×3. Nearly stable: chin-width, chin-bones, head-fat.
  - **Body-affecting:** head-scale-vert, forehead-scale-vert (both change height), head-scale-depth, head-scale-horiz, chin-height.
  - **Eye risks (numbers only):** eye-scale reshapes the eyelids around a fixed eyeball; head-scale-horiz deforms the eyeballs.

## 5. Face review status (from your visual check)
- A front close-up showed no obvious severe eye protrusion or clipping, and no confirmed gap at the inner eye corner.
- **No side-profile or wireframe inspection has been done.**
- **Eye geometry stays unchanged for now.**

## 6. Stopping point
The planned next step from the last report: visual review in Blender, front and side, starting with F1:
- `eye-trans-up`, `eye-scale` and `eyebrows-trans-up`, each `_minus1` / `_plus1` against `reference_0.glb`;
- then 3–4 stable nose, mouth or cheek controls.

That review decides the curated control list. **Only the front view is done.**

## 7. Open questions and risks

**Confirmed:**
- The production fit is unchanged and frozen.
- The adult-fit fallback exceeds 60 s (F2 60.9 s, F3 68.1 s, M1 65.4 s in Stage 1).
- The torso doesn't move for the stable face controls.
- Reported chest/hip values can shift by about 16 mm when the head height changes, because
  the slice heights are tied to body height (the mechanism is a hypothesis; the shift is measured).
- The GLB has no UVs.

**Not verified:**
- that eye-scale, eye-trans-up and the eyebrow controls are free of side-view or eyelid problems;
- that a UV-based iris will look acceptable;
- why M1 fails in the adult fit;
- why runtimes vary between runs (F2's production fit took 12 s once and 21 s another time);
- the male vs female visual differences;
- rendering quality in a browser.

**Pending decisions:** the curated face-control list and ranges; the eye-colour approach;
how to speed up the adult-fit fallback.

## 8. One next step
In Blender, view F1's `eye-scale_minus1.glb`, `eye-scale_plus1.glb` and `reference_0.glb` from the **side** (numpad 3) with the **Wireframe** and **Face Orientation** overlays, and record what you see. Nothing in the module changes.

---

## Next-session prompt (ready to paste)

```
Context: Palestinian Virtual Fitting Platform. I own the Python/FastAPI backend, Anny avatar
generation/fitting and React integration; teammates own clothing UI and 3D clothing assets.
Keep using Anny (no technology migration).

Workspace: D:\BarahaAvatarModule (Git branch main, nothing committed, .venv with Python 3.12.6,
module-local caches in var\cache, API port 8002 only). docs\STATUS.md is current up to the
adult-fit experiment; later results (Stage 1 adult validation, face discovery, face sweep) are
not yet recorded there.

State:
- Body baseline v1 is FROZEN (production fit + neck/upper-arm/thigh/calf measurement fixes;
  61 tests pass; 6 baseline GLB hashes reproduced). Known limits: circumference errors up to 8%,
  F3 weight out of reach, shoulder width is a joint estimate, fitted bodies 24-50% child-shaped.
- Adult-fit approach exists only as diagnostic tools (backend\tools\adult_fit_*.py); fallback path
  exceeds 60 s. It is an essential pre-handover task, NOT to be ported now.
- Face phase: sweep results in var\face_sweep\20261011_192618\ (results.json, run_log.txt, 102 GLBs).
  Body-stable controls: nose x5, mouth x3, chin-prominent, cheek-bones, cheek-volume,
  forehead-temple, eyebrows x2. Body-affecting: head-scale-vert, forehead-scale-vert,
  head-scale-depth, head-scale-horiz, chin-height. eye-scale reshapes lids around a fixed eyeball;
  head-scale-horiz deforms eyeballs. Exported GLB has no UVs (model has eye UVs).
- Visual review: front close-up showed no obvious severe eye clipping; side-profile and wireframe
  review NOT done. Leave eye geometry unchanged.

Next step to discuss (do not start without my approval): side-view + wireframe review of F1
eye-scale / eye-trans-up / eyebrows-trans-up (±1 vs reference_0.glb), then decide the curated
face-control list.

Safety rules:
- Never modify D:\BarahaAvatar, D:\AvatarGitHub\avatar, D:\BarahaAvatar_Experiment
  (or D:\Avatar_Backups). Read-only access only when needed.
- Work only inside D:\BarahaAvatarModule; outputs, logs, GLBs and caches stay inside it;
  never overwrite existing outputs or delete cache entries.
- Do not change the production fit, API behaviour, defaults or the frozen body baseline;
  do not port experimental code; do not commit, reset, stash or switch branches.
- No package installs or asset downloads without approval; no server on any port except 8002,
  and stop if 8002 is busy (never kill another process).
- Plan first, state exact files and commands, wait for my approval for each step; report
  failures honestly and distinguish verified facts from assumptions.
- Keep reports concise; one recommended next step at a time.
```
