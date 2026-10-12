# Prompt for the next Claude Code session

Copy everything below the line into a new conversation.

---

You are continuing my graduation project: a Virtual Fitting Platform for Palestinian clothing
shops. I own the Python/FastAPI backend, avatar generation and measurement fitting with **Anny**
(NAVER LABS), GLB export and the integration architecture. My teammates own the React clothing UI
and the 3D clothing assets. Anny stays the avatar technology; do not evaluate alternatives.

## Read first (do not repeat completed work)

1. `D:\BarahaAvatarModule\docs\PROJECT_HANDOFF_REPORT.md` — the full current status (2026-10-12).
2. `D:\BarahaAvatarModule\docs\STATUS.md` — verified module steps (current up to the adult-fit
   experiment).
3. Older history only if needed: `D:\BarahaAvatar_Experiment\outputs\handoff\SESSION_REPORT.md`
   (2026-10-10).

## Locations

- **Current module (work only here):** `D:\BarahaAvatarModule` (Git branch `main`, **no commits**,
  everything untracked). Backend `backend\`, frontend `frontend\`, outputs `var\`.
  `.venv` = Python 3.12.6. Module caches in `var\cache\`.
- **Protected — never modify:** `D:\BarahaAvatar` (original backend), `D:\AvatarGitHub\avatar`
  (teammates' frontend), `D:\BarahaAvatar_Experiment`, `D:\Avatar_Backups\*`.
- Ports: backend `127.0.0.1:8002`, frontend Vite `127.0.0.1:5173`.
  **Neither server is running.** At the end of the last session Claude Code stopped both the
  Vite frontend (5173) and the backend (8002) because the system was critically low on memory.
  The code and the saved verification outputs under `var\` remain in place. Starting them again
  needs my approval: backend from `backend\` with `..\.venv\Scripts\python.exe -B api.py`,
  frontend from `frontend\` with `npm run dev`. The backend loads two Anny models at startup, so
  close other memory-heavy programs first.

## Current state in short

- Module backend = production code + measurement-reading fixes + T1 skin colour fix + the
  integrated appearance pipeline (85 tests OK). Production fitting algorithm unchanged.
- Adult fit and arm pose exist only as prototypes. Eye colour, skin tone, face presets, female lip
  tint, mouth width -0.35 (female) and eyebrows **are integrated** (see below).
- Eye texture and the female lip-tint colour were reviewed positively by me in the browser.
- Eyebrows: the first version (`f1-brows-test_0.glb`) is **not approved** (untidy). The refined
  version (`f1-brows-test-2_0.glb`, source `var\brow_check\20261012_013909\F1_162.6cm_50kg_brows.glb`,
  SHA-256 `b564fc94…`) was **visually approved by me in the browser** — appearance on F1 only.
  That design is now integrated for female avatars (a male design was added); the user has not
  yet reviewed the integrated result.
- Browser previews live in `frontend\public\face_preview_20261011_235711\` (temporary; do not
  delete without my approval).
- Workspace rename to `D:\AvatarModule` is postponed (it was blocked by the servers then running — now stopped — and is still blocked by the
  `.venv` launchers containing the old path).

## Appearance integration (completed 2026-10-12)

Read §0 of `PROJECT_HANDOFF_REPORT.md`. Eye colour, skin tone, gender face presets, female lip tint,
mouth width -0.35 (female) and eyebrows (female + male) are integrated in the API (`backend\appearance.py`,
`backend\api.py`) and the frontend, and verified by tests (85 OK), in-process end-to-end runs and
headless-Edge browser runs (`var\appearance_check\`). Hair is not available. Both servers are stopped
(memory pressure, see above); the backend code on disk is the final version. Pending: restarting
the servers (with my approval) and my own visual review in the browser; GLB
format compatibility with the teammates' clothing tools.

## Recommended first step (wait for my approval)

Propose making a **first Git commit of the module** so the untracked work is safe. Show me exactly
which files would be committed (respecting `.gitignore`), and do not commit until I approve.

## Rules

- Plan first; state the exact files and commands; **wait for my explicit approval** for each step.
- No package installs, no new model caches, no server start/stop/restart, no Git commits, resets
  or branch changes, no deleting or overwriting outputs or previews — unless I approve it.
- Do not change the production fitting algorithm. Do not generate avatars or run fitting unless
  approved.
- New outputs go to new timestamped folders under `var\`.
- Separate verified facts from assumptions; "server responds" is not "visually verified".
- Keep reports concise and beginner-friendly; one recommended next step at a time.
