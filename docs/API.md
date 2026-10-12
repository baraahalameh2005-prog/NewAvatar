# Avatar API — frontend handoff

Backend: FastAPI app in `backend/api.py`. Base URL in development: `http://127.0.0.1:8002`.

Copied from the production backend (`D:\BarahaAvatar`). Differences in this module are
marked **Module:** below.

## Run

Run from `D:\BarahaAvatarModule\backend` (tested; the server loads the plain Anny model and the
cached face model at startup):

```powershell
..\.venv\Scripts\python.exe api.py
```

**Status (2026-10-12):** the server is **not running**. Claude Code stopped it (and the Vite frontend)
because the system was critically low on memory. The code and saved verification outputs remain in
place; start it again with the command above. It loads two Anny models at startup, so it needs more
memory than the plain-model version.

Host and port come from `backend/app/config.py` (127.0.0.1:8002; override with
`AVATAR_HOST` / `AVATAR_PORT`).

The Anny model loads once at server startup (a few seconds). Generation requests
are processed one at a time; a request takes roughly the fitting time (about 10-15 s
measured for one female case on this machine, more for other bodies).

**Module:** generated GLBs are written to `var/generated/`; the Anny, Warp and PyTorch
caches to `var/cache/anny`, `var/cache/warp` and `var/cache/torchinductor`. Each can be
moved with `AVATAR_OUTPUT_DIR`, `ANNY_CACHE_DIR`, `WARP_CACHE_PATH` or
`TORCHINDUCTOR_CACHE_DIR`, but never into the protected projects.

## Tests

From `D:\BarahaAvatarModule\backend`:

```powershell
$env:AVATAR_SKIP_MODEL_TESTS = "1"
..\.venv\Scripts\python.exe -B -m unittest tests.test_config tests.test_avatar
```

`AVATAR_SKIP_MODEL_TESTS=1` skips the tests that load the real model.
The tests never write `var/generated/` or a shared `generated_avatar.glb`.

## Endpoints

### `POST /generate-avatar`

Request (JSON):

| Field | Type | Required | Range / format |
|---|---|---|---|
| `height` | number | yes | 100-230 cm |
| `weight` | number | yes | 25-250 kg |
| `waist` | number | yes | 40-200 cm |
| `chest` | number | yes | 50-200 cm |
| `hip` | number | yes | 50-200 cm |
| `gender` | string | no (default `"female"`) | `"female"` or `"male"`, case-insensitive |
| `skin_tone` | string | no (default `"#c8997e"`) | `"#RRGGBB"` (sRGB) |
| `eye_color` | string | no (default `"brown"`) | `"brown"`, `"hazel"`, `"green"`, `"blue"`, `"gray"`, case-insensitive |

Other fields are ignored and do not change the avatar.
Invalid input returns **422** with a `detail` list.

Response (200):

```jsonc
{
  "message": "Avatar generated successfully",
  "avatar_url": "/models/generated/<avatar_id>.glb",   // unique for this generation
  "avatar_id": "<32 hex chars>",
  "legacy_avatar_url": null,                           // Module: always null (see below)
  "parameters": { "gender": "female", "height": 0.5, "weight": 0.6, "age": 0.55, "muscle": 0.65, "proportions": 0.62 },
  "appearance": { "gender_style": "female", "skin_tone": "#c8997e", "eye_color": "brown",
                  "face_controls": { "mouth-scale-horiz-incr": -0.35, "...": "..." },
                  "features": ["eye_texture", "eyebrows", "face_preset", "lip_tint"], "not_available": ["hair"] },
  "measurements": { "height_m": 1.626, "mass_kg": 49.99, "waist_m": 0.699, "chest_m": 0.865, "hip_m": 0.940, "...": "..." },
  "failed_measurements": [],
  "accuracy": {
    "waist": { "requested": 71.8, "actual": 69.92, "unit": "cm",
               "absolute_error": 1.88, "percentage_error": 2.62,
               "control": "fitted indirectly (age, muscle, proportions)", "status": "measured" }
    // also height, weight, chest, hip
  },
  "warnings": ["waist: 69.92 cm vs requested 71.80 cm (2.62% off), outside the fitting tolerance."],
  "fit_time_seconds": 11.5
}
```

- `measurements` values are in metres (`*_m`) and kilograms (`mass_kg`).
  `shoulder_width_m`, `arm_length_m` and `leg_length_m` are joint-centre estimates,
  not tape measurements. They are reported, not controlled.
- `accuracy` uses `abs(actual - requested) / abs(requested) * 100`.
- `warnings` lists targets that were not reached within the fitting tolerance
  (height 0.01 cm, weight 0.01 kg, circumferences 0.2%) or could not be measured.
  Waist and hip errors of a few percent are a known limitation of the current model setup.

### Loading the avatar in three.js

```ts
const data = await response.json();
const url = `${API_URL}${data.avatar_url}`;   // unique file, no cache-buster needed
useGLTF(url) / new GLTFLoader().load(url, ...)
```

Use `avatar_url` so each user gets their own file.

**Module:** the shared `generated_avatar.glb` that production overwrote on every request is
not written, and `legacy_avatar_url` is `null`. The field is kept so existing clients do
not break on a missing key. (`WRITE_LEGACY_AVATAR` in `api.py` can turn the old behaviour
back on; it is off by default.)

The GLB is one static mesh (Y up, feet at y = 0, metres, smooth normals, no skeleton) with **two
primitives that share the same vertex attributes** (13,718 vertices): the body (material `skin`,
base-colour texture via TEXCOORD_0: skin colour, female lip tint, eyes) and the brow band (material
`skin_brows`, texture via TEXCOORD_1: painted eyebrows). Textures use a clamp-to-edge sampler.
The body fit is unchanged; after it, a gender face preset moves head vertices only (a few mm), so
`measurements` / `accuracy` describe the fitted body before the face preset. No hair or clothing.

### `GET /avatar-options`

Machine-readable list of the accepted inputs, ranges and defaults above. Build form
controls from it instead of hard-coding ranges.

### `GET /health`

`{"status": "ok", "model_loaded": true}`. Does not load the model.

### Static files

- `/models/generated/<avatar_id>.glb` — generated avatars (from `var/generated/`).
- `/models/<name>.glb` and `/test/<name>.html` — `.glb` / `.html` files directly in the
  `backend/` folder. **Module:** the backend folder holds no such files (production's
  `generated_avatar.glb`, `shirt_*.glb` and `test_avatar.html` were not copied), so these
  return 404 for now.

Other files (source code, `.git`, subfolders, hidden files) return 404.

## Known limitations

- Generated files in `generated/` are not deleted automatically. Each is about 0.5-1 MB.
- `avatar_id` is random but not secret: anyone who knows a URL can load that avatar.
- Requests are processed one at a time; a second request waits for the first.
- Hair and extra body controls are not available. Face shape follows gender automatically (not adjustable).
