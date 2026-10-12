# Baraha Avatar Module

Standalone avatar module for the Palestinian virtual-fitting project: a Python/FastAPI
backend that fits the Anny body model to user measurements and exports a GLB, and a
React viewer.

**Status:** skeleton only. See `docs/STATUS.md` for what has been built and verified.

## Layout

| Path | Contents |
|---|---|
| `backend/app/` | Backend code. `config.py` holds the port (8002) and all output/cache folders |
| `backend/tests/` | Backend tests |
| `frontend/` | React viewer (not started) |
| `docs/` | Status log; later the API contract and integration guide |
| `var/` | Created at runtime: generated GLBs and Anny/Warp caches. Not in Git |

## Safety

The module never writes into the original backend, the teammates' frontend, the
experiment workspace or the backups. `backend/app/config.py` refuses any output or
cache folder inside those projects.
