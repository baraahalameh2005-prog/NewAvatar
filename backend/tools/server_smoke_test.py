"""
Start the module API with its documented command, check a few read-only
endpoints, then stop it cleanly (Ctrl+Break, which uvicorn handles on Windows).

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\server_smoke_test.py [--generate]

Without --generate, /generate-avatar is not called. With --generate, one
invalid request (must be 422, no file) and one valid request (baseline case
F1, must reproduce the baseline GLB) are sent and checked.

Writes only to var/server_check/<timestamp>/ (server log, responses.json) plus
whatever the server itself writes inside the module (model caches, and with
--generate one GLB in var/generated/).
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import MODULE_ROOT, is_inside, settings

PYTHON = MODULE_ROOT / ".venv" / "Scripts" / "python.exe"
BASE = f"http://{settings.host}:{settings.port}"
STARTUP_TIMEOUT = 600
STOP_TIMEOUT = 60
GENERATE_TIMEOUT = 300

# Baseline case F1 and its recorded GLB (see tools/reproduce_baseline.py).
F1_REQUEST = {
    "gender": "female",
    "height": 162.62,
    "weight": 49.99,
    "waist": 71.8,
    "chest": 86.6,
    "hip": 92.15,
}
F1_SHA256 = "759f27fa5109541b007dda9ec9202c9b9ea8af57c0236c31d464cf5f6a154321"

RESPONSE_FIELDS = [
    "message",
    "avatar_url",
    "avatar_id",
    "legacy_avatar_url",
    "parameters",
    "measurements",
    "failed_measurements",
    "accuracy",
    "warnings",
    "fit_time_seconds",
]


def port_in_use():
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((settings.host, settings.port)) == 0


def request(path, payload=None, timeout=10, raw=False):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(BASE + path, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        body = e.read()
        status = e.code

    if raw:
        return status, body

    try:
        return status, json.loads(body)
    except ValueError:
        return status, body.decode("utf-8", "replace")


def glb_files():
    folder = settings.output_dir
    return sorted(p.name for p in folder.glob("*.glb")) if folder.is_dir() else []


def generation_checks():
    """Invalid and valid generation requests; returns (checks, details)."""

    import trimesh

    checks = {}
    details = {}

    # 1. Invalid request: rejected, nothing written.
    before = glb_files()
    status, body = request("/generate-avatar", {**F1_REQUEST, "height": 50})
    details["invalid_request"] = {"status": status, "body": body}
    locs = [e.get("loc") for e in body.get("detail", [])] if isinstance(body, dict) else []
    checks["invalid_is_422_on_height"] = status == 422 and ["body", "height"] in locs
    checks["invalid_writes_no_file"] = glb_files() == before

    # 2. Valid request.
    before = glb_files()
    t = time.perf_counter()
    status, data = request("/generate-avatar", F1_REQUEST, timeout=GENERATE_TIMEOUT)
    seconds = time.perf_counter() - t
    details["valid_request"] = {"status": status, "seconds": seconds, "body": data}
    checks["valid_is_200"] = status == 200

    if status != 200 or not isinstance(data, dict):
        return checks, details

    checks["response_has_all_fields"] = all(k in data for k in RESPONSE_FIELDS)
    checks["legacy_url_is_null"] = data.get("legacy_avatar_url") is None
    avatar_id = data.get("avatar_id", "")
    checks["avatar_url_format"] = data.get("avatar_url") == f"/models/generated/{avatar_id}.glb"

    path = settings.output_dir / f"{avatar_id}.glb"
    new_files = sorted(set(glb_files()) - set(before))
    details["glb_path"] = str(path)
    details["new_files"] = new_files
    checks["file_exists_inside_module"] = path.is_file() and is_inside(path, MODULE_ROOT)
    checks["only_one_new_glb"] = new_files == [path.name]
    checks["no_legacy_file"] = not (BACKEND / "generated_avatar.glb").exists()

    if not path.is_file():
        return checks, details

    content = path.read_bytes()
    status, served = request(data["avatar_url"], raw=True)
    checks["url_serves_200"] = status == 200
    checks["served_bytes_equal_file"] = served == content

    checks["glb_header_v2"] = (
        content[:4] == b"glTF"
        and int.from_bytes(content[4:8], "little") == 2
        and int.from_bytes(content[8:12], "little") == len(content)
    )

    scene = trimesh.load(path)
    meshes = list(scene.geometry.values())
    details["glb"] = {
        "bytes": len(content),
        "meshes": len(meshes),
        "vertices": sum(len(m.vertices) for m in meshes),
        "faces": sum(len(m.faces) for m in meshes),
        "height_m": float(scene.bounds[1][1] - scene.bounds[0][1]),
        "min_y": float(scene.bounds[0][1]),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    checks["one_nonempty_mesh"] = (
        len(meshes) == 1 and details["glb"]["vertices"] > 0 and details["glb"]["faces"] > 0
    )
    checks["sha256_equals_baseline_F1"] = details["glb"]["sha256"] == F1_SHA256

    return checks, details


def main():
    generate = "--generate" in sys.argv[1:]

    if port_in_use():
        sys.exit(f"Port {settings.port} is already in use; not starting a server.")

    out = MODULE_ROOT / "var" / "server_check" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    log_path = out / "server_log.txt"

    command = [str(PYTHON), "api.py"]
    print("Command:", " ".join(command), "(cwd backend)")
    print("Log:", log_path)

    log = open(log_path, "w", encoding="utf-8")
    started = time.perf_counter()
    server = subprocess.Popen(
        command,
        cwd=BACKEND,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print("Server PID:", server.pid)

    responses = {}
    checks = {}
    details = {}
    ready_seconds = None

    try:
        # /health answers only after startup, which loads the model.
        while time.perf_counter() - started < STARTUP_TIMEOUT:
            if server.poll() is not None:
                print("Server exited during startup, code", server.returncode)
                break
            try:
                request("/health")
                ready_seconds = time.perf_counter() - started
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(1)

        if ready_seconds is not None:
            print(f"Ready after {ready_seconds:.1f} s")

            for path in [
                "/health",
                "/avatar-options",
                "/",
                "/models/generated/not-a-real-id.glb",
            ]:
                status, body = request(path)
                responses[path] = {"status": status, "body": body}
                print(f"\nGET {path} -> {status}")
                print(json.dumps(body, indent=1) if not isinstance(body, str) else body)

            if generate:
                checks, details = generation_checks()
                print("\nInvalid request:", json.dumps(details.get("invalid_request"), indent=1))
                valid = details.get("valid_request", {})
                print(f"\nValid request: {valid.get('status')} in {valid.get('seconds', 0):.1f} s")
                print(json.dumps(valid.get("body"), indent=1))
                print("\nGLB:", details.get("glb_path"), json.dumps(details.get("glb"), indent=1))
                print("\nChecks:")
                for name, ok in checks.items():
                    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    finally:
        stop_result = "not running"

        if server.poll() is None:
            server.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                server.wait(timeout=STOP_TIMEOUT)
                stop_result = f"stopped cleanly after Ctrl+Break, exit code {server.returncode}"
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
                stop_result = f"did not stop within {STOP_TIMEOUT} s; killed"

        log.close()
        print("\nShutdown:", stop_result)
        print("Port still in use after shutdown:", port_in_use())

        (out / "responses.json").write_text(json.dumps({
            "command": command,
            "ready_seconds": ready_seconds,
            "responses": responses,
            "generation_checks": checks,
            "generation_details": details,
            "shutdown": stop_result,
        }, indent=1, default=float))

    ok = (
        ready_seconds is not None
        and responses.get("/health", {}).get("status") == 200
        and responses.get("/avatar-options", {}).get("status") == 200
        and (not generate or (checks and all(checks.values())))
    )
    print("\nALL CHECKS PASSED:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
