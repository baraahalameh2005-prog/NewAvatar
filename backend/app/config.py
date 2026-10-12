"""
Module configuration: server address, output and cache folders.

Every folder the module writes to is resolved here. Defaults stay inside
the module; each can be overridden with an environment variable, but never
to a location inside one of the protected projects.
"""

import os
from pathlib import Path


# backend/app/config.py -> module root is three levels up.
MODULE_ROOT = Path(__file__).resolve().parents[2]

# Projects this module must never write into.
PROTECTED_ROOTS = (
    Path(r"D:\BarahaAvatar"),
    Path(r"D:\AvatarGitHub\avatar"),
    Path(r"D:\BarahaAvatar_Experiment"),
    Path(r"D:\Avatar_Backups"),
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002


def _normalize(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def is_inside(path, root):
    """True if path is root itself or lies below it (not just a name prefix)."""

    path, root = _normalize(path), _normalize(root)

    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        # Different drives.
        return False


def check_writable(path):
    """Return path as a Path; raise ValueError if it is inside a protected project."""

    for root in PROTECTED_ROOTS:
        if is_inside(path, root):
            raise ValueError(
                f"Refusing to use {path}: it is inside the protected folder {root}."
            )

    return Path(path)


def _path_setting(name, default):
    return check_writable(os.environ.get(name) or default)


class Settings:
    def __init__(self):
        var_dir = MODULE_ROOT / "var"

        self.host = os.environ.get("AVATAR_HOST") or DEFAULT_HOST
        self.port = int(os.environ.get("AVATAR_PORT") or DEFAULT_PORT)

        # Generated GLBs, one file per request.
        self.output_dir = _path_setting("AVATAR_OUTPUT_DIR", var_dir / "generated")

        # Caches written by Anny and NVIDIA Warp on first use.
        self.anny_cache_dir = _path_setting("ANNY_CACHE_DIR", var_dir / "cache" / "anny")
        self.warp_cache_dir = _path_setting("WARP_CACHE_PATH", var_dir / "cache" / "warp")

        # PyTorch creates this folder when torch is imported (torch._dynamo
        # resolves it at import); without the setting it goes to the user's
        # TEMP folder.
        self.torchinductor_cache_dir = _path_setting(
            "TORCHINDUCTOR_CACHE_DIR", var_dir / "cache" / "torchinductor"
        )

        # The production API also overwrites a shared generated_avatar.glb;
        # the module does not.
        self.write_legacy_avatar = False


def apply_environment(settings):
    """
    Point Anny, Warp and PyTorch at the module's cache folders.

    Must run before `import torch` and `import anny`: both read their cache
    folder at import time. Values already set in the environment are kept
    (and were checked by Settings).
    """

    os.environ.setdefault("ANNY_CACHE_DIR", str(settings.anny_cache_dir))
    os.environ.setdefault("WARP_CACHE_PATH", str(settings.warp_cache_dir))
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", str(settings.torchinductor_cache_dir)
    )


settings = Settings()
