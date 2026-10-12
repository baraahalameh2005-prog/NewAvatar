"""
Configuration safety tests. Standard library only; writes only an empty
temporary folder.

Run from the backend folder:
    python -B -m unittest tests.test_config -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import config


ENV_KEYS = (
    "AVATAR_HOST",
    "AVATAR_PORT",
    "AVATAR_OUTPUT_DIR",
    "ANNY_CACHE_DIR",
    "WARP_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
)


def all_paths(s):
    return (
        s.output_dir,
        s.anny_cache_dir,
        s.warp_cache_dir,
        s.torchinductor_cache_dir,
    )


def clean_env(**values):
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(values)

    return mock.patch.dict(os.environ, env, clear=True)


class DefaultsTest(unittest.TestCase):
    def test_module_root(self):
        self.assertEqual(config.MODULE_ROOT, Path(__file__).resolve().parents[2])
        self.assertTrue((config.MODULE_ROOT / "backend" / "app" / "config.py").is_file())

    def test_defaults(self):
        with clean_env():
            s = config.Settings()

        self.assertEqual(s.host, "127.0.0.1")
        self.assertEqual(s.port, 8002)
        self.assertFalse(s.write_legacy_avatar)

        for path in all_paths(s):
            self.assertTrue(config.is_inside(path, config.MODULE_ROOT), path)

        self.assertEqual(s.output_dir, config.MODULE_ROOT / "var" / "generated")
        self.assertEqual(
            s.torchinductor_cache_dir,
            config.MODULE_ROOT / "var" / "cache" / "torchinductor",
        )

    def test_loading_settings_creates_no_folders(self):
        # An empty stand-in for the module root: whatever already exists in
        # the real var/ folder cannot hide a folder created by Settings.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)

            with clean_env(), mock.patch.object(config, "MODULE_ROOT", root):
                s = config.Settings()

            self.assertEqual(list(root.iterdir()), [])

            for path in all_paths(s):
                self.assertTrue(config.is_inside(path, root), path)


class ProtectedPathTest(unittest.TestCase):
    def test_protected_roots_and_children_rejected(self):
        for root in config.PROTECTED_ROOTS:
            for path in (root, root / "generated", root / "a" / "b.glb"):
                with self.subTest(path=path):
                    with self.assertRaises(ValueError):
                        config.check_writable(path)

    def test_case_and_separator_variants_rejected(self):
        for path in (r"d:\barahaavatar\generated", "D:/BarahaAvatar/generated"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    config.check_writable(path)

    def test_name_prefix_is_not_inside(self):
        # D:\BarahaAvatarModule starts with "D:\BarahaAvatar" but is a different folder.
        self.assertFalse(config.is_inside(r"D:\BarahaAvatarModule", r"D:\BarahaAvatar"))
        self.assertEqual(
            config.check_writable(r"D:\BarahaAvatarModule\var"),
            Path(r"D:\BarahaAvatarModule\var"),
        )

    def test_env_override_into_protected_folder_rejected(self):
        for key in (
            "AVATAR_OUTPUT_DIR",
            "ANNY_CACHE_DIR",
            "WARP_CACHE_PATH",
            "TORCHINDUCTOR_CACHE_DIR",
        ):
            with self.subTest(key=key):
                with clean_env(**{key: r"D:\BarahaAvatar_Experiment\anny_cache"}):
                    with self.assertRaises(ValueError):
                        config.Settings()

    def test_shared_user_cache_is_not_the_default(self):
        shared = Path.home() / ".cache" / "anny"

        with clean_env():
            s = config.Settings()

        self.assertFalse(config.is_inside(s.anny_cache_dir, shared))


class OverrideTest(unittest.TestCase):
    def test_port_and_output_override(self):
        target = config.MODULE_ROOT / "var" / "other"

        with clean_env(AVATAR_PORT="9000", AVATAR_OUTPUT_DIR=str(target)):
            s = config.Settings()

        self.assertEqual(s.port, 9000)
        self.assertEqual(s.output_dir, target)

    def test_apply_environment_sets_caches(self):
        with clean_env():
            s = config.Settings()
            config.apply_environment(s)

            self.assertEqual(os.environ["ANNY_CACHE_DIR"], str(s.anny_cache_dir))
            self.assertEqual(os.environ["WARP_CACHE_PATH"], str(s.warp_cache_dir))
            self.assertEqual(
                os.environ["TORCHINDUCTOR_CACHE_DIR"],
                str(s.torchinductor_cache_dir),
            )

    def test_apply_environment_keeps_existing_values(self):
        existing = str(config.MODULE_ROOT / "var" / "custom_anny")

        with clean_env(ANNY_CACHE_DIR=existing):
            s = config.Settings()
            config.apply_environment(s)

            self.assertEqual(os.environ["ANNY_CACHE_DIR"], existing)


if __name__ == "__main__":
    unittest.main()
