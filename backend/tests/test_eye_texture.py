"""
Tests for the experimental eye-texture prototype (tools/eye_texture_check.py).

Run from the backend folder:

    ..\\.venv\\Scripts\\python.exe -B -m unittest tests.test_eye_texture -v

They load Anny and evaluate one recorded body (no fitting). Set
AVATAR_SKIP_MODEL_TESTS=1 to skip them. Exports go to a temporary folder.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np


class EyeTextureImageTests(unittest.TestCase):

    def test_texture_regions(self):
        import eye_texture_check as etc

        img = np.asarray(etc.eye_texture("#c8997e"))

        self.assertEqual(img.shape, (etc.TEX_H, etc.TEX_W, 3))
        # Skin block holds the sRGB skin tone as entered.
        np.testing.assert_array_equal(img[64, 64], [200, 153, 126])
        # Eye square: pupil at the centre, sclera near the edge.
        np.testing.assert_array_equal(img[64, 192], etc.PUPIL)
        np.testing.assert_array_equal(img[64, 130], etc.SCLERA)


@unittest.skipIf(
    os.environ.get("AVATAR_SKIP_MODEL_TESTS") == "1",
    "AVATAR_SKIP_MODEL_TESTS=1",
)
class EyeTextureExportTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import eye_texture_check as etc
        from anny.face_segmentation import get_face_segmentation_mask
        from generate_avatar import DEFAULT_SKIN_TONE, build

        cls.etc = etc
        cls.model = etc.anny.Anny()
        cls.faces = cls.model.faces.detach().cpu().numpy()
        p = etc.CASES["F1_162.6cm_50kg"]["parameters"]
        cls.output = build(model=cls.model, gender=etc.ftm.gender_value(p["gender"]), height=p["height"],
                           weight=p["weight"], age=p["age"], muscle=p["muscle"], proportions=p["proportions"])
        cls.rest = cls.output["rest_vertices"][0].detach().cpu().numpy()
        cls.v = etc.export_frame(cls.rest)
        cls.pieces = etc.eyeballs(cls.faces, len(cls.v))
        masks = {s: get_face_segmentation_mask(cls.model, [f"eye_front.{s}"]).cpu().numpy() for s in ("L", "R")}
        cls.uv, cls.info = etc.eye_uv(cls.v, cls.faces, cls.pieces, masks)
        cls.image = etc.eye_texture(DEFAULT_SKIN_TONE)

    def test_eyeballs_and_uv(self):
        self.assertEqual(sorted(len(p) for p in self.pieces), [72, 72])
        self.assertTrue(((self.uv >= 0) & (self.uv <= 1)).all())

        eyes = np.concatenate(self.pieces)
        others = np.setdiff1d(np.arange(len(self.v)), eyes)
        np.testing.assert_array_equal(self.uv[others], np.tile(self.etc.SKIN_UV, (len(others), 1)))
        # Eye UVs stay inside the eye square (right half).
        self.assertTrue((self.uv[eyes, 0] >= 0.5).all())

        for side in ("L", "R"):
            self.assertLess(self.info[side]["angle_from_straight_ahead_deg"], 15.0)

    def test_geometry_bytes_identical_to_production_export(self):
        from generate_avatar import export_avatar

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "base.glb"
            textured = Path(folder) / "eyes.glb"
            export_avatar(self.output, self.model, base)
            self.etc.export_textured(self.rest, self.faces, self.uv, self.image, textured)

            a, b = self.etc.glb_parts(base), self.etc.glb_parts(textured)

            for key in ("POSITION", "NORMAL", "indices"):
                self.assertEqual(a[key][0], b[key][0], key)

            self.assertEqual((b["meshes"], b["primitives"]), (1, 1))
            self.assertTrue(b["header_ok"])
            self.assertEqual(b["TEXCOORD_0"][1]["count"], len(self.v))
            self.assertNotIn("skins", b["gltf"])


if __name__ == "__main__":
    unittest.main()
