"""
Tests for appearance.py (face presets, eye colour, skin tone, female lip tint, brows).

Run from the backend folder:

    ..\\.venv\\Scripts\\python.exe -B -m unittest tests.test_appearance -v

AppearanceModelTests load Anny twice (plain and face model) and evaluate recorded
bodies (no fitting). Set AVATAR_SKIP_MODEL_TESTS=1 to skip them. Exports go to a
temporary folder.
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import appearance as ap


class AppearanceOptionTests(unittest.TestCase):

    def test_eye_colors(self):
        self.assertEqual(ap.normalize_eye_color(" Blue "), "blue")
        self.assertIn(ap.DEFAULT_EYE_COLOR, ap.EYE_COLORS)
        for value in ("purple", "", None, 3):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ap.normalize_eye_color(value)

    def test_appearance_for(self):
        female = ap.appearance_for("female", "#c8997e", "green")
        male = ap.appearance_for("male", "#c8997e", "brown")

        self.assertEqual(female.face_controls, ap.FACE_PRESETS["female"])
        self.assertEqual(male.face_controls, ap.FACE_PRESETS["male"])
        self.assertIn("lip_tint", female.summary()["features"])
        self.assertNotIn("lip_tint", male.summary()["features"])
        with self.assertRaises(ValueError):
            ap.appearance_for("female", "pink", "brown")

    def test_face_preset_values_are_moderate(self):
        for sex, preset in ap.FACE_PRESETS.items():
            for name, value in preset.items():
                with self.subTest(sex=sex, control=name):
                    self.assertLessEqual(abs(value), 0.5)
                    self.assertIn(name, {label.removeprefix("l-").removeprefix("r-") for label in ap.FACE_MODEL_LABELS})

    def test_face_model_labels_match_cached_reference(self):
        reference = json.loads((ROOT / "tools" / "stage5_ac67_reference.json").read_text())["local_changes_list"]

        self.assertEqual(ap.FACE_MODEL_LABELS, reference)

    def test_base_texture(self):
        skin = (200, 153, 126)
        for color, (iris, _) in ap.EYE_COLORS.items():
            with self.subTest(color=color):
                img = np.asarray(ap.base_texture("#c8997e", color, True))
                np.testing.assert_array_equal(img[64, 2], skin)                 # skin end of the strip
                np.testing.assert_array_equal(img[64, 192], ap.PUPIL)            # pupil at the eye centre
                np.testing.assert_array_equal(img[64, 130], ap.SCLERA)           # sclera at the eye edge
                self.assertLessEqual(np.abs(img[64, 192 + 20].astype(int) - iris).max(), 1)
        female = np.asarray(ap.base_texture("#c8997e", "brown", True))
        male = np.asarray(ap.base_texture("#c8997e", "brown", False))
        self.assertGreater(np.abs(female[64, ap.STRIP_LAST].astype(int) - skin).max(), 20)   # lip tint
        self.assertTrue((male[:, : ap.TEX_W // 2] == skin).all())                             # male: flat skin

    def test_brow_texture_edges_are_skin(self):
        for tone in ("#c8997e", "#6b4430", "#e2bfa3"):
            skin = np.array(ap.parse_skin_tone(tone)[:3])
            for sex in ("female", "male"):
                with self.subTest(tone=tone, sex=sex):
                    img = np.asarray(ap.brow_texture(tone, sex)).astype(int)
                    border = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3),
                                             img[:, :8].reshape(-1, 3), img[:, -8:].reshape(-1, 3)])
                    self.assertLessEqual(np.abs(border - skin).max(), 1)
                    self.assertGreater(np.abs(img - skin).max(), 40)               # brows are visible

    def test_brow_colour_follows_dark_skin(self):
        design = ap.BROWS["female"]
        light = ap.brow_color(design, np.array([200.0, 153, 126]))
        dark = ap.brow_color(design, np.array([70.0, 45, 32]))

        np.testing.assert_array_equal(light, design.rgb)
        self.assertTrue((dark < np.array(design.rgb)).all())


@unittest.skipIf(
    os.environ.get("AVATAR_SKIP_MODEL_TESTS") == "1",
    "AVATAR_SKIP_MODEL_TESTS=1",
)
class AppearanceModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import anny

        import fit_to_measurements as ftm
        from generate_avatar import build

        cls.build = staticmethod(build)
        cls.plain = anny.Anny()
        cls.face = ap.FaceModel()
        cases = {c["case"]: c for c in json.loads((ROOT / "tools" / "baseline_cases.json").read_text())["cases"]}
        cls.bodies = {}
        for name in ("F1_162.6cm_50kg", "M2_175cm_72kg"):
            p = cases[name]["parameters"]
            cls.bodies[p["gender"]] = dict(gender=ftm.gender_value(p["gender"]), height=p["height"], weight=p["weight"],
                                           age=p["age"], muscle=p["muscle"], proportions=p["proportions"])

    def test_zero_controls_reproduce_the_plain_model(self):
        kw = self.bodies["female"]
        plain = self.build(model=self.plain, **kw)["rest_vertices"]
        zero = self.build(model=self.face.controls({}), **kw)["rest_vertices"]

        self.assertEqual(float((plain - zero).abs().max()), 0.0)

    def test_presets_move_only_the_head(self):
        for sex, kw in self.bodies.items():
            with self.subTest(sex=sex):
                plain = self.build(model=self.plain, **kw)
                face = self.build(model=self.face.controls(ap.FACE_PRESETS[sex]), **kw)
                rp = plain["rest_vertices"][0].detach().cpu().numpy()
                rf = face["rest_vertices"][0].detach().cpu().numpy()
                d = np.linalg.norm(rf - rp, axis=1)
                neck = list(self.plain.bone_labels).index("neck01")
                neck_z = float(plain["rest_bone_poses"][0, neck, 2, 3])

                self.assertGreater(d.max(), 0.001)                      # the preset does something
                self.assertLess(d.max(), 0.01)                          # but stays moderate (< 1 cm)
                self.assertLess(d[rp[:, 2] < neck_z].max(), 0.0002)     # body below the neck: < 0.2 mm
                self.assertEqual(float(d[self.face.eye_index].max()), 0.0)

    def test_export(self):
        for sex, kw in self.bodies.items():
            with self.subTest(sex=sex), tempfile.TemporaryDirectory() as folder:
                appearance = ap.appearance_for(sex, "#6b4430", "blue")
                output = self.build(model=self.face.controls(appearance.face_controls), **kw)
                path = Path(folder) / "avatar.glb"
                lay = ap.export_appearance_avatar(output, self.face, path, appearance)

                self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["avatar.glb"])
                data = path.read_bytes()
                self.assertEqual(data[:4], b"glTF")
                self.assertEqual(struct.unpack("<I", data[8:12])[0], len(data))
                n = struct.unpack("<I", data[12:16])[0]
                gltf = json.loads(data[20:20 + n])
                prims = gltf["meshes"][0]["primitives"]

                self.assertEqual(len(gltf["meshes"]), 1)
                self.assertEqual(len(prims), 2)
                self.assertEqual(prims[0]["attributes"], prims[1]["attributes"])
                self.assertNotIn("skins", gltf)
                self.assertEqual(gltf["accessors"][prims[0]["attributes"]["POSITION"]]["count"], len(lay.vertices))
                count = sum(gltf["accessors"][p["indices"]]["count"] for p in prims)
                self.assertEqual(count, self.face.faces.size)
                self.assertEqual([gltf["materials"][p["material"]]["name"] for p in prims], ["skin", "skin_brows"])
                self.assertEqual(gltf["materials"][prims[1]["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"], 1)
                # textures clamp at the edges (no wrap-around bleeding in the browser)
                self.assertEqual(gltf["samplers"], [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
                self.assertTrue(all(t.get("sampler") == 0 for t in gltf["textures"]))
                self.assertGreater(int(lay.brow_faces.sum()), 400)
                self.assertTrue(((lay.uv1 >= 0) & (lay.uv1 <= 1)).all())


if __name__ == "__main__":
    unittest.main()
