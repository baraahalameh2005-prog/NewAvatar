"""
Tests for the experimental arm-pose prototype (tools/arm_pose_check.py).

Run from the backend folder:

    ..\\.venv\\Scripts\\python.exe -B -m unittest tests.test_arm_pose -v

They load Anny and evaluate one recorded body (no fitting). Set
AVATAR_SKIP_MODEL_TESTS=1 to skip them. Exports go to a temporary folder.
"""

import json
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


@unittest.skipIf(
    os.environ.get("AVATAR_SKIP_MODEL_TESTS") == "1",
    "AVATAR_SKIP_MODEL_TESTS=1",
)
class ArmPoseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import arm_pose_check as apc
        from measurements import BodyMeasurements

        cls.apc = apc
        cls.model = apc.anny.Anny()
        cls.rig = apc.Rig(cls.model)
        cls.meas = BodyMeasurements(cls.model)
        cls.faces = cls.model.faces.detach().cpu().numpy()
        cls.case, cls.output, cls.phenotype = apc.load_body(cls.model, "F2")
        cls.rest = cls.output["rest_vertices"]

    def posed(self, degrees):
        return self.apc.posed_output(self.model, self.rig, self.phenotype, self.output["rest_bone_poses"], degrees)

    def test_zero_world_rotation_reproduces_rest_pose(self):
        out = self.posed(0.0)

        self.assertLess(float((out["vertices"] - self.rest).abs().max()), 1e-12)

    def test_default_zero_pose_is_not_the_rest_pose(self):
        # Why the prototype uses the "world" parameterization.
        out = self.model(phenotype_kwargs=self.phenotype)

        self.assertGreater(float((out["vertices"] - self.rest).abs().max()), 0.01)

    def test_pose_does_not_change_measurements_or_rest_geometry(self):
        out = self.posed(15.0)

        self.assertTrue(bool((out["rest_vertices"] == self.rest).all()))
        self.assertEqual(self.meas.measure(out), self.meas.measure(self.output))

    def test_rotation_angle_and_mirror_symmetry(self):
        rest_angle = self.apc.frontal_angle(self.output["rest_bone_poses"], self.rig, "upperarm01", "lowerarm01")
        out = self.posed(15.0)
        angle = self.apc.frontal_angle(out["bone_poses"], self.rig, "upperarm01", "lowerarm01")

        for side in ("L", "R"):
            self.assertAlmostEqual(rest_angle[side] - angle[side], 15.0, places=6)

        rest_export = self.apc.to_export_frame(self.rest[0].detach().cpu().numpy())
        regions = self.apc.Regions(self.rig, self.faces, rest_export)
        v = self.apc.to_export_frame(out["vertices"][0].detach().cpu().numpy())
        mirror_error = np.abs(v - v[regions.mirror] * np.array([-1, 1, 1])).max()

        # The rest mesh itself is symmetric to about 0.01 mm.
        self.assertLess(float(mirror_error), 1e-4)

    def test_posed_export_is_a_valid_static_glb(self):
        import trimesh

        from generate_avatar import export_avatar

        out = self.posed(15.0)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "posed.glb"
            export_avatar({"rest_vertices": out["vertices"]}, self.model, path)

            content = path.read_bytes()
            self.assertEqual(content[:4], b"glTF")
            self.assertEqual(int.from_bytes(content[4:8], "little"), 2)
            self.assertEqual(int.from_bytes(content[8:12], "little"), len(content))

            length = int.from_bytes(content[12:16], "little")
            gltf = json.loads(content[20:20 + length])
            self.assertNotIn("skins", gltf)
            self.assertNotIn("animations", gltf)

            mesh = trimesh.load(path, force="mesh", process=False)
            expected = self.apc.to_export_frame(out["vertices"][0].detach().cpu().numpy())

            self.assertEqual(len(mesh.vertices), self.rest.shape[1])
            self.assertEqual(len(mesh.faces), len(self.faces))
            # GLB positions are float32.
            self.assertLess(float(np.abs(np.asarray(mesh.vertices) - expected).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
