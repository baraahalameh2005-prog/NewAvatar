import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from app.config import apply_environment, settings

# PyTorch and Anny read their cache folders when they are imported.
apply_environment(settings)

import torch
import trimesh
import anny


DT = torch.float64

# Default skin colour of the exported avatar (sRGB hex).
DEFAULT_SKIN_TONE = "#c8997e"

_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_skin_tone(skin_tone):
    """Convert "#RRGGBB" (or "RRGGBB") to an RGBA list of 0-255 ints."""

    match = _HEX_COLOR.match(skin_tone.strip()) if isinstance(skin_tone, str) else None

    if match is None:
        raise ValueError(
            f"Invalid skin tone {skin_tone!r}; expected a hex colour like '#c8997e'."
        )

    digits = match.group(1)

    return [int(digits[i:i + 2], 16) for i in (0, 2, 4)] + [255]


def srgb_to_linear(channels):
    """sRGB channel values (0-255) to linear values (0.0-1.0), IEC 61966-2-1."""

    c = np.asarray(channels, dtype=float) / 255.0

    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def skin_base_color(skin_tone):
    """glTF baseColorFactor (linear RGBA, 0.0-1.0) for an sRGB hex skin tone."""

    rgba = parse_skin_tone(skin_tone)

    return [float(v) for v in srgb_to_linear(rgba[:3])] + [1.0]


def make_phenotype(
    model,
    gender,
    height,
    weight,
    age,
    muscle,
    proportions,
):
    phenotype = {
        key: torch.full(
            (1,),
            0.5,
            dtype=DT,
        )
        for key in model.phenotype_labels
    }

    phenotype["gender"] = torch.full(
        (1,),
        gender,
        dtype=DT,
    )

    phenotype["height"] = torch.full(
        (1,),
        height,
        dtype=DT,
    )

    phenotype["weight"] = torch.full(
        (1,),
        weight,
        dtype=DT,
    )

    phenotype["age"] = torch.full(
        (1,),
        age,
        dtype=DT,
    )

    phenotype["muscle"] = torch.full(
        (1,),
        muscle,
        dtype=DT,
    )

    phenotype["proportions"] = torch.full(
        (1,),
        proportions,
        dtype=DT,
    )

    return phenotype


def build(
    model,
    gender,
    height,
    weight,
    age,
    muscle,
    proportions,
):
    phenotype = make_phenotype(
        model=model,
        gender=gender,
        height=height,
        weight=weight,
        age=age,
        muscle=muscle,
        proportions=proportions,
    )

    output = model(
        phenotype_kwargs=phenotype
    )

    return output


def export_avatar(
    output,
    model,
    path,
    skin_tone=DEFAULT_SKIN_TONE,
):
    """
    Write the avatar as a GLB with smooth normals and a skin material.

    The file is written to a temporary name first and then swapped in,
    so a viewer never loads a half-written GLB.
    """

    vertices = (
        output["rest_vertices"][0]
        .detach()
        .cpu()
        .numpy()
    )

    faces = (
        model.faces
        .detach()
        .cpu()
        .numpy()
    )

    # Convert Anny's coordinate system
    # to the coordinate system used by the viewer.
    vertices = np.stack(
        [
            vertices[:, 0],
            vertices[:, 2],
            -vertices[:, 1],
        ],
        axis=1,
    )

    # Put the feet on the ground.
    vertices[:, 1] -= vertices[:, 1].min()

    material = trimesh.visual.material.PBRMaterial(
        name="skin",
        # glTF stores colour factors as linear values; skin_tone is sRGB.
        baseColorFactor=skin_base_color(skin_tone),
        metallicFactor=0.0,
        roughnessFactor=0.65,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(material=material),
        process=False,
    )

    # Smooth shading: area-weighted vertex normals.
    mesh.vertex_normals

    path = Path(path)
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")

    try:
        mesh.export(
            temp_path,
            file_type="glb",
            include_normals=True,
        )

        _replace_file(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _replace_file(source, destination, attempts=10):
    # On Windows the target can be briefly locked while it is being
    # served; retry for a short while before giving up.
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise

            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female",
    )

    parser.add_argument(
        "--height-param",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight-param",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--age",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--muscle",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--proportions",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--out",
        default="avatar.glb",
    )

    args = parser.parse_args()

    print("Loading Anny...")

    model = anny.Anny()

    gender = (
        1.0
        if args.gender == "female"
        else 0.0
    )

    print("Building personalized avatar...")

    output = build(
        model=model,
        gender=gender,
        height=args.height_param,
        weight=args.weight_param,
        age=args.age,
        muscle=args.muscle,
        proportions=args.proportions,
    )

    print("Exporting avatar...")

    export_avatar(
        output=output,
        model=model,
        path=args.out,
    )

    result = {
        "gender": args.gender,
        "height_param": round(
            args.height_param,
            4,
        ),
        "weight_param": round(
            args.weight_param,
            4,
        ),
        "age": round(
            args.age,
            4,
        ),
        "muscle": round(
            args.muscle,
            4,
        ),
        "proportions": round(
            args.proportions,
            4,
        ),
        "output": args.out,
    }

    print()
    print("AVATAR GENERATED")
    print("----------------")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()