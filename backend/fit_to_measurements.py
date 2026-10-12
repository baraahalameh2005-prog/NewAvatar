import argparse
import json
import math
import time

import numpy as np

from app.config import apply_environment, settings

# PyTorch and Anny read their cache folders when they are imported.
apply_environment(settings)

import torch
import anny


DT = torch.float64

PARAM_MIN = 0.001
PARAM_MAX = 0.999

# Anny's age anchors (anny/models/phenotype.py) are
# baby 0.0, child 1/3, young adult 2/3 and old 1.0.
# Anny's default age, 0.5, lies halfway between child and young adult.
# The fit never goes below that default, so the circumferences are
# not matched by blending in child body proportions.
AGE_MIN = 0.5
AGE_MAX = PARAM_MAX

# Inner height / weight solves stop at these tolerances.
HEIGHT_TOLERANCE_M = 1e-4
WEIGHT_TOLERANCE_KG = 1e-2

# The shape fit stops early once every target (circumferences,
# height, weight) is within this relative error.
GOOD_ENOUGH_RELATIVE_ERROR = 0.002

# Finite-difference step for the shape parameters. A step this size
# keeps the Jacobian well above the noise left by the inner solves.
DIFF_STEP = 1e-2

GENDER_VALUES = {
    "male": 0.0,
    "female": 1.0,
}

# requested name -> (result key, scale to the requested unit, unit)
TARGET_MEASUREMENTS = {
    "height": ("height_m", 100.0, "cm"),
    "weight": ("mass_kg", 1.0, "kg"),
    "waist": ("waist_m", 100.0, "cm"),
    "chest": ("chest_m", 100.0, "cm"),
    "hip": ("hip_m", 100.0, "cm"),
}

# How each requested measurement is reached by the fit.
TARGET_CONTROL = {
    "height": "solved directly (height parameter), shape assists if out of range",
    "weight": "solved directly (weight parameter), shape assists if out of range",
    "waist": "fitted indirectly (age, muscle, proportions)",
    "chest": "fitted indirectly (age, muscle, proportions)",
    "hip": "fitted indirectly (age, muscle, proportions)",
}


def normalize_gender(gender):
    """Return "male" or "female"; raise ValueError for anything else."""

    if isinstance(gender, str):
        value = gender.strip().lower()

        if value in GENDER_VALUES:
            return value

    raise ValueError(
        f"Invalid gender {gender!r}; expected 'male' or 'female'."
    )


def gender_value(gender):
    return GENDER_VALUES[normalize_gender(gender)]


def make_phenotype(
    model,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = {
        k: torch.full(
            (1,),
            0.5,
            dtype=DT,
        )
        for k in model.phenotype_labels
    }

    ph["gender"] = torch.full(
        (1,),
        gender,
        dtype=DT,
    )

    ph["age"] = torch.full(
        (1,),
        age,
        dtype=DT,
    )

    ph["height"] = torch.full(
        (1,),
        height,
        dtype=DT,
    )

    ph["weight"] = torch.full(
        (1,),
        weight,
        dtype=DT,
    )

    ph["muscle"] = torch.full(
        (1,),
        muscle,
        dtype=DT,
    )

    ph["proportions"] = torch.full(
        (1,),
        proportions,
        dtype=DT,
    )

    return ph


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate(
    model,
    measurements,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = make_phenotype(
        model,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=ph
    )

    result = measurements.measure(
        output
    )

    return output, result


# ============================================================
# LIGHT EVALUATION
# ============================================================

def evaluate_anthropometry(
    model,
    measurements,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = make_phenotype(
        model,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=ph
    )

    base = measurements.anthropometry(
        output["rest_vertices"]
    )

    height_m = float(
        base["height"].item()
    )

    mass_kg = float(
        base["mass"].item()
    )

    return output, height_m, mass_kg


# ============================================================
# MONOTONIC 1-D SOLVER
# ============================================================

def solve_monotonic(
    func,
    target,
    start,
    tolerance,
    lo=PARAM_MIN,
    hi=PARAM_MAX,
    max_evaluations=12,
):
    """
    Find x in [lo, hi] with func(x) == target for increasing func.

    func(x) returns (value, payload). Uses secant steps from a warm
    start, kept inside a shrinking bracket, with bisection as the
    fallback. Returns (x, value, payload, evaluations) for the best
    point seen.
    """

    a, b = lo, hi
    best = None
    history = []

    def run(x):
        nonlocal a, b, best

        value, payload = func(x)
        history.append((x, value))

        if value < target:
            a = max(a, x)
        else:
            b = min(b, x)

        if best is None or abs(value - target) < abs(best[1] - target):
            best = (x, value, payload)

        return value

    x = min(max(start, lo), hi)
    value = run(x)

    while (
        abs(value - target) > tolerance
        and len(history) < max_evaluations
        and b - a > 1e-9
    ):
        if len(history) == 1:
            step = 0.02 if value < target else -0.02
            x_next = x + step
        else:
            (x0, y0), (x1, y1) = history[-2], history[-1]

            if y1 != y0:
                x_next = x1 + (target - y1) * (x1 - x0) / (y1 - y0)
            else:
                x_next = (a + b) / 2.0

        if not (a < x_next < b):
            tried = {x for x, _ in history}

            if x_next >= b and b >= hi and hi not in tried:
                # Target may lie beyond the upper bound: check it once
                # instead of creeping towards it.
                x_next = hi
            elif x_next <= a and a <= lo and lo not in tried:
                x_next = lo
            else:
                x_next = (a + b) / 2.0

        x = x_next
        value = run(x)

    return best[0], best[1], best[2], len(history)


# ============================================================
# SOLVE HEIGHT
# ============================================================

def solve_height(
    model,
    measurements,
    gender,
    age,
    weight,
    muscle,
    proportions,
    target_height,
    start=0.5,
    details=False,
):
    def func(height):
        output, current_height, current_mass = evaluate_anthropometry(
            model,
            measurements,
            gender,
            age,
            height,
            weight,
            muscle,
            proportions,
        )

        return current_height, (output, current_height, current_mass)

    solved = solve_monotonic(
        func,
        target_height,
        start,
        HEIGHT_TOLERANCE_M,
    )

    return solved if details else solved[0]


# ============================================================
# SOLVE WEIGHT
# ============================================================

def solve_weight(
    model,
    measurements,
    gender,
    age,
    height,
    muscle,
    proportions,
    target_weight,
    start=0.5,
    details=False,
):
    def func(weight):
        output, current_height, current_mass = evaluate_anthropometry(
            model,
            measurements,
            gender,
            age,
            height,
            weight,
            muscle,
            proportions,
        )

        return current_mass, (output, current_height, current_mass)

    solved = solve_monotonic(
        func,
        target_weight,
        start,
        WEIGHT_TOLERANCE_KG,
    )

    return solved if details else solved[0]


# ============================================================
# EVALUATE BODY SHAPE
# ============================================================

def evaluate_shape(
    model,
    measurements,
    gender,
    target_height,
    target_weight,
    age,
    muscle,
    proportions,
    start=None,
    stats=None,
):
    """
    Solve height and weight for the given shape, then measure the body.

    start is an optional (height, weight) warm start, usually the
    solution for the previous, nearby shape. stats, if given, counts
    the Anny evaluations used.
    """

    height, weight = start if start is not None else (0.5, 0.5)

    # Step 1: solve height
    height, _, _, n_height = solve_height(
        model,
        measurements,
        gender,
        age,
        weight,
        muscle,
        proportions,
        target_height,
        start=height,
        details=True,
    )

    # Step 2: solve weight
    weight, _, payload, n_weight = solve_weight(
        model,
        measurements,
        gender,
        age,
        height,
        muscle,
        proportions,
        target_weight,
        start=weight,
        details=True,
    )

    output, current_height, _ = payload
    n_height_again = 0

    # Step 3: weight changes height a little; solve height again
    # only if it moved outside the tolerance.
    if abs(current_height - target_height) > HEIGHT_TOLERANCE_M:
        height, _, payload, n_height_again = solve_height(
            model,
            measurements,
            gender,
            age,
            weight,
            muscle,
            proportions,
            target_height,
            start=height,
            details=True,
        )

        output = payload[0]

    if stats is not None:
        stats["anny_evaluations"] += (
            n_height + n_weight + n_height_again
        )

    # Step 4: measure the body of the last evaluation, which already
    # uses the solved height and weight.
    result = measurements.measure(
        output
    )

    return output, result, height, weight


# ============================================================
# ACCURACY
# ============================================================

def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentage_error(requested, actual):
    """
    abs(actual - requested) / abs(requested) * 100.

    Returns None when either value is missing or not finite, or the
    requested value is zero.
    """

    if not (_finite(requested) and _finite(actual)):
        return None

    requested = float(requested)
    actual = float(actual)

    if requested == 0.0:
        return None

    return abs(actual - requested) / abs(requested) * 100.0


def accuracy_report(targets, result):
    """
    Compare requested measurements with the measured body.

    targets maps names from TARGET_MEASUREMENTS to requested values
    in their user units (cm or kg). A measurement the pipeline could
    not produce is reported as "measurement_failed", never dropped.
    """

    report = {}

    for name, requested in targets.items():
        key, scale, unit = TARGET_MEASUREMENTS[name]

        raw = result.get(key) if result else None

        entry = {
            "unit": unit,
            "requested": (
                float(requested)
                if _finite(requested)
                else None
            ),
            "actual": None,
            "absolute_error": None,
            "percentage_error": None,
            "control": TARGET_CONTROL[name],
            "status": "measurement_failed",
        }

        if _finite(raw):
            actual = float(raw) * scale

            entry["actual"] = actual
            entry["status"] = "measured"

            if entry["requested"] is not None:
                entry["absolute_error"] = abs(
                    actual - entry["requested"]
                )

                entry["percentage_error"] = percentage_error(
                    entry["requested"],
                    actual,
                )

        report[name] = entry

    return report


# ============================================================
# FIT BODY SHAPE
# ============================================================

class _GoodEnough(Exception):
    pass


def fit(
    model,
    measurements,
    gender,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
    stats=None,
):
    try:
        from scipy.optimize import least_squares
    except ImportError:
        raise RuntimeError(
            "scipy is required. Install it with: "
            "python -m pip install scipy"
        )

    if stats is None:
        stats = {}

    stats.setdefault("anny_evaluations", 0)
    stats.setdefault("shape_evaluations", 0)

    target_height = height_cm / 100.0
    target_weight = weight_kg

    # Height and weight are solved inside every shape evaluation and
    # their residuals are normally ~0. They stay in the objective so
    # that, when a target is out of reach of the height or weight
    # parameter alone, the shape parameters help close the gap
    # instead of the miss going unnoticed.
    targets = np.array(
        [
            waist_cm / 100.0,
            chest_cm / 100.0,
            hip_cm / 100.0,
            target_height,
            target_weight,
        ],
        dtype=np.float64,
    )

    # age, muscle, proportions
    x0 = np.array(
        [
            AGE_MIN,
            0.5,
            0.5,
        ],
        dtype=np.float64,
    )

    lower = np.array(
        [
            AGE_MIN,
            PARAM_MIN,
            PARAM_MIN,
        ],
        dtype=np.float64,
    )

    upper = np.array(
        [
            AGE_MAX,
            PARAM_MAX,
            PARAM_MAX,
        ],
        dtype=np.float64,
    )

    state = {
        "start": (0.5, 0.5),
        "cache": {},
        "best": None,
    }

    def run(params):
        key = tuple(float(x) for x in params)

        if key in state["cache"]:
            return state["cache"][key]

        stats["shape_evaluations"] += 1

        output, result, height, weight = evaluate_shape(
            model,
            measurements,
            gender,
            target_height,
            target_weight,
            params[0],
            params[1],
            params[2],
            start=state["start"],
            stats=stats,
        )

        state["start"] = (height, weight)

        values = [
            result.get("waist_m"),
            result.get("chest_m"),
            result.get("hip_m"),
            result.get("height_m"),
            result.get("mass_kg"),
        ]

        entry = {
            "params": np.array(params, dtype=np.float64),
            "output": output,
            "result": result,
            "height": height,
            "weight": weight,
            "residual": None,
        }

        if all(_finite(v) for v in values):
            entry["residual"] = (
                np.array(values, dtype=np.float64) - targets
            ) / targets

            best = state["best"]

            if best is None or (
                np.linalg.norm(entry["residual"])
                < np.linalg.norm(best["residual"])
            ):
                state["best"] = entry

        state["cache"][key] = entry

        return entry

    def objective(params):
        entry = run(params)
        n = stats["shape_evaluations"]

        if entry["residual"] is None:
            print(
                f"  evaluation {n:>3}"
                " | invalid measurement"
                f" | age {params[0]:.3f}"
                f" | muscle {params[1]:.3f}"
                f" | proportions {params[2]:.3f}"
            )

            # Large residual steers the optimizer away; this point is
            # never chosen as the result.
            return np.full(
                len(targets),
                10.0,
                dtype=np.float64,
            )

        r = entry["residual"]

        if n % 5 == 0:
            print(
                f"  evaluation {n:>3}"
                f" | error {np.linalg.norm(r):.5f}"
                f" | age {params[0]:.3f}"
                f" | muscle {params[1]:.3f}"
                f" | proportions {params[2]:.3f}"
            )

        if np.max(np.abs(r)) <= GOOD_ENOUGH_RELATIVE_ERROR:
            raise _GoodEnough()

        return r

    print("Optimizing body shape...")

    stats["stop_reason"] = "optimizer converged"

    try:
        least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            method="trf",
            diff_step=DIFF_STEP,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-6,
            max_nfev=30,
            verbose=0,
        )
    except _GoodEnough:
        stats["stop_reason"] = "all targets within tolerance"

    # Use the best measured point instead of evaluating the
    # optimizer's final point again.
    best = state["best"]

    if best is None:
        stats["stop_reason"] = "no valid measurement found"
        best = run(x0)

    age, muscle, proportions = best["params"]
    height = best["height"]
    weight = best["weight"]

    print()
    print("Solved parameters:")

    print(
        f"  Height      : {height:.4f}"
    )

    print(
        f"  Weight      : {weight:.4f}"
    )

    print(
        f"  Age         : {age:.4f}"
    )

    print(
        f"  Muscle      : {muscle:.4f}"
    )

    print(
        f"  Proportions : {proportions:.4f}"
    )

    print(
        f"  Shape evaluations : {stats['shape_evaluations']}"
        f" | Anny evaluations : {stats['anny_evaluations']}"
        f" | {stats['stop_reason']}"
    )

    return best["output"], best["result"], np.array(
        [
            height,
            weight,
            age,
            muscle,
            proportions,
        ],
        dtype=np.float64,
    )


# ============================================================
# FASTAPI REUSABLE FUNCTION
# ============================================================

def fit_body(
    model,
    measurements,
    gender,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
):
    """
    Reusable function for FastAPI.

    Takes the user's body measurements and returns
    the fitted Anny parameters and the generated body data,
    plus an accuracy report comparing the measured body with
    the requested values.
    """

    gender = normalize_gender(gender)

    # Start measuring the fitting time.
    start_time = time.perf_counter()

    stats = {}

    output, result, params = fit(
        model=model,
        measurements=measurements,
        gender=GENDER_VALUES[gender],
        height_cm=height_cm,
        weight_kg=weight_kg,
        waist_cm=waist_cm,
        chest_cm=chest_cm,
        hip_cm=hip_cm,
        stats=stats,
    )

    # Calculate total fitting time.
    elapsed = time.perf_counter() - start_time

    print()

    print(
        f"Fitting time: {elapsed:.2f} seconds"
    )

    print(
        f"Fitting time: {elapsed / 60:.2f} minutes"
    )

    accuracy = accuracy_report(
        {
            "height": height_cm,
            "weight": weight_kg,
            "waist": waist_cm,
            "chest": chest_cm,
            "hip": hip_cm,
        },
        result,
    )

    return {
        "output": output,
        "result": result,
        "accuracy": accuracy,
        "fit_info": {
            "seconds": elapsed,
            **stats,
        },
        "parameters": {
            "gender": gender,
            "height": float(params[0]),
            "weight": float(params[1]),
            "age": float(params[2]),
            "muscle": float(params[3]),
            "proportions": float(params[4]),
        },
    }


# ============================================================
# COMMAND LINE
# ============================================================

def print_accuracy(accuracy):
    print(
        f"{'name':<7} {'requested':>10} {'actual':>10}"
        f" {'abs_err':>9} {'pct_err':>8}  status"
    )

    for name, entry in accuracy.items():
        if entry["actual"] is None:
            print(
                f"{name:<7} {entry['requested']:>10.2f}"
                f" {'-':>10} {'-':>9} {'-':>8}  {entry['status']}"
            )
            continue

        print(
            f"{name:<7} {entry['requested']:>10.2f}"
            f" {entry['actual']:>10.2f}"
            f" {entry['absolute_error']:>9.2f}"
            f" {entry['percentage_error']:>7.2f}%"
            f"  {entry['unit']}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female",
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--waist",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--chest",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--hip",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    print("Loading Anny...")

    model = anny.Anny()

    from measurements import BodyMeasurements

    measurements = BodyMeasurements(model)

    print()
    print("FITTING BODY SHAPE")
    print("------------------")
    print()

    print("Target:")

    print(
        f"  Height : {args.height:.2f} cm"
    )

    print(
        f"  Weight : {args.weight:.2f} kg"
    )

    print(
        f"  Waist  : {args.waist:.2f} cm"
    )

    print(
        f"  Chest  : {args.chest:.2f} cm"
    )

    print(
        f"  Hip    : {args.hip:.2f} cm"
    )

    print()

    fitted = fit_body(
        model=model,
        measurements=measurements,
        gender=args.gender,
        height_cm=args.height,
        weight_kg=args.weight,
        waist_cm=args.waist,
        chest_cm=args.chest,
        hip_cm=args.hip,
    )

    result = fitted["result"]
    params = fitted["parameters"]

    print()
    print("ACCURACY")
    print("--------")

    print_accuracy(fitted["accuracy"])

    print()
    print("RESULT")
    print("------")

    clean = {
        key: (
            round(float(value), 4)
            if value is not None
            else None
        )
        for key, value in result.items()
        if key != "hip_z"
    }

    print(
        json.dumps(
            clean,
            indent=2,
        )
    )

    print()
    print("PARAMETERS")
    print("----------")

    print(
        json.dumps(
            {
                "height": round(
                    params["height"],
                    4,
                ),
                "weight": round(
                    params["weight"],
                    4,
                ),
                "age": round(
                    params["age"],
                    4,
                ),
                "muscle": round(
                    params["muscle"],
                    4,
                ),
                "proportions": round(
                    params["proportions"],
                    4,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
