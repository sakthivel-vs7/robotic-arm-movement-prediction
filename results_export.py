
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

L1 = L2 = L3 = 1.0
JOINT_MIN = -90.0
JOINT_MAX = 90.0
TOLERANCE = 0.02


def forward_kinematics(angles):
    a1, a2, a3 = np.radians(angles)

    x1 = L1 * np.cos(a1)
    y1 = L1 * np.sin(a1)

    x2 = x1 + L2 * np.cos(a1 + a2)
    y2 = y1 + L2 * np.sin(a1 + a2)

    x3 = x2 + L3 * np.cos(a1 + a2 + a3)
    y3 = y2 + L3 * np.sin(a1 + a2 + a3)

    return float(x3), float(y3)


def plan_target(tx, ty, initial_angles):
    def objective(angles):
        x, y = forward_kinematics(angles)
        return [x - tx, y - ty]

    starts = [
        initial_angles,
        [0, 0, 0],
        [45, -45, 45],
        [-45, 45, -45],
        [60, 60, -60],
        [-60, -60, 60],
        [90, -90, 90],
        [-90, 90, -90]
    ]

    best_angles = None
    best_error = float("inf")

    for start in starts:
        result = least_squares(
            objective,
            x0=np.clip(
                np.array(start, dtype=float),
                JOINT_MIN,
                JOINT_MAX
            ),
            bounds=(
                [JOINT_MIN] * 3,
                [JOINT_MAX] * 3
            ),
            max_nfev=2000
        )

        px, py = forward_kinematics(result.x)
        error = float(np.hypot(px - tx, py - ty))

        if error < best_error:
            best_error = error
            best_angles = result.x.copy()

    px, py = forward_kinematics(best_angles)
    return best_angles, px, py, best_error


targets = [
    (2.0, 1.0),
    (1.0, 1.0),
    (2.5, 0.5),
    (0.0, 2.0),
    (3.0, 3.0)
]

initial_angles = np.array([30.0, 20.0, -15.0])
rows = []

for tx, ty in targets:
    angles, px, py, error = plan_target(
        tx, ty, initial_angles
    )

    rows.append({
        "target_x": tx,
        "target_y": ty,
        "planned_theta1": angles[0],
        "planned_theta2": angles[1],
        "planned_theta3": angles[2],
        "planned_x": px,
        "planned_y": py,
        "position_error": error,
        "within_tolerance": error <= TOLERANCE
    })

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(rows)
df.to_csv("results/target_experiments.csv", index=False)

print("Experiment tests completed!")
print(df.to_string(index=False))
print("\nSaved to results/target_experiments.csv")
print(
    "\nTargets within tolerance:",
    int(df["within_tolerance"].sum()),
    "out of",
    len(df)
)
