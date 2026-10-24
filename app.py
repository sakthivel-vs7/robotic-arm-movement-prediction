
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

st.set_page_config(
    page_title="Intelligent Robotic Arm Simulator",
    page_icon="🤖",
    layout="wide"
)

# Arm configuration
L1 = L2 = L3 = 1.0
JOINT_MIN = -90.0
JOINT_MAX = 90.0
TOLERANCE = 0.02


@st.cache_resource
def load_model():
    return joblib.load("models/robot_arm_model.pkl")


model = load_model()


def forward_kinematics(angles):
    """Calculate the arm joints and endpoint from angles."""
    a1, a2, a3 = np.radians(angles)

    x1 = L1 * np.cos(a1)
    y1 = L1 * np.sin(a1)

    x2 = x1 + L2 * np.cos(a1 + a2)
    y2 = y1 + L2 * np.sin(a1 + a2)

    x3 = x2 + L3 * np.cos(a1 + a2 + a3)
    y3 = y2 + L3 * np.sin(a1 + a2 + a3)

    xs = [0, x1, x2, x3]
    ys = [0, y1, y2, y3]

    return xs, ys


def solve_target(target_x, target_y, initial_angles):
    """Find a target pose while respecting joint limits."""

    def objective(angles):
        xs, ys = forward_kinematics(angles)
        return [xs[-1] - target_x, ys[-1] - target_y]

    solution = least_squares(
        objective,
        x0=np.clip(
            np.array(initial_angles, dtype=float),
            JOINT_MIN,
            JOINT_MAX
        ),
        bounds=(
            [JOINT_MIN] * 3,
            [JOINT_MAX] * 3
        ),
        max_nfev=2000
    )

    angles = solution.x
    xs, ys = forward_kinematics(angles)

    error = float(np.hypot(
        xs[-1] - target_x,
        ys[-1] - target_y
    ))

    return angles, xs, ys, error


# ---------------- Dashboard ----------------

st.title("🤖 Intelligent Robotic Arm Movement Prediction")
st.caption(
    "Machine Learning | Inverse Kinematics | Joint-Limited 2D Simulation"
)

st.sidebar.header("Current Joint Angles")

theta1 = st.sidebar.slider("Joint 1 (°)", -90, 90, 30)
theta2 = st.sidebar.slider("Joint 2 (°)", -90, 90, 20)
theta3 = st.sidebar.slider("Joint 3 (°)", -90, 90, -15)

current_angles = [theta1, theta2, theta3]
current_xs, current_ys = forward_kinematics(current_angles)
current_x, current_y = current_xs[-1], current_ys[-1]

st.sidebar.divider()
st.sidebar.header("Target Position")

target_x = st.sidebar.slider("Target X", -3.0, 3.0, 1.5, 0.1)
target_y = st.sidebar.slider("Target Y", -3.0, 3.0, 1.0, 0.1)

# ML prediction for current joint angles
prediction = model.predict(np.array([current_angles]))[0]
pred_x, pred_y = float(prediction[0]), float(prediction[1])

ml_error = float(np.hypot(
    pred_x - current_x,
    pred_y - current_y
))

# Find a joint-limited target pose
planned_angles, planned_xs, planned_ys, target_error = solve_target(
    target_x, target_y, current_angles
)

reachable = target_error <= TOLERANCE

st.subheader("Live Position Analysis")

c1, c2, c3 = st.columns(3)
c1.metric("Current X", f"{current_x:.3f}")
c2.metric("Current Y", f"{current_y:.3f}")
c3.metric("ML Prediction Error", f"{ml_error:.4f} units")

st.subheader("Target Planning")

c1, c2, c3 = st.columns(3)
c1.metric("Target X", f"{target_x:.2f}")
c2.metric("Target Y", f"{target_y:.2f}")
c3.metric("Target Position Error", f"{target_error:.4f} units")

if reachable:
    st.success("Target reached within the simulation tolerance.")
else:
    st.warning(
        "Target could not be reached within 0.02 units "
        "using the configured joint limits. Try another target."
    )

# Movement progress is a visual interpolation between poses.
# It is not a physical trajectory or collision-checked motion plan.
progress = st.slider(
    "Movement Preview Progress (%)",
    min_value=0,
    max_value=100,
    value=100,
    step=5
)

fraction = progress / 100.0

preview_angles = (
    np.array(current_angles)
    + fraction * (planned_angles - np.array(current_angles))
)

preview_xs, preview_ys = forward_kinematics(preview_angles)
preview_x, preview_y = preview_xs[-1], preview_ys[-1]

left, right = st.columns([3, 2])

with left:
    st.subheader("Robotic Arm Movement Preview")

    fig, ax = plt.subplots(figsize=(7, 6))

    # Current pose
    ax.plot(
        current_xs, current_ys,
        marker="o", linewidth=3,
        label="Current arm"
    )

    # Interpolated preview pose
    ax.plot(
        preview_xs, preview_ys,
        marker="o", linewidth=3,
        linestyle="--",
        label=f"Preview pose ({progress}%)"
    )

    # Requested target
    ax.scatter(
        target_x, target_y,
        marker="*", s=220,
        label="Requested target"
    )

    # Planned endpoint
    ax.scatter(
        planned_xs[-1], planned_ys[-1],
        marker="x", s=120, linewidth=3,
        label="Planned endpoint"
    )

    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    ax.legend(fontsize=8)

    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Joint Angle Comparison")

    angles_table = pd.DataFrame({
        "Joint": ["Joint 1", "Joint 2", "Joint 3"],
        "Current (°)": current_angles,
        "Planned (°)": planned_angles
    })

    st.dataframe(
        angles_table.style.format({
            "Current (°)": "{:.2f}",
            "Planned (°)": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Position Comparison")

    position_table = pd.DataFrame({
        "Coordinate": ["X", "Y"],
        "Target": [target_x, target_y],
        "Planned": [planned_xs[-1], planned_ys[-1]],
        "Preview": [preview_x, preview_y]
    })

    st.dataframe(
        position_table.style.format({
            "Target": "{:.3f}",
            "Planned": "{:.3f}",
            "Preview": "{:.3f}"
        }),
        hide_index=True,
        use_container_width=True
    )

st.divider()
st.subheader("System Information")

st.write(f"- Joint limits: {JOINT_MIN:.0f}° to {JOINT_MAX:.0f}°")
st.write(f"- Link lengths: {L1}, {L2}, {L3} units")
st.write(f"- Maximum theoretical reach: {L1 + L2 + L3:.1f} units")
st.write(f"- Target tolerance: {TOLERANCE:.2f} units")

st.info(
    "The Random Forest model predicts the current endpoint from synthetic "
    "data. Numerical inverse kinematics calculates the target pose. "
    "The progress slider previews interpolation between poses; it does not "
    "simulate elapsed time, velocity, acceleration, or collision avoidance. "
    "This software does not control a physical robot."
)
