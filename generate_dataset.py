
import os
import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Create output folder
os.makedirs("data", exist_ok=True)

# Generate simulated robotic arm joint angles
num_samples = 5000

theta1 = np.random.uniform(-90, 90, num_samples)
theta2 = np.random.uniform(-90, 90, num_samples)
theta3 = np.random.uniform(-90, 90, num_samples)

# Length of each arm segment
L1, L2, L3 = 1.0, 1.0, 1.0

# Convert degrees to radians
t1 = np.radians(theta1)
t2 = np.radians(theta2)
t3 = np.radians(theta3)

# Calculate end-effector position using forward kinematics
x = (
    L1 * np.cos(t1)
    + L2 * np.cos(t1 + t2)
    + L3 * np.cos(t1 + t2 + t3)
)

y = (
    L1 * np.sin(t1)
    + L2 * np.sin(t1 + t2)
    + L3 * np.sin(t1 + t2 + t3)
)

# Create dataset
df = pd.DataFrame({
    "theta1": theta1,
    "theta2": theta2,
    "theta3": theta3,
    "end_x": x,
    "end_y": y
})

# Save dataset
output_path = "data/robot_movement.csv"
df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"Saved to: {output_path}")
print("\nFirst five rows:")
print(df.head())
