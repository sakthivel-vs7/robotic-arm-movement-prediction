
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load saved predictions
results = pd.read_csv("data/predictions.csv")

# Create output folder
os.makedirs("results", exist_ok=True)

# Plot actual vs predicted X and Y coordinates
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    results["actual_x"],
    results["predicted_x"],
    alpha=0.5
)
axes[0].plot(
    [results["actual_x"].min(), results["actual_x"].max()],
    [results["actual_x"].min(), results["actual_x"].max()],
    linestyle="--"
)
axes[0].set_title("Actual vs Predicted X Position")
axes[0].set_xlabel("Actual X")
axes[0].set_ylabel("Predicted X")
axes[0].grid(True)

axes[1].scatter(
    results["actual_y"],
    results["predicted_y"],
    alpha=0.5
)
axes[1].plot(
    [results["actual_y"].min(), results["actual_y"].max()],
    [results["actual_y"].min(), results["actual_y"].max()],
    linestyle="--"
)
axes[1].set_title("Actual vs Predicted Y Position")
axes[1].set_xlabel("Actual Y")
axes[1].set_ylabel("Predicted Y")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=300)
plt.show()

print("Graph saved to results/actual_vs_predicted.png")
