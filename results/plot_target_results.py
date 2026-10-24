
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results
df = pd.read_csv("results/target_experiments.csv")

# Create output folder
os.makedirs("results", exist_ok=True)

# Separate successful and failed targets
passed = df[df["within_tolerance"] == True]
failed = df[df["within_tolerance"] == False]

# Plot target positions
plt.figure(figsize=(9, 6))

plt.scatter(
    passed["target_x"],
    passed["target_y"],
    color="green",
    s=100,
    label="Reachable targets"
)

plt.scatter(
    failed["target_x"],
    failed["target_y"],
    color="red",
    s=100,
    marker="X",
    label="Unreached targets"
)

# Label each target
for _, row in df.iterrows():
    plt.annotate(
        f'({row["target_x"]}, {row["target_y"]})',
        (row["target_x"], row["target_y"]),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Target X Position")
plt.ylabel("Target Y Position")
plt.title("Robotic Arm Target Position Testing")
plt.axhline(0)
plt.axvline(0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# Save graph
output_path = "results/target_testing_graph.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Target testing graph created successfully!")
print(f"Saved to: {output_path}")
