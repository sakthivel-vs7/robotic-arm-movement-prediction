
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/robot_movement.csv")

# Input features: robotic arm joint angles
X = data[["theta1", "theta2", "theta3"]]

# Output labels: end-effector position
y = data[["end_x", "end_y"]]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training the model...")
model.fit(X_train, y_train)

# Predict end-effector positions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/robot_arm_model.pkl")

# Save actual and predicted positions for plotting
results = pd.DataFrame({
    "actual_x": y_test["end_x"].to_numpy(),
    "actual_y": y_test["end_y"].to_numpy(),
    "predicted_x": predictions[:, 0],
    "predicted_y": predictions[:, 1]
})

results.to_csv("data/predictions.csv", index=False)

print("\nModel training completed!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2 Score: {r2:.6f}")
print("\nModel saved to models/robot_arm_model.pkl")
print("Predictions saved to data/predictions.csv")
