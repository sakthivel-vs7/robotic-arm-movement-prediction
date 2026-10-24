# Intelligent Robotic Arm Movement Prediction Using Machine Learning

## Project Overview
This project predicts the end-effector position of a 3-link robotic arm using machine learning and demonstrates robotic arm movement through simulation.

## Features
- Synthetic dataset generation using Python
- Random Forest Regression model for position prediction
- Actual vs. predicted position visualization
- Target position planning using inverse kinematics
- Interactive Streamlit dashboard
- Target testing results and graphical visualization

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- SciPy
- Joblib

## Project Structure
- `generate_dataset.py` — Generates the robotic arm dataset
- `train_model.py` — Trains and saves the machine learning model
- `plot_results.py` — Generates the actual vs. predicted graph
- `results_export.py` — Tests target positions and exports results
- `app.py` — Runs the interactive dashboard
- `data/` — Contains generated datasets and predictions
- `models/` — Stores the trained model
- `results/` — Contains graphs and experiment results

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python generate_dataset.py
```

Train the model:

```bash
python train_model.py
```

Run the dashboard:

```bash
python -m streamlit run app.py
```

## Experiment Results
- Target positions tested: 5
- Reachable targets successfully reached within tolerance: 4
- Success rate: 80%

The target `(3, 3)` is unreachable because its distance from the origin exceeds the maximum reach of the three-link arm.

## Project Scope
This is a simulation-based educational project. It does not control a physical robotic arm.
