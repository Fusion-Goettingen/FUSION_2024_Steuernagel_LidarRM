"""
Contains constants regarding the experimental evaluation.
"""
import numpy as np

# Global seed used for experiments
EXPERIMENT_SEED = 42

# Number of Monte Carlo runs to run experiments for
N_MONTE_CARLO_RUNS = 50

# Number of Monte Carlo runs for the runtime evaluation
N_RUNS_RUNTIME = 25

# Maximum length (number of steps) of individual trajectories
TRAJECTORY_LENGTH = 50

# Names of trackers to grab for the experiments
TRACKER_NAME_LIST = [
    "Feldmann",
    "TruncatedRM",
    "AlqaderiEKF",
    "GM-MEMEKF*",
    "Li",
    "LiAdapted",
    "Ours",
]

# Only visualize every n-th step
VISUALIZE_EVERY_N_STEPS = 5

# Sensor rate, should be fixed at 10
_HZ = 10

# Process noise covariance matrix, dims: [x, y, vel_x, vel_y, theta, length, width]
PROCESS_NOISE_COVARIANCE_MATRIX = np.diag([0, 0, 0.75 / _HZ, 3 / _HZ, 0, 0, 0])

# Sample initial state uniformly from the following intervals:
INITIAL_RANGES = np.array([
    [-80, -30],
    [-80, -30],
    [5 / _HZ, 7 / _HZ],
    [-3 / _HZ, 3 / _HZ],
    [0, 2 * np.pi],
    [2.5, 6],
    [1.4, 2.4]
])

# Optional fixed initial pose used in some experiments and visualizations
FIX_INITIAL_POSE = np.array([
    [10, 0, 5.5, 5.5, np.pi / 4, 4.7, 1.8]
]).reshape((7,))
