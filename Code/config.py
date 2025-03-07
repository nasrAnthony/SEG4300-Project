import torch

### 🏗 General Project Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
SEED = 42  # Set seed for reproducibility

### 📂 File Paths
DATA_PATH = "data/statcast_data.csv"
MODEL_SAVE_DIR = "saved_models/"
LOGS_DIR = "logs/"

### 🎯 Hyperparameters
# LSTM Model Hyperparameters (Pitch Sequence Model)
LSTM_INPUT_DIM = 10
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

# MLP Model Hyperparameters (Pitch Count Model)
MLP_HIDDEN_DIM = 64

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
LOSS_FUNCTION = "MSE"  # Loss function for regression
WEIGHT_DECAY = 1e-5  # Regularization

### 🧩 Categorical Mappings
PITCH_TYPES = {
    0: "Fastball",
    1: "Curveball",
    2: "Slider",
    3: "Changeup",
    4: "Cutter",
    5: "Splitter",
    6: "Knuckleball"
}

OUTCOME_TYPES = {
    0: "Ball",
    1: "Strike",
    2: "Contact"
}

### ⚾ Game Simulation Settings
MAX_PITCH_COUNT = 120  # Max pitches before pitcher is pulled
INNINGS = 9  # Standard MLB game length
BULLPEN_USAGE_FACTOR = 1.0  # Adjusts likelihood of replacing a pitcher

### 📊 Logging & Debugging
LOG_INTERVAL = 10  # How often to log training progress
SAVE_MODEL_EVERY = 5  # Save model every X epochs
DEBUG_MODE = False  # Enable/disable debugging prints

### 📌 Data Processing
FEATURE_COLUMNS = [
    "pitch_type", "release_speed", "release_spin_rate", "effective_speed", 
    "plate_x", "plate_z", "inning", "outs_when_up", "balls", "strikes", 
    "batter", "pitcher", "vx0", "vy0", "vz0", "ax", "ay", "az"
]

CATEGORICAL_FEATURES = ["pitcher", "batter", "inning_topbot", "stand", "p_throws"]
NUMERICAL_FEATURES = ["release_speed", "release_spin_rate", "plate_x", "plate_z", "effective_speed"]

SCALING_METHOD = "standard"  # Choose between "minmax" or "standard"
MISSING_VALUE_STRATEGY = "median"  # Fill missing values with median or mean
