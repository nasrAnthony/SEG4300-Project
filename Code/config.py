import torch

### 🏗 General Project Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
SEED = 42  # Set seed for reproducibility

### 📂 File Paths
DATA_PATH = r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\partclean_statcast_15to24.csv"  # Raw dataset
TRAINED_MODELS_DIR = "saved_models/"  # Directory for saving trained models
LOGS_DIR = "logs/"  # Training logs
TRAIN_FILE = r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\train_data.pt"
TEST_FILE = r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\test_data.pt"

### 🎯 Hyperparameters
# 🔹 LSTM Model for Pitch Sequence Prediction
LSTM_INPUT_DIM = 20  # Adjusted for more features
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

# 🔹 MLP Model for Total Pitch Count Prediction
MLP_HIDDEN_DIM = 64

# 🔹 Training Settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-5  # Regularization
SEQ_LENGTH = 20

# 🔹 Loss Function
LOSS_FUNCTION = "MSE"  # Options: "MSE" (Regression), "CrossEntropy" (Classification)

### ⚾ Game Simulation Settings
MAX_PITCH_COUNT = 120  # Maximum pitches before pitcher is pulled
INNINGS = 9  # Standard MLB game length
BULLPEN_USAGE_FACTOR = 1.0  # Controls pitcher substitution likelihood

### 📊 Logging & Debugging
LOG_INTERVAL = 10  # How often to log training progress
SAVE_MODEL_EVERY = 5  # Save model every X epochs
DEBUG_MODE = False  # Enable/disable debugging prints

### 🏷 Feature Lists
# 🔹 Categorical Features (Will Be Embedded)
CATEGORICAL_FEATURES = [
    "pitch_type", "pitcher", "batter", "events", "description", "inning_topbot", "stand", "p_throws", "game_pk"
]

# 🔹 Numerical Features (Will Be Scaled)
NUMERICAL_FEATURES = [
    "release_speed", "release_spin_rate", "release_extension",
    "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "balls", "strikes", "outs_when_up", "effective_speed", "delta_run_exp"
]

# 🔹 All Features (For Data Processing)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

### 🔹 Target Variables (Model Outputs)
TARGET_FEATURES = [
    "pitch_type", "release_speed", "release_spin_rate", "release_extension",
    "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "balls", "strikes", "outs_when_up", "effective_speed", "delta_run_exp"
]

### 🔹 Feature Scaling Options
SCALING_METHOD = "standard"  # Choose between "minmax" or "standard"
MISSING_VALUE_STRATEGY = "median"  # Fill missing values with median or mean

### 🎭 Categorical Mappings
PITCH_TYPES = {
    0: "Fastball",
    1: "Curveball",
    2: "Slider",
    3: "Changeup",
    4: "Cutter",
    5: "Splitter",
    6: "Knuckleball"
}