import torch

#General Project Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  #GPU if available
SEED = 42  #Set seed for reproducibility

#File Paths
DATA_PATH = r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\ungrouped_clean_statcast_15to24.csv"  #Raw dataset
TRAINED_MODELS_DIR = "saved_models/"  #Directory for saving trained models
LOGS_DIR = "logs/"  #Training logs
TRAIN_FILE = r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\train_data.pt"
TEST_FILE = r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\test_data.pt"

#Hyperparameters
#LSTM Model
LSTM_INPUT_DIM = 20 #DONT KNOW YET, ITS THE DIM OF x_t state
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

#MLP Model for regression and classification (specific pitch features an pitch results)
MLP_HIDDEN_DIM = 64

#Training Settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

#Loss Function
LOSS_FUNCTION = "MSE"  #MSE (Regression), CrossEntropy (Classification)


#Logging & Debugging ??? NOT SURE IF NEEDED
LOG_INTERVAL = 10  # How often to log training progress
SAVE_MODEL_EVERY = 5  # Save model every X epochs
DEBUG_MODE = False  # Enable/disable debugging prints

#Feature Lists
PREPITCH_FEATURES = [
    "pitcher", "batter", "inning", "inning_topbot", "stand", "p_throws", "on_1b", "on_2b", "on_3b", "outs_when_up", "home_score", "away_score", "balls", "strikes"
]

SPECIFIC_PITCH_FEATURES = [
    "pitch_type", "release_speed", "release_spin_rate", "release_extension", "plate_x", "plate_z"
]

PITCH_RESULT_FEATURES = [
    "events", "description"
]

FEATURE_COLUMNS = PREPITCH_FEATURES + SPECIFIC_PITCH_FEATURES + PITCH_RESULT_FEATURES

X_NUMERICAL_FEATURES = [
    "home_score", "away_score", "release_speed", "release_spin_rate", "release_extension", "plate_x", "plate_z"
]

X_NUMERICAL_IDX = [
    8, 9, 16, 17, 18, 19, 20
]

TARGET_FEATURES = [
    "pitch_type", "release_speed", "release_spin_rate", "release_extension",
    "plate_x", "plate_z", "events", "description"
]

#Categorical Mappings
PITCH_TYPES = {
    0: "Fastball",
    1: "Curveball",
    2: "Slider",
    3: "Changeup",
    4: "Cutter",
    5: "Splitter",
    6: "Knuckleball"
    #more
}

EVENTS = {
    0: "strikeout",
    1: "IN_PROGRESS",
    2: "field_out",
    3: "walk",
    4: "single",
    5: "double",
    6: "sac_fly",
    7: "catcher_interf",
    8: "force_out",
    9: "hit_by_pitch",
    10: "fielders_choice",
    11: "field_error",
    12: "home_run",
    13: "grounded_into_double_play",
    14: "double_play",
    15: "strikeout_double_play",
    16: "fielders_choice_out",
    17: "truncated_pa",
    18: "sac_bunt",
    19: "triple",
    20: "triple_play",
    21: "sac_fly_double_play",
    22: "sac_bunt_double_play",
    23: "game_advisory",
    24: "intent_walk",
    25: "ejection"
}

DESCRIPTION = {
    0: "swinging_strike_blocked",
    1: "swinging_strike",
    2: "ball",
    3: "foul",
    4: "called_strike",
    5: "hit_into_play",
    6: "blocked_ball",
    7: "foul_tip",
    8: "foul_bunt",
    9: "hit_by_pitch",
    10: "bunt_foul_tip",
    11: "missed_bunt",
    12: "intent_ball"
}