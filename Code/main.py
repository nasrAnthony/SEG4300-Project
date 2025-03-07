import torch
from models.pitch_features import PitchFeaturesModel
from simulation import GameSimulator
from config import DEVICE

# Load trained model
model = PitchFeaturesModel(
    num_pitch_types=7, num_pitchers=500, num_batters=500, num_stands=2,
    num_p_throws=2, num_innings=9, input_dim=20, hidden_dim=128, num_layers=2
)
model.load_state_dict(torch.load("saved_models/pitch_features.pth"))
model.eval()

# Initialize game simulator
simulator = GameSimulator(model)

# Example game state
game_state = {"inning": 1, "outs": 0, "balls": 0, "strikes": 0, "home_score": 0, "away_score": 0, "stand": 1, "p_throws": 1}

# Example pitcher and lineup
pitcher_id = 123
lineup = [657077, 621654, 669224, 669257, 643446, 621446, 592450, 554654, 492354]

# Run game simulation
game_pitches = simulator.simulate_game(pitcher_id, lineup, game_state)
print("\nFinal Pitch Sequences:")
print(game_pitches)
