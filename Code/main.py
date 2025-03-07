import torch
from models.pitch_features import PitchFeaturesModel
from models.pitch_count import PitchCountModel
from simulation import GameSimulator

# Load models
pitch_sequence_model = PitchFeaturesModel(num_pitch_types=7, input_dim=10, hidden_dim=128, num_layers=2)
pitch_count_model = PitchCountModel(num_pitchers=500, input_dim=10)

# Load pre-trained weights (optional)
pitch_sequence_model.load_state_dict(torch.load("saved_models/pitch_sequence.pth"))
pitch_count_model.load_state_dict(torch.load("saved_models/pitch_count.pth"))

# Set to evaluation mode
pitch_sequence_model.eval()
pitch_count_model.eval()

# Initialize simulator
simulator = GameSimulator(pitch_sequence_model, pitch_count_model)

# Example game setup
game_context = {"inning": 1, "home_score": 0, "away_score": 0, "outs": 0}
pitcher_id = 123  # Example pitcher ID
lineup = [657077, 621654, 669224]  # Example batting lineup (batter IDs)

# Run the game simulation
predicted_game_pitches = simulator.simulate_game(pitcher_id, lineup, game_context)

# Output results
print(predicted_game_pitches)
