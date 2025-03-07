import torch
import random
from models.pitch_features import PitchFeaturesModel
from models.pitch_count import PitchCountModel
from data_processing import get_dataloader
from config import DEVICE

class GameSimulator:
    def __init__(self, pitch_features_model, pitch_count_model):
        self.pitch_features_model = pitch_features_model.to(DEVICE)
        self.pitch_count_model = pitch_count_model.to(DEVICE)
        self.pitcher_pitches = {}

    def predict_pitch_count(self, pitcher_id, game_features):
        """Predicts how many pitches the pitcher will throw in the game."""
        pitcher_tensor = torch.tensor([pitcher_id], dtype=torch.long, device=DEVICE)
        features_tensor = torch.tensor([game_features], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            total_pitches = self.pitch_count_model(pitcher_tensor, features_tensor).item()
        
        return round(total_pitches)

    def simulate_at_bat(self, pitcher_id, batter_id, game_context):
        """Simulates a single at-bat until a result is determined (strikeout, walk, hit)."""
        balls, strikes, outs = 0, 0, game_context["outs"]
        at_bat_sequence = []

        while outs < 3:
            # Prepare input data
            pitcher_tensor = torch.tensor([pitcher_id], dtype=torch.long, device=DEVICE)
            batter_tensor = torch.tensor([batter_id], dtype=torch.long, device=DEVICE)
            game_state_tensor = torch.tensor([[game_context["inning"], game_context["home_score"], game_context["away_score"]]], dtype=torch.float, device=DEVICE)

            # Predict next pitch
            with torch.no_grad():
                predictions = self.pitch_sequence_model(game_state_tensor)

            pitch_type_pred = torch.argmax(predictions["pitch_type"], dim=1).item()
            outcome_pred = torch.argmax(predictions["outcome"], dim=1).item()

            # Store pitch details
            pitch_details = {
                "batter_id": batter_id,
                "pitch_type": pitch_type_pred,
                "velocity": predictions["speed"].item(),
                "plate_x": predictions["location"][0].item(),
                "plate_z": predictions["location"][1].item(),
                "outcome": outcome_pred
            }
            at_bat_sequence.append(pitch_details)

            # Update game state based on pitch outcome
            if outcome_pred == 0:  # Ball
                balls += 1
                if balls == 4:  # Walk
                    return at_bat_sequence, "walk"
            elif outcome_pred == 1:  # Strike
                strikes += 1
                if strikes == 3:  # Strikeout
                    return at_bat_sequence, "strikeout"
            elif outcome_pred == 2:  # Contact (Hit or Out)
                return at_bat_sequence, "contact"

        return at_bat_sequence, "unknown"

    def simulate_game(self, pitcher_id, lineup, game_context):
        """Simulates a full game using predicted pitch sequences."""
        self.pitcher_pitches[pitcher_id] = 0
        max_pitches = self.predict_pitch_count(pitcher_id, game_context)

        pitch_sequences = []

        for inning in range(1, 10):  # Simulate 9 innings
            game_context["inning"] = inning
            for batter_id in lineup:
                if self.pitcher_pitches[pitcher_id] >= max_pitches:
                    print(f"Pitcher {pitcher_id} removed after {max_pitches} pitches.")
                    return pitch_sequences  # End simulation if pitcher is pulled

                at_bat_sequence, result = self.simulate_at_bat(pitcher_id, batter_id, game_context)
                pitch_sequences.append(at_bat_sequence)

                # Update pitch count
                self.pitcher_pitches[pitcher_id] += len(at_bat_sequence)

                # Update game state
                if result == "strikeout" or result == "walk":
                    game_context["outs"] += 1
                elif result == "contact":
                    break  # Move to next batter

                if game_context["outs"] >= 3:
                    break  # End inning

        return pitch_sequences
