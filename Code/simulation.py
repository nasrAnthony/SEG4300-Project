import torch
from config import DEVICE, MAX_PITCH_COUNT

class GameSimulator:
    """
    Simulates a full MLB game using pitch sequence predictions.
    """
    def __init__(self, pitch_features_model):
        self.pitch_features_model = pitch_features_model.to(DEVICE)
        self.pitcher_pitches = {}  # Tracks pitch counts for each pitcher

    def predict_pitch_count(self, pitcher_id, game_state):
        """
        Predicts the total number of pitches a pitcher will throw before being pulled.
        """
        pitcher_tensor = torch.tensor([pitcher_id], dtype=torch.long, device=DEVICE)
        game_state_tensor = torch.tensor([game_state], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            total_pitches = self.pitch_features_model.fc_speed(
                torch.cat([game_state_tensor, pitcher_tensor.unsqueeze(1)], dim=1) 
        ).item()
        
        return min(round(total_pitches), MAX_PITCH_COUNT)  # Limit the pitch count

    def simulate_at_bat(self, pitcher_id, batter_id, game_state):
        """
        Simulates a single at-bat until a result is determined (strikeout, walk, or contact).
        """
        balls, strikes, outs = 0, 0, game_state["outs"]
        at_bat_sequence = []

        while outs < 3:
            # Prepare input tensors
            pitcher_tensor = torch.tensor([pitcher_id], dtype=torch.long, device=DEVICE)
            batter_tensor = torch.tensor([batter_id], dtype=torch.long, device=DEVICE)
            game_state_tensor = torch.tensor([[game_state["inning"], game_state["home_score"], game_state["away_score"]]], dtype=torch.float32, device=DEVICE)

            # Predict the next pitch
            with torch.no_grad():
                predictions = self.pitch_features_model(
                    pitch_seq=game_state_tensor[:, :10],
                    pitcher_ids=pitcher_tensor,
                    batter_ids=batter_tensor,
                    stands=torch.tensor([game_state["stand"]], dtype=torch.long, device=DEVICE),
                    p_throws=torch.tensor([game_state["p_throws"]], dtype=torch.long, device=DEVICE),
                    innings=torch.tensor([game_state["inning"]], dtype=torch.long, device=DEVICE),
                    numerical_features=game_state_tensor
                )

            pitch_type_pred = torch.argmax(predictions["pitch_type"], dim=1).item()
            outcome_pred = torch.argmax(predictions["speed"], dim=1).item()  # Using speed as a proxy for outcome

            # Store pitch details
            pitch_details = {
                "batter_id": batter_id,
                "pitch_type": pitch_type_pred,
                "velocity": predictions["speed"].item(),
                "plate_x": predictions["location"][0].item(),
                "plate_z": predictions["location"][1].item(),
                "spin_rate": predictions["spin_rate"].item(),
                "extension": predictions["extension"].item(),
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

    def simulate_game(self, pitcher_id, lineup, game_state):
        """
        Simulates a full game using predicted pitch sequences.
        """
        self.pitcher_pitches[pitcher_id] = 0
        max_pitches = self.predict_pitch_count(pitcher_id, game_state)

        pitch_sequences = []

        for inning in range(1, 10):  # Simulate 9 innings
            game_state["inning"] = inning
            for batter_id in lineup:
                if self.pitcher_pitches[pitcher_id] >= max_pitches:
                    print(f"Pitcher {pitcher_id} removed after {max_pitches} pitches.")
                    return pitch_sequences  # End simulation if pitcher is pulled

                at_bat_sequence, result = self.simulate_at_bat(pitcher_id, batter_id, game_state)
                pitch_sequences.append(at_bat_sequence)

                self.pitcher_pitches[pitcher_id] += len(at_bat_sequence)

                if result in ["strikeout", "walk"]:
                    game_state["outs"] += 1
                elif result == "contact":
                    break  # Move to next batter

                if game_state["outs"] >= 3:
                    break  # End inning

        return pitch_sequences
