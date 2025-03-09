import torch
import torch.nn as nn
from layers.attention import AttentionLayer
from layers.lstm import LSTM
from layers.mlp import MLP

class PitchFeaturesModel(nn.Module):
    def __init__(self, num_pitch_types, num_pitchers, num_batters, num_stands, num_p_throws, num_innings, num_events, num_descriptions, input_dim, seq_length, hidden_dim, num_layers):
        super(PitchFeaturesModel, self).__init__()

        # 🔹 Categorical embeddings
        self.pitcher_embedding = nn.Embedding(num_pitchers, 16)
        self.batter_embedding = nn.Embedding(num_batters, 16)
        self.stand_embedding = nn.Embedding(num_stands, 4)
        self.p_throws_embedding = nn.Embedding(num_p_throws, 4)
        self.inning_embedding = nn.Embedding(num_innings, 8)
        self.event_embedding = nn.Embedding(num_events, 8)
        self.description_embedding = nn.Embedding(num_descriptions, 16)

        # 🔹 LSTM for sequential processing
        self.lstm = LSTM((input_dim * seq_length) + 72, hidden_dim, num_layers)  # Supports past pitches
        self.attention = AttentionLayer(hidden_dim)

        # 🔹 Multi-task learning outputs
        self.fc_pitch_type = MLP(hidden_dim, num_pitch_types)
        self.fc_release_speed = MLP(hidden_dim, 1)
        self.fc_release_spin_rate = MLP(hidden_dim, 1)
        self.fc_release_extension = MLP(hidden_dim, 1)
        self.fc_plate_x = MLP(hidden_dim, 1)
        self.fc_plate_z = MLP(hidden_dim, 1)
        self.fc_vx0 = MLP(hidden_dim, 1)
        self.fc_vy0 = MLP(hidden_dim, 1)
        self.fc_vz0 = MLP(hidden_dim, 1)
        self.fc_ax = MLP(hidden_dim, 1)
        self.fc_ay = MLP(hidden_dim, 1)
        self.fc_az = MLP(hidden_dim, 1)
        self.fc_balls = MLP(hidden_dim, 1)
        self.fc_strikes = MLP(hidden_dim, 1)
        self.fc_outs_when_up = MLP(hidden_dim, 1)
        self.fc_effective_speed = MLP(hidden_dim, 1)
        self.fc_delta_run_exp = MLP(hidden_dim, 1)

    def forward(self, prev_pitches, pitcher_ids, batter_ids, stands, p_throws, innings, events, descriptions, numerical_features):
        """
        Args:
        - prev_pitches: (batch_size, seq_length * num_features) - Previous `N` pitches (Optional).
        - pitcher_ids: (batch_size, 1) - Pitcher ID.
        - batter_ids: (batch_size, 1) - Batter ID.
        - stands: (batch_size, 1) - Batter handedness.
        - p_throws: (batch_size, 1) - Pitcher throwing hand.
        - innings: (batch_size, 1) - Current inning.
        - events: (batch_size, 1) - Previous pitch outcome.
        - descriptions: (batch_size, 1) - Previous pitch description.
        - numerical_features: (batch_size, num_features) - Current pitch numerical features.
        """

        # 🔹 Embed categorical variables
        pitcher_embed = self.pitcher_embedding(pitcher_ids)
        batter_embed = self.batter_embedding(batter_ids)
        stand_embed = self.stand_embedding(stands)
        p_throws_embed = self.p_throws_embedding(p_throws)
        inning_embed = self.inning_embedding(innings)
        event_embed = self.event_embedding(events)
        description_embed = self.description_embedding(descriptions)

        # 🔹 Handling previous pitches:
        if prev_pitches is not None:
            # Case 1: History exists (during training or later-game simulation)
            lstm_input = torch.cat([
                prev_pitches, pitcher_embed, batter_embed, stand_embed,
                p_throws_embed, inning_embed, event_embed, description_embed
            ], dim=1)

            # 🔹 LSTM + Attention
            lstm_out = self.lstm(lstm_input.unsqueeze(1))
            context_vector, attn_weights = self.attention(lstm_out)

        else:
            # Case 2: No history (first pitch scenario)
            lstm_input = torch.cat([
                pitcher_embed, batter_embed, stand_embed, p_throws_embed,
                inning_embed, event_embed, description_embed, numerical_features
            ], dim=1)

            context_vector = lstm_input  # Skip LSTM, use direct features

        # 🔹 Multi-task predictions
        return {
            "pitch_type": self.fc_pitch_type(context_vector),
            "release_speed": self.fc_release_speed(context_vector),
            "release_spin_rate": self.fc_release_spin_rate(context_vector),
            "release_extension": self.fc_release_extension(context_vector),
            "plate_x": self.fc_plate_x(context_vector),
            "plate_z": self.fc_plate_z(context_vector),
            "vx0": self.fc_vx0(context_vector),
            "vy0": self.fc_vy0(context_vector),
            "vz0": self.fc_vz0(context_vector),
            "ax": self.fc_ax(context_vector),
            "ay": self.fc_ay(context_vector),
            "az": self.fc_az(context_vector),
            "balls": self.fc_balls(context_vector),
            "strikes": self.fc_strikes(context_vector),
            "outs_when_up": self.fc_outs_when_up(context_vector),
            "effective_speed": self.fc_effective_speed(context_vector),
            "delta_run_exp": self.fc_delta_run_exp(context_vector),
            "attention_weights": attn_weights
        }
