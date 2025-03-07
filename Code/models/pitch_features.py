import torch
import torch.nn as nn
from layers.attention import AttentionLayer
from layers.lstm import LSTM
from layers.mlp import MLP

class PitchFeaturesModel(nn.Module):
    def __init__(self, num_pitch_types, num_pitchers, num_batters, num_stands, num_p_throws, num_innings, input_dim, hidden_dim, num_layers):
        super(PitchFeaturesModel, self).__init__()

        # Categorical embeddings
        self.pitcher_embedding = nn.Embedding(num_pitchers, 16)
        self.batter_embedding = nn.Embedding(num_batters, 16)
        self.stand_embedding = nn.Embedding(num_stands, 4)
        self.p_throws_embedding = nn.Embedding(num_p_throws, 4)
        self.inning_embedding = nn.Embedding(num_innings, 8)

        # LSTM for sequential processing
        self.lstm = LSTM(input_dim + 48, hidden_dim, num_layers)
        self.attention = AttentionLayer(hidden_dim)

        # Multi-task learning outputs
        self.fc_pitch_type = MLP(hidden_dim, num_pitch_types)
        self.fc_location = MLP(hidden_dim, 2)  # Predict plate_x, plate_z
        self.fc_speed = MLP(hidden_dim, 1)  # Predict speed
        self.fc_spin_rate = MLP(hidden_dim, 1)  # Predict spin rate
        self.fc_extension = MLP(hidden_dim, 1)  # Predict release extension

    def forward(self, pitch_seq, pitcher_ids, batter_ids, stands, p_throws, innings, numerical_features):
        """
        pitch_seq: (batch_size, seq_len) - Previous pitch sequence
        pitcher_ids: (batch_size, 1) - Pitcher IDs
        batter_ids: (batch_size, 1) - Batter IDs
        stands: (batch_size, 1) - Batter handedness
        p_throws: (batch_size, 1) - Pitcher handedness
        innings: (batch_size, 1) - Inning numbers
        numerical_features: (batch_size, seq_len, num_features) - All numerical features
        """
        # Embed categorical variables
        pitcher_embed = self.pitcher_embedding(pitcher_ids).unsqueeze(1)
        batter_embed = self.batter_embedding(batter_ids).unsqueeze(1)
        stand_embed = self.stand_embedding(stands).unsqueeze(1)
        p_throws_embed = self.p_throws_embedding(p_throws).unsqueeze(1)
        inning_embed = self.inning_embedding(innings).unsqueeze(1)

        # Expand embeddings across sequence length
        expanded_pitcher_embed = pitcher_embed.expand(-1, pitch_seq.size(1), -1)
        expanded_batter_embed = batter_embed.expand(-1, pitch_seq.size(1), -1)
        expanded_stand_embed = stand_embed.expand(-1, pitch_seq.size(1), -1)
        expanded_p_throws_embed = p_throws_embed.expand(-1, pitch_seq.size(1), -1)
        expanded_inning_embed = inning_embed.expand(-1, pitch_seq.size(1), -1)

        # Concatenate inputs
        lstm_input = torch.cat([
            pitch_seq, expanded_pitcher_embed, expanded_batter_embed,
            expanded_stand_embed, expanded_p_throws_embed, expanded_inning_embed,
            numerical_features
        ], dim=2)

        # LSTM + Attention
        lstm_out = self.lstm(lstm_input)
        context_vector, attn_weights = self.attention(lstm_out)

        # Multi-task predictions
        return {
            "pitch_type": self.fc_pitch_type(context_vector),
            "location": self.fc_location(context_vector),
            "speed": self.fc_speed(context_vector),
            "spin_rate": self.fc_spin_rate(context_vector),
            "extension": self.fc_extension(context_vector),
            "attention_weights": attn_weights
        }
