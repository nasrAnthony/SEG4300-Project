import torch
import torch.nn as nn
from layers.attention import AttentionLayer
from layers.lstm import LSTM
from layers.mlp import MLP

class PitchFeaturesModel(nn.Module):
    def __init__(self, num_pitch_types, input_dim, hidden_dim, num_layers):
        super(PitchFeaturesModel, self).__init__()
        
        self.lstm = LSTM(input_dim, hidden_dim, num_layers)
        self.attention = AttentionLayer(hidden_dim)

        # Multi-task learning heads
        self.fc_pitch_type = MLP(hidden_dim, num_pitch_types)  # Predict pitch type
        self.fc_location = MLP(hidden_dim, 2)  # Predict (plate_x, plate_z)
        self.fc_speed = MLP(hidden_dim, 1)  # Predict speed

    def forward(self, x):
        lstm_out = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)

        return {
            "pitch_type": self.fc_pitch_type(context_vector),
            "location": self.fc_location(context_vector),
            "speed": self.fc_speed(context_vector),
            "attention_weights": attn_weights
        }
