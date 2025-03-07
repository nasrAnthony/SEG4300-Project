import torch
import torch.nn as nn
from layers.mlp import MLP

class PitchCountModel(nn.Module):
    def __init__(self, num_pitchers, input_dim):
        super(PitchCountModel, self).__init__()
        self.pitcher_embedding = nn.Embedding(num_pitchers, 16)
        self.mlp = MLP(input_dim + 16, 1)

    def forward(self, pitcher_id, features):
        pitcher_embed = self.pitcher_embedding(pitcher_id)
        x = torch.cat([pitcher_embed, features], dim=1)
        return self.mlp(x)
