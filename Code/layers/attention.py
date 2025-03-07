#attention layer for lstm pitch sequence prediction model
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # Compute attention scores
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # Apply attention
        return context_vector, attn_weights
