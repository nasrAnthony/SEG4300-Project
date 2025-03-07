import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import get_dataloader
from models.pitch_features import PitchFeaturesModel
from models.pitch_count import PitchCountModel

# Load data
train_loader = get_dataloader("data/train.csv")

# Load models
pitch_sequence_model = PitchFeaturesModel(num_pitch_types=7, input_dim=10, hidden_dim=128, num_layers=2)
pitch_count_model = PitchCountModel(num_pitchers=500, input_dim=10)

# Define loss functions
loss_fn = nn.MSELoss()
optimizer = optim.Adam(list(pitch_sequence_model.parameters()) + list(pitch_count_model.parameters()), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        pitch_seq_preds = pitch_sequence_model(batch)
        pitch_count_preds = pitch_count_model(batch[:, 0].long(), batch[:, 1:])

        # Compute loss
        loss_seq = loss_fn(pitch_seq_preds["speed"], batch[:, 2])  # Predicting pitch speed
        loss_count = loss_fn(pitch_count_preds, batch[:, 3])  # Predicting total pitch count
        total_loss = loss_seq + loss_count

        # Backward pass
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")
