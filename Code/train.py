import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import get_dataloader
from models.pitch_features import PitchFeaturesModel
from config import DEVICE, DATA_PATH, LEARNING_RATE, NUM_EPOCHS

# Load dataset
train_loader = get_dataloader(DATA_PATH)

# Initialize model
model = PitchFeaturesModel(
    num_pitch_types=7, num_pitchers=500, num_batters=500, num_stands=2,
    num_p_throws=2, num_innings=9, input_dim=20, hidden_dim=128, num_layers=2
).to(DEVICE)

# Define loss function & optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for categorical_values, numerical_values in train_loader:
        categorical_values = categorical_values.to(DEVICE)
        numerical_values = numerical_values.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(
            pitch_seq=numerical_values[:, :10],
            pitcher_ids=categorical_values[:, 0],
            batter_ids=categorical_values[:, 1],
            stands=categorical_values[:, 2],
            p_throws=categorical_values[:, 3],
            innings=categorical_values[:, 4],
            numerical_features=numerical_values
        )

        # Compute loss
        loss = loss_fn(predictions["speed"], numerical_values[:, 5])
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "saved_models/pitch_features.pth")
