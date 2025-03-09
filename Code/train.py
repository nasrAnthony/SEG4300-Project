import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import load_saved_dataloaders
from models.pitch_features import PitchFeaturesModel
from config import DEVICE, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE

# Load dataset
train_loader, test_loader = load_saved_dataloaders(BATCH_SIZE)

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

    model.train()
    running_loss = 0.0

    for categorical_values, numerical_values in train_loader:
        categorical_values, numerical_values = categorical_values.to(DEVICE), numerical_values.to(DEVICE)

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

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for categorical_values, numerical_values in test_loader:
            categorical_values, numerical_values = categorical_values.to(DEVICE), numerical_values.to(DEVICE)

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
            
        test_loss = loss_fn(predictions["speed"], numerical_values[:, 5])
        total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Test Loss: {avg_test_loss:.4f}")

# Save model
torch.save(model.state_dict(), "saved_models/pitch_features_8020split.pth")
