import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multihead_pitch_prediction import MultiHeadLSTM  # Assuming your model is in model.py
import os
import time

# Training Configuration
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "best_model.pth"

# Detect device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_loader = get_dataloader(file_path="train_data.csv", batch_size=BATCH_SIZE)
val_loader = get_dataloader(file_path="val_data.csv", batch_size=BATCH_SIZE)

# Initialize model (update parameters based on your dataset)
model = MultiHeadLSTM(
    num_numeric_features=7,  # Matches num_indices
    num_low_card_cats=9,  # Matches low-cardinality categorical features (encoded)
    num_pitchers=500, num_batters=500,
    num_descriptions=30, num_events=20, num_prev_pitch_types=10,
    pitcher_emb_dim=16, batter_emb_dim=16,
    description_emb_dim=8, event_emb_dim=8, prev_pitch_emb_dim=8,
    hidden_dim=128,
    num_pitch_type_classes=10, num_description_classes=30, num_event_classes=20,
    cont_dim=3, lstm_layers=2, dropout=0.2
).to(device)

# Loss functions
criterion_classification = nn.CrossEntropyLoss()
criterion_continuous = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track best validation loss
best_val_loss = float("inf")


### 🚀 Training Function
def train_one_epoch(epoch):
    model.train()
    total_loss, total_pitch_type_loss, total_desc_loss, total_event_loss, total_cont_loss = 0, 0, 0, 0, 0
    correct_pitch_types, total_samples = 0, 0

    start_time = time.time()

    for batch in train_loader:
        (
            x_num, x_low_card_cats, x_pitcher, x_batter, x_desc, x_event, 
            x_prev_pitch, lengths, y_pitch_type, y_desc, y_event, y_cont
        ) = [tensor.to(device) for tensor in batch]

        optimizer.zero_grad()
        
        # Forward pass
        pitch_type_logits, pitch_cont_values, pitch_result_desc, pitch_result_event = model(
            x_num, x_low_card_cats, x_pitcher, x_batter, x_desc, x_event, x_prev_pitch, lengths
        )

        # Compute losses
        loss_pitch_type = criterion_classification(pitch_type_logits.view(-1, pitch_type_logits.size(-1)), y_pitch_type.view(-1))
        loss_desc = criterion_classification(pitch_result_desc.view(-1, pitch_result_desc.size(-1)), y_desc.view(-1))
        loss_event = criterion_classification(pitch_result_event.view(-1, pitch_result_event.size(-1)), y_event.view(-1))
        loss_cont = criterion_continuous(pitch_cont_values, y_cont)

        # Total loss
        loss = loss_pitch_type + loss_desc + loss_event + loss_cont
        loss.backward()
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_pitch_type_loss += loss_pitch_type.item()
        total_desc_loss += loss_desc.item()
        total_event_loss += loss_event.item()
        total_cont_loss += loss_cont.item()

        # Track classification accuracy for pitch type
        _, predicted_pitch_type = torch.max(pitch_type_logits, dim=-1)
        correct_pitch_types += (predicted_pitch_type == y_pitch_type).sum().item()
        total_samples += y_pitch_type.numel()

    avg_loss = total_loss / len(train_loader)
    pitch_type_acc = 100 * correct_pitch_types / total_samples

    end_time = time.time()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Pitch Type Acc: {pitch_type_acc:.2f}% - Time: {end_time - start_time:.2f}s")

    return avg_loss


### 🚀 Validation Function
def validate():
    model.eval()
    val_loss, val_pitch_type_loss, val_desc_loss, val_event_loss, val_cont_loss = 0, 0, 0, 0, 0
    correct_pitch_types, total_samples = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            (
                x_num, x_low_card_cats, x_pitcher, x_batter, x_desc, x_event, 
                x_prev_pitch, lengths, y_pitch_type, y_desc, y_event, y_cont
            ) = [tensor.to(device) for tensor in batch]

            pitch_type_logits, pitch_cont_values, pitch_result_desc, pitch_result_event = model(
                x_num, x_low_card_cats, x_pitcher, x_batter, x_desc, x_event, x_prev_pitch, lengths
            )

            # Compute validation losses
            loss_pitch_type = criterion_classification(pitch_type_logits.view(-1, pitch_type_logits.size(-1)), y_pitch_type.view(-1))
            loss_desc = criterion_classification(pitch_result_desc.view(-1, pitch_result_desc.size(-1)), y_desc.view(-1))
            loss_event = criterion_classification(pitch_result_event.view(-1, pitch_result_event.size(-1)), y_event.view(-1))
            loss_cont = criterion_continuous(pitch_cont_values, y_cont)

            loss = loss_pitch_type + loss_desc + loss_event + loss_cont
            val_loss += loss.item()

            # Track classification accuracy for pitch type
            _, predicted_pitch_type = torch.max(pitch_type_logits, dim=-1)
            correct_pitch_types += (predicted_pitch_type == y_pitch_type).sum().item()
            total_samples += y_pitch_type.numel()

    avg_val_loss = val_loss / len(val_loader)
    pitch_type_acc = 100 * correct_pitch_types / total_samples
    print(f"Validation Loss: {avg_val_loss:.4f} - Pitch Type Acc: {pitch_type_acc:.2f}%")

    return avg_val_loss


### 🚀 Training Loop
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(epoch)
    val_loss = validate()

    # Save the best model
    if val_loss < best_val_loss:
        print("🔥 New best model found! Saving...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_val_loss = val_loss

print("Training complete! Best model saved as", MODEL_SAVE_PATH)