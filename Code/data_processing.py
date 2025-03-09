import pandas as pd
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from config import (
    FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, DATA_PATH, 
    BATCH_SIZE, TRAIN_FILE, TEST_FILE, SEQ_LENGTH, SCALING_METHOD
)

def encode_and_scale(df):
    """
    Encodes categorical features and scales numerical features in the dataset.
    - Categorical features are label-encoded, with "UNKNOWN" handling.
    - Numerical features are scaled using StandardScaler or MinMaxScaler.
    """

    # 🔹 Handle missing categorical values by filling with "UNKNOWN"
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("UNKNOWN")

    # 🔹 Label Encoding Dictionary for Categorical Features
    label_encoders = {col: LabelEncoder() for col in CATEGORICAL_FEATURES}
    
    for col in CATEGORICAL_FEATURES:
        df[col] = label_encoders[col].fit_transform(df[col])

    # 🔹 Scale numerical features
    scaler = StandardScaler() if SCALING_METHOD == "standard" else MinMaxScaler()
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])

    return df, label_encoders, scaler

def create_pitch_sequences(df, seq_length=SEQ_LENGTH, label_encoders=None):
    """
    Creates sliding window sequences for pitches. 
    - Pads numerical features with `0.0` if missing.
    - Pads categorical features (`events`, `description`, etc.) with `"UNKNOWN"`.
    - Ensures only **past pitches by the same pitcher** are included.
    """

    sequences = []
    
    # 🔹 Iterate over each row (each pitch)
    for i in range(len(df)):
        current_pitch = df.iloc[i]
        pitcher_id = current_pitch["pitcher"]

        # Get past `N` pitches by the **same pitcher**
        prev_pitches = df[(df.index < i) & (df["pitcher"] == pitcher_id)].tail(seq_length)

        # 🔹 If not enough previous pitches, pad with zeros and "UNKNOWN"
        if len(prev_pitches) < seq_length:
            padding_needed = seq_length - len(prev_pitches)

            # Pad numerical features with `0.0`
            pad_numerical = np.zeros((padding_needed, len(NUMERICAL_FEATURES)))

            # Pad categorical features with "UNKNOWN" (converted to int label)
            unknown_cat_values = [label_encoders[col].transform(["UNKNOWN"])[0] for col in CATEGORICAL_FEATURES]
            pad_categorical = np.tile(unknown_cat_values, (padding_needed, 1))

            # Stack padded values with actual data
            prev_pitches_numerical = np.vstack([pad_numerical, prev_pitches[NUMERICAL_FEATURES].values])
            prev_pitches_categorical = np.vstack([pad_categorical, prev_pitches[CATEGORICAL_FEATURES].values])
        
        else:
            prev_pitches_numerical = prev_pitches[NUMERICAL_FEATURES].values
            prev_pitches_categorical = prev_pitches[CATEGORICAL_FEATURES].values

        # 🔹 Flatten previous pitches & combine with current pitch categorical features
        sequence = prev_pitches_numerical.flatten().tolist() + prev_pitches_categorical.flatten().tolist() + \
                   current_pitch[CATEGORICAL_FEATURES].tolist()

        target = current_pitch[NUMERICAL_FEATURES].tolist()  # Target is the current pitch's numerical features

        sequences.append((sequence, target))

    return sequences

def proc_save_dataset(file_path=DATA_PATH, test_size=0.2, seed=42):
    """
    Preprocesses, splits, and saves dataset into train and test tensor files.
    Ensures past `N` pitches are only from the **same pitcher**.
    """

    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print("Train and test datasets already exist. Skipping reprocessing.")
        return  # Don't resplit if files exist

    df = pd.read_csv(file_path)[FEATURE_COLUMNS]

    # 🔹 Encode categorical features and scale numerical features before splitting
    df, label_encoders, scaler = encode_and_scale(df)

    # 🔹 Split dataset into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    # 🔹 Create sequences of pitches (handles missing history with padding)
    train_sequences = create_pitch_sequences(train_df, label_encoders=label_encoders)
    test_sequences = create_pitch_sequences(test_df, label_encoders=label_encoders)

    # 🔹 Convert to tensors
    train_inputs, train_targets = zip(*train_sequences)
    test_inputs, test_targets = zip(*test_sequences)

    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)

    test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)

    # 🔹 Save tensors
    torch.save((train_inputs_tensor, train_targets_tensor), TRAIN_FILE)
    torch.save((test_inputs_tensor, test_targets_tensor), TEST_FILE)

def load_saved_dataloaders(batch_size=BATCH_SIZE):
    """
    Loads pre-saved train and test tensor files into DataLoaders.

    Args:
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # 🔹 Ensure the dataset is processed and saved
    proc_save_dataset()

    # 🔹 Load saved tensors
    train_inputs_tensor, train_targets_tensor = torch.load(TRAIN_FILE)
    test_inputs_tensor, test_targets_tensor = torch.load(TEST_FILE)

    # 🔹 Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_inputs_tensor, train_targets_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_inputs_tensor, test_targets_tensor)

    # 🔹 Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
