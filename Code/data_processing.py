import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from config import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, DATA_PATH, SCALING_METHOD, BATCH_SIZE

class PitchDataset(Dataset):
    """
    Custom dataset class to process and load MLB pitch data.
    """
    def __init__(self, file_path=DATA_PATH):
        self.data = pd.read_csv(file_path)[FEATURE_COLUMNS]

        # 🔹 Encode categorical features as indices
        self.label_encoders = {col: LabelEncoder() for col in CATEGORICAL_FEATURES}
        for col in CATEGORICAL_FEATURES:
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])

        # 🔹 Scale numerical features
        self.scaler = StandardScaler() if SCALING_METHOD == "standard" else MinMaxScaler()
        self.data[NUMERICAL_FEATURES] = self.scaler.fit_transform(self.data[NUMERICAL_FEATURES])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        categorical_values = torch.tensor([row[col] for col in CATEGORICAL_FEATURES], dtype=torch.long)
        numerical_values = torch.tensor([row[col] for col in NUMERICAL_FEATURES], dtype=torch.float32)
        return categorical_values, numerical_values

def get_dataloader(file_path=DATA_PATH, batch_size=BATCH_SIZE):
    """Returns a PyTorch DataLoader for the dataset."""
    dataset = PitchDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
