import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

class PitchDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        # Encode categorical features
        label_encoders = {}
        categorical_cols = ["pitcher", "batter", "game_pk"]
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le

        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ["release_speed", "release_spin_rate", "effective_speed"]
        self.data[numerical_cols] = scaler.fit_transform(self.data[numerical_cols])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)

def get_dataloader(file_path, batch_size=32):
    dataset = PitchDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
