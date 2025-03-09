from data_processing import proc_save_dataset, PitchDataset
from config import DATA_PATH

#PitchDataset(DATA_PATH)

proc_save_dataset(file_path=DATA_PATH, test_size=0.2, seed=42)