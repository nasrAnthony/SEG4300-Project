import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import get_dataloader, proc_save_dataset


proc_save_dataset(file_path=r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\partclean_statcast_15to24.csv",
                  save_path=r"C:\Users\Richard\Documents\SEG4300\Project\SEG4300-Project\processed_pcs1524.pt")