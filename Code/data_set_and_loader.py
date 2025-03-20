import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class AtBatDataset(Dataset):
    def __init__(self, X, Y):
        """
        X, Y are lists of sequences. 
        Each X[i] is an array of shape (seq_len_i, input_dim).
        Each Y[i] is a list of dicts with keys ('type', 'cont', 'result').
        """
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        #Return a single at-bat (X_seq, Y_seq)
        return self.X[idx], self.Y[idx]


def collate_fn(batch):
    """
    batch: list of (X_seq, Y_seq) for each at-bat in the batch
    """
    cat_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 21, 22]
    X_list = []
    Y_type_list = []
    Y_cont_list = []
    Y_desc_list = []
    Y_event_list = []
    lengths = []
    
    for (X_seq, Y_seq) in batch:
        #X_seq is shape (seq_len, input_dim)
        #Y_seq is list of dicts each with 'type', 'cont', 'result'
        seq_len = len(X_seq)
        lengths.append(seq_len)
        
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        #Shift categorical values up by 1
        for idx in cat_indices:
            X_tensor[:, idx] += 1

        X_list.append(X_tensor)
        
        #For Y, build separate arrays
        pitch_types = []
        pitch_conts = []
        pitch_descs = []
        pitch_events = []
        for y_t in Y_seq:
            pitch_types.append(y_t['type'])
            pitch_conts.append(y_t['cont'])
            pitch_descs.append(y_t['result_desc'])
            pitch_events.append(y_t['result_event'])

        Y_type_list.append(torch.tensor(pitch_types, dtype=torch.long))
        Y_cont_list.append(torch.tensor(pitch_conts, dtype=torch.float32))
        Y_desc_list.append(torch.tensor(pitch_descs, dtype=torch.long))
        Y_event_list.append(torch.tensor(pitch_events, dtype=torch.long))
    
    #Padding
    padded_X = pad_sequence(X_list, batch_first=True, padding_value=-999)

    cat_padding_value = torch.tensor(0, dtype=torch.float32, device=padded_X.device)
    #Replaces categorical features padding values with 0
    for idx in cat_indices:
        padded_X[:, :, idx] = torch.where(padded_X[:, :, idx] == -999, cat_padding_value, padded_X[:, :, idx])


    padded_Y_type   = pad_sequence(Y_type_list, batch_first=True, padding_value=0)
    padded_Y_cont   = pad_sequence(Y_cont_list, batch_first=True, padding_value=-999)
    padded_Y_desc   = pad_sequence(Y_desc_list, batch_first=True, padding_value=0)
    padded_Y_event  = pad_sequence(Y_event_list, batch_first=True, padding_value=0)
    
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_X, padded_Y_type, padded_Y_cont, padded_Y_desc, padded_Y_event, lengths

def create_dataloaders(X_train, Y_train, X_test, Y_test, batch_size=32):
    train_dataset = AtBatDataset(X_train, Y_train)
    test_dataset  = AtBatDataset(X_test, Y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    return train_loader, test_loader
