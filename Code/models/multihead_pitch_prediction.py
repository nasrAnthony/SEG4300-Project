import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.mlp import MLP
from layers.lstm import LSTM

class MultiHeadLSTM(nn.Module):
    def __init__(
        self,
        num_numeric_features,  #Number of numerical features
        num_pitchers, num_batters,
        num_prev_descriptions, num_prev_events, num_prev_pitch_types,
        num_low_card_cats,  #Low-cardinality categorical features (already encoded)
        num_innings, inning_emb_dim,
        pitcher_emb_dim, batter_emb_dim,
        prev_description_emb_dim, prev_event_emb_dim, prev_pitch_emb_dim,
        hidden_dim,
        #Output numbers (dont use input desc/event totals)
        num_pitch_type_classes, num_description_classes, num_event_classes,
        cont_dim, lstm_layers=1, dropout=0.0
    ):
        super().__init__()

        #Track numeric feature count
        self.num_numeric_features = num_numeric_features
        self.num_low_card_cats = num_low_card_cats  #Encoded categorical features

        #Embedding layers for high-cardinality categorical features
        self.inning_emb = nn.Embedding(num_innings, inning_emb_dim)
        self.pitcher_emb = nn.Embedding(num_pitchers, pitcher_emb_dim)
        self.batter_emb = nn.Embedding(num_batters, batter_emb_dim)
        self.prev_description_emb = nn.Embedding(num_prev_descriptions, prev_description_emb_dim)
        self.prev_event_emb = nn.Embedding(num_prev_events, prev_event_emb_dim)
        self.prev_pitch_emb = nn.Embedding(num_prev_pitch_types, prev_pitch_emb_dim)

        #Linear layer to transform low-card categorical values into a useful space
        self.low_card_transform = nn.Linear(num_low_card_cats, num_low_card_cats)

        #Compute LSTM input size (numeric + embedded features)
        self.lstm_input_dim = (
            num_numeric_features +  #Numeric features
            num_low_card_cats +  #Encoded categorical features (transformed)
            inning_emb_dim +
            pitcher_emb_dim + batter_emb_dim +
            prev_description_emb_dim + prev_event_emb_dim + prev_pitch_emb_dim
        )

        #LSTM layer
        self.lstm = LSTM(
            input_dim=self.lstm_input_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )

        #Multi-head MLP outputs
        self.pitch_type_head = MLP(hidden_dim, num_pitch_type_classes, hidden_dim=128)
        self.pitch_cont_head = MLP(hidden_dim, cont_dim, hidden_dim=128)
        self.pitch_result_head_desc = MLP(hidden_dim, num_description_classes, hidden_dim=128)
        self.pitch_result_head_event = MLP(hidden_dim, num_event_classes, hidden_dim=128)

        #Automatically detect CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  #Move model and parameters to the correct device

    def forward(self, 
                x_num, x_low_card_cats, x_inning, x_pitcher, x_batter, x_desc, x_event, x_prev_pitch, lengths):
        """
        Forward pass.

        x_num: (batch_size, seq_len, num_numeric_features)  # Numeric features (e.g., release_speed, plate_x, etc.)
        x_low_card_cats: (batch_size, seq_len, num_low_card_cats)  # Low-cardinality categorical features (encoded as numbers)
        x_inning: (batch_size, seq_len)  # Inning as an integer category
        x_pitcher, x_batter, x_desc, x_event, x_prev_pitch: (batch_size, seq_len) categorical IDs for embeddings
        lengths: (batch_size,)  # Sequence lengths for packed sequences
        """

        #Move inputs to the model's device
        x_num = x_num.to(self.device)
        x_low_card_cats = x_low_card_cats.to(self.device)
        x_inning = x_inning.to(self.device)
        x_pitcher = x_pitcher.to(self.device)
        x_batter = x_batter.to(self.device)
        x_desc = x_desc.to(self.device)
        x_event = x_event.to(self.device)
        x_prev_pitch = x_prev_pitch.to(self.device)

        #Embed categorical features (high-cardinality)
        inning_vecs = self.inning_emb(x_inning)
        pitcher_vecs = self.pitcher_emb(x_pitcher)
        batter_vecs = self.batter_emb(x_batter)
        desc_vecs = self.prev_description_emb(x_desc)
        event_vecs = self.prev_event_emb(x_event)
        prev_pitch_vecs = self.prev_pitch_emb(x_prev_pitch)

        #Transform low-cardinality categorical values (learn better representation)
        #Ensure x_low_card_cats is properly shaped for the Linear layer
        batch_size, seq_len, num_features = x_low_card_cats.shape
        x_low_card_cats = x_low_card_cats.view(-1, num_features)  #(batch_size * seq_len, num_low_card_cats)
        x_low_card_cats = self.low_card_transform(x_low_card_cats)  #Apply transformation
        x_low_card_cats = x_low_card_cats.view(batch_size, seq_len, num_features)  #Reshape back

        #Concatenate all inputs (numeric + transformed categorical + embeddings)
        x = torch.cat([
            x_num,  #Numeric features
            x_low_card_cats,  #Transformed low-card categorical features
            inning_vecs,
            pitcher_vecs, batter_vecs, desc_vecs, event_vecs, prev_pitch_vecs
        ], dim=2)

        #Handle packed sequences for LSTM
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        #LSTM output → Multi-head predictions
        bs, sl, hd = lstm_out.shape
        lstm_out_2d = lstm_out.view(bs * sl, hd)

        pitch_type_logits = self.pitch_type_head(lstm_out_2d).view(bs, sl, -1)
        pitch_cont_values = self.pitch_cont_head(lstm_out_2d).view(bs, sl, -1)
        pitch_result_desc = self.pitch_result_head_desc(lstm_out_2d).view(bs, sl, -1)
        pitch_result_event = self.pitch_result_head_event(lstm_out_2d).view(bs, sl, -1)

        return pitch_type_logits, pitch_cont_values, pitch_result_desc, pitch_result_event
