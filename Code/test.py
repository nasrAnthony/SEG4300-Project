import torch
from data_preprocessing import load_data, sort_n_group, build_seqs, encode_and_scale, test_encoding_scaling
from config import DATA_PATH

df = load_data(DATA_PATH)
df = sort_n_group(df)
X_sequences , Y_sequences = build_seqs(df)

# 🔹 Run Encoding & Scaling
X_encoded, Y_encoded, stand_encoder, pthrows_encoder, pitcher_encoder, batter_encoder, \
    pitch_type_encoder, pitch_result_desc_encoder, pitch_result_event_encoder, scaler, cont_scaler = \
    encode_and_scale(X_sequences, Y_sequences)

# 🔹 Create dictionary of encoders
encoders = {
    "stand": stand_encoder,
    "pthrows": pthrows_encoder,
    "pitcher": pitcher_encoder,
    "batter": batter_encoder,
    "last_pitch_type": pitch_type_encoder,
    "last_result_desc": pitch_result_desc_encoder,
    "last_result_event": pitch_result_event_encoder
}

# 🔹 Run the test function
test_encoding_scaling(X_sequences, Y_sequences, encoders, scaler)