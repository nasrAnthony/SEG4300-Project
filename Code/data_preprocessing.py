import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import DATA_PATH, TRAIN_FILE, TEST_FILE
X_NUMERICAL_IDX = [8, 9, 16, 17, 18, 19, 20]
#Define valid ranges for selected numeric features
VALID_RANGES = {
    "release_speed": (50, 110),        #MPH
    "release_spin_rate": (500, 3700),  #RPM
    "plate_x": (-3, 3),                #Feet
    "plate_z": (-1, 4)                 #Feet
}

#LOAD DATASET
def load_data(file_path=DATA_PATH):
    df = pd.read_csv(file_path)
    return df

#SORT AND GROUP TO BUILD SEQUENCES
def sort_n_group(df):
    #Sort by game, at_bat_number, and atbat_pitch_number
    df = df.sort_values(by=['game_pk', 'at_bat_number', 'atbat_pitch_number'])
    #Preliminary encoding
    df["stand"] = df["stand"].map({"L": 0, "R": 1})
    df["p_throws"] = df["p_throws"].map({"L": 0, "R": 1})
    df["inning"] = df["inning"] - 1
    df["inning_topbot"] = df["inning_topbot"].map({"Top": 0, "Bot": 1})
    df["on_1b"] = df["on_1b"].astype(int)
    df["on_2b"] = df["on_2b"].astype(int)
    df["on_3b"] = df["on_3b"].astype(int)
    return df

def compute_feature_medians(df):
    """
    Computes median values for all numeric features that require imputation.
    
    Returns:
      A dictionary containing median values for each feature.
    """
    numeric_cols = ['release_speed', 'release_spin_rate', 'release_extension', 
                    'plate_x', 'plate_z']

    median_values = {}
    for col in numeric_cols:
        median_values[col] = df[col].median()  #Compute median for each column

    return median_values

def build_seqs(df, median_values):
    """
    Returns: 
      X_sequences: list of at-bat sequences, each is a list of x_t vectors
      Y_sequences: list of at-bat sequences, each is a list of dicts with keys 
                   { 'type': ..., 'cont': [...], 'result': ... }
    """
    X_sequences = []
    Y_sequences = []

    grouped = df.groupby(['game_pk', 'at_bat_number'], as_index=False)

    dropped_count = 0 #Track at-bats dropped due to NaN
    dropped_count_outliers = 0  #Track at-bats dropped due to outliers
    dropped_count_invalid_results = 0  # Track at-bats dropped due to invalid results

    INVALID_RESULTS = {'strikeout', 'strikeout_double_play', 'truncated_pa', 'walk'} #Invalid results for last-pitch result while in same at-bat (these values should only appear in Y)

    for (game_id, ab_id), group in grouped:
        group = group.sort_values('atbat_pitch_number').reset_index(drop=True)

        x_seq = []
        y_seq = []

        for i in range(len(group)):
            row = group.iloc[i]

            is_first_pitch = 1 if i == 0 else 0  

            #Current pre-pitch features
            pre_pitch_feats = [
                row['balls'],
                row['strikes'],
                row['outs_when_up'],
                row['on_1b'],
                row['on_2b'],
                row['on_3b'],
                row['inning'],
                row['inning_topbot'],
                row['home_score'],
                row['away_score'],
                row['stand'],
                row['p_throws'],
                row['pitcher'],
                row['batter'],
                is_first_pitch
            ]

            #Previous pitch features
            # For i=0 (the first pitch in the at-bat), there's no "previous pitch" so use dummy
            if i == 0:
                last_pitch_feats = ['NONE', median_values['release_speed'], median_values['release_spin_rate'], median_values['release_extension'], 
                                    median_values['plate_x'], median_values['plate_z'], 'NONE', 'NONE']
            else:
                prev = group.iloc[i-1]
                last_pitch_feats = [
                    prev['pitch_type'],
                    prev['release_speed'],
                    prev['release_spin_rate'],
                    prev['release_extension'],
                    prev['plate_x'],
                    prev['plate_z'],
                    prev['events'],
                    prev['description']
                ]

            x_t = pre_pitch_feats + last_pitch_feats

            #Current pitch outputs (y_t) stored as dict
            pitch_type = row['pitch_type']
            pitch_cont = [
                row['release_speed'],
                row['release_spin_rate'],
                row['release_extension'],
                row['plate_x'],
                row['plate_z']
            ]
            pitch_result = [
                row['description'],
                row['events']
            ]

            y_t = {
                'type': pitch_type,
                'cont': pitch_cont,
                'result': pitch_result
            }

            x_seq.append(x_t)
            y_seq.append(y_t)

        
        #Check if any pitch in this at-bat has an invalid result at index 21
        contains_invalid_result = any(pitch[21] in INVALID_RESULTS for pitch in x_seq)
        if contains_invalid_result:
            dropped_count_invalid_results += 1
            continue  #Skip this at-bat

        #Check for NaNs in X (numeric features)
        x_numeric_only = np.array([[pitch[i] for i in X_NUMERICAL_IDX] for pitch in x_seq], dtype=np.float64)

        #Check for NaNs in Y_cont (all numeric values)
        y_cont_only = np.array([y_t['cont'] for y_t in y_seq], dtype=np.float64)

        #If any NaNs exist in X or Y_cont, drop this at-bat
        if np.isnan(x_numeric_only).any() or np.isnan(y_cont_only).any():
            dropped_count += 1  #Increment dropped at-bats
            continue  #Skip this at-bat

        #Check for outliers in critical numeric features
        outlier_found = False
        
        #Check X features for outliers
        for pitch in x_seq:
            speed, spin_rate, _, plate_x, plate_z = pitch[16], pitch[17], pitch[18], pitch[19], pitch[20]

            if not (VALID_RANGES["release_speed"][0] <= speed <= VALID_RANGES["release_speed"][1]):
                outlier_found = True
                break
            if not (VALID_RANGES["release_spin_rate"][0] <= spin_rate <= VALID_RANGES["release_spin_rate"][1]):
                outlier_found = True
                break
            if not (VALID_RANGES["plate_x"][0] <= plate_x <= VALID_RANGES["plate_x"][1]):
                outlier_found = True
                break
            if not (VALID_RANGES["plate_z"][0] <= plate_z <= VALID_RANGES["plate_z"][1]):
                outlier_found = True
                break

        #Check Y features for outliers
        for pitch in y_seq:
            speed, spin_rate, _, plate_x, plate_z = pitch['cont']
            
            if not (VALID_RANGES["release_speed"][0] <= speed <= VALID_RANGES["release_speed"][1]):
                outlier_found = True
                break
            if not (VALID_RANGES["release_spin_rate"][0] <= spin_rate <= VALID_RANGES["release_spin_rate"][1]):
                outlier_found = True
                break
            if not (VALID_RANGES["plate_x"][0] <= plate_x <= VALID_RANGES["plate_x"][1]):
                outlier_found = True
                break
            if not (VALID_RANGES["plate_z"][0] <= plate_z <= VALID_RANGES["plate_z"][1]):
                outlier_found = True
                break

        if outlier_found:
            dropped_count_outliers += 1
            continue  #Skip this at-bat due to outliers

        X_sequences.append(x_seq)
        Y_sequences.append(y_seq)

    print(f"Finished sequence building. Dropped {dropped_count} at-bats with NaN values.")
    print(f"Dropped {dropped_count_outliers} at-bats due to outliers.")
    print(f"Dropped {dropped_count_invalid_results} at-bats due to invalid results: {INVALID_RESULTS}")

    return X_sequences, Y_sequences

def encode_and_scale(X_sequences, Y_sequences):
    """
    Encodes categorical features and scales numerical features for both X and Y sequences.
    
    Parameters:
        X_sequences (list): Nested list of pitch sequences with numerical and categorical features.
        Y_sequences (list): Nested list of dictionaries containing type, cont (5 features), result (desc, event).
        num_indices (list): Indices of numerical features in X.
        cat_indices (list): Indices of categorical features in X.
    
    Returns:
        processed_X (list): Transformed X sequences with encoded and scaled features.
        processed_Y (dict): Dictionary containing transformed Y components.
    """

    num_indices = [8, 9, 16, 17, 18, 19, 20]
    cat_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 21, 22]
    
    # Flatten X and Y for transformation
    X_all, Y_type_all, Y_cont_all, Y_result_desc_all, Y_result_event_all = [], [], [], [], []

    for x_seq, y_seq in zip(X_sequences, Y_sequences):
        for x_t, y_t in zip(x_seq, y_seq):
            X_all.append(x_t)
            Y_type_all.append(y_t['type'])
            Y_cont_all.append(y_t['cont'])  # Expecting list of 5 numerical values
            Y_result_desc_all.append(y_t['result'][0])
            Y_result_event_all.append(y_t['result'][1])
    
    # Convert X into DataFrame
    X_df = pd.DataFrame(X_all)

    # Get column names using indices
    num_cols = [X_df.columns[i] for i in num_indices]
    cat_cols = [X_df.columns[i] for i in cat_indices]

    # Step 1: Encode categorical features in X
    label_encoders_X = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])  # Encode categorical column
        label_encoders_X[col] = le  # Store for potential inverse transform

    # Step 2: Scale numerical features in X
    scaler_X = StandardScaler()
    X_df[num_cols] = scaler_X.fit_transform(X_df[num_cols])

    # Step 3: Process Y components
    le_Y_type = LabelEncoder()
    Y_type_encoded = le_Y_type.fit_transform(Y_type_all)

    # Ensure Y_cont_all is a 2D array for scaling (shape: (num_samples, 5))
    Y_cont_array = np.array(Y_cont_all)  # Shape should be (num_samples, 5)
    
    # Check if Y_cont_array has correct dimensions
    if len(Y_cont_array.shape) == 1:  # If mistakenly 1D, reshape
        Y_cont_array = Y_cont_array.reshape(-1, 5)

    # Scale Y_cont (each column separately)
    scaler_Y_cont = StandardScaler()
    Y_cont_scaled = scaler_Y_cont.fit_transform(Y_cont_array)

    le_Y_result_desc = LabelEncoder()
    Y_result_desc_encoded = le_Y_result_desc.fit_transform(Y_result_desc_all)

    le_Y_result_event = LabelEncoder()
    Y_result_event_encoded = le_Y_result_event.fit_transform(Y_result_event_all)

    # Step 4: Rebuild sequences for X
    index = 0
    processed_X = []
    for at_bat in X_sequences:
        new_at_bat = []
        for _ in at_bat:
            new_at_bat.append(X_df.iloc[index].values.tolist())
            index += 1
        processed_X.append(new_at_bat)

    # Step 5: Rebuild sequences for Y
    processed_Y = []

    index = 0
    for at_bat in Y_sequences:
        num_pitches = len(at_bat)
        at_bat_data = []  #Store pitches for this at-bat

        for j in range(num_pitches):
            pitch_data = {
                'type': Y_type_encoded[index + j],  #Extract single pitch
                'cont': Y_cont_scaled[index + j].tolist(),
                'result_desc': Y_result_desc_encoded[index + j],
                'result_event': Y_result_event_encoded[index + j]
            }
            at_bat_data.append(pitch_data)

        processed_Y.append(at_bat_data)
        index += num_pitches

    return processed_X, processed_Y, label_encoders_X

#TRAIN AND TEST SPLIT
def split_data(X_encoded, Y_encoded, test_size=0.2, random_state=42):
    # We'll treat each at-bat as one data point for splitting
    indices = np.arange(len(X_encoded))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    X_train = [X_encoded[i] for i in train_idx]
    Y_train = [Y_encoded[i] for i in train_idx]
    X_test  = [X_encoded[i] for i in test_idx]
    Y_test  = [Y_encoded[i] for i in test_idx]

    torch.save((X_train, Y_train), TRAIN_FILE)
    torch.save((X_test, Y_test), TEST_FILE)
    
    return X_train, X_test, Y_train, Y_test
