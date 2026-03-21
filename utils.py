import os
import pandas as pd
import numpy as np

dataset = "TrackNet/Dataset/"
test_clip = "Clip1"  # Clip1 for testing

# get trajectory data for the LSTM model: use frames 0-9 as input to predict frames 10-19
def getTrajectoryTrainData():
    input_data = []
    output_data = []
    for game_folder in os.listdir(dataset):
        if game_folder.startswith("game"):
            for clip_folder in os.listdir(os.path.join(dataset, game_folder)):
                if clip_folder.startswith("Clip") and clip_folder != test_clip:
                    filename = os.path.join(dataset, game_folder, clip_folder, "Label.csv")
                    df = pd.read_csv(filename)
                    # convert "None" strings to actual np.nan float values
                    df['x-coordinate'] = pd.to_numeric(df['x-coordinate'], errors='coerce')
                    df['y-coordinate'] = pd.to_numeric(df['y-coordinate'], errors='coerce')

                    # interpolate missing values so the LSTM has continuous input
                    df['x-coordinate'] = df['x-coordinate'].interpolate(method='linear').bfill().ffill()
                    df['y-coordinate'] = df['y-coordinate'].interpolate(method='linear').bfill().ffill()
                    for i in range(0, len(df) - 20, 20):
                        input_data.append(df.iloc[i:i+10][['x-coordinate', 'y-coordinate']].values)
                        output_data.append(df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values)

    return np.array(input_data, dtype=np.float32), np.array(output_data, dtype=np.float32)


def getTrajectoryTestData():
    test_data_dict = {}
    for game_folder in os.listdir(dataset):
        if game_folder.startswith("game"):
            for clip_folder in os.listdir(os.path.join(dataset, game_folder)):
                if clip_folder.startswith("Clip") and clip_folder == test_clip:
                    filename = os.path.join(dataset, game_folder, clip_folder, "Label.csv")
                    df = pd.read_csv(filename)
                    df['x-coordinate'] = pd.to_numeric(df['x-coordinate'], errors='coerce')
                    df['y-coordinate'] = pd.to_numeric(df['y-coordinate'], errors='coerce')
                    
                    input_data = []
                    output_data = []
                    
                    # Slice into non-overlapping windows of 20 frames
                    for i in range(0, len(df) - 20, 20):
                        in_slice = df.iloc[i:i+10][['x-coordinate', 'y-coordinate']].copy()
                        out_slice = df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values
                        
                        # Interpolate ONLY the inputs so the LSTM has a continuous sequence
                        in_slice['x-coordinate'] = in_slice['x-coordinate'].interpolate(method='linear').bfill().ffill()
                        in_slice['y-coordinate'] = in_slice['y-coordinate'].interpolate(method='linear').bfill().ffill()
                        
                        input_data.append(in_slice.values)
                        output_data.append(out_slice) # Output retains the NaNs for masking!
                    
                    # Store the numpy arrays in the dictionary using the game folder name as the key
                    test_data_dict[game_folder] = (
                        np.array(input_data, dtype=np.float32), 
                        np.array(output_data, dtype=np.float32)
                    )

    return test_data_dict


# split data into training and validation sets
def splitTrainVal(input_data, output_data, val_ratio=0.1):
    total_samples = len(input_data)
    val_size = int(total_samples * val_ratio)

    train_input = input_data[:-val_size]
    train_output = output_data[:-val_size]
    val_input = input_data[-val_size:]
    val_output = output_data[-val_size:]

    return train_input, train_output, val_input, val_output