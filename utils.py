import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

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


def getVisionTrainData():
    input_paths = []
    output_data = []
    for game_folder in os.listdir(dataset):
        if game_folder.startswith("game"):
            for clip_folder in os.listdir(os.path.join(dataset, game_folder)):
                if clip_folder.startswith("Clip") and clip_folder != test_clip:
                    clip_path = os.path.join(dataset, game_folder, clip_folder)
                    img_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])
                    
                    label_file = os.path.join(clip_path, "Label.csv")
                    df = pd.read_csv(label_file)
                    
                    df['x-coordinate'] = pd.to_numeric(df['x-coordinate'], errors='coerce')
                    df['y-coordinate'] = pd.to_numeric(df['y-coordinate'], errors='coerce')
                    df['x-coordinate'] = df['x-coordinate'].interpolate(method='linear').bfill().ffill()
                    df['y-coordinate'] = df['y-coordinate'].interpolate(method='linear').bfill().ffill()

                    for i in range(0, len(img_files) - 20, 20):
                        input_imgs = []
                        for j in range(i, i+10):
                            # Store the PATH, not the image itself
                            img_path = os.path.join(clip_path, img_files[j])
                            input_imgs.append(img_path)
                            
                        input_paths.append(input_imgs)
                        output_data.append(df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values)

    # Return lists of paths, and numpy arrays for targets
    return input_paths, np.array(output_data, dtype=np.float32)

def getVisionTestData():
    test_data_dict = {}
    for game_folder in os.listdir(dataset):
        if game_folder.startswith("game"):
            for clip_folder in os.listdir(os.path.join(dataset, game_folder)):
                if clip_folder.startswith("Clip") and clip_folder == test_clip:
                    clip_path = os.path.join(dataset, game_folder, clip_folder)
                    img_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])
                    
                    label_file = os.path.join(clip_path, "Label.csv")
                    df = pd.read_csv(label_file)
                    df['x-coordinate'] = pd.to_numeric(df['x-coordinate'], errors='coerce')
                    df['y-coordinate'] = pd.to_numeric(df['y-coordinate'], errors='coerce')
                    
                    input_paths = []
                    output_data = []
                    
                    for i in range(0, len(img_files) - 20, 20):
                        input_imgs = []
                        for j in range(i, i+10):
                            img_path = os.path.join(clip_path, img_files[j])
                            input_imgs.append(img_path)
                            
                        input_paths.append(input_imgs)
                        output_data.append(df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values) 
                    
                    test_data_dict[game_folder] = (input_paths, np.array(output_data, dtype=np.float32))

    return test_data_dict

# --- NEW CUSTOM DATASET ---
class VisionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        # Resize images to 224x224 to save memory and speed up training
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Standard normalization for CNNs
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        window_paths = self.image_paths[idx]
        label = self.labels[idx]

        frames = []
        for path in window_paths:
            # Load image on the fly
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack the 10 frames into a tensor of shape (10, C, H, W)
        frames_tensor = torch.stack(frames)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return frames_tensor, label_tensor


# split data into training and validation sets
def splitTrainVal(input_data, output_data, val_ratio=0.1):
    total_samples = len(input_data)
    val_size = int(total_samples * val_ratio)

    train_input = input_data[:-val_size]
    train_output = output_data[:-val_size]
    val_input = input_data[-val_size:]
    val_output = output_data[-val_size:]

    return train_input, train_output, val_input, val_output