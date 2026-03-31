import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

dataset = "TrackNet/Dataset/"
test_clip = "Clip1"  # Clip1 for testing
VIDEO_WIDTH = 1280.0
VIDEO_HEIGHT = 720.0
RESIZED_HEIGHT = 288
RESIZED_WIDTH = 512
DEFAULT_IMAGE_SIZE = (RESIZED_HEIGHT, RESIZED_WIDTH)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_image_size(image_size):
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError("image_size must be an int or a tuple/list of (height, width).")


def build_vision_transform(image_size=DEFAULT_IMAGE_SIZE, augment=False):
    target_size = _resolve_image_size(image_size)
    transform_ops = [transforms.Resize(target_size)]

    if augment:
        transform_ops.extend([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05))
            ], p=0.6),
        ])

    transform_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if augment:
        transform_ops.append(transforms.RandomErasing(p=0.15, scale=(0.01, 0.06), ratio=(0.3, 3.3)))

    return transforms.Compose(transform_ops)


def _window_starts(sequence_len, input_steps=10, output_steps=10, stride=1):
    total_steps = input_steps + output_steps
    if sequence_len < total_steps:
        return []
    return range(0, sequence_len - total_steps + 1, max(1, int(stride)))


def _group_train_val_indices(group_ids, val_ratio=0.1, random_seed=42):
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)

    if len(unique_groups) <= 1:
        return np.arange(len(group_ids)), np.array([], dtype=np.int64)

    rng = np.random.default_rng(random_seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    val_group_count = max(1, int(round(len(unique_groups) * val_ratio)))
    val_group_count = min(val_group_count, len(unique_groups) - 1)
    val_groups = set(shuffled_groups[:val_group_count])

    val_mask = np.array([g in val_groups for g in group_ids], dtype=bool)
    val_indices = np.where(val_mask)[0]
    train_indices = np.where(~val_mask)[0]

    if len(train_indices) == 0 and len(val_indices) > 0:
        train_indices = val_indices[:1]
        val_indices = val_indices[1:]

    return train_indices, val_indices


def _random_train_val_indices(total_samples, val_ratio=0.1, random_seed=42):
    if total_samples <= 1:
        return np.arange(total_samples), np.array([], dtype=np.int64)

    indices = np.arange(total_samples)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(total_samples * val_ratio)))
    val_size = min(val_size, total_samples - 1)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def _index_data(data, indices):
    if isinstance(data, np.ndarray):
        return data[indices]
    return [data[i] for i in indices]


def normalize_coordinate_df(df):
    df['x-coordinate'] = df['x-coordinate'] / VIDEO_WIDTH
    df['y-coordinate'] = df['y-coordinate'] / VIDEO_HEIGHT
    return df

# get trajectory data for the LSTM model: use frames 0-9 as input to predict frames 10-19
def getTrajectoryTrainData(window_stride=1, return_groups=False):
    input_data = []
    output_data = []
    group_ids = []
    for game_folder in os.listdir(dataset):
        if game_folder.startswith("game"):
            for clip_folder in os.listdir(os.path.join(dataset, game_folder)):
                if clip_folder.startswith("Clip") and clip_folder != test_clip:
                    filename = os.path.join(dataset, game_folder, clip_folder, "Label.csv")
                    df = pd.read_csv(filename)
                    # convert "None" strings to actual np.nan float values
                    df['x-coordinate'] = pd.to_numeric(df['x-coordinate'], errors='coerce')
                    df['y-coordinate'] = pd.to_numeric(df['y-coordinate'], errors='coerce')
                    df = normalize_coordinate_df(df)

                    # interpolate missing values so the LSTM has continuous input
                    df['x-coordinate'] = df['x-coordinate'].interpolate(method='linear').bfill().ffill()
                    df['y-coordinate'] = df['y-coordinate'].interpolate(method='linear').bfill().ffill()
                    clip_id = f"{game_folder}/{clip_folder}"
                    for i in _window_starts(len(df), input_steps=10, output_steps=10, stride=window_stride):
                        input_data.append(df.iloc[i:i+10][['x-coordinate', 'y-coordinate']].values)
                        output_data.append(df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values)
                        group_ids.append(clip_id)

    input_arr = np.array(input_data, dtype=np.float32)
    output_arr = np.array(output_data, dtype=np.float32)
    if return_groups:
        return input_arr, output_arr, np.array(group_ids)
    return input_arr, output_arr


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
                    df = normalize_coordinate_df(df)
                    
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


def getVisionTrainData(window_stride=1, return_groups=False):
    input_paths = []
    output_data = []
    group_ids = []
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
                    df = normalize_coordinate_df(df)
                    df['x-coordinate'] = df['x-coordinate'].interpolate(method='linear').bfill().ffill()
                    df['y-coordinate'] = df['y-coordinate'].interpolate(method='linear').bfill().ffill()

                    clip_id = f"{game_folder}/{clip_folder}"
                    for i in _window_starts(len(img_files), input_steps=10, output_steps=10, stride=window_stride):
                        input_imgs = []
                        for j in range(i, i+10):
                            # Store the PATH, not the image itself
                            img_path = os.path.join(clip_path, img_files[j])
                            input_imgs.append(img_path)
                            
                        input_paths.append(input_imgs)
                        output_data.append(df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values)
                        group_ids.append(clip_id)

    # Return lists of paths, and numpy arrays for targets
    output_arr = np.array(output_data, dtype=np.float32)
    if return_groups:
        return input_paths, output_arr, np.array(group_ids)
    return input_paths, output_arr

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
                    df = normalize_coordinate_df(df)
                    
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


class VisionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, image_size=DEFAULT_IMAGE_SIZE, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or build_vision_transform(image_size=image_size, augment=augment)

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


def getFusionTrainData(window_stride=1, return_groups=False):
    input_paths = []
    input_coords = []
    output_data = []
    group_ids = []
    
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
                    df = normalize_coordinate_df(df)
                    df['x-coordinate'] = df['x-coordinate'].interpolate(method='linear').bfill().ffill()
                    df['y-coordinate'] = df['y-coordinate'].interpolate(method='linear').bfill().ffill()

                    clip_id = f"{game_folder}/{clip_folder}"
                    for i in _window_starts(len(img_files), input_steps=10, output_steps=10, stride=window_stride):
                        # 1. Image Paths
                        input_imgs = []
                        for j in range(i, i+10):
                            input_imgs.append(os.path.join(clip_path, img_files[j]))
                        
                        input_paths.append(input_imgs)
                        input_coords.append(df.iloc[i:i+10][['x-coordinate', 'y-coordinate']].values)  # Trajectory input
                        output_data.append(df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values)  # Trajectory output
                        group_ids.append(clip_id)

    input_tuple = (input_paths, np.array(input_coords, dtype=np.float32))
    output_arr = np.array(output_data, dtype=np.float32)
    if return_groups:
        return input_tuple, output_arr, np.array(group_ids)
    return input_tuple, output_arr


def getFusionTestData():
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
                    df = normalize_coordinate_df(df)
                    
                    input_paths = []
                    input_coords = []
                    output_data = []
                    
                    for i in range(0, len(img_files) - 20, 20):
                        input_imgs = []
                        for j in range(i, i+10):
                            input_imgs.append(os.path.join(clip_path, img_files[j]))
                            
                        in_slice = df.iloc[i:i+10][['x-coordinate', 'y-coordinate']].copy()
                        in_slice['x-coordinate'] = in_slice['x-coordinate'].interpolate(method='linear').bfill().ffill()
                        in_slice['y-coordinate'] = in_slice['y-coordinate'].interpolate(method='linear').bfill().ffill()
                        
                        out_slice = df.iloc[i+10:i+20][['x-coordinate', 'y-coordinate']].values
                        
                        input_paths.append(input_imgs)
                        input_coords.append(in_slice.values)
                        output_data.append(out_slice)
                    
                    test_data_dict[game_folder] = (
                        (input_paths, np.array(input_coords, dtype=np.float32)), 
                        np.array(output_data, dtype=np.float32)
                    )

    return test_data_dict


class FusionDataset(Dataset):
    def __init__(self, input_data, target_coords, transform=None, image_size=DEFAULT_IMAGE_SIZE, augment=False):
        self.image_paths, self.input_coords = input_data
        self.target_coords = target_coords
        self.transform = transform or build_vision_transform(image_size=image_size, augment=augment)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        window_paths = self.image_paths[idx]
        in_coords = self.input_coords[idx]
        out_coords = self.target_coords[idx]

        frames = []
        for path in window_paths:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames)
        in_coords_tensor = torch.tensor(in_coords, dtype=torch.float32)
        out_coords_tensor = torch.tensor(out_coords, dtype=torch.float32)

        # Return a tuple of ((images, input_trajectory), target_trajectory)
        return (frames_tensor, in_coords_tensor), out_coords_tensor


# split data into training and validation sets for regular input (list/array)
def splitTrainValRegular(input_data, output_data, val_ratio=0.1, group_ids=None, random_seed=42):
    total_samples = len(input_data)

    if group_ids is not None and len(group_ids) == total_samples:
        train_indices, val_indices = _group_train_val_indices(group_ids, val_ratio, random_seed)
    else:
        train_indices, val_indices = _random_train_val_indices(total_samples, val_ratio, random_seed)

    train_input = _index_data(input_data, train_indices)
    train_output = _index_data(output_data, train_indices)
    val_input = _index_data(input_data, val_indices)
    val_output = _index_data(output_data, val_indices)

    return train_input, train_output, val_input, val_output


# split data into training and validation sets for fusion input tuple
def splitTrainValFusion(input_data, output_data, val_ratio=0.1, group_ids=None, random_seed=42):
    image_paths, input_coords = input_data
    total_samples = len(image_paths)

    if group_ids is not None and len(group_ids) == total_samples:
        train_indices, val_indices = _group_train_val_indices(group_ids, val_ratio, random_seed)
    else:
        train_indices, val_indices = _random_train_val_indices(total_samples, val_ratio, random_seed)

    train_input = (_index_data(image_paths, train_indices), _index_data(input_coords, train_indices))
    val_input = (_index_data(image_paths, val_indices), _index_data(input_coords, val_indices))
    train_output = _index_data(output_data, train_indices)
    val_output = _index_data(output_data, val_indices)

    return train_input, train_output, val_input, val_output


# Backward-compatible wrapper
def splitTrainVal(input_data, output_data, val_ratio=0.1, group_ids=None, random_seed=42):
    if isinstance(input_data, (tuple, list)) and len(input_data) == 2:
        return splitTrainValFusion(input_data, output_data, val_ratio, group_ids=group_ids, random_seed=random_seed)
    return splitTrainValRegular(input_data, output_data, val_ratio, group_ids=group_ids, random_seed=random_seed)