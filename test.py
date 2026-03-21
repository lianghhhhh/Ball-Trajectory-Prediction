import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import VisionDataset

def testTrajectoryModel(model, test_data_dict, device):
    """
    Evaluates the model on a single clip (one game's test data).
    input_data: numpy array of shape (num_windows, 10, 2)
    output_data: numpy array of shape (num_windows, 10, 2)
    """
    model.eval()
    model.to(device)

    game_ade = {}
    game_fde = {}

    with torch.no_grad():
        for game_folder, (input_data, output_data) in test_data_dict.items():
            ade_list = []
            fde_list = []
            for i in range(len(input_data)):
                input_tensor = torch.tensor(input_data[i:i+1], dtype=torch.float32).to(device)  # Shape (1, 10, 2)
                output_tensor = torch.tensor(output_data[i:i+1], dtype=torch.float32).to(device)  # Shape (1, 10, 2)
                predictions = model(input_tensor)

                # Mask out the NaN values in the output for ADE/FDE calculation
                mask = ~torch.isnan(output_tensor)
                masked_predictions = predictions[mask].cpu().numpy().reshape(-1, 2)
                masked_output = output_tensor[mask].cpu().numpy().reshape(-1, 2)
                if masked_output.shape[0] > 0:  # Only calculate if there are valid points
                    ade = np.mean(np.linalg.norm(masked_predictions - masked_output, axis=1))
                    fde = np.linalg.norm(masked_predictions[-1] - masked_output[-1])
                    ade_list.append(ade)
                    fde_list.append(fde)
            game_ade[game_folder] = np.mean(ade_list) if ade_list else float('nan')
            game_fde[game_folder] = np.mean(fde_list) if fde_list else float('nan')

    print("Test Results:")
    for game_folder in game_ade.keys():
        print(f"{game_folder}: ADE = {game_ade[game_folder]:.4f}, FDE = {game_fde[game_folder]:.4f}")
    print(f"Average ADE: {np.nanmean(list(game_ade.values())):.4f}, Average FDE: {np.nanmean(list(game_fde.values())):.4f}")


def testVisionModel(model, test_data_dict, device):
    model.eval()
    model.to(device)

    game_ade = {}
    game_fde = {}

    with torch.no_grad():
        for game_folder, (input_data, output_data) in test_data_dict.items():
            ade_list = []
            fde_list = []
            
            # Use custom dataset for Vision Model testing
            dataset = VisionDataset(input_data, output_data)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            for input_tensor, output_tensor in loader:
                input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)
                predictions = model(input_tensor)
                
                mask = ~torch.isnan(output_tensor)
                masked_predictions = predictions[mask].cpu().numpy().reshape(-1, 2)
                masked_output = output_tensor[mask].cpu().numpy().reshape(-1, 2)
                
                if masked_output.shape[0] > 0:
                    ade = np.mean(np.linalg.norm(masked_predictions - masked_output, axis=1))
                    fde = np.linalg.norm(masked_predictions[-1] - masked_output[-1])
                    ade_list.append(ade)
                    fde_list.append(fde)

            game_ade[game_folder] = np.mean(ade_list) if ade_list else float('nan')
            game_fde[game_folder] = np.mean(fde_list) if fde_list else float('nan')

    print("Test Results:")
    for game_folder in game_ade.keys():
        print(f"{game_folder}: ADE = {game_ade[game_folder]:.4f}, FDE = {game_fde[game_folder]:.4f}")
    print(f"Average ADE: {np.nanmean(list(game_ade.values())):.4f}, Average FDE: {np.nanmean(list(game_fde.values())):.4f}")