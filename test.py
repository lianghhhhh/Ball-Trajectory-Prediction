import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import VisionDataset, FusionDataset, VIDEO_WIDTH, VIDEO_HEIGHT

def denormalize_points(points):
    points_px = points.copy()
    points_px[:, 0] *= VIDEO_WIDTH
    points_px[:, 1] *= VIDEO_HEIGHT
    return points_px

def testTrajectoryModel(model, test_data_dict, device):
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
                    masked_predictions_px = denormalize_points(masked_predictions)
                    masked_output_px = denormalize_points(masked_output)
                    ade = np.mean(np.linalg.norm(masked_predictions_px - masked_output_px, axis=1))
                    fde = np.linalg.norm(masked_predictions_px[-1] - masked_output_px[-1])
                    ade_list.append(ade)
                    fde_list.append(fde)
            game_ade[game_folder] = np.mean(ade_list) if ade_list else float('nan')
            game_fde[game_folder] = np.mean(fde_list) if fde_list else float('nan')

    print("Test Results:")
    for game_folder in game_ade.keys():
        print(f"{game_folder}: ADE = {game_ade[game_folder]:.4f}, FDE = {game_fde[game_folder]:.4f}")
    print(f"Average ADE: {np.nanmean(list(game_ade.values())):.4f}, Average FDE: {np.nanmean(list(game_fde.values())):.4f}")

    # plot trajectory for the last data point of the last game folder
    last_game_folder = list(test_data_dict.keys())[-1]
    input_data, output_data = test_data_dict[last_game_folder]
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    predictions = model(input_tensor).cpu().detach().numpy()
    plt.figure(figsize=(8, 6))
    for i in range(predictions.shape[0]):
        observed_traj = denormalize_points(input_data[i])
        pred_traj = denormalize_points(predictions[i])
        true_traj = denormalize_points(output_data[i])
        plt.plot(observed_traj[:, 0], observed_traj[:, 1], 'g', label='Observed Trajectory' if i == 0 else "")
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b', label='Predicted Trajectory' if i == 0 else "")
        plt.plot(true_traj[:, 0], true_traj[:, 1], 'r', label='True Trajectory' if i == 0 else "")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Prediction")
    plt.legend()
    plt.show()


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
                    masked_predictions_px = denormalize_points(masked_predictions)
                    masked_output_px = denormalize_points(masked_output)
                    ade = np.mean(np.linalg.norm(masked_predictions_px - masked_output_px, axis=1))
                    fde = np.linalg.norm(masked_predictions_px[-1] - masked_output_px[-1])
                    ade_list.append(ade)
                    fde_list.append(fde)

            game_ade[game_folder] = np.mean(ade_list) if ade_list else float('nan')
            game_fde[game_folder] = np.mean(fde_list) if fde_list else float('nan')

    print("Test Results:")
    for game_folder in game_ade.keys():
        print(f"{game_folder}: ADE = {game_ade[game_folder]:.4f}, FDE = {game_fde[game_folder]:.4f}")
    print(f"Average ADE: {np.nanmean(list(game_ade.values())):.4f}, Average FDE: {np.nanmean(list(game_fde.values())):.4f}")


def testFusionModel(model, test_data_dict, device):
    model.eval()
    model.to(device)

    game_ade = {}
    game_fde = {}

    with torch.no_grad():
        for game_folder, (input_data, output_data) in test_data_dict.items():
            ade_list = []
            fde_list = []
            
            # Use custom dataset for Fusion Model testing
            dataset = FusionDataset(input_data, output_data)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            for (image_tensor, traj_tensor), output_tensor in loader:
                image_tensor = image_tensor.to(device)
                traj_tensor = traj_tensor.to(device)
                output_tensor = output_tensor.to(device)
                predictions = model(image_tensor, traj_tensor)
                
                mask = ~torch.isnan(output_tensor)
                masked_predictions = predictions[mask].cpu().numpy().reshape(-1, 2)
                masked_output = output_tensor[mask].cpu().numpy().reshape(-1, 2)
                
                if masked_output.shape[0] > 0:
                    masked_predictions_px = denormalize_points(masked_predictions)
                    masked_output_px = denormalize_points(masked_output)
                    ade = np.mean(np.linalg.norm(masked_predictions_px - masked_output_px, axis=1))
                    fde = np.linalg.norm(masked_predictions_px[-1] - masked_output_px[-1])
                    ade_list.append(ade)
                    fde_list.append(fde)

            game_ade[game_folder] = np.mean(ade_list) if ade_list else float('nan')
            game_fde[game_folder] = np.mean(fde_list) if fde_list else float('nan')

    print("Test Results:")
    for game_folder in game_ade.keys():
        print(f"{game_folder}: ADE = {game_ade[game_folder]:.4f}, FDE = {game_fde[game_folder]:.4f}")
    print(f"Average ADE: {np.nanmean(list(game_ade.values())):.4f}, Average FDE: {np.nanmean(list(game_fde.values())):.4f}")