# a neural network model using car's current velocity and delta state to predict target velocity
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils import getTrajectoryTrainData, getTrajectoryTestData, getVisionTrainData, getVisionTestData
from trajectory_model import TrajectoryModel
from vision_model import VisionModel
from fusion_model import FusionModel
from train import trainTrajectoryModel, trainVisionModel
from test import testTrajectoryModel, testVisionModel

def selectMode():
    print("Select mode:")
    print("1. Train trajectory-only model")
    print("2. Inference trajectory-only model")
    print("3. Train vision-only model")
    print("4. Inference vision-only model")
    print("5. Train fusion model")
    print("6. Inference fusion model")
    mode = input("Enter mode (1-6): ")
    return mode


if __name__ == "__main__":
    mode = selectMode()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode in ["1", "2"]:
        model = TrajectoryModel()

        if mode == "1":
            print("Training trajectory-only model.")
            input_data, output_data = getTrajectoryTrainData()
            trainTrajectoryModel(model, input_data, output_data, device)

        elif mode == "2":
            print("Inference trajectory-only model.")
            test_data_dict = getTrajectoryTestData()
            model.load_state_dict(torch.load("trajectory_model.pth"))
            testTrajectoryModel(model, test_data_dict, device)

    elif mode in ["3", "4"]:
        model = VisionModel()

        if mode == "3":
            print("Training vision-only model.")
            input_data, output_data = getVisionTrainData()
            trainVisionModel(model, input_data, output_data, device)

        elif mode == "4":
            print("Inference vision-only model.")
            test_data_dict = getVisionTestData()
            model.load_state_dict(torch.load("vision_model.pth"))
            testVisionModel(model, test_data_dict, device)


    elif mode == "5":
        print("Training fusion model.")

    elif mode == "6":
        print("Inference fusion model.")

    else:
        print("Invalid mode selected. Please enter a number between 1 and 6.")