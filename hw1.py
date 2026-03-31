import torch
from vision_model import VisionModel
from fusion_model import FusionModel
from trajectory_model import TrajectoryModel
from utils import (getTrajectoryTrainData, getTrajectoryTestData, 
                   getVisionTrainData, getVisionTestData, 
                   getFusionTrainData, getFusionTestData)
from train import trainTrajectoryModel, trainVisionModel, trainFusionModel
from test import testTrajectoryModel, testVisionModel, testFusionModel

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
        model_name = "trajectory_model.pth"

        if mode == "1":
            print("Training trajectory-only model.")
            input_data, output_data, group_ids = getTrajectoryTrainData(window_stride=1, return_groups=True)
            trainTrajectoryModel(model, input_data, output_data, device, group_ids=group_ids, model_name=model_name)

        elif mode == "2":
            print("Inference trajectory-only model.")
            test_data_dict = getTrajectoryTestData()
            model.load_state_dict(torch.load(model_name))
            testTrajectoryModel(model, test_data_dict, device)

    elif mode in ["3", "4"]:
        model = VisionModel()
        model_name = "vision_model.pth"

        if mode == "3":
            print("Training vision-only model.")
            input_data, output_data, group_ids = getVisionTrainData(window_stride=1, return_groups=True)
            trainVisionModel(model, input_data, output_data, device, group_ids=group_ids, model_name=model_name)

        elif mode == "4":
            print("Inference vision-only model.")
            test_data_dict = getVisionTestData()
            model.load_state_dict(torch.load(model_name))
            testVisionModel(model, test_data_dict, device)

    elif mode in ["5", "6"]:
        model = FusionModel()
        model_name = "fusion_model.pth"

        if mode == "5":
            print("Training fusion model.")
            input_data, output_data, group_ids = getFusionTrainData(window_stride=1, return_groups=True)
            trainFusionModel(model, input_data, output_data, device, group_ids=group_ids, model_name=model_name)

        elif mode == "6":
            print("Inference fusion model.")
            test_data_dict = getFusionTestData()
            model.load_state_dict(torch.load(model_name))
            testFusionModel(model, test_data_dict, device)

    else:
        print("Invalid mode selected. Please enter a number between 1 and 6.")