# Fusion model definition combining trajectory and vision information

import torch
import torch.nn as nn
from vision_model import VisionModel
from trajectory_model import TrajectoryModel


class FusionModel(nn.Module):
    def __init__(self, trajectory_input_size=45, vision_input_size=45, hidden_size=128, output_size=45, num_layers=2, dropout=0.2):
        super(FusionModel, self).__init__()
        self.trajectory_model = TrajectoryModel(trajectory_input_size, hidden_size, hidden_size, num_layers, dropout)
        self.vision_model = VisionModel(vision_input_size, hidden_size, hidden_size, num_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, trajectory_input, vision_input):
        trajectory_output = self.trajectory_model(trajectory_input)
        vision_output = self.vision_model(vision_input)
        fused_output = torch.cat([trajectory_output, vision_output], dim=1)
        out = self.fusion(fused_output)
        return out