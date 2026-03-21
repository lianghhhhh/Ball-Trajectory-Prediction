# Trajectory-only model definition using LSTM for ball trajectory prediction

import torch.nn as nn


class TrajectoryModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_steps=10, output_size=2, num_layers=2, dropout=0.2):
        super(TrajectoryModel, self).__init__()
        self.output_steps = output_steps
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_steps * output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.model(lstm_out[:, -1, :])
        return out.reshape(-1, self.output_steps, self.output_size)