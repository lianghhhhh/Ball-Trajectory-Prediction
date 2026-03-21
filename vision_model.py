# Vision-only model definition using CNN+LSTM for ball trajectory prediction

import torch.nn as nn


class VisionModel(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, output_size=45, num_layers=2, dropout=0.2):
        super(VisionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lstm = nn.LSTM(64 * 8 * 8, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        cnn_out = self.cnn(x).view(batch_size, -1)
        lstm_out, _ = self.lstm(cnn_out.unsqueeze(1))
        out = self.model(lstm_out[:, -1, :])
        return out