# Vision-only model definition using CNN+LSTM for ball trajectory prediction

import torch.nn as nn


class VisionModel(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, output_steps=10, num_layers=2, dropout=0.2):
        super(VisionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.output_steps = output_steps
        self.lstm = nn.LSTM(64 * 8 * 8, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_steps * 2)
        )

    def forward(self, x):
        # Expected input is a sequence of image frames:
        # (B, T, H, W, C) from numpy or (B, T, C, H, W) if already channel-first.
        if x.dim() != 5:
            raise ValueError(f"VisionModel expects 5D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (1, 3, 4):
            # Convert (B, T, H, W, C) -> (B, T, C, H, W)
            x = x.permute(0, 1, 4, 2, 3).contiguous()

        batch_size, seq_len, channels, height, width = x.shape
        cnn_in = x.view(batch_size * seq_len, channels, height, width)
        cnn_out = self.cnn(cnn_in).view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(cnn_out)
        out = self.model(lstm_out[:, -1, :])
        return out.view(batch_size, self.output_steps, 2)