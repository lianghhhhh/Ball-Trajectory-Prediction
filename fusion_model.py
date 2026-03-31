# Fusion model definition combining trajectory and vision information

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_channels=3, cnn_feature_size=64, traj_feature_size=32, hidden_size=128, output_steps=10, num_layers=2, dropout=0.2):
        super(FusionModel, self).__init__()
        
        # 1. Vision Pathway (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, cnn_feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((8, 8)) # Output: (B, 64, 8, 8)
        )
        
        # Flattened CNN feature size: 64 * 8 * 8 = 4096
        self.vision_fc = nn.Linear(cnn_feature_size * 8 * 8, cnn_feature_size)
        
        # 2. Trajectory Pathway (Embedding)
        self.traj_fc = nn.Linear(2, traj_feature_size)
        
        # 3. Fusion LSTM
        # Combined input size: cnn_feature_size + traj_feature_size
        combined_size = cnn_feature_size + traj_feature_size
        self.lstm = nn.LSTM(combined_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # 4. Output MLP
        self.output_steps = output_steps
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_steps * 2)
        )

    def forward(self, x_img, x_traj):
        """
        x_img: (Batch, 10, C, H, W)
        x_traj: (Batch, 10, 2)
        """
        # --- Process Images ---
        if x_img.dim() != 5:
            raise ValueError(f"FusionModel expects 5D image input, got {tuple(x_img.shape)}")
        
        if x_img.shape[-1] in (1, 3, 4):
            x_img = x_img.permute(0, 1, 4, 2, 3).contiguous()

        batch_size, seq_len, channels, height, width = x_img.shape
        cnn_in = x_img.view(batch_size * seq_len, channels, height, width)
        
        cnn_out = self.cnn(cnn_in).view(batch_size * seq_len, -1)
        vision_features = torch.relu(self.vision_fc(cnn_out))
        vision_features = vision_features.view(batch_size, seq_len, -1) # Shape: (B, 10, 64)
        
        # --- Process Trajectories ---
        traj_features = torch.relu(self.traj_fc(x_traj)) # Shape: (B, 10, 32)
        
        # --- Fuse Modalities ---
        # Concatenate along the feature dimension (dim=2)
        fused_features = torch.cat((vision_features, traj_features), dim=2) # Shape: (B, 10, 96)
        
        # --- Sequence Prediction ---
        lstm_out, _ = self.lstm(fused_features)
        
        # Take the output from the last time step
        out = self.fc_out(lstm_out[:, -1, :])
        
        return out.view(batch_size, self.output_steps, 2)