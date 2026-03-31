# Vision-only model definition using pretrained CNN+LSTM for ball trajectory prediction

import warnings
import torch.nn as nn
import torchvision.models as tv_models


class VisionModel(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_size=128,
        output_steps=10,
        num_layers=2,
        dropout=0.2,
        backbone_name="resnet18",
        use_pretrained=True,
        freeze_backbone=False,
    ):
        super(VisionModel, self).__init__()

        if input_channels != 3:
            raise ValueError("VisionModel currently supports RGB input (input_channels=3).")

        self.output_steps = output_steps
        self.feature_extractor, feature_dim = self._build_feature_extractor(
            backbone_name=backbone_name,
            use_pretrained=use_pretrained,
        )

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_steps * 2),
        )

    def _build_feature_extractor(self, backbone_name, use_pretrained):
        if backbone_name == "mobilenet_v3_small":
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
            try:
                backbone = tv_models.mobilenet_v3_small(weights=weights)
            except Exception as exc:
                warnings.warn(
                    f"Could not load pretrained MobileNet weights ({exc}). Falling back to random init.",
                    RuntimeWarning,
                )
                backbone = tv_models.mobilenet_v3_small(weights=None)

            feature_extractor = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            return feature_extractor, 576

        if backbone_name != "resnet18":
            warnings.warn(
                f"Unsupported backbone '{backbone_name}'. Falling back to resnet18.",
                RuntimeWarning,
            )

        weights = tv_models.ResNet18_Weights.DEFAULT if use_pretrained else None
        try:
            backbone = tv_models.resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained ResNet18 weights ({exc}). Falling back to random init.",
                RuntimeWarning,
            )
            backbone = tv_models.resnet18(weights=None)

        feature_extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        return feature_extractor, 512

    def forward(self, x):
        # Expected input is a sequence of image frames:
        # (B, T, H, W, C) from numpy or (B, T, C, H, W) if already channel-first.
        if x.dim() != 5:
            raise ValueError(f"VisionModel expects 5D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (1, 3, 4):
            # Convert (B, T, H, W, C) -> (B, T, C, H, W)
            x = x.permute(0, 1, 4, 2, 3).contiguous()

        batch_size, seq_len, channels, height, width = x.shape
        cnn_in = x.reshape(batch_size * seq_len, channels, height, width)
        cnn_out = self.feature_extractor(cnn_in).reshape(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(cnn_out)
        out = self.model(lstm_out[:, -1, :])
        return out.reshape(batch_size, self.output_steps, 2)