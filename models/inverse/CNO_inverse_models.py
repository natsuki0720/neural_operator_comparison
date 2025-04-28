import torch
import torch.nn as nn
import torch.nn.functional as F

class CNOInverseClassifier(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

        self.activation = nn.Sigmoid()  # 2値分類を意識

    def forward(self, x):
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing="ij"
        )
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        x = torch.cat([x, coords], dim=1)  # (B, C+2, H, W)

        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))
        x4 = self.decoder1(self.up1(x3))
        x5 = self.decoder2(self.up2(x4))

        return self.activation(x5)
