import torch
import torch.nn as nn
import torch.fft

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # x方向のモード数
        self.modes2 = modes2  # y方向のモード数

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y) × (in_channel, out_channel, x, y) → (batch, out_channel, x, y)
        return torch.einsum("bixy, ioxy -> boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=48):  # ← 変更済み
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2

        self.fc0 = nn.Linear(3, self.width)

        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # 入力 + 座標情報
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x1 = self.conv1(x) + self.w1(x)
        x2 = self.conv2(x1) + self.w2(x1)
        x3 = self.conv3(x2) + self.w3(x2)

        x = x3.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x  # (B, H, W, 1)

    def get_grid(self, shape, device):
        B, H, W = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, W, device=device)
        gridy = torch.linspace(0, 1, H, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        grid = torch.stack((gridx, gridy), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid
    


class FNO_noGrad(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2

        self.fc0 = nn.Linear(1, self.width)  # x: (B, N, N, 1) → (B, N, N, width)

        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x1 = self.conv1(x) + self.w1(x)
        x2 = self.conv2(x1) + self.w2(x1)
        x3 = self.conv3(x2) + self.w3(x2)

        x = x3.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x  # (B, H, W, 1)

    
    