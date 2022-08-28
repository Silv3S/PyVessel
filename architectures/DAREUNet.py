import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary


class DropBlock(nn.Module):
    def __init__(self, block_size, p):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def gamma(self, x):
        input_x_size = x.shape[-1]
        dropped = (1 - self.p) / (self.block_size ** 2)
        retained = (input_x_size ** 2) / \
            ((input_x_size - self.block_size + 1) ** 2)
        return dropped * retained

    def forward(self, x):
        if self.training:
            mask_block = 1 - F.max_pool2d(
                torch.bernoulli(torch.ones_like(x) * self.gamma(x)),
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )

            x = x * mask_block * (mask_block.numel() / mask_block.sum())
        return x


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            DropBlock(5, 0.85),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        return x + residual


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            DropBlock(5, 0.85),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            DropBlock(5, 0.85),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.double_conv(x)
        return x + residual


class UpsampleConvolve(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UpsampleConvolve, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels):
        super(AttentionGate, self).__init__()
        self.features_g = nn.Sequential(
            nn.Conv2d(g_channels, x_channels, kernel_size=1),
            nn.BatchNorm2d(x_channels))
        self.features_x = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(x_channels))
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(x_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, g, x):
        g_feat = self.features_g(g)
        x_feat = self.features_x(x)
        psi = self.psi(g_feat + x_feat)
        return x * psi


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, 1, keepdim=True)
        att = torch.cat((avg_out, max_out), dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class UpsampleAttConv(nn.Module):
    def __init__(self, g_channels, x_channels):
        super().__init__()
        self.upsample_convolve = UpsampleConvolve(g_channels, x_channels)
        self.att_gate = AttentionGate(g_channels, x_channels)
        self.double_conv = DoubleConv(g_channels, x_channels)

    def forward(self, x, g):
        upsampled = self.upsample_convolve(x)
        x = self.att_gate(x, g)
        concated = torch.cat((x, upsampled), dim=1)
        return self.double_conv(concated)


class DARE_UNet(nn.Module):
    def __init__(self):
        super(DARE_UNet, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        features = [16, 32, 64, 128]
        self.encoder_1 = DoubleConv(3, features[0])
        self.encoder_2 = DoubleConv(features[0], features[1])
        self.encoder_3 = DoubleConv(features[1], features[2])

        self.bottleneck = nn.Sequential(
            SingleConv(features[2], features[3]),
            SpatialAttention(),
            SingleConv(features[3], features[3])
        )

        self.decoder_1 = UpsampleAttConv(features[3], features[2])
        self.decoder_2 = UpsampleAttConv(features[2], features[1])
        self.decoder_3 = UpsampleAttConv(features[1], features[0])

        self.final_pred = nn.Sequential(
            nn.Conv2d(features[0], 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        skip_connections = []

        # Encoder
        x = self.encoder_1(x)
        skip_connections.append(x)
        x = self.pooling(x)
        x = self.encoder_2(x)
        skip_connections.append(x)
        x = self.pooling(x)
        x = self.encoder_3(x)
        skip_connections.append(x)
        x = self.pooling(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder_1(x, skip_connections[2])
        x = self.decoder_2(x, skip_connections[1])
        x = self.decoder_3(x, skip_connections[0])

        return self.final_pred(x)


if __name__ == "__main__":
    model = DARE_UNet()
    summary(model, (3, 256, 256), device='cpu')
