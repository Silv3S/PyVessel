import torch
import torch.nn.functional as F
from torch import nn


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
            in_channels, out_channels, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
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
            in_channels, out_channels, kernel_size=1, bias=False)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            DropBlock(5, 0.85),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
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


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, 1, keepdim=True)
        att = torch.cat((avg_out, max_out), dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class DARE_UNet(nn.Module):
    def __init__(self):
        super(DARE_UNet, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DoubleConv(3, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 256)

        self.conv5 = SingleConv(256, 512)
        self.sp_att = SpatialAttention()
        self.conv6 = SingleConv(512, 512)

        self.up1 = UpsampleConvolve(512, 256)
        self.uconv1 = DoubleConv(512, 256)
        self.up2 = UpsampleConvolve(256, 128)
        self.uconv2 = DoubleConv(256, 128)
        self.up3 = UpsampleConvolve(128, 64)
        self.uconv3 = DoubleConv(128, 64)
        self.up4 = UpsampleConvolve(64, 32)
        self.uconv4 = DoubleConv(64, 32)

        self.Final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc_1 = self.conv1(x)
        enc_1_pool = self.pooling(enc_1)

        enc_2 = self.conv2(enc_1_pool)
        enc_2_pool = self.pooling(enc_2)

        enc_3 = self.conv3(enc_2_pool)
        enc_3_pool = self.pooling(enc_3)

        enc_4 = self.conv4(enc_3_pool)
        enc_4_pool = self.pooling(enc_4)

        # Bottleneck
        bot_1 = self.conv5(enc_4_pool)
        bot_2 = self.sp_att(bot_1)
        bot_3 = self.conv6(bot_2)

        # Decoder
        dec_1_ups = self.up1(bot_3)
        dec_1_cat = torch.cat((enc_4, dec_1_ups), dim=1)
        dec_1 = self.uconv1(dec_1_cat)

        dec_2_ups = self.up2(dec_1)
        dec_2_cat = torch.cat((enc_3, dec_2_ups), dim=1)
        dec_2 = self.uconv2(dec_2_cat)

        dec_3_ups = self.up3(dec_2)
        dec_3_cat = torch.cat((enc_2, dec_3_ups), dim=1)
        dec_3 = self.uconv3(dec_3_cat)

        dec_4_ups = self.up4(dec_3)
        dec_4_cat = torch.cat((enc_1, dec_4_ups), dim=1)
        dec_4 = self.uconv4(dec_4_cat)

        return self.Final(dec_4)


if __name__ == "__main__":
    x = torch.randn((4, 3, 256, 256))
    model = DARE_UNet()
    preds = model(x)
    assert preds.shape == (4, 1, 256, 256)
