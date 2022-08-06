# Original paper implementation https://github.com/juntang-zhuang/LadderNet

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.channels_not_equal = in_channels != out_channels
        if self.channels_not_equal:
            self.conv0 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding='same')
        self.conv1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.15)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.channels_not_equal:
            x = self.conv0(x)
            x = self.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + x
        return self.relu(out)


class Initial_LadderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super().__init__()
        self.out_channels = out_channels
        self.layers = layers

        self.in_conv = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=3, padding='same', bias=True)
        self.relu = nn.ReLU(inplace=True)

        scaled_out_channels = [16, 32, 64, 128]
        self.down_residuals = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_residuals = nn.ModuleList()
        for i in range(0, layers):
            upconv_oc = max(0, layers-i-1)
            self.down_residuals.append(
                ResidualBlock(scaled_out_channels[i], scaled_out_channels[i]))
            self.down_convs.append(nn.Conv2d(
                scaled_out_channels[i], scaled_out_channels[i+1], stride=2, kernel_size=3, padding=1))
            self.up_convs.append(nn.ConvTranspose2d(scaled_out_channels[layers-i], scaled_out_channels[upconv_oc], kernel_size=3,
                                                    stride=2, padding=1, output_padding=1, bias=True))
            self.up_residuals.append(
                ResidualBlock(scaled_out_channels[upconv_oc], scaled_out_channels[upconv_oc]))

        self.bottom = ResidualBlock(
            out_channels*(2**layers), out_channels*(2**layers))

    def forward(self, x):
        out = self.in_conv(x)
        out = self.relu(out)

        skip_connections_AD = []
        for i in range(0, self.layers):
            out = self.down_residuals[i](out)
            skip_connections_AD.append(out)
            out = self.down_convs[i](out)
            out = self.relu(out)

        out = self.bottom(out)
        bottom = out

        skip_connections_DA = []
        skip_connections_DA.append(bottom)

        for j in range(0, self.layers):
            out = self.up_convs[j](out) + skip_connections_AD[self.layers-j-1]
            out = self.relu(out)
            out = self.up_residuals[j](out)
            skip_connections_DA.append(out)

        return skip_connections_DA


class Middle_LadderBlock(nn.Module):
    def __init__(self, out_channels, layers):
        super().__init__()
        self.out_channels = out_channels
        self.layers = layers

        self.in_conv = ResidualBlock(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

        scaled_out_channels = [16, 32, 64, 128]
        self.down_residuals = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_residuals = nn.ModuleList()
        for i in range(0, layers):
            upconv_oc = max(0, layers-i-1)
            self.down_residuals.append(
                ResidualBlock(scaled_out_channels[i], scaled_out_channels[i]))
            self.down_convs.append(nn.Conv2d(
                scaled_out_channels[i], scaled_out_channels[i+1], stride=2, kernel_size=3, padding=1))
            self.up_convs.append(nn.ConvTranspose2d(scaled_out_channels[layers-i], scaled_out_channels[upconv_oc], kernel_size=3,
                                                    stride=2, padding=1, output_padding=1, bias=True))
            self.up_residuals.append(
                ResidualBlock(scaled_out_channels[upconv_oc], scaled_out_channels[upconv_oc]))

        self.bottom = ResidualBlock(
            out_channels*(2**layers), out_channels*(2**layers))

    def forward(self, residual_addends):
        out = self.in_conv(residual_addends[-1])

        skip_connections_AD = []
        for i in range(0, self.layers):
            out = out + residual_addends[-i-1]
            out = self.down_residuals[i](out)
            skip_connections_AD.append(out)

            out = self.down_convs[i](out)
            out = self.relu(out)

        out = self.bottom(out)
        bottom = out

        skip_connections_DA = []
        skip_connections_DA.append(bottom)

        for j in range(0, self.layers):
            out = self.up_convs[j](out) + skip_connections_AD[self.layers-j-1]
            out = self.relu(out)
            out = self.up_residuals[j](out)
            skip_connections_DA.append(out)

        return skip_connections_DA


class Final_LadderBlock(nn.Module):
    def __init__(self, out_channels, layers):
        super().__init__()
        self.block = Middle_LadderBlock(out_channels, layers)

    def forward(self, x):
        out = self.block(x)
        return out[-1]


class LadderNet(nn.Module):
    def __init__(self, layers=3, filters=16, in_channels=3):
        super().__init__()
        self.initial_block = Initial_LadderBlock(
            in_channels, filters, layers)
        # Middle_block can be repeated to increase model's parameter count
        # This may lead to overfitting, so use with caution
        # self.middle_block = Middle_LadderBlock(filters, layers)
        self.final_block = Final_LadderBlock(filters, layers)
        self.final = nn.Sequential(
            nn.Conv2d(filters, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.initial_block(x)
        # out = self.middle_block(out)
        out = self.final_block(out)
        out = self.final(out)
        return out
