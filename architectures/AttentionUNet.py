import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, channels_in=3, channels_out=1):
        super(AttentionUNet, self).__init__()
        features = [64, 128, 256, 512, 1024]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(channels_in, features[0])
        self.Conv2 = conv_block(features[0], features[1])
        self.Conv3 = conv_block(features[1], features[2])
        self.Conv4 = conv_block(features[2], features[3])
        self.Conv5 = conv_block(features[3], features[4])

        self.Up5 = up_conv(features[4], features[3])
        self.Att5 = Attention_block(features[3], features[2])
        self.Up_conv5 = conv_block(features[4], features[3])

        self.Up4 = up_conv(features[3], features[2])
        self.Att4 = Attention_block(features[2], features[1])
        self.Up_conv4 = conv_block(features[3], features[2])

        self.Up3 = up_conv(features[2], features[1])
        self.Att3 = Attention_block(features[1], features[0])
        self.Up_conv3 = conv_block(features[2], features[1])

        self.Up2 = up_conv(features[1], features[0])
        self.Att2 = Attention_block(features[0], 32)
        self.Up_conv2 = conv_block(features[1], features[0])

        self.Final = nn.Sequential(
            nn.Conv2d(features[0], channels_out, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        s4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Final(d2)
        return d1
