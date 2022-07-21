import torch
import torch.nn as nn


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


class Recurrent_block(nn.Module):
    def __init__(self, channels_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.channels_out = channels_out
        self.conv = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, channels_in, channels_out, t=2):
        super(RRCNN_block, self).__init__()
        self.Conv_1x1 = nn.Conv2d(
            channels_in, channels_out, kernel_size=1)
        self.RCNN = nn.Sequential(
            Recurrent_block(channels_out, t),
            Recurrent_block(channels_out, t)
        )

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2U_Net(nn.Module):
    def __init__(self, channels_in=3, channels_out=1, t=2):
        super(R2U_Net, self).__init__()
        features = [64, 128, 256, 512, 1024]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(channels_in, features[0], t)
        self.RRCNN2 = RRCNN_block(features[0], features[1], t)
        self.RRCNN3 = RRCNN_block(features[1], features[2], t)
        self.RRCNN4 = RRCNN_block(features[2], features[3], t)
        self.RRCNN5 = RRCNN_block(features[3], features[4], t)

        self.Up5 = up_conv(features[4], features[3])
        self.Up_RRCNN5 = RRCNN_block(features[4], features[3], t)
        self.Up4 = up_conv(features[3], features[2])
        self.Up_RRCNN4 = RRCNN_block(features[3], features[2], t)
        self.Up3 = up_conv(features[2], features[1])
        self.Up_RRCNN3 = RRCNN_block(features[2], features[1], t)
        self.Up2 = up_conv(features[1], features[0])
        self.Up_RRCNN2 = RRCNN_block(features[1], features[0], t)

        self.Conv_1x1 = nn.Conv2d(features[0], channels_out, kernel_size=1)

    def forward(self, x):
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
