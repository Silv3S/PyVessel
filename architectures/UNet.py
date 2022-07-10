import torch
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU
from torch.nn import functional as F
from torchvision.transforms import CenterCrop


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.decoder_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encoder_feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, encoder_features, x):
        (_, _, H, W) = x.shape
        encoder_features = CenterCrop([H, W])(encoder_features)
        return encoder_features


class UNet(Module):
    def __init__(self, output_dims, encoder_channels=(3, 64, 128, 256, 512, 1024),
                 decoder_channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = Conv2d(decoder_channels[-1], 1, 1)
        self.output_dims = output_dims

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0],
                           encoder_features[::-1][1:])
        out = self.head(out)
        out = F.interpolate(out, self.output_dims)
        return out
