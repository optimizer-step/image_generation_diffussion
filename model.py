import torch
from torch import nn
from torch.nn import functional as F
from config import NOISE_STEPS, MODEL_WIDTH

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.time_layer = nn.Linear(time_size, out_channels)


    def forward(self, image, time_embedding):
        features = F.silu(self.conv1(image))
        time_signal = self.time_layer(time_embedding)[:, :, None, None]
        features = features + time_signal
        features = F.silu(self.conv2(features))
        return (features + self.skip(image)) / 1.4142


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        width = MODEL_WIDTH
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128,128))

        self.start = nn.Conv2d(3, width, 3, padding = 1)
        self.down1 = ConvBlock(width, width, 128)
        self.down2 = ConvBlock(width, width*2, 128)
        self.middle = ConvBlock(width *2, width *2, 128)
        self.up2 = ConvBlock(width *4, width *2, 128)
        self.up1 = ConvBlock(width *3, width, 128)
        self.finish = nn.Conv2d(width, 3, 1)

    def forward(self, image, step):
        time_number = (step.float() / NOISE_STEPS).unsqueeze(1)
        time_embedding = self.time_mlp(time_number)
        start = self.start(image)
        skip1 = self.down1(start, time_embedding)
        smaller = F.avg_pool2d(skip1, 2)
        skip2 = self.down2(smaller, time_embedding)
        smallest = F.avg_pool2d(skip2, 2)
        middle = self.middle(smallest, time_embedding)
        bigger = F.interpolate(middle, scale_factor = 2, mode = 'nearest')
        bigger = torch.cat([bigger, skip2], dim = 1)
        bigger = self.up2(bigger, time_embedding)
        biggest = F.interpolate(bigger, scale_factor = 2, mode = 'nearest')
        biggest = torch.cat([biggest, skip1], dim = 1)
        biggest = self.up1(biggest, time_embedding)
        return self.finish(biggest)
        