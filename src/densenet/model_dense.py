import torch
import torch.nn as nn


# ==========================================================
# Dense Layer
# ==========================================================

class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate):

        super().__init__()

        self.layer = nn.Sequential(

            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

    def forward(self, x):

        new_features = self.layer(x)

        return torch.cat([x, new_features], dim=1)


# ==========================================================
# Dense Block
# ==========================================================

class DenseBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        growth_rate,
        num_layers
    ):

        super().__init__()

        layers = []

        channels = in_channels

        for _ in range(num_layers):

            layers.append(
                DenseLayer(
                    channels,
                    growth_rate
                )
            )

            channels += growth_rate

        self.block = nn.Sequential(*layers)

        self.out_channels = channels

    def forward(self, x):

        return self.block(x)


# ==========================================================
# Transition Block
# ==========================================================

class Transition(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.transition = nn.Sequential(

            nn.BatchNorm2d(in_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        )

    def forward(self, x):

        return self.transition(x)


# ==========================================================
# Dense Encoder Block
# ==========================================================

class EncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        growth_rate=16,
        num_layers=2
    ):

        super().__init__()

        self.first_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.dense = DenseBlock(
            out_channels,
            growth_rate,
            num_layers
        )

        self.transition = Transition(
            self.dense.out_channels,
            out_channels
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.first_conv(x)

        x = self.dense(x)

        x = self.transition(x)

        skip = x

        x = self.pool(x)

        return x, skip


# ==========================================================
# Decoder Block
# ==========================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        growth_rate=16,
        num_layers=2
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.first_conv = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.dense = DenseBlock(
            out_channels,
            growth_rate,
            num_layers
        )

        self.transition = Transition(
            self.dense.out_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.first_conv(x)

        x = self.dense(x)

        x = self.transition(x)

        return x


# ==========================================================
# Dense U-Net
# ==========================================================

class DenseUNet(nn.Module):

    def __init__(
        self,
        in_channels=83,
        out_channels=1
    ):

        super().__init__()

        self.enc1 = EncoderBlock(
            in_channels,
            32
        )

        self.enc2 = EncoderBlock(
            32,
            64
        )

        self.enc3 = EncoderBlock(
            64,
            128
        )

        self.enc4 = EncoderBlock(
            128,
            256
        )

        # --------------------------
        # Bottleneck
        # --------------------------

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            DenseBlock(
                512,
                growth_rate=16,
                num_layers=3
            ),

            Transition(
                560,
                512
            ),

            nn.Dropout2d(
                p=0.5
            )
        )

        # --------------------------
        # Decoder
        # --------------------------

        self.dec4 = DecoderBlock(
            512,
            256,
            256
        )

        self.dec3 = DecoderBlock(
            256,
            128,
            128
        )

        self.dec2 = DecoderBlock(
            128,
            64,
            64
        )

        self.dec1 = DecoderBlock(
            64,
            32,
            32
        )

        self.final = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        x, s1 = self.enc1(x)

        x, s2 = self.enc2(x)

        x, s3 = self.enc3(x)

        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)

        x = self.dec3(x, s3)

        x = self.dec2(x, s2)

        x = self.dec1(x, s1)

        x = self.final(x)

        return x