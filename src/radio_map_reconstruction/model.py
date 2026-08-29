from torch import Tensor, cat
from torch.nn import Conv2d, GroupNorm, Identity, MaxPool2d, Module, Sequential, SiLU, Upsample

def _group_count(channels: int, max_group: int = 8) -> int:
    groups = min(channels, max_group)
    while channels % groups != 0:
        groups -= 1
    return groups

class ResidualBlock(Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        groups = _group_count(out_channels)

        self.residual = Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            GroupNorm(groups, out_channels),
            SiLU(inplace=True),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding = 1,
                bias=False
            ),
            GroupNorm(groups, out_channels),
        )

        if in_channels == out_channels:
            self.shortcut = Identity()
        else:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

        self.activation = SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.residual(inputs) + self.shortcut(inputs))

class EncoderBlock(Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = ResidualBlock(in_channels, out_channels)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.block(input)
        output = self.pool(skip)
        return output, skip

class DecoderBlock(Module):

    def __init__(self, in_channels: int, skip_channels:int, out_channels: int) -> None:
        super().__init__()

        self.upsample = Sequential(
            Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),
            Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )
        self.residual = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, input: Tensor, skip_input: Tensor) -> Tensor:
        return self.residual(cat((self.upsample(input), skip_input), dim=1))

class ResUnet(Module):

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 1,
            base_channels: int = 32
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        feature_channels = []
        for i in range(5):
            feature_channels.append(base_channels * (2 ** i))

        self.input_block = ResidualBlock(in_channels, feature_channels[0])

        self.encoder1 = EncoderBlock(feature_channels[0], feature_channels[1])
        self.encoder2 = EncoderBlock(feature_channels[1], feature_channels[2])
        self.encoder3 = EncoderBlock(feature_channels[2], feature_channels[3])
        self.encoder4 = EncoderBlock(feature_channels[3], feature_channels[4])

        self.bottleneck = ResidualBlock(feature_channels[4], feature_channels[4])

        self.decoder1 = DecoderBlock(feature_channels[4], feature_channels[4], feature_channels[3])
        self.decoder2 = DecoderBlock(feature_channels[3], feature_channels[3], feature_channels[2])
        self.decoder3 = DecoderBlock(feature_channels[2], feature_channels[2], feature_channels[1])
        self.decoder4 = DecoderBlock(feature_channels[1], feature_channels[1], feature_channels[0])

        self.output_head = Conv2d(
            feature_channels[0] * 2,
            out_channels,
            kernel_size=1
        )

        
    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != 4:
            raise ValueError("输入数据的维度数量不等于4")

        if input.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"but received {input.shape[1]}"
            )

        skip1 = output = self.input_block(input)

        output, skip2 = self.encoder1(output)
        output, skip3 = self.encoder2(output)
        output, skip4 = self.encoder3(output)
        output, skip5 = self.encoder4(output)

        output = self.bottleneck(output)

        output = self.decoder1(output, skip5)
        output = self.decoder2(output, skip4)
        output = self.decoder3(output, skip3)
        output = self.decoder4(output, skip2)

        output = self.output_head(cat((output, skip1), dim=1))

        return output
