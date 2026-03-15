import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    """
    ConvBlock used in U-Net architecture.

    This block consists of two consecutive convolution layers
    with kernel_size 3x3, padding is set to 1 to preserve spatial dimensions,
    each followed by batch normalization and ReLU activation function.

    Parameters
    ----------
    in_c: int
    out_c: int

    Input
    -----
    x: torch.Tensor -> tensor of shape (B, in_c, H, W)
    B -> how much images in single patch, example: 8
    in_c -> input channel, example: 3 (RGB)
    H -> height
    W -> width
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Convolutional block.
        :param x: torch.Tensor
        :return: torch.Tensor -> output tensor after convolution operations.
        """

        return self.conv(x)
    
class EncoderBlock(nn.Module):
    """
    Encoder block used in contracting path of U-Net.

    This block consists of ConvBlock class,
    followed by MaxPooling with 2x2 kernel size,
    and stride set it to 2 to reduce input data size.

    Parameters
    ----------
    in_c: int
    out_c: int

    Input
    -----
    x: torch.Tensor -> Tensor of shape (B, in_c, H, W)
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pass forward to encoder block
        :param x: torch.Tensor: input Tensor
        :return:
        skip: torch.Tensor -> Feature map before downsampling (used in decoder skip connections)
        pooled: torch.Tensor -> Downsampled feature map
        """

        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled

class DecoderBlock(nn.Module):
    """
    Decoder block used for expansive path.

    This block consist of ConvBlock only.

    Parameters
    ----------
    in_c: int
    out_c: int

    Input
    -----
    x: torch.Tensor -> tensor of shape (B, in_c, H, W)
    skip: torch.Tensor

    Output
    ______
    torch.Tensor -> Tensor of shape (B, out_c, H2, W2)
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of decoder block
        """
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    Input
    ----
    in_c: int
    num_classes: int

    Output
    out: torch.Tensor -> Tensor of shape (B, in_c, H, W)

    ------

    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.d1 = EncoderBlock(in_channels, 64)
        self.d2 = EncoderBlock(64, 128)
        self.d3 = EncoderBlock(128, 256)
        self.d4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.u1 = DecoderBlock(1024 + 512, 512)
        self.u2 = DecoderBlock(512 + 256, 256)
        self.u3 = DecoderBlock(256 + 128, 128)
        self.u4 = DecoderBlock(128 + 64, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        self.__init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)

        b = self.bottleneck(p4)

        u1 = self.u1(b, s4)
        u2 = self.u2(u1, s3)
        u3 = self.u3(u2, s2)
        u4 = self.u4(u3, s1)

        out = self.out(u4)

        return out

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=1).to(device)
    print(f"Device                      : {device}")
    print(f"Total trainable parameters  : {model.count_parameters()}")
    summary(model, input_size=(3, 256, 256), device=str(device))
