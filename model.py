import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp

from config import config


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_bn=False, dropout_rate=None):
        nn.Module.__init__(self)
        pad = kernel_size // 2

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(nn.ReLU(inplace=True))

        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_bn=False, dropout_rate=None):
        nn.Module.__init__(self)
        pad = kernel_size // 2

        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                     stride=2, padding=pad, output_padding=1)
        self.res = ResidualBlock(out_ch, out_ch * 2, kernel_size, use_bn, dropout_rate)

    def forward(self, x, skip):
        out = self.up(x)
        out = out + skip
        out = self.res(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_bn, dropout_rate):
        nn.Module.__init__(self)

        layers = []
        layers.append(ConvBlock(in_ch, out_ch, 1, use_bn, None))
        layers.append(ConvBlock(out_ch, out_ch, kernel_size, use_bn, dropout_rate))
        layers.append(ConvBlock(out_ch, out_ch, kernel_size, use_bn, dropout_rate))
        layers.append(ConvBlock(out_ch, in_ch, 1, use_bn, None))

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        skip = x
        features = self.residual(x)

        return features + skip


class UNet(nn.Module):
    def __init__(self, num_classes, use_bn=False, dropout_rate=None, binary=False):
        nn.Module.__init__(self)
        self.binary = binary

        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        down1 = self.backbone.features[:2]
        self.down1 = nn.Sequential(*down1)

        down2 = self.backbone.features[2:4]
        self.down2 = nn.Sequential(*down2)

        down3 = self.backbone.features[4:7]
        self.down3 = nn.Sequential(*down3)

        down4 = self.backbone.features[7:14]
        self.down4 = nn.Sequential(*down4)

        down5 = self.backbone.features[14:]
        self.down5 = nn.Sequential(*down5)

        self.res1 = ResidualBlock(1280, 512, 3, use_bn, dropout_rate=dropout_rate)
        self.conv = ConvBlock(1280, 512, 3, use_bn, dropout_rate=dropout_rate)
        self.res2 = ResidualBlock(512, 512, 3, use_bn, dropout_rate=dropout_rate)

        self.up1 = UpBlock(512, 96, kernel_size=3, use_bn=use_bn, dropout_rate=dropout_rate)
        self.up2 = UpBlock(96, 32, kernel_size=3, use_bn=use_bn, dropout_rate=dropout_rate)
        self.up3 = UpBlock(32, 24, kernel_size=3, use_bn=use_bn, dropout_rate=dropout_rate)
        self.up4 = UpBlock(24, 16, kernel_size=3, use_bn=use_bn, dropout_rate=dropout_rate)
        self.up5 = nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.final_conv = ConvBlock(32, num_classes, kernel_size=3, use_bn=False, dropout_rate=None)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        flat = self.res1(down5)
        flat = self.conv(flat)
        flat = self.res2(flat)

        up1 = self.up1(flat, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)
        up5 = self.up5(up4)

        out = self.final_conv(up5)

        if self.binary:
            out = nn.functional.sigmoid(out)

        return out


def get_model(backbone, binary=False):
    if backbone == 'unet':
        model = UNet(num_classes=config["num_classes"], use_bn=True, dropout_rate=config["dropout_rate"], binary=binary)

    else:
        if binary:
            model = smp.Unet(backbone, encoder_weights='imagenet', classes=config["num_classes"], activation='sigmoid',
                             encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
        else:
            model = smp.Unet(backbone, encoder_weights='imagenet', classes=config["num_classes"], activation=None,
                             encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

    return model


if __name__ == "__main__":
    in_tensor = torch.rand((1, 3, 704, 1056))

    model = UNet(1)
    print(model)

    out = model(in_tensor)
    print(out.shape)
