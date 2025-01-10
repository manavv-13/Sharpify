import torch
import torch.nn as nn
import torch.optim as optim

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6):
        super(Generator, self).__init__()
        layers = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(True)
            ]
            ngf *= 2

        layers += [ResNetBlock(ngf) for _ in range(n_blocks)]

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf // 2),
                nn.ReLU(True)
            ]
            ngf //= 2

        layers += [
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True)
        ]
        for i in range(1, 4):
            layers += [
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True)
            ]
            ndf *= 2

        layers += [
            nn.Conv2d(ndf, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
