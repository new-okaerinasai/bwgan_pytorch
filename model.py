from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_filters=64, out_filters=64, kernel_size=(3, 3),  kind="up",
                 normalize=False, **kw):
        super(ResBlock, self).__init__()
        self.kind = kind
        self.kernel_size = kernel_size
        self.in_filters = in_filters
        self.out_filters = out_filters

        if self.kind == "up":
            self.resample = nn.Upsample(scale_factor=2)
        elif self.kind == "down":
            self.resample = nn.AvgPool2d(kernel_size=2)
        elif self.kind is None:
            self.resample = Identity()
        else:
            raise Exception("Unknown resample kind")

        if normalize:
            self.bn = nn.BatchNorm2d(self.out_filters, momentum=0.9)
        else:
            self.bn = Identity()

        self.conv0 = nn.Conv2d(self.in_filters, self.out_filters, (1, 1))
        self.conv1 = nn.Conv2d(self.in_filters, self.out_filters, self.kernel_size,
                               padding=self.kernel_size[0]-2)
        self.conv2 = nn.Conv2d(self.out_filters, self.out_filters, self.kernel_size,
                               padding=self.kernel_size[0]-2)

    def forward(self, x):
        skip_connection = self.resample(self.conv0(x))
        x = self.resample(x)
        x = nn.ReLU()(self.bn(self.conv1(x)))
        x = nn.ReLU()(self.bn(self.conv2(x)))
        result = x + skip_connection
        return result


class Generator(nn.Module):
    def __init__(self, z_size=128, **kw):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_size, z_size*4*4)
        self.resblock1 = ResBlock(in_filters=128, out_filters=128, kind="up", normalize=True)
        self.resblock2 = ResBlock(in_filters=128, out_filters=128, kind="up", normalize=True)
        self.resblock3 = ResBlock(in_filters=128, out_filters=128, kind="up", normalize=True)
        self.main = nn.Sequential(self.resblock1, self.resblock2, self.resblock3)
        self.conv = nn.Conv2d(128, 3, kernel_size=(1, 1))

    def forward(self, z):
        x = self.fc(z).view((-1, 128, 4, 4))
        x = self.main(x)
        x = nn.Tanh()(self.conv(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, in_filters=3, out_filters=128):
        super(Discriminator, self).__init__()

        self.resblock1 = ResBlock(in_filters=in_filters, out_filters=out_filters, kind="down")
        self.resblock2 = ResBlock(in_filters=out_filters, out_filters=out_filters, kind="down")
        self.resblock3 = ResBlock(in_filters=out_filters, out_filters=out_filters, kind=None)
        self.resblock4 = ResBlock(in_filters=out_filters, out_filters=out_filters, kind=None)
        self.main = nn.Sequential(self.resblock1, self.resblock2, self.resblock3, self.resblock4)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.main(x)
        x = nn.AvgPool2d([x.shape[-1], x.shape[-2]])(x).view((-1, 128))
        x = self.fc(x)

        return x
