import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        #first Conv2d, stride is applied
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # modify  the input tensor directly, without allocating any additional output

        # second Conv2d, stride is not applied

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # modify  the input tensor directly, without allocating any additional output

        # 1x1 convolution for the shortcut connection
        # avoid using 1x1 if the output and the input have the same shape
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward (self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out= self.relu(out)

        #second time
        out = self.conv2(out)
        out = self.bn2(out)
        # create shortcut (skip connection)
        #by adding residual or x to the second  conv2 after bn2, we create a shortcut connection that allows the inpit to skip
        #some layers and be added directly to the output -> improve: gradient flow and prevent overfitting and preserve the identity of the input.
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1 = self.make_resblock(64, 64, stride=1)
        self.resblock2 = self.make_resblock(64, 128, stride=2)
        self.resblock3 = self.make_resblock(128, 256, stride=2)
        self.resblock4 = self.make_resblock(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def make_resblock(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out