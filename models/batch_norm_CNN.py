import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First Convolutional Layer: Input channels=3 (for RGB images), Output channels=128, Kernel size=3x3, Padding=1
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Add MaxPooling layer with kernel size=2x2 and stride=2
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second Convolutional Layer: Input channels=128, Output channels=64, Kernel size=3x3, Padding=1
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Add MaxPooling layer with kernel size=2x2 and stride=2
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third Convolutional Layer: Input channels=64, Output channels=32, Kernel size=3x3, Padding=1
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Fourth Convolutional Layer: Input channels=32, Output channels=16, Kernel size=3x3, Padding=1
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        # Fifth Convolutional Layer: Input channels=16, Output channels=3, Kernel size=3x3, Padding=1
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x


class ImprovedSimpleAE(nn.Module):
    def __init__(self):
        super(ImprovedSimpleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Add MaxPooling layer with kernel size=2x2 and stride=2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Add MaxPooling layer with kernel size=2x2 and stride=2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))  # Additional Convolutional Layer

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
