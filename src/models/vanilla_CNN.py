import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First Convolutional Layer: Input channels=3 (for RGB images), Output channels=128, Kernel size=5x5, Padding=1
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=1)
    
        # Second Convolutional Layer: Input channels=128, Output channels=64, Kernel size=3x3, Padding=1
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Third Convolutional Layer: Input channels=64, Output channels=3, Kernel size=1x1, Padding=1
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x

class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x