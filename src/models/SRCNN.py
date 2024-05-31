import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First Convolutional Layer: Input channels=3 (for RGB images), Output channels=64, Kernel size=9x9
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)

        # Second Convolutional Layer: Input channels=64, Output channels=32, Kernel size=1x1
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)

        # Third Convolutional Layer: Input channels=32, Output channels=3, Kernel size=5x5
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class SimpleSRCNN(nn.Module):
    def __init__(self):
        super(SimpleSRCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4),  # Input channels=3 (for RGB images), Output channels=32, Kernel size=9x9
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Output channels=64, Kernel size=5x5
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output channels=128, Kernel size=3x3
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),  # Input channels=128, Output channels=64, Kernel size=3x3
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2),  # Input channels=64, Output channels=32, Kernel size=5x5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=9, padding=4),  # Input channels=32, Output channels=3, Kernel size=9x9
            nn.ReLU(True))
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Usage example:
# srcnn_model = SRCNN()
# input_tensor = torch.rand((batch_size, 3, height, width))  # Example input tensor
# output_tensor = srcnn_model(input_tensor)

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the SRCNN architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2, bias=False)

        # Initialize the weights using Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)


    
    def forward(self, x):
        # Pass the input through the SRCNN layers
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.Sigmoid()(x)

        return x