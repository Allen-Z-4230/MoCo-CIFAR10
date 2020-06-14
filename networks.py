import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim


class Encoder(nn.Module):
    """Encoder convolutional network used for MoCo.
    Architecture: 2 convolutional layers, 1 fully connected layer.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        args = {'stride':1, 'padding':1}  # arguments for both convolutional layers
        self.conv1 = nn.Conv2d(3, 36, 3, **args)
        self.conv2 = nn.Conv2d(36, 72, 3, **args)
        self.fc1 = nn.Linear(18432, 4000)
        self.fc2 = nn.Linear(4000, 128)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = F.relu(x)

        # layer 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        #fully-connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x) # nomralizes all the features to 0-1
        return x


class ConvNet(nn.Module):
    """Standard convolutional net for baseline
    Architecture: 2 convolutional layers, 3 fully connected layers.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # 3 input image channels, 10 output channels, 3x3 square convolution
        # Convolution layers
        args = {'stride':1, 'padding':1}  # arguments for both convolutional layers
        self.conv1 = nn.Conv2d(3, 10, 3, **args)
        self.conv2 = nn.Conv2d(10, 20, 3, **args)

        #pooling layer
        self.pool = nn.AvgPool2d(2, 2)

        #ReLU
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(1280, 640)
        self.fc2 = nn.Linear(640, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        #first block
        x = self.pool(self.relu(self.conv1(x)))

        #second block
        x = self.pool(self.relu(self.conv2(x)))

        #flatten
        x = x.view(-1, 1280)

        #fully-connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x
