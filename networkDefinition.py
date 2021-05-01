import torch.nn as nn
import torch
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, height, width):
        self.width = width
        self.height = height
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 12, 7)
        self.conv3 = nn.Conv2d(12, 6, 7)
        self.lin1 = nn.Linear(6498, 6*((((width-3)//2)-3)//2)*((((height-3)//2)-3)//2))
        self.lin2 = nn.Linear(6*((((width-3)//2)-3)//2)*((((height-3)//2)-3)//2), 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 1, 6498)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(120, 200)
        self.lin2 = nn.Linear(200, 420)
        self.conv1 = nn.Conv2d(14, 28, 2)
        self.conv2 = nn.Conv2d(28, 40, 2)
        self.conv3 = nn.Conv2d(104, 70, 3, padding=1)
        self.conv4 = nn.Conv2d(70, 25, 3, padding=1)
        self.conv5 = nn.Conv2d(25, 1, 3, padding=1)
    def forward(self, x, songData, otherSlices):
        x = torch.cat((x, songData), 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = x.view(-1, 14, 5, 6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.cat((x, otherSlices), 1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(65, 40, 3)
        self.conv2 = nn.ConvTranspose2d(40, 20, 3)
        self.conv3 = nn.ConvTranspose2d(20, 6, 3)
        self.lin1 = nn.Linear(540, 150)
        self.lin2 = nn.Linear(150, 50)
        self.lin3 = nn.Linear(50, 2)
    def forward(self, x, songData, otherSlices):
        
        x = torch.cat((x.unsqueeze(1), otherSlices), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1, 540)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softmax(self.lin3(x))
        return x
