import os
import torch
import torch.nn as nn
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Sequential(
            nn.Linear(128*6*3, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    spects = torch.rand(size=(8, 2, 33, 61))
    MODEL = CNN()
    print(MODEL)
    print("number of model parameters:", sum([np.prod(p.size()) for p in MODEL.parameters()]))
    x = MODEL(spects)
    print(x.size())
