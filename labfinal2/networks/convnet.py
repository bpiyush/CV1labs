"""Defines ConvNet architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    # Complete the code using LeNet-5
    # reference: https://ieeexplore.ieee.org/document/726791
    def __init__(self, in_channels, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels = 6, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.linear1 = nn.Linear(in_features = 400, out_features = 120)
        self.linear2 = nn.Linear(in_features = 120, out_features = 84)
        self.linear3 = nn.Linear(in_features = 84, out_features = num_classes)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.act_fn = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act_fn(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.pool(x)

        x = x.view(x.size()[0], -1)

        x = self.linear1(x)
        x = self.act_fn(x)
    
        x = self.linear2(x)
        x = self.act_fn(x)

        x = self.linear3(x)

        return x
    

if __name__ == "__main__":
    num_classes = 10
    net = ConvNet(in_channels=3, num_classes=num_classes)
    x = torch.randn((1, 3, 32, 32))
    y = net(x)
    assert y.shape == torch.Size([1, num_classes])