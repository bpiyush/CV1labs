"""Defines the TwoLayerNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwolayerNet(nn.Module):
    # assign layer objects to class attributes
    # nn.init package contains convenient initialization methods
    # http://pytorch.org/docs/master/nn.html#torch-nn-init
    def __init__(self, num_inputs, num_hidden, num_classes):
        '''
        :param input_size: 3*32*32
        :param hidden_size: decide by yourself e.g. 1024, 512, 128 ...
        :param num_classes: 
        '''
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        super(TwolayerNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, num_classes)
        
    def forward(self,x):
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.act_fn(x)
        scores = self.linear2(x)
        return scores


class TwolayerNetAdditionalLayers(nn.Module):
    # assign layer objects to class attributes
    # nn.init package contains convenient initialization methods
    # http://pytorch.org/docs/master/nn.html#torch-nn-init
    def __init__(self, num_inputs, num_hidden: list, num_classes):
        '''
        :param input_size: 3*32*32
        :param hidden_size: decide by yourself e.g. 1024, 512, 128 ...
        :param num_classes: 
        '''
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        super(TwolayerNetAdditionalLayers, self).__init__()
        self.act_fn = nn.ReLU()

        self.linear_layers = [nn.Linear(num_inputs, num_hidden[0])]
        for i in range(1, len(num_hidden)):
            self.linear_layers.append(nn.Linear(num_hidden[i - 1], num_hidden[i]))
            self.linear_layers.append(self.act_fn)
        self.linear_layers = nn.Sequential(*self.linear_layers)

        self.final = nn.Linear(num_hidden[-1], num_classes)

    def forward(self,x):
        # flatten
        x = x.view(x.shape[0], -1)

        x = self.linear_layers(x)
        scores = self.final(x)

        return scores


if __name__ == "__main__":
    num_classes = 10
    net = TwolayerNet(num_inputs = 3*32*32, num_hidden = 50, num_classes = num_classes)
    x = torch.randn((1, 3, 32, 32))
    y = net(x)
    assert y.shape == torch.Size([1, num_classes])


    net = TwolayerNetAdditionalLayers(num_inputs = 3*32*32, num_hidden = [1024, 512, 128], num_classes = num_classes)
    y = net(x)
    assert y.shape == torch.Size([1, num_classes])