#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn

class DenseNN(nn.Module):
    def __init__(self, input_size, hyperparam, num_classes):
        super(DenseNN, self).__init__()
        
        # flatten input to 1D array
        self.flatten = nn.Flatten()

        # define layers
        self.fc1 = nn.Linear(input_size, hyperparam)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hyperparam, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

