import torch.nn as nn
import torch

# This is our Deep Q-learning model:
#We take a stack of 4 frames as input
#It passes through 3 convnets
#Then it is flatened
#Finally it passes through 2 FC layers
#It outputs a Q value for each actions

class DQN(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        linear = nn.linear
        
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels = 4,
            out_channels = 32,
            kernel_size = 8,
            filters = 32,
            stride = 4),
            nn.ELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels = 4,
            out_channels = 64,
            kernel_size = 4,
            filters = 64,
            stride = 2),
            nn.ELU(),
            nn.BatchNorm2d(64),

            nn.Conv2D(in_channels = 4,
            out_channels = 128,
            kernel_size = 4,
            filters = 128),
            nn.BatchNorm2d(128),
            nn.ELU(),
            Flatten(),

            linear(
                7 * 7 * 64,
                256),
            nn.ELU(),
            linear(
                256,
                448),
            nn.ELU()
            )
    
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
