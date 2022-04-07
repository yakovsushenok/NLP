"""
    file : heads
    authors : 21112254, 16008937, 20175911, 21180859

"""

import torch
import torch.nn as nn

class FullyConnectedLayerHead(nn.Module):

    """
    Class for Decoders. 

    Here we pass the output from the encoder through a number 
    of fullly connected layers. 

    """

    def __init__(self, num_class, embed_dim):
        super(FullyConnectedLayerHead, self).__init__()

        # num_class depends on the task
        self.fc1 = nn.Linear(embed_dim, 60)
        self.fc2 = nn.Linear(60, num_class)

    def forward(self, input):

        #pass input through fully connected layers
        out = self.fc1(input)
        return self.fc2(out)