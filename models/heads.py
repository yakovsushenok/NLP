import torch
import torch.nn as nn

class FullyConnectedLayerHead(nn.Module):

    def __init__(self, num_class, embed_dim):
        super(FullyConnectedLayerHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 60)
        self.fc2 = nn.Linear(60, num_class)

    def forward(self, input):
        out = self.fc1(input)
        return self.fc2(out)