import torch
import torch.nn as nn

class FullyConnectedLayerHead(nn.Module):

    def __init__(self, num_class, embed_dim):
        super(FullyConnectedLayerHead, self).__init__()
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input):
        #print(input.shape)
        out = self.fc(input)
        #print(out.shape)
        return out