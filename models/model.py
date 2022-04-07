"""
    file : model
    authors : 21112254, 16008937, 20175911, 21180859

"""

import torch
import torch.nn as nn
from models.utils import get_body, get_heads

class Model(nn.Module):
    """
    Class for formulate model. 

    Here we combine an encoder with a specific series of task decoders 
    depending on the input configuration

    """
    
    def __init__(self, config: dict,vocab,mapping):
        super(Model, self).__init__()
        self.model = config["Model"]
        self.tasks = config["Tasks"]
        self.encoder = get_body(self.model,vocab)
        self.decoders = get_heads(self.tasks,mapping)
         

    def forward(self, x,text_len):
        output = self.encoder(x,text_len)
        return {task:self.decoders[task](output) for task in self.tasks }