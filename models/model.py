import torch
import torch.nn as nn
from models.utils import get_body, get_heads

class Model(nn.Module):
    
    def __init__(self, config: dict,vocab,mapping):
        super(Model, self).__init__()
        self.model = config["Model"]
        self.tasks = config["Tasks"]
        self.encoder = get_body(self.model,vocab)
        self.decoders = get_heads(self.tasks,mapping)
         

    def forward(self, x,text_len):
        output = self.encoder(x,text_len)
        return {task:self.decoders[task](output) for task in self.tasks }