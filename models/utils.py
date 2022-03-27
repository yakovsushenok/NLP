import torch
import torch.nn as nn
from transformers import BertModel
#could put in if


def get_body(model,vocab):

    if model == "Conv":
        from models.bodys import Conv
        vocab_size, embed_dim = len(vocab), 300
        return Conv(vocab_size, embed_dim)

    
def get_heads(tasks,mapping):

    return torch.nn.ModuleDict({task: get_head(task,mapping[task]) for task in tasks})

def get_head(task,mapping): 

    if task in ['genre','topic']:
        from models.heads import FullyConnectedLayerHead
        num_class, input_size = len(mapping), 360
        return FullyConnectedLayerHead(num_class, input_size)

    if task in ['violence','romantic','sadness', 'feeling','danceability','energy']:
        from models.heads import FullyConnectedLayerHead
        num_class, input_size = 1, 360
        return FullyConnectedLayerHead(num_class, input_size)
