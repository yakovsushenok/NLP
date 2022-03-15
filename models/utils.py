import torch
import torch.nn as nn
#could put in if


def get_body(model,vocab):

    if model == "TCM":
        from models.bodys import TextClassificationModelBody
        vocab_size, embed_dim = len(vocab), 64
        return TextClassificationModelBody(vocab_size, embed_dim)

    if model == "LSTM":
        from models.bodys import LSTMBody
        vocab_size, embed_dim = len(vocab), 300
        hidden_dim = 128
        return LSTMBody(vocab_size, embed_dim,hidden_dim)
    
def get_heads(tasks,mapping):

    return torch.nn.ModuleDict({task: get_head(task,mapping[task]) for task in tasks})

def get_head(task,mapping): 

    if task == "genre":
        from models.heads import FullyConnectedLayerHead
        num_class, embed_dim = len(mapping), 60
        return FullyConnectedLayerHead(num_class, embed_dim)
