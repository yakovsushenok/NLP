"""
    file : utils
    authors : 21112254, 16008937, 20175911, 21180859

"""

import torch
import torch.nn as nn

################# util functions for model formulation ########################

def get_body(model,vocab):
    """
    Find the encoder to use for the model
    
    Parameters
    ----------
    model : string
        name of encoder to use
    vocab : array
        transformed tokens
    
    Returns
    -------
    model : class for the corresponding encoder

    """
    if model == "Conv":
        from models.bodys import Conv
        vocab_size, embed_dim = len(vocab), 512
        return Conv(vocab_size, embed_dim)

    if model == "BERT":
        from models.bodys import BERT
        num_output = 360
        return BERT(num_output)

    
def get_heads(tasks,mapping):
    """
    Find the decoders for a given model
    
    Parameters
    ----------
    tasks : list
        list of tasks included on model
    mapping : dict
        dictionary of all the class variables 
        and the correspinding class value
    
    Returns
    -------
    Decoders : dictionary of task specific decoders

    """

    return torch.nn.ModuleDict({task: get_head(task,mapping[task]) for task in tasks})

def get_head(task,mapping): 
    """
    Find the decoder for a given model
    
    Parameters
    ----------
    task : string
        name of task
    mapping : dict
        dictionary of all the class variables 
        and the correspinding class value
    
    Returns
    -------
    Decoder : class for the corresponding decoder

    """

    if task in ['genre','topic']:
        from models.heads import FullyConnectedLayerHead
        num_class, input_size = len(mapping), 360
        return FullyConnectedLayerHead(num_class, input_size)

    if task in ['violence','romantic','sadness', 'feelings','danceability','energy']:
        from models.heads import FullyConnectedLayerHead
        num_class, input_size = 1, 360
        return FullyConnectedLayerHead(num_class, input_size)
