"""
    file : criterion
    authors : 21112254, 16008937, 20175911, 21180859

"""


import torch
import torch.nn as nn

def get_loss(task):
    """
    Find the loss function for a given task
    
    Parameters
    ----------
    task : string
        name of task

    Returns
    -------
    Loss: class for the corresponding loss function

    """
    
    if  task in ['genre','topic']:
        return torch.nn.CrossEntropyLoss()

    if task in ['violence','romantic','sadness', 'feelings','danceability','energy']:
        return torch.nn.MSELoss()


class Criterion(nn.Module):
    """
    Class to combine losses for model. 

    Here we use the tasks list  to calculate the given losses for the seperate heads

    """
    def __init__(self, tasks):
        super(Criterion, self).__init__()
        self.tasks = tasks
        self.loss_fncts = torch.nn.ModuleDict({task: get_loss(task) for task in tasks})

    
    def forward(self, prediction, truth):
        loss_dict = {task: self.loss_fncts[task](prediction[task], truth[task]) for task in self.tasks}
        loss_dict['total'] = torch.sum(torch.stack([loss_dict[task] for task in self.tasks]))
        return loss_dict