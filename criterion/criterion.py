import torch
import torch.nn as nn

def get_loss(task):
    
    if  task in ['genre','topic']:
        return torch.nn.CrossEntropyLoss()

    if task in ['violence','romantic','sadness', 'feeling','danceability','energy']:
        return torch.nn.MSELoss()


class Criterion(nn.Module):
    def __init__(self, tasks):
        super(Criterion, self).__init__()
        self.tasks = tasks
        self.loss_fncts = torch.nn.ModuleDict({task: get_loss(task) for task in tasks})

    
    def forward(self, prediction, truth):
        loss_dict = {task: self.loss_fncts[task](prediction[task], truth[task]) for task in self.tasks}
        loss_dict['total'] = torch.sum(torch.stack([loss_dict[task] for task in self.tasks]))
        return loss_dict