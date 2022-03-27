import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np

class F1(nn.Module):
    def __init__(self,task):
        super(F1, self).__init__()
        self.task = task 
 
    def forward(self, prediction, truth):
        pred = np.argmax(prediction,axis=1)
        truth = np.array(truth)
        if self.task == 'genre':
            return f1_score(truth,pred,labels=[0,1,2,3,4,5,6],average=None,zero_division=1)
        else:
            return f1_score(truth,pred,average='weighted',zero_division=1)

def get_metric(task):
    
    if task in ['genre','topic']:
        return F1(task)
        #return torch.nn.CrossEntropyLoss()

    if task in ['violence','romantic','sadness', 'feeling','danceability','energy']:
        return torch.nn.MSELoss()


class Metrics(nn.Module):
    def __init__(self, tasks):
        super(Metrics, self).__init__()
        self.tasks = tasks
        self.metric_fncts = torch.nn.ModuleDict({task: get_metric(task) for task in tasks})

    def forward(self, prediction, truth):
        metric_dict = {task: self.metric_fncts[task](prediction[task], truth[task]) for task in self.tasks}
        return metric_dict