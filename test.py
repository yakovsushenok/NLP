"""
    file : test
    authors : 21112254, 16008937, 20175911, 21180859

"""

import torch
import numpy as np

def evaluate(config, model, iterator, metric):
    """
    Function to evaluate a model given test data
    
    Parameters
    ----------
    config : dict
        dictionary of configuration to run
    model : class
        class of model
    iterator: class
        test dataloader
    metric : class
        class of metrics to report
    
    Returns
    -------
    dict : dictionary of results

    """
    metric_dict = create_metric_dict(config)
    model.eval()
    with torch.no_grad():
        for idx, mini_batch in enumerate(iterator):
            predicted_labels = model(mini_batch['lyrics'],mini_batch['attention_mask'])
            task_targets = {task:mini_batch[task]for task in config["Tasks"]}
            metrics = metric(predicted_labels, task_targets)
            conf_mat = generate_conf_mat(predicted_labels['genre'],task_targets['genre'].cpu())
            metric_dict = update_metric_dict(idx,metric_dict,metrics,conf_mat)
    return metric_dict

def create_metric_dict(config):
    """
    Function initialise results dict

    """
    dict_ = {'cm': np.zeros((5,5)),'genre':0}
    if 'violence' in config["Tasks"]:
        dict_['violence'] = 0
    if 'romantic' in config["Tasks"]:
        dict_['romantic'] = 0
    if 'sadness' in config["Tasks"]:
        dict_['sadness'] = 0
    if 'feelings' in config["Tasks"]:
        dict_['feelings'] = 0
    if 'topic' in config["Tasks"]:
        dict_['topic'] = 0
    if 'energy' in config["Tasks"]:
        dict_['energy'] = 0
    if 'danceability' in config["Tasks"]:
        dict_['danceability'] = 0
    return dict_

def generate_conf_mat(predicted,target):
    """
    Function to create confusion matrix

    """
    mat = np.zeros((5,5))
    for i in range(len(target)):
        pred = np.argmax(predicted[i])
        #if pred != target[i]:
        mat[pred,target[i]] +=1
    return mat

def update_metric_dict(idx,dict,metrics,conf_mat):
    """
    Function update metric dictionary 

    """
    dict['cm'] = np.add(dict['cm'],conf_mat)
    dict['genre'] = (1/(idx+1)) *( (idx)*dict['genre'] + metrics['genre'])
    if 'violence' in dict:
        dict['violence'] = (1/(idx+1)) *( (idx)*dict['violence'] + metrics['violence'])
    if 'romantic' in dict:
        dict['romantic'] = (1/(idx+1)) *( (idx)*dict['romantic'] + metrics['romantic'])
    if 'sadness' in dict:
        dict['sadness'] = (1/(idx+1)) *( (idx)*dict['sadness'] + metrics['sadness'])
    if 'feelings' in dict:
        dict['feelings'] = (1/(idx+1)) *( (idx)*dict['feelings'] + metrics['feelings'])
    if 'topic' in dict:
        dict['topic'] = (1/(idx+1)) *( (idx)*dict['topic'] + metrics['topic'])
    if 'energy' in dict:
        dict['energy'] = (1/(idx+1)) *( (idx)*dict['energy'] + metrics['energy'])
    if 'danceability' in dict:
        dict['danceability'] = (1/(idx+1)) *( (idx)*dict['danceability'] + metrics['danceability'])
    return dict

