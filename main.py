"""
    file : main
    authors : 21112254, 16008937, 20175911, 21180859

"""

import torch
import pandas as pd
from criterion.criterion import Criterion
from dataload import get_dataloader, get_dataset, create_mappings, get_vocab,get_bert_dataloader
from models.model import Model
from train import model_train
from utils import preprocess_class_im,save_conf_mat,update_results,save_class_breakdown
from test import evaluate
from metrics import Metrics
import time
import torch
import pandas as pd
from transformers import AdamW


def experiment(config,batch_size,num_epoch,lr_,df,data_mapping):
    """
    Function to run single experiment

    Returns:
    dict - results from experiment

    """

    # create and split data set 
    full_dataset = get_dataset(config,df)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


    # get dataloader and optimizers
    if config['Model'] == "Conv":
      vocab = get_vocab(full_dataset)
      train_dataloader = get_dataloader(train_dataset,batch_size,data_mapping,vocab)
      test_dataloader = get_dataloader(test_dataset,batch_size,data_mapping,vocab)


    else:
      vocab = None
      train_dataloader = get_bert_dataloader(train_dataset,batch_size,data_mapping)
      test_dataloader = get_bert_dataloader(test_dataset,batch_size,data_mapping)


    # create model details
    net = Model(config,vocab,data_mapping)
    criterion = Criterion(config['Tasks'])
    metrics = Metrics(config['Tasks'])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr_,momentum=0.9)

    # train model
    for epoch in range(num_epoch):
        model_eval = model_train(config,net,criterion,optimizer,train_dataloader,epoch)

    # evaulate model
    eval_metrics = evaluate(config,model_eval,test_dataloader,metrics)
    return eval_metrics

def main():
    """
    Main fuction 

    Details to run:

    CONFIG['model'] : the model to run
    Values : 'Conv' or 'BERT'

    CONFIG['Tasks'] : Tasks to consider
    Values : ['genre','violence','romantic','sadness',
     ,'danceability']

     Always include 'genre' 

     Exmaples:
     {'Model':'Conv','Tasks':['genre']}
     {'Model':'BERT','Tasks':['genre','violence','romantic']}
     {'Model':'Conv','Tasks':['genre','romantic']}


    """
    CONFIG = {'Model':'Conv','Tasks':['genre','romantic']}
    DATA = pd.read_csv( "./tcc_ceds_music.csv")
    BATCH_SIZE = 32

    if CONFIG['Model']=="BERT":
        NUM_EPOCH = 3
        LR = 1e-5
        NUM_EX = 5
    else: 
        NUM_EPOCH = 10
        LR = 1
        NUM_EX = 5


    results= None

    ## address class imbalances
    df = preprocess_class_im(DATA)

    ## find mappings 
    data_mapping = create_mappings(df,CONFIG)

    ## run experiments
    for i in range(NUM_EX):
        print("EXPERIMENT " + str(i+1))
        experiment_results = experiment(CONFIG,BATCH_SIZE,NUM_EPOCH,LR,df,data_mapping)
        results = update_results(results,experiment_results,i)   

    ## save results
    save_conf_mat(CONFIG,results['mean']['cm'],data_mapping)

    ## print results
    print(results)


if __name__ == '__main__':
    main()
    