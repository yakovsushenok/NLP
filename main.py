import torch
import pandas as pd
import numpy as np
from criterion.criterion import Criterion
from dataload import get_dataloader, get_dataset, create_mappings, get_vocab
from models.model import Model
from train import model_train
from utils import preprocess_class_im,save_conf_mat,update_results
from test import evaluate
from metrics import Metrics
import time
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer

def experiment(config,batch_size,num_epoch,lr_,df,data_mapping):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = get_dataset(config,df)
    vocab = get_vocab(full_dataset)

    net = Model(config,vocab,data_mapping).to(device)
    criterion = Criterion(config['Tasks'])
    metrics = Metrics(config['Tasks'])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr_)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = get_dataloader(train_dataset,batch_size,data_mapping,vocab)
    test_dataloader = get_dataloader(test_dataset,batch_size,data_mapping,vocab)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        model_eval = model_train(config,net,criterion,optimizer,train_dataloader,epoch)

    eval_metrics = evaluate(config,model_eval,test_dataloader,metrics)
    return eval_metrics

def main():

    ## PARAMS 
    ## ['genre','violence','romantic','sadness',
    ## 'feeling','danceability','topic','energy']
    CONFIG = {'Model':'Conv','Tasks':['genre']}
    DATA = pd.read_csv( "/home/cwatts/NLP/data/tcc_ceds_music.csv")
    BATCH_SIZE = 32
    NUM_EPOCH = 35
    LR = 5
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
    
    
    
    
    #tokenizer = get_tokenizer('basic_english')
    #token_lens = []
    #for txt in df['lyrics']:
        #tokens = tokenizer(txt)
        #token_lens.append(len(tokens))
    #sns.distplot(token_lens)
    #plt.xlim([0, 512])
    #plt.xlabel('Token count')

    #plt.savefig("matplotlib.png") 
