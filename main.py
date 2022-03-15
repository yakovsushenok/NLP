import torch
from torchtext import data
import pandas as pd
import re 
from torch.utils.data import DataLoader, Dataset
from criterion.criterion import Criterion
from dataload import get_dataloader, get_dataset, create_mappings, get_vocab
from models.model import Model
from train import model_train
import time
import torch
import torchtext
from torchtext import data
import pandas as pd
import re 
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {'Model':'LSTM','Tasks':['genre']}

    ##test
    df = pd.read_csv( "/home/cwatts/NLP/data/tcc_ceds_music.csv")
    print(df.columns)
    
    BATCH_SIZE = 32
    NUM_EPOCH = 20
    LR = 5

    full_dataset = get_dataset(config,df)
    data_mapping = create_mappings(df,config)
    vocab = get_vocab(full_dataset)

    net = Model(config,vocab,data_mapping).to(device)
    criterion = Criterion(config['Tasks'])
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = get_dataloader(train_dataset,BATCH_SIZE,data_mapping,vocab)
    test_dataloader = get_dataloader(test_dataset,BATCH_SIZE,data_mapping,vocab)

    for epoch in range(NUM_EPOCH):
        epoch_start_time = time.time()
        model_eval = model_train(config,net,criterion,optimizer,train_dataloader,epoch)


if __name__ == '__main__':
    main()