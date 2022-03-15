import torch
from torchtext import data
import pandas as pd
import re 
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_map(col_):
    unique = col_.unique()
    return {unique[i]:i for i in range(len(unique))}

def create_mappings(df,config):
    #this will need change for regression tasks!!
    return {task:gen_map(df[task]) for task in config['Tasks']}

class MyDataset(Dataset):

    def __init__(self,config,df):
        self.lyrics = df['lyrics']
        self.genre_bool = 'genre' in config['Tasks']
        if self.genre_bool:
            self.genre = df['genre']

    def __len__(self):
        return len(self.lyrics)
  
    def __getitem__(self,idx):
        if idx >= len(self.lyrics):
            return None
        sample = {}
        sample['lyrics'] = self.lyrics[idx]
        if self.genre_bool:
            sample['genre'] = self.genre[idx]
        return sample

def get_dataset(config,df):

    dataset = MyDataset(config,df)
    return dataset


tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    # pytorch implementation
    for sample in data_iter:
        if sample == None:
            break 
        else:
            yield tokenizer(sample['lyrics'])

def get_vocab(dataset):
    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

## check vocab
#print(vocab(['here', 'is', 'an', 'example']))

def get_dataloaderOLD(dataset, batch_size,mapping,vocab):

    lyrics_pipeline = lambda x: vocab(tokenizer(x))
    genre_pipeline = lambda x: mapping['genre'][x]

    def collate_batch(batch):
    
        lyrics_list = []
        genre_list = []
        offsets = [0]
        text_len =[]
        
        for sample in batch:
            processed_lyrics = torch.tensor(lyrics_pipeline(sample['lyrics']), dtype=torch.int64)
            lyrics_list.append(processed_lyrics)
            if 'genre' in sample:
                genre_list.append(genre_pipeline(sample['genre']))            
            offsets.append(processed_lyrics.size(0))
        genre_list = torch.tensor(genre_list, dtype=torch.int64)
        text_len = torch.tensor(offsets[1:])
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        #lyrics_list = torch.tensor(lyrics_list)
        #lyrics_list = torch.cat(lyrics_list)
        results={'lyrics':lyrics_list,'genre':genre_list.to(device),'offsets':offsets.to(device),'text_len':text_len.to(device)}
        return results

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

def get_dataloader(dataset, batch_size,mapping,vocab):

    lyrics_pipeline = lambda x: vocab(tokenizer(x))
    genre_pipeline = lambda x: mapping['genre'][x]

    def collate_batch(batch):
    
        lyrics_ids = [torch.Tensor(lyrics_pipeline(sample['lyrics'])).long() for sample in batch]
        lyrics_ids = nn.utils.rnn.pad_sequence(lyrics_ids, padding_value=vocab['<pad>'], batch_first=True)
        genre_ids = torch.Tensor([genre_pipeline(sample['genre']) for sample in batch]).int()
        #is this the right implementation of len
        text_len = torch.Tensor([len(sample['lyrics']) for sample in batch])
        batch = {'lyrics': lyrics_ids,
             'genre': genre_ids,
             'text_len': text_len}
        return batch

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

