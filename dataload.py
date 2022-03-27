import torch
from torchtext import data
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset(config,df):

    if config['Model'] == "Conv":
        dataset = ConvDataset(config,df)
    return dataset

def gen_map(col_):
    unique = col_.unique()
    return {unique[i]:i for i in range(len(unique))}

def create_mappings(df,config):
    return {task:gen_map(df[task]) for task in config['Tasks']}

class ConvDataset(Dataset):

    def __init__(self,config,df):
        self.lyrics = df['lyrics']

        self.genre_bool = 'genre' in config['Tasks']
        self.violence_bool = 'violence' in config['Tasks']
        self.dance_bool = 'danceability' in config['Tasks']
        self.energy_bool = 'energy' in config['Tasks']
        self.topic_bool = 'topic' in config['Tasks']
        self.feelings_bool = 'feelings' in config['Tasks']
        self.sadness_bool = 'sadness' in config['Tasks']
        self.romantic_bool = 'romantic' in config['Tasks']

        if self.genre_bool:
            self.genre = df['genre']
        if self.violence_bool:
            self.violence = df['violence']
        if self.dance_bool:
            self.dance = df['danceability']
        if self.energy_bool:
            self.energy = df['energy']
        if self.topic_bool:
            self.topic = df['topic']
        if self.feelings_bool:
            self.feelings = df['feelings']
        if self.sadness_bool:
            self.sadness = df['sadness']
        if self.romantic_bool:
            self.romantic = df['romantic']

    def __len__(self):
        return len(self.lyrics)
  
    def __getitem__(self,idx):
        if idx >= len(self.lyrics):
            return None
        sample = {}
        sample['lyrics'] = self.lyrics[idx]
        if self.genre_bool:
            sample['genre'] = self.genre[idx]
        if self.violence_bool:
            sample['violence'] = self.violence[idx]
        if self.dance_bool:
            sample['danceability'] = self.dance[idx]
        if self.energy_bool:
            sample['energy'] = self.energy[idx]
        if self.topic_bool:
            sample['topic'] = self.topic[idx]
        if self.feelings_bool:
            sample['feelings'] = self.feelings[idx]
        if self.sadness_bool:
            sample['sadness'] = self.sadness[idx]
        if self.romantic_bool:
            sample['romantic'] = self.romantic[idx]
        return sample

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


def get_dataloader(dataset, batch_size,mapping,vocab):

    lyrics_pipeline = lambda x: vocab(tokenizer(x))
    genre_pipeline = lambda x: mapping['genre'][x]
    topic_pipeline = lambda x: mapping['topic'][x]

    def collate_batch(batch):
    
        lyrics_ids = [torch.Tensor(lyrics_pipeline(sample['lyrics'])).long() for sample in batch]
        lyrics_ids = nn.utils.rnn.pad_sequence(lyrics_ids, padding_value=vocab['<pad>'], batch_first=True)
        genre_ids = torch.Tensor([genre_pipeline(sample['genre']) for sample in batch]).long()
        text_len = torch.Tensor([len(sample['lyrics']) for sample in batch])
        batch_output = {'lyrics': lyrics_ids,
             'genre': genre_ids,
             'text_len': text_len}
        if 'violence' in batch[0]:
            violence_ids = torch.Tensor([sample['violence'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['violence'] = violence_ids
        if 'romantic' in batch[0]:
            romantic_ids = torch.Tensor([sample['romantic'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['romantic'] = romantic_ids
        if 'sadness' in batch[0]:
            sadness_ids = torch.Tensor([sample['sadness'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['sadness'] = sadness_ids
        if 'feelings' in batch[0]:
            feelings_ids = torch.Tensor([sample['feelings'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['feelings'] = feelings_ids
        if 'romantic' in batch[0]:
            romantic_ids = torch.Tensor([sample['romantic'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['romantic'] = romantic_ids
        if 'energy' in batch[0]:
            energy_ids = torch.Tensor([sample['energy'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['energy'] = energy_ids
        if 'danceability' in batch[0]:
            dance_ids = torch.Tensor([sample['danceability'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['danceability'] = dance_ids
        if 'topic' in batch[0]:
            topic_ids = torch.Tensor([topic_pipeline(sample['topic']) for sample in batch]).long()
            batch_output['topic'] = topic_ids
        
        return batch_output

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

