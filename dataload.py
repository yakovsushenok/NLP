"""
    file : dataload
    authors : 21112254, 16008937, 20175911, 21180859

"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####### Data untils functions #############

def get_dataset(config,df):

    if config['Model'] == "Conv":
        dataset = ConvDataset(config,df)

    if config['Model'] == "BERT":
        PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        dataset = BERTDataset(config,df,tokenizer,300)
    
    return dataset

def gen_map(col_):
    unique = col_.unique()
    return {unique[i]:i for i in range(len(unique))}

def create_mappings(df,config):
    return {task:gen_map(df[task]) for task in config['Tasks']}

########## Dataset classes ######################

class BERTDataset(Dataset):
  """
  This class is adopted from the pytorch Dataset class. We use it for loading the data for the BERT model.
  """  
  def __init__(self,config, df, tokenizer, max_len):
    self.lyrics = df['lyrics']
    self.genre = df['genre']
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.violence_bool = 'violence' in config['Tasks']
    self.dance_bool = 'danceability' in config['Tasks']
    self.sadness_bool = 'sadness' in config['Tasks']
    self.romantic_bool = 'romantic' in config['Tasks']
    self.topic_bool = 'topic' in config['Tasks']
    if self.violence_bool:
      self.violence = df['violence']
    if self.dance_bool:
      self.dance = df['danceability']
    if self.sadness_bool:
      self.sadness = df['sadness']
    if self.romantic_bool:
      self.romantic = df['romantic']
    if self.topic_bool:
      self.topic = df['topic']


  def __len__(self):
    return len(self.lyrics)
  
  
  def __getitem__(self, idx):
    sample = {}
    lyrics = str(self.lyrics[idx])
    genre = self.genre[idx]
    encoding = self.tokenizer.encode_plus(
      lyrics,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )
    sample['lyrics'] = lyrics
    sample['lyrics_ids'] = encoding['input_ids'].flatten()
    sample['attention_mask'] = encoding['attention_mask'].flatten()
    sample['genre'] = genre
    if self.violence_bool:
      sample['violence'] = self.violence[idx]
    if self.dance_bool:
      sample['danceability'] = self.dance[idx]
    if self.sadness_bool:
      sample['sadness'] = self.sadness[idx]
    if self.romantic_bool:
      sample['romantic'] = self.romantic[idx]
    if self.topic_bool:
      sample['topic'] = self.topic[idx]
    return sample

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
        if self.topic_bool:
            sample['topic'] = self.topic[idx]
        return sample

############ Dataloaders ################

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

def get_bert_dataloader(ds,batch_size,mapping):

    genre_pipeline = lambda x: mapping['genre'][x]
    topic_pipeline = lambda x: mapping['topic'][x]

    def collate_batch(batch):
    
        genre_ids = torch.Tensor([genre_pipeline(sample['genre']) for sample in batch]).long()
        #attention_mask = torch.Tensor([sample['attention_mask'] for sample in batch])
        attention_mask = torch.Tensor([sample['attention_mask'].flatten().tolist()[:300] for sample in batch]).to(torch.int64)
        lyrics_ids = torch.Tensor([sample['lyrics_ids'].tolist()[:300] for sample in batch]).to(torch.int64)
        batch_output = {'lyrics': lyrics_ids,
             'genre': genre_ids,
             'attention_mask': attention_mask}
        if 'violence' in batch[0]:
            violence_ids = torch.Tensor([sample['violence'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['violence'] = violence_ids
        if 'romantic' in batch[0]:
            romantic_ids = torch.Tensor([sample['romantic'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['romantic'] = romantic_ids
        if 'sadness' in batch[0]:
            sadness_ids = torch.Tensor([sample['sadness'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['sadness'] = sadness_ids
        if 'romantic' in batch[0]:
            romantic_ids = torch.Tensor([sample['romantic'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['romantic'] = romantic_ids
        if 'danceability' in batch[0]:
            dance_ids = torch.Tensor([sample['danceability'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['danceability'] = dance_ids
        if 'topic' in batch[0]:
            topic_ids = torch.Tensor([topic_pipeline(sample['topic']) for sample in batch]).long()
            batch_output['topic'] = topic_ids

        
        return batch_output
    return DataLoader(
    ds,
    batch_size=batch_size,
    #num_workers=4,
    shuffle=True,
    collate_fn=collate_batch
  )

def get_dataloader(dataset, batch_size,mapping,vocab):

    

    def preprocess(passage):
      remove_punc = re.sub(r'[^\w\s]', ' ', passage)
      remove_und_sc = remove_punc.replace('_', ' ')
      remove_non_eng = re.sub(r'[^\x00-\x7F]+',' ', remove_und_sc)
      digi = r'[0-9]'
      remove_num = re.sub(digi, ' ', remove_non_eng)
      lower_text = remove_num.lower()
      tokenization = word_tokenize(lower_text)
      lemma = WordNetLemmatizer()
      tokens = []

      for word in tokenization:
        lemmatized_word = lemma.lemmatize(word)
        tokens.append(lemmatized_word)

      return tokens

    lyrics_pipeline = lambda x: vocab(preprocess(x))
    genre_pipeline = lambda x: mapping['genre'][x]
    topic_pipeline = lambda x: mapping['topic'][x]

    def collate_batch(batch):
    
        lyrics_ids = [torch.Tensor(lyrics_pipeline(sample['lyrics'])).long() for sample in batch]
        lyrics_ids = nn.utils.rnn.pad_sequence(lyrics_ids, padding_value=vocab['<pad>'], batch_first=True)
        genre_ids = torch.Tensor([genre_pipeline(sample['genre']) for sample in batch]).long()
        text_len = torch.Tensor([len(sample['lyrics']) for sample in batch])
        batch_output = {'lyrics': lyrics_ids,
             'genre': genre_ids,
             'attention_mask': text_len}
        if 'violence' in batch[0]:
            violence_ids = torch.Tensor([sample['violence'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['violence'] = violence_ids
        if 'romantic' in batch[0]:
            romantic_ids = torch.Tensor([sample['romantic'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['romantic'] = romantic_ids
        if 'sadness' in batch[0]:
            sadness_ids = torch.Tensor([sample['sadness'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['sadness'] = sadness_ids
        if 'romantic' in batch[0]:
            romantic_ids = torch.Tensor([sample['romantic'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['romantic'] = romantic_ids
        if 'danceability' in batch[0]:
            dance_ids = torch.Tensor([sample['danceability'] for sample in batch]).to(torch.float32).unsqueeze(1)
            batch_output['danceability'] = dance_ids
        
        return batch_output

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)