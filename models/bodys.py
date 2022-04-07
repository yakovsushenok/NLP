"""
    file : bodys
    authors : 21112254, 16008937, 20175911, 21180859

"""

from typing import Text
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import transformers
from transformers import BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############### Classes for encoders #########################

class EmbeddingLayer(nn.Module):
    """
    Class to instantiate Embedding Layer

    """
    def __init__(self, vocab_size, embed_dim,pad_index =2):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_index)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text):
        embedded = self.embedding(text)
        return embedded


class Conv(nn.Module):
    """
    Class for CNN-MTL encoder. 

    """
    def __init__(self,vocab_size, embed_dim,dropout=0.25):
        super(Conv, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.conv2 = nn.Conv1d(in_channels=embed_dim,out_channels=120,kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=embed_dim,out_channels=120,kernel_size=4)
        self.conv6 = nn.Conv1d(in_channels=embed_dim,out_channels=120,kernel_size=6)
        self.convs = [self.conv2,self.conv4,self.conv6]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, text,text_len):
        # pass input through embedding layer
        embedded = self.dropout(self.embedding(text)) 

        # transform output to be able to be convoluted
        embedded = embedded.permute(0,2,1) 

        # perform 3 parallel convolution and ReLU
        convs = [self.relu(conv(embedded)) for conv in self.convs] 

        # take the max of the filter outputs
        max_conv = [conv.max(dim=-1).values for conv in convs] 

        # concatenrate then pass through dropout layer
        cat = self.dropout(torch.cat(max_conv, dim=-1)) 
        return cat

class BERT(nn.Module):
    """
    Class for BERT-MTL encoder. 

    """
    def __init__(self, n_classes):
      super(BERT, self).__init__()

      self.bert = BertModel.from_pretrained('bert-base-cased')
      self.drop = nn.Dropout(p=0.3)
      self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):

      # pass input and attention mask through BERT model
      output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      # pass output through dropout layer
      output = self.drop(output[1])
      return self.fc(output)
   
