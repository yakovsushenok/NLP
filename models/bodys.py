from typing import Text
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import BertModel


class EmbeddingLayer(nn.Module):
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
        embedded = self.dropout(self.embedding(text))
        embedded = embedded.permute(0,2,1)
        convs = [self.relu(conv(embedded)) for conv in self.convs]
        pooled = [conv.max(dim=-1).values for conv in convs]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        return cat
   
