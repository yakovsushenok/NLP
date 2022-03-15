from typing import Text
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class TextClassificationModelBody(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(TextClassificationModelBody, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return embedded

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim,pad_index =2):
        super(EmbeddingLayer, self).__init__()
        #self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, padding_idx = pad_index)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_index)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text):
        #embedded = self.embedding(text, offsets)
        embedded = self.embedding(text)
        return embedded

class LSTMLayer(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers =1,dropout=0,bidirectional=False):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True,batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, text):

        return self.lstm(text)


class EmbeddingBody(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingBody, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text)
        return embedded

class LSTMBodyOLD(nn.Module):
    def __init__(self,vocab_size, embed_dim,hidden_size,num_layers =1,dropout=0,bidirectional=False):
        super(LSTMBody, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.lstm = LSTMLayer(120, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, text,text_len):
        embedded = self.embedding(text)
        print(text.shape)
        print(embedded.shape)
        packed_embedded = pack_padded_sequence(embedded, text_len, batch_first=True) 
        packed_output, _ = self.lstm(packed_embedded)
        ##do i need to unsort
        return packed_output

class Conv(nn.Module):
    def __init__(self,vocab_size, embed_dim,hidden_size,num_layers =1,dropout=0.25,bidirectional=False):
        super(LSTMBody, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim,out_channels=120,kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=embed_dim,out_channels=120,kernel_size=5)
        self.conv5 = nn.Conv1d(in_channels=embed_dim,out_channels=120,kernel_size=7)
        self.convs = [self.conv1,self.conv4,self.conv5]
        self.max_pool1 = nn.MaxPool1d(2)
        #self.max_pool1 = nn.MaxPool1d(pool_size)
        self.fc1 = nn.Linear(24448 , 60)
        self.fc2 = nn.Linear(60, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text,text_len):
        # ids = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, embedding dim]
        embedded = embedded.permute(0,2,1)
        # embedded = [batch size, embedding dim, seq len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        # pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        return cat

class LSTMBody(nn.Module):
    def __init__(self,vocab_size, embed_dim,hidden_size,num_layers =1,dropout=0.25,bidirectional=False):
        super(LSTMBody, self).__init__()
        self.hidden_dim = hidden_size
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        #self.conv2 = nn.Conv2d(in_channels=1,out_channels=128,kernel_size=(3,embed_dim))
        #self.conv3 = nn.Conv2d(in_channels=1,out_channels=128,kernel_size=(8,embed_dim))
        
        self.conv1 = nn.Conv1d(in_channels=embed_dim,out_channels=32,kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=embed_dim,out_channels=32,kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=embed_dim,out_channels=32,kernel_size=4)
        self.convs = [self.conv1,self.conv4,self.conv5]
        #self.lstm = LSTMLayer(120, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.lstm = nn.LSTM(input_size=32, 
                            hidden_size=self.hidden_dim,                             
                            bidirectional=True,
                            batch_first=True)
        self.max_pool1 = nn.MaxPool1d(2)
        #self.max_pool1 = nn.MaxPool1d(pool_size)
        self.fc1 = nn.Linear(256 , 60)
        self.fc2 = nn.Linear(60, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text,text_len):
        hidden = (torch.zeros(2, text.shape[0], self.hidden_dim),
                torch.zeros(2, text.shape[0], self.hidden_dim))
        embedded = self.dropout(self.embedding(text))
        embedded = embedded.permute(0,2,1)
        # embedded = [batch size, embedding dim, seq len]
        #conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(F.elu(conv(embedded)),1).permute(0, 2, 1) for conv in self.convs]
        # pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        _, hidden = self.lstm(cat,hidden)
        out = torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim=1)
        return self.dropout(self.relu(self.fc1(out)))
       
