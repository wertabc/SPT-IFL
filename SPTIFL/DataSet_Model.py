import math

import torch
import torch.nn as nn

from torch.utils.data import Dataset


class DataSet_Lables(Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)


class DataSet_noLables(Dataset):
    def __init__(self, text_list):
        self.text_list = text_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        return text

    def __len__(self):
        return len(self.text_list)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1200):
        super(PositionalEncoding, self).__init__()
        #         实现Dropout正则化减少过拟合
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_(nn.Module):
    def __init__(self, word_list, embedding_dim=50):
        super(Transformer_,self).__init__()
        self.em = nn.Embedding(len(word_list) + 1, embedding_dim=embedding_dim)  # 对0也需要编码
        self.pos = PositionalEncoding(embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        # self.fc1 = nn.Linear(embedding_dim, 128)
        # self.fc2 = nn.Linear(128, 2)

    def forward(self, inputs):
        x = self.em(inputs)                                                                                             # [16,1200,50]
        x = self.pos(x)                                                                                                 # [16,1200,50]
        x = x.float()                                                                                                   # [16,1200,50]
        x = self.transformer_encoder(x)                                                                                 # [16,1200,50]
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view(-1,3600)                                                                                             # [16,3600]
        return x



class LSTM_(nn.Module):
    def __init__(self, word_list, embed_dim, hidden_size):
        super(LSTM_, self).__init__()
        self.em = nn.Embedding(len(word_list) + 1, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        # self.fc1 = nn.Linear(hidden_size, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.hidden_size = hidden_size

    def forward(self, inputs):
        bz = inputs.shape[1]
        h0 = torch.zeros((1, bz, self.hidden_size))
        c0 = torch.zeros((1, bz, self.hidden_size))
        x = self.em(inputs)  # [16,1200,24]
        r_o, _ = self.rnn(x, (h0, c0))  # [16,1200,24]
        x = self.pool1(x)  # [16,1200,4]
        x = self.pool2(x)  # [16,1200,1]
        # r_o = r_o[-1]                                                                                                  #[16,1200,24]
        x = x.view(-1, 1200)  # [16,28800]  两层池化层之后---->[16,1200]
        return x


class GRU_(nn.Module):
    def __init__(self, word_list, embed_dim, hidden_size, num_layers=2):
        super(GRU_, self).__init__()

        self.em = nn.Embedding(len(word_list) + 1, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        # self.linear1 = nn.Linear(hidden_size, 128)
        # self.linear2 = nn.Linear(128, 2)

    def forward(self, inputs):
        x = self.em(inputs)  # inputs is input, size (seq_len, batch, input_size)               #[16,1200,50]
        x = x.float()
        x, _ = self.gru(
            x)  # x is outuput, size (seq_len, batch, hidden_size)                                          #[16,1200,20]
        x = self.pool1(x)  # [16,1200,4]
        x = self.pool2(x)  # [16,1200,1]
        x = x.view(-1, 1200)
        return x
