import math
import sys
import re
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
device =torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

data = pd.read_csv(r"C:\Users\林夕\Desktop\Sorghum bicolor\Sorghum bicolor基准数据集.csv")
sequences = data['Seq']
labels = data['label'].values

pat = re.compile('[ACGTacgt]')


def del_end_xing(text):  # 用于删除序列末尾的*
    text = pat.findall(text)
    text = [each.lower() for each in text]
    return text


x = sequences.apply(del_end_xing)

word_set = set()

for lst in x:
    for word in lst:
        word_set.add(word)

word_list = list(word_set)
word_index = dict([(each, word_list.index(each) + 1) for each in word_list])

text = x.apply(lambda x: [word_index.get(word, 0) for word in x])


text_len = 1200

pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
pad_text = np.array(pad_text)

#划分
pad_text,labels=torch.LongTensor(pad_text),torch.LongTensor(labels)
#x_train, x_test, y_train, y_test = train_test_split(pad_text, labels, test_size=0.3)



class Mydataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)

data_ds = Mydataset(pad_text,labels)
#train_ds = Mydataset(x_train, y_train)
#test_ds = Mydataset(x_test, y_test)

batch_size = 16

data_dl =torch.utils.data.DataLoader(data_ds,batch_size=batch_size,shuffle=False)
#train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
#test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# embed_dim = 2 ** (int(np.log2(len(word_list) ** 0.25)) + 2)  # 经验值

embed_dim = 50
hidden_size = 20

class Net(nn.Module):
    def __init__(self, word_list, embed_dim, hidden_size, num_layers=2):
        super().__init__()

        self.em = nn.Embedding(len(word_list) + 1, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers)
        self.pool1=nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        self.linear1 = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, inputs):
        x = self.em(inputs)  # inputs is input, size (seq_len, batch, input_size)               #[16,1200,50]
        x = x.float()
        x, _ = self.gru(x)  # x is outuput, size (seq_len, batch, hidden_size)                  #[16,1200,20]
        x = self.pool1(x)                                                                       #[16,1200,4]
        x =self.pool2(x)                                                                        #[16,1200,1]
        x =x.view(-1,1200)
        #x = torch.sum(x, dim=0)                                                                #[1200,20]
        #x = F.relu(self.linear1(x))                                                            #[1200,128]
        #x = self.linear2(x)                                                                    #[1200,2]
        return x

model = Net(word_list, embed_dim, hidden_size)                  #实例化
model = model.to(device)

loss = nn.CrossEntropyLoss()
loss = loss.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with torch.no_grad():
    outputs=[]
    for x,y in data_dl:
        out = model(x)
        outputs.append(out)
    print(outputs)
    print(outputs[0].shape)
    print(len(outputs))

outputs1 = torch.concat(outputs,axis =0)
print(outputs1)

deep_fea_GRU=pd.DataFrame(outputs1)
print(deep_fea_GRU)
deep_fea_GRU.to_csv(r'C:\Users\林夕\Desktop\Sorghum bicolor\GRU提取Sorghum bicolor深度特征.csv')
