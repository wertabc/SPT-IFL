import math
import re

import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, auc, matthews_corrcoef, precision_score, recall_score, confusion_matrix, roc_curve,roc_auc_score
device =torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

data=pd.read_csv(r"C:\Users\林夕\Desktop\Zea mays\Zea_mays_test13775基准数据集.csv")
sequences = data['seq']
labels = data['label'].values
print(labels)

pat = re.compile('[AGCTagct]')

def pre_process(text):
    text = pat.findall(text)
    text = [each.lower() for each in text]
    return text

x = sequences.apply(pre_process)
#print(x)
word_set = set()

for lst in x:
    for word in lst:
        word_set.add(word)

word_list = list(word_set)
word_index = dict([(each, word_list.index(each) + 1) for each in word_list])
#print(word_list)
#print(word_index)

text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
print(text)

text_len = 1000

pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
pad_text = np.array(pad_text)


pad_text,labels=torch.LongTensor(pad_text),torch.LongTensor(labels)
x_train, x_test, y_train, y_test = train_test_split(pad_text, labels, test_size=0.1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
print(x_train.shape)
#x_train=x_train.reshape(32,1000,-1)
#print(x_train.shape)
print(y_train.shape)
#y_train=to_categorical(y_train,2)
#y_test=to_categorical(y_test,2)
#pad_text,labels=torch.tensor(pad_text).float(),torch.LongTensor(labels)

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
train_ds = Mydataset(x_train, y_train)
test_ds = Mydataset(x_test, y_test)
# print(train_ds[0])


# batch_size = [8,16,32]
batch_size = 16

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
data_dl =torch.utils.data.DataLoader(data_ds,batch_size=batch_size,shuffle=True)

print(len(train_dl))

class PositionalEncoding(nn.Module):
    "Implement the PE function."

#     d_model：表示输入向量的维度（通常也是 Transformer 模型中的隐藏层维度）。
#     dropout：表示应用于位置编码的 dropout 概率。
#     max_len：表示输入文本的最大长度。
    def __init__(self, d_model, dropout=0.1, max_len=text_len) :
        super(PositionalEncoding, self).__init__()
#         实现Dropout正则化减少过拟合
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
#
#
# # embed_dim = 2 ** (int(np.log2(len(word_list) ** 0.25)) + 2)  # 经验值
#
embedding_dim = 50
print('embedding_dim:', embedding_dim)


# nhead = [8,10,12]
# num_layers = [3,4,5]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.em = nn.Embedding(len(word_list) + 1, embedding_dim=50)  # 对0也需要编码
        self.pos = PositionalEncoding(embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 2)


    def forward(self, inputs):
        x = self.em(inputs)                                  #[16,1000,50]
        x = self.pos(x)                                     #[16,1000,50]
        x = x.float()                                       #[16,1000,50]
        x = self.transformer_encoder(x)                     #[16,1000,50]
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view(-1, 3000)                                   #[16,3000]
        x = torch.sum(x, dim=0)                             #[1000,50]
        x = F.relu(self.fc1(x))                             #[1000,128]
        x = self.fc2(x)                                     #[1000,2]
        return x





model = Net()
model = model.to(device)

loss = nn.CrossEntropyLoss()
loss = loss.to(device)
criterion=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
# %%time
model.train()
#训练10轮
TOTAL_EPOCHS=100
#记录损失函数
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_dl):
        x = x.float() #输入必须未float类型
        y = y.long() #结果标签必须未long类型
        #清零
        optimizer.zero_grad()
        outputs = model(x)
        #计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, np.mean(losses)))

from sklearn.metrics import recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score

model.eval()
correct = 0
total = 0
for i,(x, y) in enumerate(val_dl):
    x = x.float()
    y = y.long()
    outputs = model(x).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
    f1 = f1_score(y,predicted)
    mcc = matthews_corrcoef(y, predicted)
    roc_auc = roc_auc_score(y, predicted)
    conf_matrix = confusion_matrix(y,predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)
    # 计算Recall
    recall = recall_score(y, predicted)
    specificity = tn / (tn + fp)
print('准确率: %.4f %%' % (100 * correct / total))
print(f'Val f1 score:{f1},Val Roc Auc score:{roc_auc},Val Mcc:{mcc},Val NPV:{npv},Val PPV:{ppv},Val recall:{recall},Val specificity:{specificity}')

model.eval()
correct = 0
total = 0
for i,(x, y) in enumerate(test_dl):
    x = x.float()
    y = y.long()
    outputs = model(x).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
    f1 = f1_score(y,predicted)
    mcc = matthews_corrcoef(y, predicted)
    roc_auc = roc_auc_score(y, predicted)
    conf_matrix = confusion_matrix(y,predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)
    # 计算Recall
    recall = recall_score(y, predicted)
    specificity = tn / (tn + fp)
print('准确率: %.4f %%' % (100 * correct / total))
print(f'Test f1 score:{f1},Test Roc Auc score:{roc_auc},Test Mcc:{mcc},Test NPV:{npv},Test PPV:{ppv},Test recall:{recall},Test specificity:{specificity}')

with torch.no_grad():
    outputs=[]
    for x,y in data_dl:
        out = model(x)
        outputs.append(out)
        #print(out.shape)
    print(outputs)
    print(outputs[0].shape)
    print(len(outputs))
    #sys.exit()
#r1 = [i.reshape(16000,1) for i in out] 
outputs1 = torch.concat(outputs,axis =0)
print(outputs1)

deep_fea_tran =pd.DataFrame(outputs1)
deep_fea_tran.to_csv(r"C:\Users\林夕\Desktop\重提测试集深度特征\Zea_mays-transformer.csv")
