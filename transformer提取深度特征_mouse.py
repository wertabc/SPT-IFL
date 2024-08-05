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

data=pd.read_excel(r"D:\human_and_mouse\mouse\Mouse.xlsx")
sequences = data['Sequence']
labels = data['label'].values
# print(labels)

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

text_len = 1400

pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
pad_text = np.array(pad_text)
# pad_text

pad_text,labels=torch.LongTensor(pad_text),torch.LongTensor(labels)
x_train, x_test, y_train, y_test = train_test_split(pad_text, labels, test_size=0.4)
x_val,x_test,y_val,y_test = train_test_split(x_test,y_test,test_size=0.5)

# print(x_train.shape)
#x_train=x_train.reshape(32,1000,-1)
#print(x_train.shape)
# print(y_train.shape)
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

# data_ds = Mydataset(pad_text,labels)
train_ds = Mydataset(x_train, y_train)
val_ds = Mydataset(x_val, y_val)
test_ds = Mydataset(x_test, y_test)
# print(train_ds[0])


# batch_size = [8,16,32]
batch_size = 16

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
# data_dl =torch.utils.data.DataLoader(data_ds,batch_size=batch_size,shuffle=True)
# print(len(train_dl))

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

# embed_dim = 2 ** (int(np.log2(len(word_list) ** 0.25)) + 2)  # 经验值

embedding_dim = 50
print('embedding_dim:', embedding_dim)


# nhead = [8,10,12]
# num_layers = [3,4,5]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.em = nn.Embedding(len(word_list) + 1, embedding_dim=50)  # 对0也需要编码
        self.pos = PositionalEncoding(embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, inputs):
        x = self.em(inputs)
        x = self.pos(x)
        x = x.float()
        x = self.transformer_encoder(x)
        x = torch.sum(x, dim=0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
model = model.to(device)

loss = nn.CrossEntropyLoss()
loss = loss.to(device)
criterion=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
def fit(model, optimizer, train_dl,val_dl, test_dl):
    tr_correct = 0  # 预测正确的个数
    tr_total = 0  # 总样本数
    tr_loss = 0
    tr_TP = 0
    tr_TN = 0
    tr_FP = 0
    tr_FN = 0

    model.train()  # 训练模式
    for x, y in train_dl:
        x = x.permute(1, 0)
        # x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss_value = loss(y_pred, y)
        # flood=(loss_value - 0.002).abs() + 0.002  # 洪泛函数：防止过拟合
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)  # 在预测结果 y_pred 的每个样本上找到具有最大值的索引
            tr_correct += (y_pred == y).sum().item()
            tr_TP += ((y_pred == y) & (y == 1)).sum().item()
            tr_FN += ((y_pred != y) & (y == 1)).sum().item()
            tr_FP += ((y_pred != y) & (y == 0)).sum().item()
            tr_TN += ((y_pred == y) & (y == 0)).sum().item()
            tr_total += len(y)
            tr_loss += loss_value.item()  # 最后的loss还要除以batch数

    """1个epoch训练结束后，计算训练集的各个指标"""
    epoch_tr_loss = tr_loss / len(train_dl)
    epoch_tr_accuracy = tr_correct / tr_total
    epoch_tr_MCC = (tr_TP * tr_TN - tr_TP * tr_FN) / (
        math.sqrt((tr_TP + tr_FP) * (tr_TP + tr_FN) * (tr_TN + tr_FP) * (tr_TN + tr_FN)))
    epoch_tr_SE = tr_TP / (tr_TP + tr_FN)
    epoch_tr_SPC = tr_TN / (tr_TN + tr_FP)
    epoch_tr_PPV = tr_TP / (tr_TP + tr_FP)
    epoch_tr_NPV = tr_TN / (tr_TN + tr_FN)
    epoch_tr_recall = tr_TP / (tr_TP + tr_FN)
    epoch_tr_precision = tr_TP / (tr_TP + tr_FP)
    epoch_tr_F1 = (2 * epoch_tr_precision * epoch_tr_recall) / (epoch_tr_precision + epoch_tr_recall)

    val_correct = 0  # 预测正确的个数
    val_total = 0  # 总样本数
    val_loss = 0
    val_TP = 0
    val_TN = 0
    val_FP = 0
    val_FN = 0

    model.eval()  # 验证模式
    with torch.no_grad():  # 是一个上下文管理器，用于在块中禁用梯度计算，以减少内存使用并加快计算速度
        for x, y in val_dl:
            x = x.permute(1, 0)
            # x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss_value = loss(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            te_correct += (y_pred == y).sum().item()
            te_TP += ((y_pred == y) & (y == 1)).sum().item()
            te_FN += ((y_pred != y) & (y == 1)).sum().item()
            te_FP += ((y_pred != y) & (y == 0)).sum().item()
            te_TN += ((y_pred == y) & (y == 0)).sum().item()
            te_total += len(y)
            te_loss += loss_value.item()

    """1个epoch训练结束后，计算测试集的各个指标"""
    epoch_val_loss = val_loss / len(val_dl)
    epoch_val_accuracy = val_correct / val_total
    epoch_val_MCC = (val_TP * val_TN - val_TP * val_FN) / (
        math.sqrt((val_TP + val_FP) * (val_TP + val_FN) * (val_TN + val_FP) * (val_TN + val_FN)))
    epoch_val_SE = val_TP / (val_TP + val_FN)
    epoch_val_SPC = val_TN / (val_TN + val_FP)
    epoch_val_PPV = val_TP / (val_TP + val_FP)
    epoch_val_NPV = val_TN / (val_TN + val_FN)
    epoch_val_recall = val_TP / (val_TP + val_FN)
    epoch_val_precision = val_TP / (val_TP + val_FP)
    epoch_val_F1 = (2 * epoch_val_precision * epoch_val_recall) / (epoch_val_precision + epoch_val_recall)

    te_correct = 0  # 预测正确的个数
    te_total = 0  # 总样本数
    te_loss = 0
    te_TP = 0
    te_TN = 0
    te_FP = 0
    te_FN = 0

    model.eval()  # 评估模式
    with torch.no_grad():  # 是一个上下文管理器，用于在块中禁用梯度计算，以减少内存使用并加快计算速度
        for x, y in test_dl:
            x = x.permute(1, 0)
            # x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss_value = loss(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            te_correct += (y_pred == y).sum().item()
            te_TP += ((y_pred == y) & (y == 1)).sum().item()
            te_FN += ((y_pred != y) & (y == 1)).sum().item()
            te_FP += ((y_pred != y) & (y == 0)).sum().item()
            te_TN += ((y_pred == y) & (y == 0)).sum().item()
            te_total += len(y)
            te_loss += loss_value.item()

    """1个epoch训练结束后，计算测试集的各个指标"""
    epoch_te_loss = te_loss / len(test_dl)
    epoch_te_accuracy = te_correct / te_total
    epoch_te_MCC = (te_TP * te_TN - te_TP * te_FN) / (
        math.sqrt((te_TP + te_FP) * (te_TP + te_FN) * (te_TN + te_FP) * (te_TN + te_FN)))
    epoch_te_SE = te_TP / (te_TP + te_FN)
    epoch_te_SPC = te_TN / (te_TN + te_FP)
    epoch_te_PPV = te_TP / (te_TP + te_FP)
    epoch_te_NPV = te_TN / (te_TN + te_FN)
    epoch_te_recall = te_TP / (te_TP + te_FN)
    epoch_te_precision = te_TP / (te_TP + te_FP)
    epoch_te_F1 = (2 * epoch_te_precision * epoch_te_recall) / (epoch_te_precision + epoch_te_recall)

    return epoch_tr_loss, epoch_tr_accuracy, epoch_tr_MCC, epoch_tr_SE, epoch_tr_F1, epoch_tr_SPC, epoch_tr_PPV, epoch_tr_NPV, epoch_te_loss, epoch_te_accuracy, epoch_te_MCC, epoch_te_SE, epoch_te_SPC, epoch_te_PPV, epoch_te_F1, epoch_te_NPV,epoch_val_loss, epoch_val_accuracy, epoch_val_MCC, epoch_val_SE, epoch_val_SPC, epoch_val_PPV, epoch_val_F1, epoch_val_NPV
#     return epoch_tr_loss, epoch_tr_accuracy,epoch_te_loss, epoch_te_accuracy

tr_loss = []
tr_accuracy = []
tr_MCC = []
tr_SE = []
tr_SPC = []
tr_PPV=[]
tr_NPV=[]
tr_AUC=[]
tr_F1=[]

val_loss = []
val_accuracy = []
val_MCC = []
val_SE = []
val_SPC = []
val_PPV=[]
val_NPV=[]
val_AUC=[]
val_F1=[]

te_loss = []
te_accuracy = []
te_MCC = []
te_SE = []
te_SPC = []
te_PPV=[]
te_NPV=[]
te_AUC=[]
te_F1=[]
from sklearn import metrics

epochs = 80

for epoch in range(epochs):
    print(f'{epoch} : ', end='')
    #     epoch_tr_loss, epoch_tr_accuracy,epoch_te_loss, epoch_te_accuracy= fit(model, optimizer, train_dl, test_dl)
    epoch_tr_loss, epoch_tr_accuracy, epoch_tr_MCC, epoch_tr_SE, epoch_tr_F1, epoch_tr_SPC, epoch_tr_PPV, epoch_tr_NPV, epoch_te_loss, epoch_te_accuracy, epoch_te_MCC, epoch_te_SE, epoch_te_SPC, epoch_te_PPV, epoch_te_F1, epoch_te_NPV,epoch_val_loss, epoch_val_accuracy, epoch_val_MCC, epoch_val_SE, epoch_val_SPC, epoch_val_PPV, epoch_val_F1, epoch_val_NPV = fit(
        model, optimizer, train_dl, val_dl,test_dl)
    tr_loss.append(epoch_tr_loss)
    tr_accuracy.append(epoch_tr_accuracy)
    tr_MCC.append(epoch_tr_MCC)
    tr_SE.append(epoch_tr_SE)
    tr_SPC.append(epoch_tr_SPC)
    tr_PPV.append(epoch_tr_PPV)
    tr_NPV.append(epoch_tr_NPV)
    tr_F1.append(epoch_tr_F1)
    "tr_AUC.append(epoch_tr_AUC)"

    val_loss.append(epoch_val_loss)
    val_accuracy.append(epoch_val_accuracy)
    print(epoch_val_accuracy)
    val_MCC.append(epoch_val_MCC)
    val_SE.append(epoch_val_SE)
    val_SPC.append(epoch_val_SPC)
    val_PPV.append(epoch_val_PPV)
    val_NPV.append(epoch_val_NPV)
    val_F1.append(epoch_val_F1)
    "te_AUC.append(epoch_te_AUC)"

    te_loss.append(epoch_te_loss)
    te_accuracy.append(epoch_te_accuracy)
    print(epoch_te_accuracy)
    te_MCC.append(epoch_te_MCC)
    te_SE.append(epoch_te_SE)
    te_SPC.append(epoch_te_SPC)
    te_PPV.append(epoch_te_PPV)
    te_NPV.append(epoch_te_NPV)
    te_F1.append(epoch_te_F1)
    "te_AUC.append(epoch_te_AUC)"

column_name = ['loss', 'accuracy', 'MCC', 'SE', 'SPC', 'PPV', 'NPV','F1']

tr_loss = pd.Series(tr_loss)
tr_accuracy = pd.Series(tr_accuracy)
tr_MCC = pd.Series(tr_MCC)
tr_SE = pd.Series(tr_SE)
tr_SPC = pd.Series(tr_SPC)
tr_PPV = pd.Series(tr_PPV)
tr_NPV = pd.Series(tr_NPV)
tr_F1 = pd.Series(tr_F1)

tr_result = pd.concat([tr_loss, tr_accuracy, tr_MCC, tr_SE, tr_SPC, tr_PPV, tr_NPV,tr_F1], axis=1)
tr_result.columns = column_name

column_name = ['loss', 'accuracy', 'MCC', 'SE', 'SPC', 'PPV', 'NPV','F1']

val_loss = pd.Series(val_loss)
val_accuracy = pd.Series(val_accuracy)
val_MCC = pd.Series(val_MCC)
val_SE = pd.Series(val_SE)
val_SPC = pd.Series(val_SPC)
val_PPV = pd.Series(val_PPV)
val_NPV = pd.Series(val_NPV)
val_F1= pd.Series(val_F1)

val_result = pd.concat([val_loss, val_accuracy, val_MCC, val_SE, val_SPC, val_PPV, val_NPV,val_F1], axis=1)
val_result.columns = column_name

column_name = ['loss', 'accuracy', 'MCC', 'SE', 'SPC', 'PPV', 'NPV','F1']

te_loss = pd.Series(te_loss)
te_accuracy = pd.Series(te_accuracy)
te_MCC = pd.Series(te_MCC)
te_SE = pd.Series(te_SE)
te_SPC = pd.Series(te_SPC)
te_PPV = pd.Series(te_PPV)
te_NPV = pd.Series(te_NPV)
te_F1= pd.Series(te_F1)

te_result = pd.concat([te_loss, te_accuracy, te_MCC, te_SE, te_SPC, te_PPV, te_NPV,te_F1], axis=1)
te_result.columns = column_name
#te_result.index = [*range(1, epochs + 1)]

print(te_result)
print(val_result)
val_result.to_csv(r"D:\林夕\Documents\Tran_mouse_val.csv")
te_result.to_csv(r"D:\林夕\Documents\Tran_mouse_test.csv")




# with torch.no_grad():
#     outputs=[]
#     for x,y in data_dl:
#         out = model(x)
#         outputs.append(out)
#         #print(out.shape)
#     print(outputs)
#     print(outputs[0].shape)
#     print(len(outputs))
#     #sys.exit()
# #r1 = [i.reshape(16000,1) for i in out]
# outputs1 = torch.concat(outputs,axis =0)
# print(outputs1)
#
# deep_fea_tran =pd.DataFrame(outputs1)
# deep_fea_tran.to_csv(r"C:\Users\林夕\Desktop\重提测试集深度特征\Zea_mays-transformer.csv")
