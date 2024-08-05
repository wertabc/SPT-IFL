import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score,accuracy_score
import pandas as pd
import numpy as np
import sys
from src.earlyStopping import EarlyStopping

# 假设你有一个名为df的DataFrame，其中包含特征和目标列
# 请替换"features"和"target"为你实际的列名

# 1. 数据准备和标准化
#
data1 = pd.read_csv(r"D:\1d深度特征\基础数据集_tran_fea_171_v.csv")
# data1 = pd.read_csv(r"D:\1d深度特征\基础数据集_gru_fea_191_v.csv")
# data1 = pd.read_csv(r"D:\1d深度特征\基础数据集_lstm_fea_195_v.csv")
data2 = pd.read_csv(r"D:\plant_2d_deep_fea\deep_fea_plant_2d_vit_v_106.csv")
data3 = pd.read_csv(r"D:\RNA特征融合文件夹\基准数据集_plant.csv")
data4 = pd.read_csv(r"D:\RNA特征融合文件夹\86个特征_plant.csv")
labels = data3['label']
onedimension_deep = data1.iloc[:,1:]
twodimensions_deep = data2.iloc[:,1:]
onedimension_hand_fea = data4.iloc[:,1:]
data0=pd.concat([onedimension_hand_fea,onedimension_deep,twodimensions_deep,labels],axis = 1)
# data0=pd.concat([twodimensions_deep,labels],axis = 1)
# data0=pd.concat([onedimension_deep,labels],axis = 1)

# 提取特征和目标
# x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size = 0.2,random_state = 66)

# 独立测试集
solanum_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Solanum tuberosum_16564.csv")
solanum_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Solanum tuberosum_86.csv")
solanum_data2=pd.read_csv(r"D:\1d深度特征\Solanum tuberosum_transformer_v.csv")
# solanum_data2=pd.read_csv(r"D:\1d深度特征\Solanum tuberosum_lstm_v.csv")
# solanum_data2=pd.read_csv(r"D:\1d深度特征\Solanum tuberosum_gru_v.csv")
solanum_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Solanum tuberosum_2d_vit_v.csv")
solanum_labels=solanum_data0['label']
solanum_onedimension_hand = solanum_data1.iloc[:,1:]
solanum_onedimensiondeep = solanum_data2.iloc[:,1:]
solanum_twodimensionsdeep = solanum_data3.iloc[:,1:]
solanum=pd.concat([solanum_onedimension_hand,solanum_onedimensiondeep,solanum_twodimensionsdeep,solanum_labels],axis =1)
# solanum=pd.concat([solanum_onedimension_hand,solanum_labels],axis =1)
# solanum=pd.concat([solanum_twodimensionsdeep,solanum_labels],axis =1)
# solanum = pd.concat([solanum_onedimensiondeep, solanum_labels], axis=1)

# print(X_test)
# sys.exit()

sorghum_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Sorghum bicolor_17314.csv")
sorghum_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Sorghum bicolor_86.csv")
sorghum_data2=pd.read_csv(r"D:\1d深度特征\Sorghum bicolor_transformer_v.csv")
# sorghum_data2=pd.read_csv(r"D:\1d深度特征\Sorghum bicolor_lstm_v.csv")
# sorghum_data2=pd.read_csv(r"D:\1d深度特征\Sorghum bicolor_gru_v.csv")
sorghum_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Sorghum bicolor_2d_vit_v.csv")
sorghum_labels=sorghum_data0['label']
sorghum_onedimension_hand = sorghum_data1.iloc[:,1:]
sorghum_onedimensiondeep = sorghum_data2.iloc[:,1:]
sorghum_twodimensionsdeep = sorghum_data3.iloc[:,1:]
sorghum=pd.concat([sorghum_onedimension_hand,sorghum_onedimensiondeep,sorghum_twodimensionsdeep,sorghum_labels],axis =1)
# sorghum=pd.concat([sorghum_onedimension_hand,sorghum_labels],axis =1)
# sorghum=pd.concat([sorghum_twodimensionsdeep,sorghum_labels],axis =1)
# sorghum=pd.concat([sorghum_onedimensiondeep,sorghum_labels],axis =1)

zea_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Zea mays_14812.csv")
zea_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Zea mays_86.csv")
zea_data2=pd.read_csv(r"D:\1d深度特征\Zea mays_transformer_v.csv")
# zea_data2=pd.read_csv(r"D:\1d深度特征\Zea mays_lstm_v.csv")
# zea_data2=pd.read_csv(r"D:\1d深度特征\Zea mays_gru_v.csv")
zea_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Zea mays_2d_vit_v.csv")
zea_labels=zea_data0['label']
zea_onedimension_hand = zea_data1.iloc[:,1:]
zea_onedimensiondeep = zea_data2.iloc[:,1:]
zea_twodimensionsdeep = zea_data3.iloc[:,1:]
zea=pd.concat([zea_onedimension_hand,zea_onedimensiondeep,zea_twodimensionsdeep,zea_labels],axis =1)
# zea=pd.concat([zea_onedimension_hand,zea_labels],axis =1)
# zea=pd.concat([zea_twodimensionsdeep,zea_labels],axis =1)
# zea=pd.concat([zea_onedimensiondeep,zea_labels],axis =1)

goss_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Gossypium darwinii_11244.csv")
goss_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Gossypium darwinii_86.csv")
goss_data2=pd.read_csv(r"D:\1d深度特征\Gossypium darwinii-transformer_v.csv")
# goss_data2=pd.read_csv(r"D:\1d深度特征\Gossypium darwinii-lstm_v.csv")
# goss_data2=pd.read_csv(r"D:\1d深度特征\Gossypium darwinii-gru_v.csv")
goss_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Gossypium darwinii_2d_vit_v.csv")
goss_labels=goss_data0['label']
goss_onedimension_hand = goss_data1.iloc[:,1:]
goss_onedimensiondeep = goss_data2.iloc[:,1:]
goss_twodimensionsdeep = goss_data3.iloc[:,1:]
goss=pd.concat([goss_onedimension_hand,goss_onedimensiondeep,goss_twodimensionsdeep,goss_labels],axis =1)
# goss=pd.concat([goss_onedimension_hand,goss_labels],axis =1)
# goss=pd.concat([goss_twodimensionsdeep,goss_labels],axis =1)
# goss=pd.concat([goss_onedimensiondeep,goss_labels],axis =1)

ori_data = pd.concat([data0,solanum,sorghum,zea,goss],axis=0)

ori_data_x = ori_data.drop(['label'],axis = 1)
# ori_data_x.reset_index(drop=True)
labels = ori_data['label'].values
x_train,x_val,y_train,y_val = train_test_split(ori_data_x,labels,test_size = 0.2,random_state = 66)

target = torch.tensor(labels,dtype=int)
pos_count = (target == 1).sum().item()
neg_count = (target == 0).sum().item()
total_count = pos_count + neg_count
pos_weight = total_count / (2.0 * pos_count)
neg_weight = total_count / (2.0 * neg_count)
w = torch.tensor([pos_weight,neg_weight],dtype = float)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_val_scaled = scaler.transform(x_val)
#


eye_matrix = torch.eye(2)
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train).view(-1, 1) # Assuming y_train is a column vector
y_train_tensor1 = torch.tensor(eye_matrix[y_train_tensor.view(-1)].view(-1, 2))
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.LongTensor(y_val).view(-1, 1)
y_val_tensor1= torch.tensor(eye_matrix[y_val_tensor.view(-1)].view(-1, 2))






# 独立测试集
test1_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Cicer_arietinum_4198.csv")
test1_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Cicer arietinum_86.csv")
test1_data2=pd.read_csv(r"D:\1d深度特征\Cicer_arietinum-transformer_v.csv")
# test1_data2=pd.read_csv(r"D:\1d深度特征\Cicer_arietinum_gru_v.csv")
# test1_data2=pd.read_csv(r"D:\1d深度特征\Cicer_arietinum_lstm_v.csv")
test1_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Cicer_arietinum_2d_vit_v.csv")

test2_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Lactuca sativa_9364.csv")
test2_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Lactuca sativ_86.csv")
test2_data2=pd.read_csv(r"D:\1d深度特征\Lactuca sativa-transformer_v.csv")
# test2_data2=pd.read_csv(r"D:\1d深度特征\Lactuca sativa-gru_v.csv")
# test2_data2=pd.read_csv(r"D:\1d深度特征\Lactuca sativa_lstm_v.csv")
test2_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Lactuca sativa_2d_vit_v.csv")

test3_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Manihot esculenta_5616.csv")
test3_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Manihot esculenta_86.csv")
test3_data2=pd.read_csv(r"D:\1d深度特征\Manihot esculenta-transformer_v.csv")
# test3_data2=pd.read_csv(r"D:\1d深度特征\Manihot esculenta-gru_v.csv")
# test3_data2=pd.read_csv(r"D:\1d深度特征\Manihot esculenta_lstm_v.csv")
test3_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Manihot esculenta_2d_vit_v.csv")

test4_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Musa acuminata_4122.csv")
test4_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Musa acuminata_86.csv")
test4_data2=pd.read_csv(r"D:\1d深度特征\Musa acuminata-transformer_v.csv")
# test4_data2=pd.read_csv(r"D:\1d深度特征\Musa acuminata-gru_v.csv")
# test4_data2=pd.read_csv(r"D:\1d深度特征\Musa acuminata_lstm_v.csv")
test4_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Musa acuminata_2d_vit_v.csv")

test5_data0=pd.read_csv(r"D:\植物独立测试集基础数据\Nymphaea colorata_3416.csv")
test5_data1=pd.read_csv(r"D:\植物独立测试集86个特征\Nymphaea colorata_86.csv")
test5_data2=pd.read_csv(r"D:\1d深度特征\Nymphaea colorata-transformer_v.csv")
# test5_data2=pd.read_csv(r"D:\1d深度特征\Nymphaea colorata_gru_v.csv")
# test5_data2=pd.read_csv(r"D:\1d深度特征\Nymphaea colorata_lstm_v.csv")
test5_data3=pd.read_csv(r"D:\植物独立测试集二维深度特征\deep_fea_Nymphaea colorata_2d_vit_v.csv")

y_test1=test1_data0['label']
test1_onedimension_hand = test1_data1.iloc[:,1:]
test1_onedimensiondeep = test1_data2.iloc[:,1:]
test1_twodimensionsdeep = test1_data3.iloc[:,1:]
x_test1=pd.concat([test1_onedimension_hand,test1_onedimensiondeep,test1_twodimensionsdeep],axis =1)
# x_test1=pd.concat([test1_onedimension_hand],axis =1)
# x_test1=pd.concat([test1_twodimensionsdeep],axis =1)
# x_test1=pd.concat([test1_onedimensiondeep],axis =1)
X_test1_scaled = scaler.transform(x_test1)
X_test1_tensor = torch.FloatTensor(X_test1_scaled)
y_test1_tensor = torch.LongTensor(y_test1).view(-1, 1)
y_test1_tensor1= torch.tensor(eye_matrix[y_test1_tensor.view(-1)].view(-1, 2))

y_test2=test2_data0['label']
test2_onedimension_hand = test2_data1.iloc[:,1:]
test2_onedimensiondeep = test2_data2.iloc[:,1:]
test2_twodimensionsdeep = test2_data3.iloc[:,1:]
x_test2=pd.concat([test2_onedimension_hand,test2_onedimensiondeep,test2_twodimensionsdeep],axis =1)
# x_test2=pd.concat([test2_onedimension_hand],axis =1)
# x_test2=pd.concat([test2_twodimensionsdeep],axis =1)
# x_test2=pd.concat([test2_onedimensiondeep],axis =1)
X_test2_scaled = scaler.transform(x_test2)
X_test2_tensor = torch.FloatTensor(X_test2_scaled)
y_test2_tensor = torch.LongTensor(y_test2).view(-1, 1)
y_test2_tensor1= torch.tensor(eye_matrix[y_test2_tensor.view(-1)].view(-1, 2))

y_test3=test3_data0['label']
test3_onedimension_hand = test3_data1.iloc[:,1:]
test3_onedimensiondeep = test3_data2.iloc[:,1:]
test3_twodimensionsdeep = test3_data3.iloc[:,1:]
x_test3=pd.concat([test3_onedimension_hand,test3_onedimensiondeep,test3_twodimensionsdeep],axis =1)
# x_test3=pd.concat([test3_onedimension_hand],axis =1)
# x_test3=pd.concat([test3_twodimensionsdeep],axis =1)
# x_test3=pd.concat([test3_onedimensiondeep],axis =1)
X_test3_scaled = scaler.transform(x_test3)
X_test3_tensor = torch.FloatTensor(X_test3_scaled)
y_test3_tensor = torch.LongTensor(y_test3).view(-1, 1)
y_test3_tensor1= torch.tensor(eye_matrix[y_test3_tensor.view(-1)].view(-1, 2))

y_test4=test4_data0['label']
test4_onedimension_hand = test4_data1.iloc[:,1:]
test4_onedimensiondeep = test4_data2.iloc[:,1:]
test4_twodimensionsdeep = test4_data3.iloc[:,1:]
x_test4=pd.concat([test4_onedimension_hand,test4_onedimensiondeep,test4_twodimensionsdeep],axis =1)
# x_test4=pd.concat([test4_onedimension_hand],axis =1)
# x_test4=pd.concat([test4_twodimensionsdeep],axis =1)
# x_test4=pd.concat([test4_onedimensiondeep],axis =1)
X_test4_scaled = scaler.transform(x_test4)
X_test4_tensor = torch.FloatTensor(X_test4_scaled)
y_test4_tensor = torch.LongTensor(y_test4).view(-1, 1)
y_test4_tensor1= torch.tensor(eye_matrix[y_test4_tensor.view(-1)].view(-1, 2))

y_test5=test5_data0['label']
test5_onedimension_hand = test5_data1.iloc[:,1:]
test5_onedimensiondeep = test5_data2.iloc[:,1:]
test5_twodimensionsdeep = test5_data3.iloc[:,1:]
x_test5=pd.concat([test5_onedimension_hand,test5_onedimensiondeep,test5_twodimensionsdeep],axis =1)
# x_test5=pd.concat([test5_onedimension_hand],axis =1)
# x_test5=pd.concat([test5_twodimensionsdeep],axis =1)
# x_test5=pd.concat([test5_onedimensiondeep],axis =1)
X_test5_scaled = scaler.transform(x_test5)
X_test5_tensor = torch.FloatTensor(X_test5_scaled)
y_test5_tensor = torch.LongTensor(y_test5).view(-1, 1)
y_test5_tensor1= torch.tensor(eye_matrix[y_test5_tensor.view(-1)].view(-1, 2))


# 2. 构建神经网络模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        # x = self.dropout(x)
        x = self.activation(self.layer3(x))
        # x = self.dropout(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x


# 初始化模型和优化器
model = BinaryClassifier(input_size=x_train.shape[1])
# criterion = nn.BCELoss()
# criterion = nn.BCELoss(weight= w,reduction='mean')
criterion = nn.BCEWithLogitsLoss(weight= w,reduction='mean')
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.AdamW(model.parameters(), lr=0.001)


for patience in range(5,51,5):
    # 初始化早停对象
    early_stopping = EarlyStopping(verbose=True,patience=patience)  #patience的取值对验证集表现的影响

    # 训练模型
    train_losses = []
    val_losses = []
    # 模型训练
    for epoch in range(6000):  # 设定最大的训练周期数
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)

        _, predicted = torch.max(outputs.data,1)

        # predicted = predicted.reshape(-1,1)
        loss = criterion(outputs, y_train_tensor1)

        loss.backward()
        optimizer.step()
        # scheduler.step()
        train_losses.append(loss.item())

        # 在验证集上计算损失
        model.eval()
        # f1_total = 0
        # precision_total = 0
        # mcc_total = 0
        # roc_auc_total = 0
        # conf_matrix_total = np.zeros((2, 2))
        # npv_total=0
        # ppv_total=0
        # recall_total = 0
        # specificity_total = 0
        with torch.no_grad():
            outputs_val = model(X_val_tensor)
            _, predicted = torch.max(outputs_val.data, 1)
            # predicted = predicted.reshape(-1, 1)
            # print(y_val.shape)
            # sys.exit()
            # accuracy = (predicted == y_val_tensor).sum().item() / len(y_val)
            accuracy = accuracy_score(predicted,y_val_tensor)
            # accuracy = accuracy_score(y_val_tensor, predicted)
            # acc_list.append(accuracy)
            # 计算F1 Score
            f1 = f1_score(predicted,y_val_tensor)
            # f1_total+=f1
            precision = precision_score(predicted,y_val_tensor)
            # precision_total += precision
            mcc = matthews_corrcoef(predicted,y_val_tensor)
            # mcc_total+=mcc
            roc_auc = roc_auc_score(predicted,y_val_tensor)
            # roc_auc_total+=roc_auc
            conf_matrix = confusion_matrix(predicted,y_val_tensor)
            # conf_matrix_total += conf_matrix
            tn, fp, fn, tp = conf_matrix.ravel()
            npv = tn / (tn + fn)
            # npv_total+=npv
            ppv = tp / (tp + fp)
            # ppv_total+=ppv
            # 计算Recall
            recall = recall_score(predicted,y_val_tensor)
            # recall_total = recall_score(y_val_tensor, predicted)
            specificity = tn / (tn + fp)
            # specificity_total += specificity
            # _, predicted = torch.max(outputs.data, 1)
            # predicted = predicted.reshape(-1, 1)
            val_loss = criterion(outputs_val,y_val_tensor1)
        val_losses.append(val_loss.item())
        # 早停检查
        early_stopping(val_loss, model)
        # print(f"Epoch [{epoch}/{6000}]")
        if early_stopping.early_stop:
            print("Early stopping")
            print(f'Val Accuracy: {accuracy},Val f1 score:{f1},'
                  f'Val Roc Auc score:{roc_auc},Val Mcc:{mcc},Val NPV:{npv},'
                  f'Val PPV:{ppv},Val recall:{recall},Val specificity:{specificity}')
            break
    # acc_list.append(accuracy)
    # 绘制训练和验证损失曲线
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Curve')
    # plt.legend()
    # plt.savefig(f"./pic_mouse/{patience}_lstm_logit.png")
    # plt.close()
    # plt.show()
    # plt.show()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test1_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(predicted,y_test1_tensor)
        # 计算F1 Score
        f1 = f1_score(predicted,y_test1_tensor)
        # f1_total+=f1
        precision = precision_score(predicted,y_test1_tensor)
        # precision_total += precision
        mcc = matthews_corrcoef(predicted,y_test1_tensor)
        # mcc_total+=mcc
        roc_auc = roc_auc_score(predicted,y_test1_tensor)
        # roc_auc_total+=roc_auc
        conf_matrix = confusion_matrix(predicted,y_test1_tensor)
        # conf_matrix_total += conf_matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        npv = tn / (tn + fn)
        # npv_total+=npv
        ppv = tp / (tp + fp)
        # ppv_total+=ppv
        # 计算Recall
        recall = recall_score(predicted,y_test1_tensor)
        # recall_total += recall
        specificity = tn / (tn + fp)
        # specificity_total += specificity
    print("TEST1:")
    print(
        f'Test Accuracy: {accuracy},f1 score:{f1},Test Roc Auc score:{roc_auc},Test Mcc:{mcc},Test NPV:{npv},Test PPV:{ppv},Test recall:{recall},Test specificity:{specificity}')

# 5. 模型评估
# 模型测试
    model.eval()
    with torch.no_grad():
        outputs = model(X_test2_tensor)
        _, predicted = torch.max(outputs.data,1)
        accuracy = accuracy_score(predicted,y_test2_tensor)
        # 计算F1 Score
        f1 = f1_score(predicted,y_test2_tensor)
        # f1_total+=f1
        precision = precision_score(predicted,y_test2_tensor)
        # precision_total += precision
        mcc = matthews_corrcoef(predicted,y_test2_tensor)
        # mcc_total+=mcc
        roc_auc = roc_auc_score(predicted,y_test2_tensor)
        # roc_auc_total+=roc_auc
        conf_matrix = confusion_matrix(predicted,y_test2_tensor)
        # conf_matrix_total += conf_matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        npv = tn / (tn + fn)
        # npv_total+=npv
        ppv = tp / (tp + fp)
        # ppv_total+=ppv
        # 计算Recall
        recall = recall_score(predicted,y_test2_tensor)
        # recall_total += recall
        specificity = tn / (tn + fp)
        # specificity_total += specificity
    print("TEST2:")
    print(f'Test Accuracy: {accuracy},f1 score:{f1},Test Roc Auc score:{roc_auc},Test Mcc:{mcc},Test NPV:{npv},Test PPV:{ppv},Test recall:{recall},Test specificity:{specificity}')

    model.eval()
    with torch.no_grad():
        outputs = model(X_test3_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(predicted,y_test3_tensor)
        # 计算F1 Score
        f1 = f1_score(predicted,y_test3_tensor)
        # f1_total+=f1
        precision = precision_score(predicted,y_test3_tensor)
        # precision_total += precision
        mcc = matthews_corrcoef(predicted,y_test3_tensor)
        # mcc_total+=mcc
        roc_auc = roc_auc_score(predicted,y_test3_tensor)
        # roc_auc_total+=roc_auc
        conf_matrix = confusion_matrix(predicted,y_test3_tensor)
        # conf_matrix_total += conf_matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        npv = tn / (tn + fn)
        # npv_total+=npv
        ppv = tp / (tp + fp)
        # ppv_total+=ppv
        # 计算Recall
        recall = recall_score(predicted,y_test3_tensor)
        # recall_total += recall
        specificity = tn / (tn + fp)
        # specificity_total += specificity
    print("TEST3:")
    print(
        f'Test Accuracy: {accuracy},f1 score:{f1},Test Roc Auc score:{roc_auc},Test Mcc:{mcc},Test NPV:{npv},Test PPV:{ppv},Test recall:{recall},Test specificity:{specificity}')

    model.eval()
    with torch.no_grad():
        outputs = model(X_test4_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(predicted,y_test4_tensor)
        # 计算F1 Score
        f1 = f1_score(predicted,y_test4_tensor)
        # f1_total+=f1
        precision = precision_score(predicted,y_test4_tensor)
        # precision_total += precision
        mcc = matthews_corrcoef(predicted,y_test4_tensor)
        # mcc_total+=mcc
        roc_auc = roc_auc_score(predicted,y_test4_tensor)
        # roc_auc_total+=roc_auc
        conf_matrix = confusion_matrix(predicted,y_test4_tensor)
        # conf_matrix_total += conf_matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        npv = tn / (tn + fn)
        # npv_total+=npv
        ppv = tp / (tp + fp)
        # ppv_total+=ppv
        # 计算Recall
        recall = recall_score(predicted,y_test4_tensor)
        # recall_total += recall
        specificity = tn / (tn + fp)
        # specificity_total += specificity
    print("TEST4:")
    print(
        f'Test Accuracy: {accuracy},f1 score:{f1},Test Roc Auc score:{roc_auc},Test Mcc:{mcc},Test NPV:{npv},Test PPV:{ppv},Test recall:{recall},Test specificity:{specificity}')

    model.eval()
    with torch.no_grad():
        outputs = model(X_test5_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(predicted,y_test5_tensor)
        # 计算F1 Score
        f1 = f1_score(predicted,y_test5_tensor)
        # f1_total+=f1
        precision = precision_score(predicted,y_test5_tensor)
        # precision_total += precision
        mcc = matthews_corrcoef(predicted,y_test5_tensor)
        # mcc_total+=mcc
        roc_auc = roc_auc_score(predicted,y_test5_tensor)
        # roc_auc_total+=roc_auc
        conf_matrix = confusion_matrix(predicted,y_test5_tensor)
        # conf_matrix_total += conf_matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        npv = tn / (tn + fn)
        # npv_total+=npv
        ppv = tp / (tp + fp)
        # ppv_total+=ppv
        # 计算Recall
        recall = recall_score(predicted,y_test5_tensor)
        # recall_total += recall
        specificity = tn / (tn + fp)
        # specificity_total += specificity
    print("TEST5:")
    print(
        f'Test Accuracy: {accuracy},f1 score:{f1},Test Roc Auc score:{roc_auc},Test Mcc:{mcc},Test NPV:{npv},Test PPV:{ppv},Test recall:{recall},Test specificity:{specificity}')



