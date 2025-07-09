import re

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .DataSet_Model import DataSet_Lables,DataSet_noLables,Transformer_,GRU_,LSTM_
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
def prepare_data_plant(data_plant,text_len=1200):

    # 提取序列和标签
    sequences = data_plant[0]

    # 正则表达式匹配
    pat = re.compile('[AGCTagct]')

    # 预处理函数
    def pre_process(text):
        text = pat.findall(text)
        text = [each.lower() for each in text]
        return text

    # 应用预处理
    x = sequences.apply(pre_process)
    # 创建单词集合
    word_set = set()
    for lst in x:
        for word in lst:
            word_set.add(word)
    # 创建单词索引映射
    word_list = list(word_set)
    word_index = {each: word_list.index(each) + 1 for each in word_list}
    # 将单词转换为索引
    text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
    # 填充文本到固定长度
    pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
    pad_text = np.array(pad_text)

    # 创建数据集实例
    data_ds = DataSet_noLables(pad_text)

    return data_ds,word_list,pad_text


def select_features_by_model_importance(deep_feature,num):
    n_samples, n_features = deep_feature.shape
    num_features_to_select = num
    # 随机选择特征
    np.random.seed(42)
    selected_indices = np.random.choice(n_features, num_features_to_select, replace=False).tolist()

    # 选择特征
    selected_features_dataset = deep_feature.iloc[:,selected_indices]
    new_column_names = [f'{i+1}' for i in range(len(selected_indices))]
    # 将选择出的特征转换为 DataFrame
    selected_features_dataset.columns = new_column_names
    return selected_features_dataset







def get_deep_feature(seq):
    # 数据准备
    data_ds, word_list, pad_text = prepare_data_plant(seq)
    dataLoader = DataLoader(data_ds, batch_size=16, shuffle=False)

    # 模型初始化
    transformer = Transformer_(word_list)
    lstm = LSTM_(word_list, embed_dim=24, hidden_size=20)
    gru = GRU_(word_list, embed_dim=50, hidden_size=20)

    # 模型设置为评估模式
    transformer.eval()
    lstm.eval()
    gru.eval()

    transformer_features = []
    lstm_features = []
    gru_features = []

    labels_list =[]
    # 迭代数据加载器，提取特征
    with torch.no_grad():
        for text in dataLoader:

            texts = text  # 假设批次中的数据是文本数据和标签

            # 获取Transformer特征
            transformer_output = transformer(texts)
            transformer_features.append(transformer_output)


            # 获取LSTM特征
            lstm_output = lstm(texts)
            lstm_features.append(lstm_output)

            # 获取GRU特征
            gru_output = gru(texts)
            gru_features.append(gru_output)

    # 将各批次的特征拼接在一起

    transformer_features = select_features_by_model_importance(pd.DataFrame(torch.concat(transformer_features,dim=0))
                                                       ,num=171)

    lstm_features = select_features_by_model_importance(pd.DataFrame(torch.concat(lstm_features, dim=0))
                                                , num=195)
    gru_features = select_features_by_model_importance(pd.DataFrame(torch.concat(gru_features, dim=0))
                                               , num=191)


    return [transformer_features, lstm_features, gru_features],pad_text






