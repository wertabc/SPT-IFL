import pandas as pd
import numpy as np
import re
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_excel(r"D:\mouse_m21\mouse_m21.xlsx")

seq=data['Sequence']
labels = data.label.values

pat = re.compile('[ACGTacgt]')

def del_end_xing(text):  # 用于删除序列末尾的*
    text = pat.findall(str(text))
    text = [each.lower() for each in text]
    return text


import sys

# 读取数据的每个批次大小
batch_size = 8
# 计算总行数
total_rows = len(seq)
# print(total_rows)

# 计算批次数
num_batches = (total_rows // batch_size) + 1

# 分批次读取数据
c = 0
for batch_number in range(num_batches):
    start_index = batch_number * batch_size
    end_index = min((batch_number + 1) * batch_size, total_rows)

    # 从 DataFrame 中获取当前批次的数据
    current_batch = seq.iloc[start_index:end_index]

    # 在这里可以对当前批次的数据进行处理或使用
    x = current_batch.apply(del_end_xing)
    word_set = set()
    for lst in x:
        for word in lst:
            word_set.add(word)

    word_list = list(word_set)
    word_index = dict([(each, word_list.index(each) + 1) for each in word_list])
    text = x.apply(lambda x: [word_index.get(word, 0) for word in x])
    text_len = 1500  # 序列长度的中位数为1344  植物600；小鼠1500；人类1400
    pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
    pad_text = np.array(pad_text)


    # PAA
    transformer = PiecewiseAggregateApproximation(window_size=1)
    a = np.arange(1, 1501)
    # print(a)
    a = np.array(a)
    for i in pad_text:
        c += 1
        i = np.array(i)
        # print(i)
        I = np.concatenate((a.reshape(1, -1), i.reshape(1, -1)), axis=0)
        # print(I)
        # I.reshape(-1,1400)
        I = I.reshape(-1, 1500)
        # print(I)

        result_I = transformer.transform(I)
        # Scaling in interval [0,1]
        scaler = MinMaxScaler()
        scaled_I = scaler.transform(result_I)
        arccos_I = np.arccos(scaled_I[1, :])
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(a, arccos_I)
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5, 2]) # Less radial ticks
        ax.set_rlabel_position(-22.5) # Move radial labels away from plotted line
        ax.grid(True)
        ax.set_title('Polar coordinates', va='bottom')
        plt.show()
        field = [a + b for a in arccos_I for b in arccos_I]
        gram = np.cos(field).reshape(-1, 1500)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gram)
    if c>5:
        break
print("Completed!")