import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField

data = pd.read_csv(r"D:\植物独立测试集基础数据\Zea mays_14812.csv")
seq=data['Sequence']
labels = data.label.values
pat = re.compile('[ACGTacgt]')
def del_end_xing(text):  # 用于删除序列末尾的*
    text = pat.findall(str(text))
    text = [each.lower() for each in text]
    return text
# 读取数据的每个批次大小
batch_size = 500
# 计算总行数
total_rows = len(seq)
print(total_rows)
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
    text_len = 600  # 序列长度的中位数为1344  植物600；小鼠1500；人类1400
    pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
    pad_text = np.array(pad_text)
    a = np.arange(1, 601)
    a = np.array(a)

    for i in pad_text:
        c += 1


        gaf = MarkovTransitionField(image_size=600)

        x = i.reshape(1,-1)
        img_gaf = gaf.fit_transform(x)
        img_gaf = img_gaf.reshape(-1,600)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_gaf)
        if c<=7406:
            plt.savefig(r"C:\MTF_plant\Zea mays\0\%d.png"%c)
            plt.close()
        else:
            plt.savefig(r"C:\MTF_plant\Zea mays\1\%d.png"%c)
            plt.close()


