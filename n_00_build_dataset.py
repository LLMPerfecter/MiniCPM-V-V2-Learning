"""
创建适合输入模型的数据集
1. 模型：MiniCPM-V-2模型数据集合准备
2. 现有数据：json数据，包含：id/image/convertsations
3. converstaions包含 人输出的信息，以及模型回答的内容
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

"""创建 MiniCPM-V-2 模型数据集"""
#创建成这样的json样本集合：{"query": "55555", "response": "66666", "images": ["image_path"]}
# 1. 模型：MiniCPM-V-2模型数据集合准备
# 2. 现有数据：json数据，包含：id/image/convertsations
# 3. converstaions包含 人输出的信息，以及模型回答的内容
# 4. 模型数据集：{"query": "55555", "response": "66666", "images": ["image_path"]}

# 数据创建函数
def create_dataset(data, data_image_path, path):
    """
    :param data: 包含id/image/convertsations的json数据
    :param path: 模型数据集路径
    :return:
    """
    # 1. 读取数据
    with open(data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 2. 创建模型数据集
    dataset = []
    for i in tqdm(range(len(data))):
        query = data[i]['conversations'][0]["value"].replace("<image>\n","")  # 查询语句
        response = data[i]['conversations'][-1]["value"]  # 回答语句
        # images = [os.path.join(path, data[i]['image']) for _ in range(3)]  # 图片路径
        images = [data_image_path + data[i]['image']]  # 图片路径
        # print(images)
        dataset.append({'query': query, 'response': response, 'images': images})
    # 3. 保存模型数据集
    with open('MiniCPM_V2_Dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)
        print('数据集创建完成！')
        print('数据集路径：', 'MiniCPM_V2_Dataset.json')
        print('数据集大小：', len(dataset))
        print('数据集格式：{"query": "55555", "response": "66666", "images": ["图片路径"]}')  
        print('数据集样例：', dataset[0])


# 调用函
if __name__ == '__main__':
    # 模型数据集路径
    path = '../data/MiniCPM_V2_Dataset'
    # 数据集路径
    data = '../data/cont_rev_fine.json'
    data_images_path = '../MiniCPM-Model-Train/data/'
    create_dataset(data, data_images_path, path)
