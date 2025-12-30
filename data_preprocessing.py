import pandas as pd
from sklearn.model_selection import train_test_split
import json

import csv
import os
import pandas as pd

def create_padded_message(message, total_length=80):
    # 计算消息中的中文字符数量
    chinese_chars_count = sum(1 for char in message if '\u4e00' <= char <= '\u9fff')
    
    # 调整填充长度
    adjusted_length = total_length - len(message) - chinese_chars_count
    left_padding = adjusted_length // 2
    right_padding = adjusted_length - left_padding
    return '=' * left_padding + ' ' + message + ' ' + '=' * right_padding+'\n'



def process_faq(file_test, file_faq, scene):
    faqs_set = set()

    # 读取 file_train 并提取第二列，跳过第一行
    with open(file_test, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行（标题行）
        for row in reader:
            if len(row) > 1:  # 确保行中有至少两个元素
                faqs_set.add(row[1])  # 添加第二列的内容

    print(create_padded_message(f"处理 {scene}，提取并去重第二列完成, 共 {len(faqs_set)} 条数据"))

    # 将去重后的列表写入 file_faq
    with open(file_faq, 'w', encoding='utf-8') as file:
        file.write('id,faq')
        for index,faq in enumerate(faqs_set):
            file.write(f'\n{index},{faq}')

    print(create_padded_message(f"写入 {scene} 完成, 共 {len(faqs_set)} 条数据"))

def process_data(file_train, file_test, output_dir, file_faq, scene):
    # 确保输出目录存在
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # 读取并处理训练和测试文件
    def process_file(file):
        df = pd.read_csv(file)
        
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]  # 使用iloc
        else:
            raise ValueError(f"文件 {file} 应该大于两列")
        df.columns = ['query', 'label_name']
        return df

    print(create_padded_message(f"开始处理 {scene} 训练文件"))
    train_df = process_file(file_train)
    print(create_padded_message(f"{scene} 训练文件处理完成, 共 {len(train_df)} 行数据"))

    print(create_padded_message(f"开始处理 {scene} 测试文件"))
    test_df = process_file(file_test)
    print(create_padded_message(f"{scene} 测试文件处理完成, 共 {len(test_df)} 行数据"))

    
    name2id = {}
    id2name = {}
    
    # 读取 FAQ 文件并创建映射
    with open(file_faq, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if line:
                ID,name = line.split(',',1)
                name2id[name]=ID
                id2name[ID]=name

    # 保存 name2id 和 id2name 字典到 JSON 文件
    with open(os.path.join(output_dir, f"{scene}_name2id.json"), 'w', encoding='utf-8') as f:
        json.dump(name2id, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, f"{scene}_id2name.json"), 'w', encoding='utf-8') as f:
        json.dump(id2name, f, ensure_ascii=False, indent=4)

    print(create_padded_message(f"{scene} name2id 和 id2name 字典保存完成"))

    # 更新 DataFrame
    for df in [train_df, test_df]:
        df['label'] = df['label_name'].map(name2id)
        df.dropna(subset=['label'], inplace=True)  # 删除'label'列中含有NaN的行


    # 保存处理后的文件
    train_df.to_csv(os.path.join(output_dir, 'train/train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test/test.csv'), index=False)

    print(create_padded_message(f"{scene} 训练和测试文件保存完成"))

    return name2id, id2name




# 数据处理的场景
scenes = ['hitch_driver', 'hitch_passenger', 'taxi_driver', 'taxi_passenger','scenic']
for scene in scenes:
    file_train = f'../data/source/{scene}_data/train/train.csv'
    file_test = f'../data/source/{scene}_data/test/test.csv'
    out_dir = f'../data/process/{scene}_data'
    file_faq = f'../data/process/{scene}_data/faqs.csv'
    process_faq(file_train, file_faq,scene)
    process_data(file_train, file_test,out_dir,file_faq,scene)
    
    

