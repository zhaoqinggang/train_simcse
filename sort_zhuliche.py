import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch.nn.functional as F
import json
import os
import shutil
import glob
import random
import numpy as np
import torch
import sys
import csv
# 配置参数
seed = 42
batch_size = 128
num_epochs = 20
hidden_size = 768
num_classes = 185
early_stop_patience = 10
LR = 5e-5
TRAIN = True  # 根据需要设定是否训练
LOAD = False  # 根据需要设定是否载入预训练模型
EVALUATE = False  # 根据需要设定是否评估
PREDICT = False  # 根据需要设定是否预测
best_val_accuracy = 0.00
tokenizer_name_path = '../../pretrained_model/chinese-roberta-wwm-ext'
max_length = 32
split =0.05
############################################################################
model_version ='zhuliche_data'
pre_path = '../../models/'+model_version
BETTER_MODEL_PATH = '../../models/'+'better_model_'+model_version
SAVE_PATH = pre_path+'/pipei'+model_version+'.pt'
faqs_path = f'../../data/process/{model_version}/faq.csv'
faqs_id2name_path = f'../../data/process/{model_version}/{model_version}_id2name.json'


# 检查路径是否存在
if not os.path.exists(pre_path):
    # 路径不存在，创建路径
    os.makedirs(pre_path)

# 检查路径是否存在
if not os.path.exists(BETTER_MODEL_PATH):
    # 路径不存在，创建路径
    os.makedirs(BETTER_MODEL_PATH)

# 检查文件是否存在
if os.path.exists(SAVE_PATH):
    if not LOAD:
        print(f"=====模型{SAVE_PATH}存在,为避免被覆盖转移置{BETTER_MODEL_PATH}=====")
        # 文件存在，将文件移动到better_model文件夹
        # 获取SAVE_PATH同级所有文件
        dir_path = os.path.dirname(SAVE_PATH)
        files = glob.glob(os.path.join(dir_path, '*'))

        # 遍历所有文件，将文件移动到BETTER_MODEL_PATH
        for file in files:
            shutil.move(file, BETTER_MODEL_PATH)
    else:
        print("加载模型")

    
    

# 设置Python的随机种子
random.seed(seed)

# 设置NumPy的随机种子
np.random.seed(seed)

# 设置PyTorch的随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True




class FaqsDataset:
    def __init__(self, data, tokenizer,max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][1]
        label = self.data[idx][0]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(int(label))
        }


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
#         print(self.data.index)
#         print(index in self.data.index)
        query = self.data.loc[index, 'query']
        label = self.data.loc[index, 'label']

        encoded_input = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }



# 定义SimCSE模型
class SimCSE(nn.Module):
    def __init__(self, model_name_path,hidden_size, num_classes,dropout_rate=0.):
        super(SimCSE, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_path)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]

        
        return embeddings

    
def evaluate(model, dataloader, criterion,stacked_tensor):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    correct_top1, correct_top2, correct_top5 = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            
            embeddings = model(input_ids, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 现在你可以进行矩阵乘法了
            logits1 = torch.mm(embeddings, stacked_tensor)##[batch,188]  
            logits = logits1/0.05

            
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            
            _, predicted_top2 = torch.topk(logits, k=2, dim=1)
            _, predicted_top5 = torch.topk(logits, k=5, dim=1)

            expanded_labels2 = labels.view(-1, 1).expand_as(predicted_top2)
            expanded_labels5 = labels.view(-1, 1).expand_as(predicted_top5)

            correct += (predicted == labels).sum().item()
            correct_top1 += (predicted == labels).sum().item()
            correct_top2 += (predicted_top2 == expanded_labels2).any(dim=1).sum().item()
            correct_top5 += (predicted_top5 == expanded_labels5).sum().item()
            
    accuracy = correct / total
    accuracy_top1 = correct_top1 / total
    accuracy_top2 = correct_top2 / total
    accuracy_top5 = correct_top5 / total
    average_loss = total_loss / len(dataloader)

    return accuracy, average_loss,accuracy_top1, accuracy_top2, accuracy_top5
    
    

def train(model, dataloader, optimizer, criterion,epoch):
    model.train()
    global best_val_accuracy
    total_loss = 0
    correct, total = 0, 0
    early_stop_counter = 0
    global early_stop_patience  # 设定早停的耐心值，即连续多少个epoch验证集的loss没有改善时停止训练
    
    for batch_idx,source in tqdm(enumerate(dataloader)):
        input_ids = source['input_ids'].to(device)
        attention_mask = source['attention_mask'].to(device)
        labels = source['label'].to(device)

        optimizer.zero_grad()
##########===================================#########################################################
        embeddings = model(input_ids, attention_mask)
        
        tensor_list = []
        for _,source_faqs in enumerate(faqs_loader):
            faqs_input_ids = source_faqs['input_ids'].to(device)
            faqs_attention_mask = source_faqs['attention_mask'].to(device)
            faqs_labels = source_faqs['label'].to(device)
            
            faqs_embeddings = model(faqs_input_ids, faqs_attention_mask)
            tensor_list.append(faqs_embeddings)            
        # 假设 tensor_list 是一个包含了188个1*768张量的列表
        # tensor_list = [tensor1, tensor2, ..., tensor188]

        # 使用 torch.stack() 函数将这些张量拼接成一个新的张量
        stacked_tensor = torch.stack(tensor_list)

        # stacked_tensor 的形状现在应该是 [188, 1, 768]，你可以使用 .squeeze() 方法去掉维度为1的维度
        stacked_tensor = stacked_tensor.squeeze()
            
        # 对 b 进行转置操作，使其形状变为 [768, 188]
        stacked_tensor = stacked_tensor.t()

        # 对 embeddings 的每一行进行归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 对 stacked_tensor 的每一列进行归一化
        stacked_tensor = F.normalize(stacked_tensor, p=2, dim=0)

        
        # 现在你可以进行矩阵乘法了
        logits1 = torch.mm(embeddings, stacked_tensor)##[batch,188]
        
        
        logits = logits1/0.05
        
        
        
##########===================================#########################################################
    
        loss = criterion(logits, labels)
        total_loss += loss.item()

        _, predicted = torch.max(logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
#         logger.info('开始计算验证集的效果。。。')
#         val_accuracy, val_loss,_,_,_ = evaluate(model, dev_loader, criterion,stacked_tensor)
#         logger.info('开始计算测试集的效果。。。')
#         test_accuracy, test_loss,_,_,_ = evaluate(model, test_loader, criterion,stacked_tensor)
#         print(val_accuracy,test_accuracy)
        
        
        loss.backward()
        optimizer.step()
        
        train_loss = total_loss / len(dataloader)
        train_accuracy = correct / total

    # 评估
        if batch_idx %50  == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs}::Batch {batch_idx+1}/{len(dataloader)}||||Train Loss: {train_loss:.4f}::Train Accuracy: {train_accuracy*100:.2f}%')
            logger.info('开始计算验证集的效果。。。')
            val_accuracy, val_loss,_,_,_ = evaluate(model, dev_loader, criterion,stacked_tensor)
            logger.info('开始计算测试集的效果。。。')
            test_accuracy, test_loss,_,_,_ = evaluate(model, test_loader, criterion,stacked_tensor)
            print(val_accuracy,test_accuracy)
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stop_counter = 0
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f'Val Loss: {val_loss:.4f} | Higher val Accuracy: {val_accuracy*100:.2f}% | saved model,test Accuracy: {test_accuracy*100:.2f}%')

                with open(pre_path+'/train_state.txt','w') as file:
                    file.write(f"Train Loss: {train_loss:.4f}\n")
                    file.write(f"Train Accuracy: {train_accuracy*100:.2f}%\n")
                    file.write(f"Dev Loss: {val_loss:.4f}\n")
                    file.write(f"Dev Accuracy: {val_accuracy*100:.2f}%\n")
                    file.write(f"Test Loss: {test_loss:.4f}\n")
                    file.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")

            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print('Early stopping triggered.')
                    break



    
def predict(model, texts, tokenizer):
    model.eval()
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)        
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
        with open(faqs_id2name_path, 'r') as f:
            id_to_name = json.load(f)
        # 将predicted_labels映射到predicted_names
        predicted_names = list(map(lambda label: id_to_name.get(str(label)), predicted_labels))
        
                
        my_list = probabilities.cpu().numpy().tolist()
        # 使用列表的元素作为键，索引作为值创建字典
        my_dict = [{index: value for index,value in enumerate(x) } for x in my_list]

        # 使用字典1的值作为键，字典2的值作为值创建新字典
        new_dict = [{ name :  y[int(key)]   for key, name in id_to_name.items()  if int(key) < num_classes }    for y in my_dict]
        
    return predicted_names,new_dict




    
    
    

if __name__ == '__main__':

    
   #===============预处理模型和数据================================================================
    print("="*20+"start"+"="*20)
    
    
    # 初始化模型和tokenizer
    model = SimCSE(tokenizer_name_path,hidden_size, num_classes)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name_path)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    
    train_path = f'../../data/process/{model_version}/train/train.csv'
    test_path = f'../../data/process/{model_version}/test/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data, dev_data = train_test_split(train_df, test_size=split, random_state=seed)
    

    # 重置train_data的索引
    train_data = train_data.reset_index(drop=True)
    dev_data = dev_data.reset_index(drop=True)
    


    # 初始化列表
    faqs_data = []

    # 打开并读取文件
    with open(faqs_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            faqs_data.append((row[0], row[1]))  # 将两列数据作为元组加入列表
            
    faqs_dataset = FaqsDataset(faqs_data,tokenizer, max_length)
    train_dataset = CustomDataset(train_data, tokenizer, max_length)
    dev_dataset = CustomDataset(dev_data, tokenizer, max_length)
    test_dataset = CustomDataset(test_data, tokenizer, max_length)
    
    faqs_loader = DataLoader(faqs_dataset, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
        

    print(f"train_data:{len(train_data)}, dev_data:{len(dev_data)},test_data:{len(test_data)}")
    print("="*20+"数据处理完毕"+"="*20)
    
    if LOAD:
        print("="*20+"load"+"="*20)
        model.load_state_dict(torch.load(SAVE_PATH))
        print("="*20+"模型加载完毕"+"="*20)
        
   ###===============训练======================================================================================
    if TRAIN:
        print("="*20+"train"+"="*20)
        for epoch in range(num_epochs):
            LR = (1-0.1)**epoch * LR
            optimizer = optim.Adam(model.parameters(), lr=LR)
            train(model, train_loader,optimizer, criterion,epoch)
        print(f"num_epochs:{num_epochs},best_val_accuracy:{best_val_accuracy}")
        print("="*20+"模型训练完毕"+"="*20)
          

    if EVALUATE:
        # 在验证集上评估模型
        with torch.no_grad():
            for batch in DataLoader(val_dataset, batch_size=batch_size):
                inputs = tokenizer(batch['sentence'], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                embeddings = model(**inputs)
                # 这里的评估逻辑取决于你的具体任务

    if PREDICT:
        # 进行预测
        with torch.no_grad():
            inputs = tokenizer(['这是一个句子', '这是另一个句子'], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            embeddings = model(**inputs)
            # 这里的预测逻辑取决于你的具体任务
