import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
from tqdm import tqdm
from .config import Config

class NewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=48):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_PATH)
        self.label2id = {label: i for i, label in enumerate(config.LABELS)}
        self.id2label = {i: label for i, label in enumerate(config.LABELS)}
    
    def load_data(self, file_path, has_label=True):
        """加载数据"""
        df = pd.read_csv(file_path, sep='\t', header=None)
        
        if has_label:
            # 训练集和验证集格式: 文本\t标签
            df.columns = ['text', 'label']
            texts = df['text'].tolist()
            labels = [self.label2id[label] for label in df['label'].tolist()]
            return texts, labels
        else:
            # 测试集格式: 仅有文本
            df.columns = ['text']
            texts = df['text'].tolist()
            return texts, None
    
    def analyze_data(self, file_path, has_label=True):
        """分析数据集信息"""
        if has_label:
            texts, labels = self.load_data(file_path, has_label=True)
            
            # 计算类别分布
            label_names = [self.id2label[label] for label in labels]
            label_counts = pd.Series(label_names).value_counts()
            print(f"类别分布:\n{label_counts}")
            
            # 计算文本长度统计
            text_lengths = [len(text) for text in texts]
            print(f"文本长度: 最小 {min(text_lengths)}, 最大 {max(text_lengths)}, 平均 {sum(text_lengths)/len(text_lengths):.2f}")
            
            return len(texts), label_counts
        else:
            texts, _ = self.load_data(file_path, has_label=False)
            
            # 计算文本长度统计
            text_lengths = [len(text) for text in texts]
            print(f"文本长度: 最小 {min(text_lengths)}, 最大 {max(text_lengths)}, 平均 {sum(text_lengths)/len(text_lengths):.2f}")
            
            return len(texts), None
    
    def create_dataloaders(self):
        """创建训练、验证和测试数据加载器"""
        # 加载训练集
        train_texts, train_labels = self.load_data(self.config.TRAIN_FILE)
        train_dataset = NewsDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_SEQ_LENGTH
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        # 加载验证集
        dev_texts, dev_labels = self.load_data(self.config.DEV_FILE)
        dev_dataset = NewsDataset(
            texts=dev_texts,
            labels=dev_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_SEQ_LENGTH
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        # 加载测试集
        test_texts, _ = self.load_data(self.config.TEST_FILE, has_label=False)
        test_dataset = NewsDataset(
            texts=test_texts,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_SEQ_LENGTH
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        return train_dataloader, dev_dataloader, test_dataloader
