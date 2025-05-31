import torch
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from .utils import calculate_accuracy, ensure_dir
import os
import time

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # 确保模型保存目录存在
        ensure_dir(config.MODEL_SAVE_DIR)
        
    def train(self, train_dataloader, dev_dataloader):
        """训练模型"""
        # 优化器
        optimizer = AdamW(self.model.parameters(), 
                          lr=self.config.LEARNING_RATE, 
                          weight_decay=self.config.WEIGHT_DECAY)
        
        # 计算总训练步数
        total_steps = len(train_dataloader) * self.config.NUM_EPOCHS
        
        # 学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * self.config.WARMUP_PROPORTION),
            num_training_steps=total_steps
        )
        
        # 记录训练历史
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        # 记录最佳验证准确率
        best_val_acc = 0.0
        
        # 开始训练
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_steps = 0
            
            for batch in tqdm(train_dataloader, desc="Training"):
                # 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # 计算准确率
                preds = logits.detach().cpu().numpy()
                label_ids = labels.cpu().numpy()
                tmp_train_acc = calculate_accuracy(preds, label_ids)
                
                train_loss += loss.item()
                train_acc += tmp_train_acc
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            avg_train_acc = train_acc / train_steps
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Training Accuracy: {avg_train_acc:.4f}")
            
            # 验证阶段
            val_loss, val_acc = self.evaluate(dev_dataloader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.4f}")
                self.save_model()
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # 计算准确率
                preds = logits.detach().cpu().numpy()
                label_ids = labels.cpu().numpy()
                tmp_val_acc = calculate_accuracy(preds, label_ids)
                
                val_loss += loss.item()
                val_acc += tmp_val_acc
                val_steps += 1
        
        return val_loss / val_steps, val_acc / val_steps
    
    def predict(self, dataloader):
        """预测"""
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        return all_preds
    
    def save_model(self):
        """保存模型"""
        self.model.save_pretrained(self.config.MODEL_SAVE_DIR)
        print(f"Model saved to {self.config.MODEL_SAVE_DIR}")
    
    def load_model(self, model_class):
        """加载模型"""
        self.model = model_class.from_pretrained(self.config.MODEL_SAVE_DIR)
        self.model.to(self.device)
        print(f"Model loaded from {self.config.MODEL_SAVE_DIR}")
