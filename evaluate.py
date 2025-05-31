import torch
import numpy as np
import pandas as pd
from transformers import BertConfig
from src.config import Config
from src.data_processor import DataProcessor
from src.model import NewsClassifier
from src.trainer import Trainer
from src.utils import set_seed, plot_confusion_matrix


def main():
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    data_processor = DataProcessor(Config)
    
    # 创建数据加载器
    print("Creating dataloaders...")
    _, dev_dataloader, _ = data_processor.create_dataloaders()
    
    # 加载模型配置
    print("Loading model configuration...")
    model_config = BertConfig.from_pretrained(
        Config.MODEL_SAVE_DIR,
        num_labels=len(Config.LABELS)
    )
    
    # 创建模型
    print("Loading model...")
    model = NewsClassifier.from_pretrained(
        Config.MODEL_SAVE_DIR,
        config=model_config
    )
    model.to(device)
    
    # 创建训练器
    trainer = Trainer(model, Config, device)
    
    # 评估模型
    print("Evaluating model...")
    val_loss, val_acc = trainer.evaluate(dev_dataloader)
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # 获取预测结果和真实标签，用于绘制混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, Config.LABELS)
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()
