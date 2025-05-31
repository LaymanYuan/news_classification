import torch
import os
import pandas as pd
from transformers import BertConfig, AutoTokenizer
from src.config import Config
from src.data_processor import DataProcessor
from src.model import NewsClassifier
from src.trainer import Trainer
from src.utils import set_seed, plot_training_history

def main():
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 检查CUDA是否可用
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    data_processor = DataProcessor(Config)
    
    # 分析数据
    print("\n分析训练集:")
    train_count, train_dist = data_processor.analyze_data(Config.TRAIN_FILE)
    print(f"训练集样本数: {train_count}")
    
    print("\n分析验证集:")
    dev_count, dev_dist = data_processor.analyze_data(Config.DEV_FILE)
    print(f"验证集样本数: {dev_count}")
    
    print("\n分析测试集:")
    test_count, _ = data_processor.analyze_data(Config.TEST_FILE, has_label=False)
    print(f"测试集样本数: {test_count}")
    
    # 创建数据加载器
    print("\nCreating dataloaders...")
    train_dataloader, dev_dataloader, test_dataloader = data_processor.create_dataloaders()
    
    # 加载预训练模型配置
    print("\nLoading model configuration...")
    model_config = BertConfig.from_pretrained(
        Config.PRETRAINED_MODEL_PATH,
        num_labels=len(Config.LABELS)
    )
    
    # 创建模型
    print("Initializing model...")
    model = NewsClassifier.from_pretrained(
        Config.PRETRAINED_MODEL_PATH,
        config=model_config
    )
    model.to(device)
    
    # 创建训练器
    trainer = Trainer(model, Config, device)
    
    # 训练模型
    print("\nStarting training...")
    history = trainer.train(train_dataloader, dev_dataloader)
    
    # 绘制训练历史
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['train_accs'],
        history['val_accs']
    )
    
    print(f"Best validation accuracy: {history['best_val_acc']:.4f}")
    print("Training completed!")

if __name__ == "__main__":
    main()
