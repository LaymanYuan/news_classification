import torch
import pandas as pd
from transformers import BertConfig
from src.config import Config
from src.data_processor import DataProcessor
from src.model import NewsClassifier
from src.trainer import Trainer
from src.utils import set_seed

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
    _, _, test_dataloader = data_processor.create_dataloaders()
    
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
    
    # 预测
    print("Predicting...")
    predictions = trainer.predict(test_dataloader)
    
    # 将数字标签转换为文本标签
    predicted_labels = [Config.LABELS[pred] for pred in predictions]
    
    # 加载测试数据
    test_texts, _ = data_processor.load_data(Config.TEST_FILE, has_label=False)
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'label': predicted_labels
    })
    
    # 保存提交文件
    submission_df.to_csv('result.txt', index=False, header=False)
    print("Predictions saved to result.txt")

if __name__ == "__main__":
    main()
