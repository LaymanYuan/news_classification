import os

# # 类别标签
# LABELS = ["财经", "彩票", "房产", "股票", "家居", "教育", 
#           "科技", "社会", "时尚", "时政", "体育", "星座", 
#           "游戏", "娱乐"]

# # 训练集类别分布（与LABELS顺序一致）
# LABEL_COUNTS = [33389, 6830, 18045, 138959, 29328, 37743, 
#                 146637, 45765, 12032, 56778, 118440, 3221, 
#                 21936, 83368]
# TOTAL_SAMPLES = 752471
# NUM_CLASSES = len(LABELS)

class Config:
    # 数据路径
    TRAIN_FILE = os.path.join("data", "train.txt")
    # 数据增强后的训练集路径
    # TRAIN_FILE = os.path.join("data", "train_merge.txt")
    DEV_FILE = os.path.join("data", "dev.txt")
    TEST_FILE = os.path.join("data", "test.txt")
    
    # 预训练模型路径
    PRETRAINED_MODEL_PATH = os.path.join("pretrained_models", "roberta-wwm-ext-large")
    
    # 模型保存路径
    MODEL_SAVE_DIR = os.path.join("models", "best_model")
    
    # 训练参数
    MAX_SEQ_LENGTH = 48
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    LEARNING_RATE = 4e-5
    WARMUP_PROPORTION = 0.1
    WEIGHT_DECAY = 0.01

    # 类别标签
    LABELS = ["财经", "彩票", "房产", "股票", "家居", "教育", 
              "科技", "社会", "时尚", "时政", "体育", "星座", 
              "游戏", "娱乐"]
    
    # 计算类别权重
    # CLASS_WEIGHTS = [(TOTAL_SAMPLES / (NUM_CLASSES * c)) ** 0.5 for c in LABEL_COUNTS]
    
    # 随机种子
    SEED = 42
