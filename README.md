# 中文新闻标题分类项目

## 项目简介

本项目是一个基于深度学习的中文新闻标题文本分类系统，旨在自动识别新闻标题所属的类别。项目使用RoBERTa预训练模型作为基础架构，通过微调实现14个新闻类别的精确分类。

## 环境要求

```
Python >= 3.8
torch >= 1.9.0
transformers >= 4.21.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
tqdm >= 4.62.0
```

## 项目结构

```
news-classification/
├── data/                           # 数据目录
│   ├── train.txt                   # 训练集
│   ├── dev.txt                     # 验证集
│   ├── test.txt                    # 测试集
│   ├── train_merge.txt             # 增强后的训练集
│   ├── augmented_data.txt          # 生成模型增强数据
│   └── augmentation_log.txt        # 数据增强日志
├── src/                            # 核心代码包
│   ├── __init__.py                 # 包初始化文件
│   ├── config.py                   # 全局配置参数
│   ├── data_processor.py           # 数据处理和加载
│   ├── model.py                    # 模型定义
│   ├── trainer.py                  # 训练器类
│   └── utils.py                    # 工具函数
├── models/                         # 模型保存目录
│   └── best_model/                 # 最优模型文件
├── pretrained_models/              # 预训练模型目录
│   ├── roberta-wwm-ext-large/      # RoBERTa预训练模型
│   └── Qwen2.5-7B-Instruct/        # Qwen生成模型
├── train.py                        # 主训练脚本
├── evaluate.py                     # 模型评估脚本
├── predict.py                      # 预测脚本
├── mask_aug.py                     # BERT掩码数据增强
├── qwen_aug.py                     # Qwen生成数据增强
└── frame.txt                       # 项目结构说明
```

## 数据格式

**训练集和验证集格式**：
```
新闻标题+\t+类别标签
示例：网易第三季度业绩低于分析师预期	科技
```

**测试集格式**：
```
新闻标题
示例：北京君太百货璀璨秋色 满100省353020元
```

## 模型架构

### 基础模型
- **预训练模型**：RoBERTa-WWM-EXT-Large
- **模型参数**：约3.5亿参数
- **最大序列长度**：48 tokens
- **分类器**：线性层（hidden_size → num_classes）

### 关键参数配置
```python
MAX_SEQ_LENGTH = 48        # 最大序列长度
BATCH_SIZE = 128           # 批次大小
NUM_EPOCHS = 5             # 训练轮数
LEARNING_RATE = 4e-5       # 学习率
WARMUP_PROPORTION = 0.1    # 预热比例
WEIGHT_DECAY = 0.01        # 权重衰减
```

### 模型特点
- 使用预训练的中文RoBERTa模型
- 添加Dropout层防止过拟合
- 支持加权损失函数处理类别不平衡
- 使用AdamW优化器和线性学习率调度

## 如何运行

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers pandas numpy matplotlib seaborn scikit-learn tqdm

# 创建项目目录
mkdir news-classification && cd news-classification
```

### 2. 数据准备

将训练数据放置在 `data/` 目录下：
- `train.txt`：训练集
- `dev.txt`：验证集  
- `test.txt`：测试集

### 3. 预训练模型下载
下载RoBERTa-WWM-EXT-Large模型文件到 `pretrained_models/roberta-wwm-ext-large/` 目录

### 4. 模型训练
```bash
# 基础训练
python train.py

# 查看训练过程中的输出信息
# 训练完成后会在models/best_model/保存最优模型
```

### 5. 模型评估
```bash
# 在验证集上评估模型
python evaluate.py

# 输出验证准确率和混淆矩阵图片
```

### 6. 模型预测
```bash
# 对测试集进行预测
python predict.py

# 结果保存在result.txt文件中
```

## 数据增强

项目提供两种数据增强策略：

### 1. BERT掩码增强
```bash
# 使用BERT MLM进行数据增强
python mask_aug.py --strategy minority --target_ratio 1.5

# 参数说明：
# --strategy: 增强策略 (minority/balanced/all)
# --target_ratio: 目标样本比例
# --max_samples: 每类最大样本数
# --show_examples: 展示增强示例
```

### 2. Qwen生成模型增强
```bash
# 使用Qwen2.5生成新标题
python qwen_aug.py

# 需要先下载Qwen2.5-7B-Instruct模型
# 生成的所有新标题样本保存在data/augmented_data.txt
```

## 输出文件

- `result.txt`：测试集预测结果
- `training_history.png`：训练历史曲线
- `confusion_matrix.png`：混淆矩阵图
- `models/best_model/`：最优模型文件
- `data/augmented_data.txt`：数据增强结果

## 扩展功能

- 支持自定义类别权重处理样本不平衡
- 可配置的数据增强策略
- 多种评估指标和可视化
- 模块化设计便于扩展

## 问题排查

### 常见问题
1. **CUDA内存不足**：减小BATCH_SIZE
2. **模型加载失败**：检查预训练模型路径
3. **数据格式错误**：确认tab分隔符和编码格式
4. **准确率较低**：尝试数据增强或调整超参数

### 日志查看

- 训练日志：控制台输出
- 增强日志：`data/augmentation_log.txt`
- 错误信息：检查Python异常输出
