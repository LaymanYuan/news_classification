import torch
import pandas as pd
import numpy as np
import random
from transformers import BertTokenizer, BertForMaskedLM
from collections import Counter, defaultdict
from tqdm import tqdm
import os
import argparse
from src.config import Config

class BertDataAugmentor:
    def __init__(self, config):
        self.config = config
        # 加载tokenizer和MLM模型
        print("Loading tokenizer and MLM model...")
        self.tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_PATH)
        self.mlm_model = BertForMaskedLM.from_pretrained(config.PRETRAINED_MODEL_PATH)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlm_model.to(self.device)
        self.mlm_model.eval()
        
        print(f"Using device: {self.device}")
        
        # 创建标签映射
        self.label2id = {label: i for i, label in enumerate(config.LABELS)}
        self.id2label = {i: label for i, label in enumerate(config.LABELS)}
        
    def load_and_analyze_data(self, train_file):
        """加载并分析训练数据"""
        print("Loading and analyzing training data...")
        
        # 读取数据
        df = pd.read_csv(train_file, sep='\t', header=None, names=['text', 'label'])
        
        # 统计类别分布
        class_counts = df['label'].value_counts()
        total_samples = len(df)
        
        print(f"总样本数: {total_samples}")
        print("类别分布:")
        for label, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {label}: {count} 样本 ({percentage:.1f}%)")
        
        # 识别少样本类别（低于平均值的类别）
        avg_samples = class_counts.mean()
        minority_classes = class_counts[class_counts < avg_samples * 0.8].index.tolist()
        
        print(f"\n平均样本数: {avg_samples:.0f}")
        print(f"少样本类别: {minority_classes}")
        
        return df, class_counts, minority_classes
    
    def bert_mlm_augment_text(self, text, num_variants=3, mask_prob=0.15, top_k=10):
        """使用BERT MLM对单个文本进行增强"""
        augmented_texts = set()  # 使用set避免重复
        
        # 预处理文本
        text = text.strip()
        if len(text) == 0:
            return []
        
        for attempt in range(num_variants * 2):  # 多尝试几次以确保得到足够的变体
            try:
                # 分词
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) <= 2:  # 文本太短，跳过
                    continue
                
                # 找到可以mask的位置（排除特殊token和标点）
                maskable_positions = []
                for i, token in enumerate(tokens):
                    # 排除特殊token、标点符号、数字
                    if (not token.startswith('[') and 
                        not token.startswith('##') and 
                        len(token) > 1 and 
                        not token.isdigit() and
                        token not in ['，', '。', '！', '？', '、', '；', '：']):
                        maskable_positions.append(i)
                
                if len(maskable_positions) == 0:
                    continue
                
                # 随机选择要mask的位置
                num_to_mask = max(1, min(3, int(len(maskable_positions) * mask_prob)))
                mask_positions = random.sample(maskable_positions, 
                                             min(num_to_mask, len(maskable_positions)))
                
                # 创建masked版本
                masked_tokens = tokens.copy()
                original_tokens = []
                for pos in mask_positions:
                    original_tokens.append(tokens[pos])
                    masked_tokens[pos] = '[MASK]'
                
                # 构建输入
                input_text = '[CLS] ' + ' '.join(masked_tokens) + ' [SEP]'
                input_ids = self.tokenizer.encode(
                    input_text, 
                    max_length=self.config.MAX_SEQ_LENGTH, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                # 获取mask位置在input_ids中的索引
                mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                
                if len(mask_indices) == 0:
                    continue
                
                # 预测
                with torch.no_grad():
                    outputs = self.mlm_model(input_ids)
                    predictions = outputs.logits
                
                # 为每个mask位置选择top_k候选
                new_tokens = masked_tokens.copy()
                for i, mask_idx in enumerate(mask_indices):
                    if i < len(mask_positions):
                        # 获取top_k预测
                        top_k_tokens = torch.topk(predictions[0, mask_idx], top_k)
                        
                        # 随机选择一个（偏向概率高的）
                        probs = torch.softmax(top_k_tokens.values, dim=0)
                        selected_idx = torch.multinomial(probs, 1).item()
                        predicted_token_id = top_k_tokens.indices[selected_idx].item()
                        predicted_token = self.tokenizer.decode([predicted_token_id])
                        
                        # 如果预测的token合理，则替换
                        if (predicted_token and 
                            predicted_token != original_tokens[i] and
                            len(predicted_token.strip()) > 0 and
                            not predicted_token.startswith('[') and
                            predicted_token != '[UNK]'):
                            new_tokens[mask_positions[i]] = predicted_token.strip()
                
                # 重建文本
                new_text = ''.join(new_tokens).replace(' ', '')
                
                # 质量检查
                if (new_text != text and 
                    len(new_text) > 2 and 
                    len(new_text) <= len(text) * 2 and  # 避免文本过长
                    new_text not in augmented_texts):
                    augmented_texts.add(new_text)
                
                # 如果已经有足够的变体，提前退出
                if len(augmented_texts) >= num_variants:
                    break
                    
            except Exception as e:
                # 如果某次增强失败，继续尝试
                continue
        
        return list(augmented_texts)[:num_variants]
    
    def augment_class_data(self, texts, labels, target_label, target_samples):
        """为特定类别增强数据"""
        print(f"\n正在增强类别: {target_label}")
        
        # 获取该类别的所有样本
        class_texts = [texts[i] for i, label in enumerate(labels) if label == target_label]
        current_count = len(class_texts)
        needed_samples = max(0, target_samples - current_count)
        
        print(f"  现有样本: {current_count}")
        print(f"  目标样本: {target_samples}")
        print(f"  需要增强: {needed_samples}")
        
        if needed_samples <= 0:
            return [], []
        
        augmented_texts = []
        augmented_labels = []
        
        # 计算每个原始样本需要生成的增强样本数
        samples_per_original = max(1, needed_samples // current_count + 1)
        
        with tqdm(total=len(class_texts), desc=f"增强{target_label}类别") as pbar:
            for original_text in class_texts:
                # 为每个原始样本生成多个增强样本
                variants = self.bert_mlm_augment_text(
                    original_text, 
                    num_variants=samples_per_original,
                    mask_prob=random.uniform(0.1, 0.25),  # 随机mask比例
                    top_k=random.randint(5, 15)  # 随机top_k
                )
                
                for variant in variants:
                    augmented_texts.append(variant)
                    augmented_labels.append(target_label)
                    
                    # 如果已经有足够的样本，停止
                    if len(augmented_texts) >= needed_samples:
                        break
                
                pbar.update(1)
                
                if len(augmented_texts) >= needed_samples:
                    break
        
        print(f"  实际生成: {len(augmented_texts)} 个增强样本")
        return augmented_texts[:needed_samples], augmented_labels[:needed_samples]
    
    def augment_dataset(self, train_file, output_file, strategy='minority', target_ratio=1.5, max_samples_per_class=3000):
        """增强整个数据集"""
        print("=" * 60)
        print("开始数据增强")
        print("=" * 60)
        
        # 加载和分析数据
        df, class_counts, minority_classes = self.load_and_analyze_data(train_file)
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # 确定增强策略
        if strategy == 'minority':
            # 只增强少样本类别
            classes_to_augment = minority_classes
            avg_samples = class_counts.mean()
            target_samples_per_class = int(avg_samples * target_ratio)
        elif strategy == 'balanced':
            # 平衡所有类别
            classes_to_augment = self.config.LABELS
            max_samples = class_counts.max()
            target_samples_per_class = min(max_samples, max_samples_per_class)
        else:
            # 增强所有类别
            classes_to_augment = self.config.LABELS
            target_samples_per_class = max_samples_per_class
        
        print(f"\n增强策略: {strategy}")
        print(f"目标样本数: {target_samples_per_class}")
        print(f"需要增强的类别: {classes_to_augment}")
        
        # 收集所有增强数据
        all_augmented_texts = []
        all_augmented_labels = []
        
        for label in classes_to_augment:
            current_count = class_counts.get(label, 0)
            if current_count < target_samples_per_class:
                aug_texts, aug_labels = self.augment_class_data(
                    texts, labels, label, target_samples_per_class
                )
                all_augmented_texts.extend(aug_texts)
                all_augmented_labels.extend(aug_labels)
        
        # 合并原始数据和增强数据
        final_texts = texts + all_augmented_texts
        final_labels = labels + all_augmented_labels
        
        # 创建最终数据框
        final_df = pd.DataFrame({
            'text': final_texts,
            'label': final_labels
        })
        
        # 保存增强后的数据
        final_df.to_csv(output_file, sep='\t', header=False, index=False)
        
        # 输出统计信息
        print("\n" + "=" * 60)
        print("数据增强完成")
        print("=" * 60)
        print(f"原始样本数: {len(texts)}")
        print(f"增强样本数: {len(all_augmented_texts)}")
        print(f"总样本数: {len(final_texts)}")
        print(f"增强后数据保存到: {output_file}")
        
        # 显示增强后的类别分布
        final_class_counts = final_df['label'].value_counts()
        print("\n增强后类别分布:")
        for label in self.config.LABELS:
            original_count = class_counts.get(label, 0)
            final_count = final_class_counts.get(label, 0)
            increase = final_count - original_count
            print(f"  {label}: {original_count} → {final_count} (+{increase})")
        
        return final_df
    
    def sample_and_show_augmentations(self, train_file, num_samples=5):
        """展示增强效果示例"""
        print("=" * 60)
        print("数据增强效果示例")
        print("=" * 60)
        
        df = pd.read_csv(train_file, sep='\t', header=None, names=['text', 'label'])
        
        # 从每个类别随机采样一些文本进行展示
        for label in random.sample(self.config.LABELS, min(3, len(self.config.LABELS))):
            class_texts = df[df['label'] == label]['text'].tolist()
            if class_texts:
                sample_text = random.choice(class_texts)
                augmented = self.bert_mlm_augment_text(sample_text, num_variants=3)
                
                print(f"\n类别: {label}")
                print(f"原文: {sample_text}")
                for i, aug_text in enumerate(augmented, 1):
                    print(f"增强{i}: {aug_text}")


def main():
    parser = argparse.ArgumentParser(description='BERT数据增强')
    parser.add_argument('--strategy', choices=['minority', 'balanced', 'all'], 
                       default='minority', help='增强策略')
    parser.add_argument('--target_ratio', type=float, default=1.5, 
                       help='目标样本比例（相对于平均值）')
    parser.add_argument('--max_samples', type=int, default=3000, 
                       help='每个类别最大样本数')
    parser.add_argument('--output_suffix', type=str, default='augmented', 
                       help='输出文件后缀')
    parser.add_argument('--show_examples', action='store_true', 
                       help='展示增强示例')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    # 创建增强器
    augmentor = BertDataAugmentor(Config)
    
    # 展示增强示例（可选）
    if args.show_examples:
        augmentor.sample_and_show_augmentations(Config.TRAIN_FILE)
        return
    
    # 执行数据增强
    output_file = Config.TRAIN_FILE.replace('.txt', f'_{args.output_suffix}.txt')
    
    augmented_df = augmentor.augment_dataset(
        train_file=Config.TRAIN_FILE,
        output_file=output_file,
        strategy=args.strategy,
        target_ratio=args.target_ratio,
        max_samples_per_class=args.max_samples
    )
    
    print(f"\n数据增强完成！增强后的训练文件: {output_file}")


if __name__ == "__main__":
    main()