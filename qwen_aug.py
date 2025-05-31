import os
import random
import re
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from tqdm import tqdm

class NewsDataAugmentor:
    def __init__(self, model_path="./pretrained_models/Qwen2.5-7B-Instruct", data_path="data/train.txt"):
        self.model_path = model_path
        self.data_path = data_path
        self.log_file = "data/augmentation_log.txt"
        self.output_file = "data/augmented_data.txt"
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载模型和分词器
        self.load_model()
        
        # 加载原始数据
        self.load_data()
        
        # 类别权重（用于平衡采样）
        self.setup_category_weights()
        
    def load_model(self):
        """加载Qwen2.5模型和分词器"""
        self.logger.info("加载模型和分词器...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def load_data(self):
        """加载原始训练数据"""
        self.logger.info("加载原始数据集...")
        try:
            self.data = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            title, category = parts
                            self.data.append({'title': title, 'category': category})
            
            # 统计类别分布
            self.category_counts = Counter([item['category'] for item in self.data])
            self.logger.info(f"数据加载完成，总计 {len(self.data)} 条样本")
            self.logger.info(f"类别分布: {dict(self.category_counts)}")
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def setup_category_weights(self):
        """设置类别权重，用于平衡采样"""
        # 计算逆频率权重
        total_samples = len(self.data)
        self.category_weights = {}
        
        for category, count in self.category_counts.items():
            # 使用平方根来缓解权重差异过大
            self.category_weights[category] = (total_samples / count) ** 0.5
        
        # 归一化权重
        total_weight = sum(self.category_weights.values())
        for category in self.category_weights:
            self.category_weights[category] /= total_weight
        
        self.logger.info(f"类别权重: {self.category_weights}")
    
    def generate_text(self, prompt, max_length=150, temperature=0.7):
        """使用模型生成文本"""
        try:
            # 格式化输入
            messages = [
                {"role": "system", "content": "你是一个专业的中文文本处理助手。"},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt")
            if torch.cuda.is_available():
                model_inputs = model_inputs.to('cuda')
            
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"文本生成失败: {e}")
            return ""
    
    def extract_keywords(self, title, category):
        """从原始标题中提取关键词"""
        prompt = f"""请从以下新闻标题中提取3-5个最重要的中文关键词。这些关键词应该能够概括新闻的核心内容。

新闻标题：{title}
新闻类别：{category}

请只返回关键词，用逗号分隔，不要包含任何其他内容。"""
        
        keywords_text = self.generate_text(prompt, max_length=50, temperature=0.3)
        
        # 清理和处理关键词
        keywords = []
        if keywords_text:
            # 提取中文关键词
            raw_keywords = re.findall(r'[\u4e00-\u9fff]+', keywords_text)
            keywords = [kw for kw in raw_keywords if len(kw) >= 2 and len(kw) <= 10][:5]
        
        return keywords
    
    def generate_new_title(self, keywords, category):
        """基于关键词和类别生成新标题"""
        if not keywords:
            return ""
            
        keywords_str = "、".join(keywords)
        
        prompt = f"""请基于以下关键词和新闻类别，生成一个自然、流畅的中文新闻标题。标题应该：
1. 长度在10-25个字符之间
2. 语言自然流畅
3. 符合新闻标题的表达习惯
4. 与给定类别相关

关键词：{keywords_str}
新闻类别：{category}

请只返回一个新闻标题，不要包含任何其他内容。"""
        
        new_title = self.generate_text(prompt, max_length=80, temperature=0.8)
        
        # 清理生成的标题
        new_title = self.clean_title(new_title)
        
        return new_title
    
    def clean_title(self, title):
        """清理和过滤生成的标题"""
        if not title:
            return ""
        
        # 移除多余的标点和符号
        title = re.sub(r'[^\u4e00-\u9fff\w\s，。！？：；""''（）【】《》]', '', title)
        
        # 移除多余的空格
        title = re.sub(r'\s+', '', title)
        
        # 移除引号等包围符号
        title = title.strip('"""''《》【】（）')
        
        # 检查中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', title))
        total_chars = len(title)
        
        if total_chars == 0 or chinese_chars / total_chars < 0.8:
            return ""
        
        # 检查长度
        if len(title) < 5 or len(title) > 50:
            return ""
        
        return title
    
    def weighted_sample(self):
        """基于权重进行采样"""
        categories = list(self.category_weights.keys())
        weights = list(self.category_weights.values())
        
        # 随机选择类别
        selected_category = random.choices(categories, weights=weights)[0]
        
        # 从该类别中随机选择一个样本
        category_samples = [item for item in self.data if item['category'] == selected_category]
        selected_sample = random.choice(category_samples)
        
        return selected_sample
    
    def augment_data(self, num_samples=10000):
        """执行数据扩充"""
        self.logger.info(f"开始数据扩充，目标生成 {num_samples} 个样本...")
        
        generated_samples = []
        successful_count = 0
        
        # 清空日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        with tqdm(total=num_samples, desc="生成新样本") as pbar:
            attempt = 0
            while successful_count < num_samples and attempt < num_samples * 3:  # 最多尝试3倍数量
                attempt += 1
                
                try:
                    # 采样原始数据
                    sample = self.weighted_sample()
                    original_title = sample['title']
                    category = sample['category']
                    
                    # 提取关键词
                    keywords = self.extract_keywords(original_title, category)
                    
                    if not keywords:
                        continue
                    
                    # 生成新标题
                    new_title = self.generate_new_title(keywords, category)
                    
                    if not new_title:
                        continue
                    
                    # 避免生成与原标题相同的内容
                    if new_title == original_title:
                        continue
                    
                    # 记录成功的样本
                    generated_samples.append({'title': new_title, 'category': category})
                    successful_count += 1
                    
                    # 写入日志
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{successful_count:6d}] 类别: {category:8s}\n")
                        f.write(f"         原标题: {original_title}\n")
                        f.write(f"         关键词: {', '.join(keywords)}\n")
                        f.write(f"         新标题: {new_title}\n")
                        f.write("-" * 60 + "\n")
                    
                    pbar.update(1)
                    
                    # 每100个样本保存一次
                    if successful_count % 100 == 0:
                        self.save_augmented_data(generated_samples)
                    
                except Exception as e:
                    self.logger.error(f"生成第 {attempt} 个样本时出错: {e}")
                    continue
        
        # 最终保存
        self.save_augmented_data(generated_samples)
        
        self.logger.info(f"数据扩充完成！成功生成 {successful_count} 个新样本")
        
        # 统计生成的类别分布
        generated_category_counts = Counter([item['category'] for item in generated_samples])
        self.logger.info(f"生成的类别分布: {dict(generated_category_counts)}")
        
        return generated_samples
    
    def save_augmented_data(self, generated_samples):
        """保存扩充后的数据"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for sample in generated_samples:
                f.write(f"{sample['title']}\t{sample['category']}\n")
        
        self.logger.info(f"已保存 {len(generated_samples)} 个扩充样本到 {self.output_file}")

def main():
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("使用CPU")
    
    # 创建数据扩充器
    augmentor = NewsDataAugmentor()
    
    # 执行数据扩充
    # 可以根据需要调整生成的样本数量
    generated_samples = augmentor.augment_data(num_samples=100000)
    
    print(f"数据扩充完成！生成了 {len(generated_samples)} 个新样本")
    print(f"日志文件: {augmentor.log_file}")
    print(f"输出文件: {augmentor.output_file}")

if __name__ == "__main__":
    main()