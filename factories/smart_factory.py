"""
AI增强的智能Alpha工厂
使用训练好的Transformer模型指导Alpha生成
"""
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.alpha_transformer import AlphaTransformerModel
from models.tokenizer import AlphaTokenizer
from data.preprocessor import AlphaDataPreprocessor
from utils.wq_client import WorldQuantClient
from config import config
from typing import List, Tuple, Dict
import pickle
import random


class SmartAlphaFactory:
    """AI增强的智能Alpha工厂"""
    
    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.wq_client = WorldQuantClient()
        
        # 加载模型
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str = None):
        """加载训练好的模型"""
        if model_path is None:
            model_path = os.path.join(config.transformer.model_save_dir, 'best_model.pt')
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("请先训练模型或提供正确的模型路径")
            return False
        
        # 加载tokenizer和scaler
        preprocessor = AlphaDataPreprocessor()
        try:
            preprocessor.tokenizer.load(os.path.join('./data/processed', 'tokenizer.pkl'))
            with open(os.path.join('./data/processed', 'scaler.pkl'), 'rb') as f:
                preprocessor.scaler = pickle.load(f)
            
            self.tokenizer = preprocessor.tokenizer
            self.scaler = preprocessor.scaler
        except:
            print("无法加载tokenizer和scaler")
            return False
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 从checkpoint中读取配置信息
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
            print(f"从checkpoint读取vocab_size: {vocab_size}")
        else:
            vocab_size = self.tokenizer.vocab_size
            print(f"从tokenizer读取vocab_size: {vocab_size}")
        
        # 检查是否匹配
        if vocab_size != self.tokenizer.vocab_size:
            print(f"⚠️ 警告：模型vocab_size({vocab_size}) 与 tokenizer({self.tokenizer.vocab_size}) 不匹配")
            print(f"这通常是因为重新采集数据后需要重新训练模型")
            print(f"建议：在Tab 3重新训练模型")
            return False
        
        # 假设知道特征数量（从训练时保存）
        num_features = checkpoint.get('num_features', 8)
        
        self.model = AlphaTransformerModel(
            vocab_size=vocab_size,
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_encoder_layers,
            dim_feedforward=config.transformer.dim_feedforward,
            dropout=config.transformer.dropout,
            max_seq_length=config.transformer.max_seq_length,
            num_features=num_features
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型加载成功: {model_path}")
        print(f"  - Vocab Size: {vocab_size}")
        print(f"  - Num Features: {num_features}")
        return True
    
    def predict_alpha_score(self, expressions: List[str]) -> np.ndarray:
        """预测Alpha表达式的分数"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 编码表达式
        encoded_list = []
        features_list = []
        
        for expr in expressions:
            # Token编码
            encoded = self.tokenizer.encode(expr, config.transformer.max_seq_length)
            encoded_list.append(encoded)
            
            # 提取特征
            features = self.tokenizer.extract_features(expr)
            features_list.append(list(features.values()))
        
        X = np.array(encoded_list)
        X_features = np.array(features_list)
        X_features = self.scaler.transform(X_features)
        
        # 预测 - 确保所有tensor在同一设备
        with torch.no_grad():
            X_tensor = torch.LongTensor(X).to(self.device)
            X_feat_tensor = torch.FloatTensor(X_features).to(self.device)
            
            predictions = self.model(X_tensor, X_feat_tensor)
            scores = predictions.cpu().numpy().flatten()
        
        return scores
    
    def generate_and_rank_alphas(
        self,
        datafields_df,
        generation_size: int = 10000,
        top_k: int = 1000
    ) -> List[Tuple[str, int, float]]:
        """
        生成Alpha并使用AI模型排序
        
        Returns:
            List of (expression, decay, predicted_score)
        """
        print(f"生成{generation_size}个候选Alpha...")
        
        # 使用传统工厂生成候选Alpha
        alpha_list = self.wq_client.generate_first_order_alphas(
            datafields_df, 
            max_count=generation_size
        )
        
        if self.model is None:
            print("警告：模型未加载，返回随机排序的Alpha")
            random.shuffle(alpha_list)
            return [(expr, decay, 0.0) for expr, decay in alpha_list[:top_k]]
        
        # 分批预测（避免内存溢出）
        batch_size = 256
        all_expressions = [expr for expr, _ in alpha_list]
        all_scores = []
        
        print(f"使用AI模型评分...")
        for i in range(0, len(all_expressions), batch_size):
            batch_exprs = all_expressions[i:i+batch_size]
            batch_scores = self.predict_alpha_score(batch_exprs)
            all_scores.extend(batch_scores)
        
        # 组合表达式、decay和预测分数
        ranked_alphas = [
            (expr, decay, score) 
            for (expr, decay), score in zip(alpha_list, all_scores)
        ]
        
        # 按预测分数降序排序
        ranked_alphas.sort(key=lambda x: x[2], reverse=True)
        
        print(f"排序完成，返回Top {top_k}个高潜力Alpha")
        print(f"Top 1 预测分数: {ranked_alphas[0][2]:.4f}")
        print(f"Top {top_k} 预测分数: {ranked_alphas[top_k-1][2]:.4f}")
        
        return ranked_alphas[:top_k]
    
    def smart_backtest_workflow(
        self,
        datafields_df,
        generation_size: int = 5000,
        backtest_top_k: int = 300,
        batch_size: int = 3
    ):
        """
        智能回测工作流
        1. 生成大量候选Alpha
        2. AI模型预测排序
        3. 只回测Top-K高分Alpha
        """
        print("\n=== 智能回测工作流 ===")
        
        # 生成并排序
        ranked_alphas = self.generate_and_rank_alphas(
            datafields_df,
            generation_size=generation_size,
            top_k=backtest_top_k
        )
        
        # 准备回测
        backtest_list = [(expr, decay) for expr, decay, _ in ranked_alphas]
        
        print(f"\n准备回测Top {len(backtest_list)}个Alpha...")
        print("是否开始回测? (y/n)")
        
        # 实际使用时可以自动开始
        # self.wq_client.submit_simulations(backtest_list, batch_size=batch_size)
        
        return backtest_list, ranked_alphas
    
    def analyze_prediction_quality(
        self,
        expressions: List[str],
        true_sharpes: List[float]
    ) -> Dict:
        """分析预测质量"""
        predicted_scores = self.predict_alpha_score(expressions)
        
        # 计算相关性
        correlation = np.corrcoef(predicted_scores, true_sharpes)[0, 1]
        
        # 计算Top-K准确率
        def top_k_accuracy(pred, true, k=100):
            top_k_pred_indices = np.argsort(pred)[-k:]
            top_k_true_indices = np.argsort(true)[-k:]
            overlap = len(set(top_k_pred_indices) & set(top_k_true_indices))
            return overlap / k
        
        top100_acc = top_k_accuracy(predicted_scores, true_sharpes, 100)
        top50_acc = top_k_accuracy(predicted_scores, true_sharpes, 50)
        
        results = {
            'correlation': correlation,
            'top_100_accuracy': top100_acc,
            'top_50_accuracy': top50_acc,
            'mean_predicted_score': np.mean(predicted_scores),
            'std_predicted_score': np.std(predicted_scores)
        }
        
        print("\n=== 预测质量分析 ===")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        
        return results


# 测试代码
if __name__ == "__main__":
    # 创建智能工厂
    factory = SmartAlphaFactory()
    
    # 尝试加载模型
    model_loaded = factory.load_model()
    
    if model_loaded:
        # 获取数据字段
        print("\n获取数据字段...")
        df = factory.wq_client.get_available_datafields()
        
        # 测试生成和排序
        print("\n测试智能生成和排序...")
        ranked = factory.generate_and_rank_alphas(
            df,
            generation_size=100,  # 测试用小数量
            top_k=10
        )
        
        print("\nTop 10 Alphas:")
        for i, (expr, decay, score) in enumerate(ranked[:10]):
            print(f"{i+1}. Score: {score:.4f}, Decay: {decay}")
            print(f"   {expr[:100]}...")
    else:
        print("请先训练模型")
