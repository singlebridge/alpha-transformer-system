"""
数据预处理模块
为Transformer训练准备数据集
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.tokenizer import AlphaTokenizer
from config import config


class AlphaDataPreprocessor:
    """Alpha数据预处理器"""
    
    def __init__(self):
        self.tokenizer = AlphaTokenizer()
        self.scaler = StandardScaler()
        self.processed_dir = "./data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def prepare_training_data(
        self,
        data_path: str = None,
        target_metric: str = 'sharpe',
        use_seed_alphas: bool = True,
        seed_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据（增强版）
        
        Args:
            data_path: 数据文件路径
            target_metric: 目标指标 ('sharpe', 'fitness', 'combined')
            use_seed_alphas: 是否注入高质量种子Alpha
            seed_ratio: 种子Alpha占比（默认30%）
            
        Returns:
            训练、验证、测试数据集
        """
        # 加载数据
        df = None  # 初始化
        
        try:
            from data.collector import AlphaDataCollector
            collector = AlphaDataCollector()
            
            if data_path:
                df = pd.read_csv(data_path)
            else:
                df = collector.load_existing_data()
            
            if df is None or len(df) == 0:
                raise ValueError("没有可用的数据！请先在Tab 1采集数据。")
            
            print(f"开始预处理数据，原始记录数: {len(df)}")
            
            # ✨ 新增：注入高质量种子Alpha
            if use_seed_alphas:
                try:
                    from data.seed_alphas import augment_with_seed_alphas
                    print(f"\n✨ 注入高质量种子Alpha（占比{seed_ratio*100:.0f}%）...")
                    df = augment_with_seed_alphas(df, seed_ratio=seed_ratio)
                except Exception as e:
                    print(f"⚠️ 种子Alpha注入失败: {e}")
                    print(f"  详细错误: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    print("继续使用原始数据...")
        except Exception as e:
            raise ValueError(f"数据加载失败: {str(e)}")
        
        # 数据清洗
        df = self._clean_data(df, filter_low_quality=True)
        print(f"清洗后记录数: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("清洗后没有可用数据！请检查数据质量或降低筛选标准。")
        
        # 构建tokenizer词汇表
        print("构建分词器词汇表...")
        self.tokenizer.build_vocab_from_expressions(df['expression'].tolist())
        
        # 编码表达式
        print("编码表达式...")
        encoded_expressions = []
        for expr in df['expression']:
            encoded = self.tokenizer.encode(expr, config.transformer.max_seq_length)
            encoded_expressions.append(encoded)
        
        X = np.array(encoded_expressions)
        
        # 提取表达式特征
        print("提取表达式特征...")
        feature_list = []
        for expr in df['expression']:
            features = self.tokenizer.extract_features(expr)
            feature_list.append(list(features.values()))
        
        X_features = np.array(feature_list)
        X_features_scaled = self.scaler.fit_transform(X_features)
        
        # 准备目标变量
        y = self._prepare_targets(df, target_metric)
        
        # 分割数据集
        print("分割数据集...")
        train_data, val_data, test_data = self._split_dataset(
            X, X_features_scaled, y, df
        )
        
        # 保存预处理结果
        self._save_preprocessed_data(train_data, val_data, test_data)
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'tokenizer': self.tokenizer,
            'scaler': self.scaler,
            'feature_names': list(self.tokenizer.extract_features(df['expression'].iloc[0]).keys())
        }
    
    def _clean_data(self, df: pd.DataFrame, filter_low_quality: bool) -> pd.DataFrame:
        """数据清洗"""
        # 移除空表达式
        df = df[df['expression'].notna()].copy()
        df = df[df['expression'].str.len() > 0].copy()
        
        # 移除异常值
        df = df[df['sharpe'].notna()].copy()
        df = df[df['fitness'].notna()].copy()
        df = df[np.isfinite(df['sharpe'])].copy()
        df = df[np.isfinite(df['fitness'])].copy()
        
        # 过滤低质量数据（可选）
        if filter_low_quality:
            # 移除sharpe和fitness都很差的
            df = df[
                (df['sharpe'].abs() > 0.1) | 
                (df['fitness'].abs() > 0.1)
            ].copy()
            
            # 移除持仓数过少的
            df = df[(df['longCount'] + df['shortCount']) > 50].copy()
        
        # 去重
        df = df.drop_duplicates(subset=['expression']).copy()
        
        return df.reset_index(drop=True)
    
    def _prepare_targets(self, df: pd.DataFrame, target_metric: str) -> np.ndarray:
        """准备目标变量"""
        if target_metric == 'sharpe':
            y = df['sharpe'].values
        elif target_metric == 'fitness':
            y = df['fitness'].values
        elif target_metric == 'combined':
            # 综合指标：sharpe和fitness的加权组合
            y = 0.6 * df['sharpe'].values + 0.4 * df['fitness'].values
        else:
            raise ValueError(f"不支持的目标指标: {target_metric}")
        
        return y.astype(np.float32)
    
    def _split_dataset(
        self, 
        X: np.ndarray, 
        X_features: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame
    ) -> Tuple[Dict, Dict, Dict]:
        """分割数据集"""
        # 第一次分割：训练集 vs (验证集+测试集)
        train_size = config.transformer.train_ratio
        val_size = config.transformer.val_ratio
        test_size = config.transformer.test_ratio
        
        X_train, X_temp, X_feat_train, X_feat_temp, y_train, y_temp, df_train, df_temp = \
            train_test_split(
                X, X_features, y, df,
                test_size=(val_size + test_size),
                random_state=42,
                shuffle=True
            )
        
        # 第二次分割：验证集 vs 测试集
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, X_feat_val, X_feat_test, y_val, y_test, df_val, df_test = \
            train_test_split(
                X_temp, X_feat_temp, y_temp, df_temp,
                test_size=(1 - val_ratio),
                random_state=42,
                shuffle=True
            )
        
        train_data = {
            'X': X_train,
            'X_features': X_feat_train,
            'y': y_train,
            'df': df_train
        }
        
        val_data = {
            'X': X_val,
            'X_features': X_feat_val,
            'y': y_val,
            'df': df_val
        }
        
        test_data = {
            'X': X_test,
            'X_features': X_feat_test,
            'y': y_test,
            'df': df_test
        }
        
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        
        return train_data, val_data, test_data
    
    def _save_preprocessed_data(self, train_data, val_data, test_data):
        """保存预处理后的数据"""
        save_path = os.path.join(self.processed_dir, "dataset.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'train': train_data,
                'val': val_data,
                'test': test_data
            }, f)
        
        print(f"预处理数据已保存至: {save_path}")
        
        # 保存tokenizer和scaler
        tokenizer_path = os.path.join(self.processed_dir, "tokenizer.pkl")
        scaler_path = os.path.join(self.processed_dir, "scaler.pkl")
        
        self.tokenizer.save(tokenizer_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Tokenizer和Scaler已保存")
    
    def load_preprocessed_data(self) -> Dict:
        """加载预处理后的数据"""
        dataset_path = os.path.join(self.processed_dir, "dataset.pkl")
        tokenizer_path = os.path.join(self.processed_dir, "tokenizer.pkl")
        scaler_path = os.path.join(self.processed_dir, "scaler.pkl")
        
        # 加载数据集
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        # 加载tokenizer
        self.tokenizer.load(tokenizer_path)
        
        # 加载scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("预处理数据加载完成")
        return data


# 测试代码
if __name__ == "__main__":
    # 加载采集的数据
    from collector import AlphaDataCollector
    
    collector = AlphaDataCollector()
    df = collector.load_existing_data()
    
    if len(df) > 0:
        # 预处理
        preprocessor = AlphaDataPreprocessor()
        data = preprocessor.prepare_training_data(df, target_metric='combined')
        
        print("\n=== 数据集信息 ===")
        print(f"训练集大小: {len(data['train']['y'])}")
        print(f"验证集大小: {len(data['val']['y'])}")
        print(f"测试集大小: {len(data['test']['y'])}")
        print(f"词汇表大小: {preprocessor.tokenizer.vocab_size}")
        print(f"特征维度: {data['train']['X_features'].shape[1]}")
    else:
        print("请先运行collector.py采集数据")
