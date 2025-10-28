"""
增强版Alpha工厂 - 支持多数据集和二阶Alpha
利用WorldQuant Brain的丰富数据集
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import random
from typing import List, Tuple, Dict
from utils.wq_client import WorldQuantClient
from config import config


class EnhancedAlphaFactory:
    """增强版Alpha工厂 - 支持多种数据集和高阶Alpha"""
    
    def __init__(self):
        self.wq_client = WorldQuantClient()
        self.available_datasets = None
        self.dataset_fields_cache = {}
    
    def get_datasets_info(self) -> pd.DataFrame:
        """获取所有可用数据集"""
        if self.available_datasets is None:
            self.available_datasets = self.wq_client.get_available_datasets()
        return self.available_datasets
    
    def get_fields_by_category(self, category: str = 'all') -> pd.DataFrame:
        """
        按类别获取数据字段（暂时简化版，使用全部字段）
        
        Args:
            category: 'fundamental', 'technical', 'price', 'volume', 'all'
        """
        try:
            # 暂时简化：所有类别都返回全部字段
            # TODO: 未来可以实现按数据集分类
            print(f"获取数据字段（类别: {category}）")
            return self.wq_client.get_available_datafields()
        except Exception as e:
            print(f"获取字段失败: {e}")
            raise
    
    def get_fields_by_dataset(self, dataset_id: str) -> pd.DataFrame:
        """按数据集ID获取字段"""
        if dataset_id not in self.dataset_fields_cache:
            self.dataset_fields_cache[dataset_id] = \
                self.wq_client.get_available_datafields(dataset_id=dataset_id)
        return self.dataset_fields_cache[dataset_id]
    
    def generate_diversified_alphas(
        self,
        generation_size: int = 5000,
        use_datasets: List[str] = None,
        mix_strategy: str = 'balanced'
    ) -> List[Tuple[str, int]]:
        """
        生成多样化的Alpha表达式
        
        Args:
            generation_size: 生成数量
            use_datasets: 使用的数据集ID列表（None=自动选择）
            mix_strategy: 'balanced'（均衡）, 'fundamental'（财务为主）, 'technical'（技术为主）
        
        Returns:
            List of (expression, decay)
        """
        print(f"生成多样化Alpha，策略: {mix_strategy}")
        
        alpha_list = []
        
        if use_datasets is None:
            # 自动选择关键数据集
            use_datasets = self._get_recommended_datasets(mix_strategy)
        
        # 按策略分配生成比例
        if mix_strategy == 'balanced':
            ratios = {
                'price_volume': 0.3,
                'fundamental': 0.3,
                'technical': 0.2,
                'mixed': 0.2
            }
        elif mix_strategy == 'fundamental':
            ratios = {
                'fundamental': 0.6,
                'price_volume': 0.2,
                'mixed': 0.2
            }
        elif mix_strategy == 'technical':
            ratios = {
                'technical': 0.5,
                'price_volume': 0.3,
                'mixed': 0.2
            }
        else:
            ratios = {'price_volume': 1.0}
        
        # 按比例生成不同类型的Alpha
        for category, ratio in ratios.items():
            count = int(generation_size * ratio)
            if count > 0:
                category_alphas = self._generate_category_alphas(category, count)
                alpha_list.extend(category_alphas)
        
        # 随机打乱
        random.shuffle(alpha_list)
        
        print(f"生成完成: 共{len(alpha_list)}个Alpha，涵盖{len(set([a[0].split('(')[0] for a in alpha_list[:100]]))}种操作符")
        
        return alpha_list[:generation_size]
    
    def _get_recommended_datasets(self, strategy: str) -> List[str]:
        """获取推荐的数据集"""
        # 这里可以根据实际可用的数据集动态调整
        common_datasets = {
            'balanced': ['stockfundamentals_free', 'sharadar_sf1', 'technical_indicators'],
            'fundamental': ['stockfundamentals_free', 'sharadar_sf1', 'sharadar_sf3'],
            'technical': ['technical_indicators', 'price_volume']
        }
        return common_datasets.get(strategy, ['stockfundamentals_free'])
    
    def _generate_category_alphas(
        self, 
        category: str, 
        count: int
    ) -> List[Tuple[str, int]]:
        """生成特定类别的Alpha（多策略混合版）"""
        
        try:
            print(f"生成{category}类别Alpha，数量: {count}")
            
            # 策略分配（多样化）
            strategy_split = {
                'seed_mutation': int(count * 0.4),      # 40% 从种子Alpha变异
                'simple_combination': int(count * 0.3), # 30% 简单组合
                'first_order': int(count * 0.3)         # 30% 原有逻辑
            }
            
            all_alphas = []
            
            # 策略1：种子Alpha变异 ⭐⭐⭐⭐⭐
            try:
                from data.seed_alphas import ALL_SEED_ALPHAS
                from factories.alpha_mutator import generate_diverse_alphas_from_seeds
                
                seed_alphas = generate_diverse_alphas_from_seeds(
                    seed_alphas=ALL_SEED_ALPHAS,
                    target_count=strategy_split['seed_mutation'],
                    variants_per_seed=3
                )
                all_alphas.extend(seed_alphas)
                print(f"  - 种子变异: {len(seed_alphas)}个")
            except Exception as e:
                print(f"  ⚠️ 种子变异失败: {e}")
            
            # 策略2：简单组合生成 ⭐⭐⭐⭐
            try:
                from factories.alpha_mutator import generate_simple_combinations
                
                df = self.wq_client.get_available_datafields()
                if df is not None and len(df) > 0:
                    # 提取字段名
                    datafields = df['id'].tolist()[:100]  # 取前100个字段
                    
                    # 扩展操作符列表
                    operators = [
                        'ts_mean', 'ts_std_dev', 'ts_min', 'ts_max',
                        'ts_sum', 'ts_rank', 'ts_arg_min', 'ts_arg_max',
                        'ts_delta', 'rank', 'zscore'
                    ]
                    
                    simple_alphas = generate_simple_combinations(
                        datafields=datafields,
                        operators=operators,
                        count=strategy_split['simple_combination']
                    )
                    all_alphas.extend(simple_alphas)
                    print(f"  - 简单组合: {len(simple_alphas)}个")
            except Exception as e:
                print(f"  ⚠️ 简单组合失败: {e}")
            
            # 策略3：原有first_order_factory ⭐⭐⭐
            try:
                df = self.wq_client.get_available_datafields()
                if df is not None and len(df) > 0:
                    first_order = self.wq_client.generate_first_order_alphas(
                        df, 
                        max_count=strategy_split['first_order']
                    )
                    all_alphas.extend(first_order)
                    print(f"  - 原有逻辑: {len(first_order)}个")
            except Exception as e:
                print(f"  ⚠️ 原有逻辑失败: {e}")
            
            if not all_alphas:
                raise ValueError(f"所有生成策略都失败了")
            
            print(f"✓ {category}类别生成{len(all_alphas)}个Alpha（多策略混合）")
            return all_alphas
            
        except Exception as e:
            print(f"⚠️ 生成{category}类别Alpha失败: {e}")
            raise
    
    def generate_second_order_alphas(
        self,
        base_alphas: List[str],
        generation_size: int = 1000
    ) -> List[Tuple[str, int]]:
        """
        生成二阶Alpha（基于一阶Alpha的组合）
        
        Args:
            base_alphas: 基础Alpha表达式列表
            generation_size: 生成数量
        """
        print(f"生成二阶Alpha，基于{len(base_alphas)}个基础Alpha")
        
        from machine_lib import login
        from machine_lib import second_order_factory
        
        s = login()
        
        # 使用原始的second_order_factory
        second_order_alphas = second_order_factory(
            base_alphas[:min(50, len(base_alphas))],  # 限制基础Alpha数量
            config.factory.ts_ops,
            generation_size
        )
        
        # 转换为(expression, decay)格式
        result = [(alpha, random.choice([0, 1, 3, 5])) for alpha in second_order_alphas[:generation_size]]
        
        print(f"二阶Alpha生成完成: {len(result)}个")
        return result
    
    def recommend_dataset_selection(self) -> Dict:
        """推荐数据集选择方案"""
        datasets_info = self.get_datasets_info()
        
        recommendations = {
            '初学者': {
                'datasets': ['stockfundamentals_free'],
                'description': '免费财务数据，包含基本的revenue、earnings等',
                'alpha_types': '财务比率、盈利能力'
            },
            '进阶用户': {
                'datasets': ['stockfundamentals_free', 'sharadar_sf1'],
                'description': '财务数据 + Sharadar核心财务数据',
                'alpha_types': '财务分析、价值投资'
            },
            '专业用户': {
                'datasets': ['stockfundamentals_free', 'sharadar_sf1', 'technical_indicators'],
                'description': '财务 + 技术指标的全面组合',
                'alpha_types': '量价结合、多因子策略'
            },
            '研究人员': {
                'datasets': 'all_available',
                'description': '使用所有可用数据集',
                'alpha_types': '探索性研究、创新策略'
            }
        }
        
        return recommendations


# 使用示例
if __name__ == "__main__":
    factory = EnhancedAlphaFactory()
    
    # 查看可用数据集
    print("=== 可用数据集 ===")
    datasets = factory.get_datasets_info()
    if len(datasets) > 0:
        print(datasets[['id', 'name']].head(10))
    
    # 生成多样化Alpha
    print("\n=== 生成多样化Alpha ===")
    alphas = factory.generate_diversified_alphas(
        generation_size=100,
        mix_strategy='balanced'
    )
    
    print("\n示例Alpha:")
    for i, (expr, decay) in enumerate(alphas[:10]):
        print(f"{i+1}. Decay={decay}: {expr[:80]}...")
    
    # 推荐方案
    print("\n=== 数据集推荐方案 ===")
    recommendations = factory.recommend_dataset_selection()
    for level, info in recommendations.items():
        print(f"\n{level}:")
        print(f"  数据集: {info['datasets']}")
        print(f"  说明: {info['description']}")
        print(f"  适用: {info['alpha_types']}")
