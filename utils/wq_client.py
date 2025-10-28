"""
WorldQuant Brain API客户端
封装原有machine_lib的功能，提供更友好的接口
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from machine_lib import (
    login, get_datasets, get_datafields, process_datafields,
    first_order_factory, load_task_pool_single, single_simulate,
    get_alphas, prune, check_submission, view_alphas
)
from config import config
from typing import List, Tuple, Dict
import pandas as pd


class WorldQuantClient:
    """WorldQuant Brain API客户端"""
    
    def __init__(self):
        self.session = None
        self.login()
    
    def login(self):
        """登录"""
        print("登录WorldQuant Brain...")
        self.session = login()
        return self.session
    
    def get_available_datafields(self, dataset_id: str = None, search: str = None) -> pd.DataFrame:
        """
        获取可用的数据字段（增强版）
        
        Args:
            dataset_id: 数据集ID（如'stockfundamentals_free'）
            search: 搜索关键词（如'revenue', 'earnings'）
        """
        from machine_lib import get_datafields as get_datafields_raw
        
        s = login()
        df = get_datafields_raw(
            s,
            instrument_type='EQUITY',
            region=config.wq.region,
            delay=1,
            universe=config.wq.universe,
            dataset_id=dataset_id or '',
            search=search or ''
        )
        return df
    
    def get_available_datasets(self) -> pd.DataFrame:
        """获取可用的数据集列表"""
        from machine_lib import get_datasets
        
        s = login()
        df = get_datasets(
            s,
            instrument_type='EQUITY',
            region=config.wq.region,
            delay=1,
            universe=config.wq.universe
        )
        return df
    
    def generate_first_order_alphas(
        self, 
        datafields: pd.DataFrame,
        max_count: int = None
    ) -> List[Tuple[str, int]]:
        """生成一阶Alpha表达式"""
        max_count = max_count or config.factory.max_first_order
        
        # 预处理数据字段
        processed_fields = process_datafields(datafields)
        
        # 生成表达式
        alphas = first_order_factory(processed_fields, config.factory.ts_ops)
        
        # 限制数量并添加初始decay
        alphas = alphas[:max_count]
        alpha_list = [(alpha, 6) for alpha in alphas]
        
        print(f"生成一阶表达式数量: {len(alpha_list)}")
        return alpha_list
    
    def submit_simulations(
        self,
        alpha_list: List[Tuple[str, int]],
        batch_size: int = 3,
        start_index: int = 0
    ):
        """批量提交回测"""
        pools = load_task_pool_single(alpha_list, batch_size)
        
        print(f"开始回测，共{len(pools)}个批次")
        single_simulate(
            pools, 
            config.wq.neutralization,
            config.wq.region,
            config.wq.universe,
            start_index
        )
    
    def fetch_alphas_by_performance(
        self,
        start_date: str,
        end_date: str,
        min_sharpe: float = None,
        min_fitness: float = None,
        max_count: int = 200,
        for_submission: bool = False
    ) -> List:
        """根据性能指标获取Alpha"""
        min_sharpe = min_sharpe or config.factory.min_sharpe
        min_fitness = min_fitness or config.factory.min_fitness
        
        usage = "submit" if for_submission else "track"
        
        alphas = get_alphas(
            start_date, end_date,
            min_sharpe, min_fitness,
            config.wq.region,
            max_count,
            usage
        )
        
        return alphas
    
    def prune_similar_alphas(
        self,
        alpha_recs: List,
        field_prefix: str,
        keep_top_k: int = None
    ) -> List:
        """剪枝相似Alpha"""
        keep_top_k = keep_top_k or config.factory.prune_keep_num
        
        pruned = prune(alpha_recs, field_prefix, keep_top_k)
        
        print(f"剪枝前: {len(alpha_recs)}, 剪枝后: {len(pruned)}")
        return pruned
    
    def check_alphas_for_submission(
        self,
        alpha_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """检查Alpha是否可提交"""
        gold_bag = []
        check_submission(alpha_ids, gold_bag, 0)
        
        print(f"可提交Alpha数量: {len(gold_bag)}")
        return gold_bag
    
    def display_alpha_info(self, gold_bag: List[Tuple[str, float]]):
        """显示Alpha信息"""
        view_alphas(gold_bag)


# 测试代码
if __name__ == "__main__":
    client = WorldQuantClient()
    
    # 测试获取数据字段
    df = client.get_available_datafields()
    print(f"可用数据字段数量: {len(df)}")
    print(df.head())
