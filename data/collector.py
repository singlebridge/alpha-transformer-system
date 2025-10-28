"""
历史Alpha数据采集模块
从WorldQuant Brain获取历史回测数据用于训练
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

# 导入原有的机器学习库函数
from machine_lib import login, locate_alpha
from config import config


class AlphaDataCollector:
    """Alpha历史数据采集器"""
    
    def __init__(self):
        self.session = None
        self.data_dir = "./data/raw"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_historical_alphas(
        self, 
        start_date: str,
        end_date: str,
        min_alphas: int = 1000,
        max_alphas: int = 5000,
        min_sharpe: float = None,
        min_fitness: float = None,
        include_all_years: bool = False
    ) -> pd.DataFrame:
        """
        采集历史Alpha数据（增强版）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD 或 MM-DD格式)
            end_date: 结束日期 (YYYY-MM-DD 或 MM-DD格式)
            min_alphas: 最小采集数量
            max_alphas: 最大采集数量
            min_sharpe: 最小Sharpe阈值（可选）
            min_fitness: 最小Fitness阈值（可选）
            include_all_years: 是否包含所有年份（不限定2025年）
            
        Returns:
            包含Alpha表达式和性能指标的DataFrame
        """
        print(f"开始采集历史Alpha数据: {start_date} 至 {end_date}")
        print(f"采集配置: min_sharpe={min_sharpe}, min_fitness={min_fitness}, include_all_years={include_all_years}")
        self.session = login()
        
        alpha_records = []
        
        # 分批次采集
        for offset in tqdm(range(0, max_alphas, 100), desc="采集进度"):
            # 构建查询URL (包含所有状态的Alpha)
            if include_all_years:
                # 跨年度采集（不限定年份）
                url = (
                    f"{config.wq.api_base_url}/users/self/alphas?"
                    f"limit=100&offset={offset}"
                    f"&settings.region={config.wq.region}"
                    f"&order=-dateCreated"
                    f"&hidden=false"
                    f"&type!=SUPER"
                )
            else:
                # 处理日期格式
                if len(start_date.split('-')) == 3:
                    # 完整格式 YYYY-MM-DD
                    start_full = start_date
                    end_full = end_date
                else:
                    # MM-DD格式，默认2025年
                    start_full = f"2025-{start_date}"
                    end_full = f"2025-{end_date}"
                
                url = (
                    f"{config.wq.api_base_url}/users/self/alphas?"
                    f"limit=100&offset={offset}"
                    f"&dateCreated>={start_full}T00:00:00-04:00"
                    f"&dateCreated<={end_full}T23:59:59-04:00"
                    f"&settings.region={config.wq.region}"
                    f"&order=-dateCreated"
                    f"&hidden=false"
                    f"&type!=SUPER"
                )
            
            try:
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    alphas = data.get('results', [])
                    
                    if not alphas:
                        print(f"offset {offset} 无更多数据")
                        break
                    
                    for alpha in alphas:
                        try:
                            record = self._extract_alpha_info(alpha)
                            if record:
                                # 应用Sharpe和Fitness筛选
                                if min_sharpe is not None and record['sharpe'] < min_sharpe:
                                    continue
                                if min_fitness is not None and record['fitness'] < min_fitness:
                                    continue
                                alpha_records.append(record)
                        except Exception as e:
                            print(f"解析Alpha失败: {e}")
                            continue
                    
                    # 避免频繁请求
                    time.sleep(0.5)
                    
                else:
                    print(f"请求失败: {response.status_code}")
                    if response.status_code == 429:  # 限流
                        print("触发限流，等待60秒...")
                        time.sleep(60)
                        self.session = login()
                    
            except Exception as e:
                print(f"采集出错: {e}")
                time.sleep(5)
            
            # 达到最小数量后可以提前结束
            if len(alpha_records) >= min_alphas:
                break
        
        # 转换为DataFrame
        df = pd.DataFrame(alpha_records)
        print(f"\n采集完成！共获取 {len(df)} 条Alpha记录")
        
        # 保存原始数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.data_dir, f"alphas_{timestamp}.csv")
        df.to_csv(save_path, index=False)
        print(f"数据已保存至: {save_path}")
        
        return df
    
    def _extract_alpha_info(self, alpha_data: Dict) -> Dict:
        """提取Alpha关键信息"""
        try:
            # 基本信息
            alpha_id = alpha_data.get('id')
            expression = alpha_data.get('regular', {}).get('code')
            
            if not expression:
                return None
            
            # 性能指标
            is_metrics = alpha_data.get('is', {})
            settings = alpha_data.get('settings', {})
            
            record = {
                'alpha_id': alpha_id,
                'expression': expression,
                'sharpe': is_metrics.get('sharpe', 0),
                'fitness': is_metrics.get('fitness', 0),
                'turnover': is_metrics.get('turnover', 0),
                'margin': is_metrics.get('margin', 0),
                'longCount': is_metrics.get('longCount', 0),
                'shortCount': is_metrics.get('shortCount', 0),
                'decay': settings.get('decay', 0),
                'neutralization': settings.get('neutralization', ''),
                'universe': settings.get('universe', ''),
                'status': alpha_data.get('status', ''),
                'dateCreated': alpha_data.get('dateCreated', ''),
            }
            
            return record
            
        except Exception as e:
            print(f"提取信息失败: {e}")
            return None
    
    def load_existing_data(self, filename: str = None) -> pd.DataFrame:
        """加载已有的采集数据"""
        if filename is None:
            # 加载最新的文件
            files = [f for f in os.listdir(self.data_dir) if f.startswith('alphas_') and f.endswith('.csv')]
            if not files:
                print("未找到已有数据文件")
                return pd.DataFrame()
            filename = sorted(files)[-1]
        
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        print(f"加载数据: {filepath}, 共 {len(df)} 条记录")
        return df
    
    def get_statistics(self, df: pd.DataFrame):
        """显示数据统计信息"""
        print("\n=== 数据统计 ===")
        print(f"总记录数: {len(df)}")
        print(f"\nSharpe统计:")
        print(df['sharpe'].describe())
        print(f"\nFitness统计:")
        print(df['fitness'].describe())
        print(f"\nTurnover统计:")
        print(df['turnover'].describe())
        
        # 高质量Alpha统计
        high_quality = df[(df['sharpe'] > 1.0) & (df['fitness'] > 0.8)]
        print(f"\n高质量Alpha (Sharpe>1.0, Fitness>0.8): {len(high_quality)} 条")
        
        # 可提交Alpha统计
        submittable = df[(df['sharpe'] > 1.25) & (df['fitness'] > 1.0)]
        print(f"可提交Alpha (Sharpe>1.25, Fitness>1.0): {len(submittable)} 条")


# 主函数
if __name__ == "__main__":
    collector = AlphaDataCollector()
    
    # 采集最近3个月的数据
    today = datetime.now()
    three_months_ago = today - timedelta(days=90)
    
    start_date = three_months_ago.strftime("%m-%d")
    end_date = today.strftime("%m-%d")
    
    print(f"采集时间范围: {start_date} 到 {end_date}")
    
    # 执行采集
    df = collector.collect_historical_alphas(
        start_date=start_date,
        end_date=end_date,
        min_alphas=500,
        max_alphas=2000
    )
    
    # 显示统计信息
    collector.get_statistics(df)
