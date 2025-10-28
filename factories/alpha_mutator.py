"""
Alpha表达式变异器
用于从种子Alpha生成多样化的变体
"""
import random
import re
from typing import List, Tuple


class AlphaMutator:
    """Alpha表达式变异器"""
    
    def __init__(self):
        # 可替换的操作符（同义替换）
        self.operator_synonyms = {
            'ts_mean': ['ts_sum', 'ts_product'],
            'ts_min': ['ts_max', 'ts_arg_min'],
            'ts_max': ['ts_min', 'ts_arg_max'],
            'rank': ['zscore', 'ts_rank'],
        }
        
        # 外层包装操作
        self.wrappers = [
            'rank({expr})',
            'zscore({expr})',
            '(-1 * {expr})',
            'abs({expr})',
            'sign({expr})',
            '(1 / ({expr} + 0.001))',
            'log(abs({expr}) + 1)',
        ]
        
        # 可调整的参数范围
        self.param_adjustments = [-10, -5, -2, 2, 5, 10, 20]
    
    def mutate_expression(self, expression: str, num_variants: int = 3) -> List[str]:
        """
        从一个表达式生成多个变体
        
        Args:
            expression: 原始Alpha表达式
            num_variants: 生成变体数量
        
        Returns:
            变体表达式列表
        """
        variants = []
        
        # 方法1：调整数字参数
        variants.extend(self._mutate_parameters(expression, max_variants=num_variants))
        
        # 方法2：替换操作符
        variants.extend(self._replace_operators(expression, max_variants=num_variants))
        
        # 方法3：添加外层包装
        variants.extend(self._add_wrappers(expression, max_variants=num_variants))
        
        # 方法4：组合变异
        if len(variants) > 2:
            variants.extend(self._combine_mutations(variants[:2]))
        
        # 去重并限制数量
        variants = list(set(variants))
        return variants[:num_variants]
    
    def _mutate_parameters(self, expr: str, max_variants: int = 3) -> List[str]:
        """变异数字参数"""
        variants = []
        
        # 提取所有数字
        numbers = re.findall(r'\d+', expr)
        if not numbers:
            return variants
        
        # 对每个数字生成变体（限制数量）
        for i, num in enumerate(numbers[:min(3, len(numbers))]):
            if i >= max_variants:
                break
            
            try:
                original_num = int(num)
                new_num = original_num + random.choice(self.param_adjustments)
                
                # 确保参数合理
                if new_num <= 0:
                    new_num = max(1, original_num // 2)
                if new_num > 1000:
                    new_num = min(500, original_num)
                
                # 替换（只替换第一次出现）
                new_expr = expr.replace(num, str(new_num), 1)
                if new_expr != expr:
                    variants.append(new_expr)
            except:
                continue
        
        return variants
    
    def _replace_operators(self, expr: str, max_variants: int = 3) -> List[str]:
        """替换操作符（同义替换）"""
        variants = []
        
        for op, synonyms in self.operator_synonyms.items():
            if op in expr and len(variants) < max_variants:
                for syn in synonyms:
                    new_expr = expr.replace(op, syn, 1)
                    if new_expr != expr:
                        variants.append(new_expr)
                        if len(variants) >= max_variants:
                            break
        
        return variants
    
    def _add_wrappers(self, expr: str, max_variants: int = 3) -> List[str]:
        """添加外层包装"""
        variants = []
        
        # 避免过度嵌套（最多3层）
        if expr.count('(') > 10:
            return variants
        
        # 随机选择包装器
        selected_wrappers = random.sample(
            self.wrappers, 
            min(max_variants, len(self.wrappers))
        )
        
        for wrapper in selected_wrappers:
            new_expr = wrapper.format(expr=expr)
            variants.append(new_expr)
        
        return variants
    
    def _combine_mutations(self, variants: List[str]) -> List[str]:
        """组合多种变异"""
        combined = []
        
        if len(variants) < 2:
            return combined
        
        # 简单组合（避免过度复杂）
        expr1, expr2 = variants[0], variants[1]
        
        # 避免表达式过长
        if len(expr1) + len(expr2) > 200:
            return combined
        
        # 生成组合
        combinations = [
            f"({expr1} + {expr2})",
            f"({expr1} * {expr2})",
            f"({expr1} - {expr2})",
            f"correlation({expr1}, {expr2}, 10)",
        ]
        
        combined.extend(combinations[:2])  # 只取前2个
        return combined


def generate_diverse_alphas_from_seeds(
    seed_alphas: List[str],
    target_count: int = 1000,
    variants_per_seed: int = 3
) -> List[Tuple[str, int]]:
    """
    从种子Alpha生成多样化的变体
    
    Args:
        seed_alphas: 种子Alpha列表
        target_count: 目标生成数量
        variants_per_seed: 每个种子的变体数量
    
    Returns:
        (expression, decay)元组列表
    """
    mutator = AlphaMutator()
    result = []
    
    # 随机选择种子
    seeds_to_use = min(target_count // variants_per_seed + 1, len(seed_alphas))
    selected_seeds = random.sample(seed_alphas, seeds_to_use)
    
    for seed in selected_seeds:
        # 添加原始种子
        result.append((seed, random.choice([0, 1, 3, 5, 6])))
        
        # 生成变体
        variants = mutator.mutate_expression(seed, num_variants=variants_per_seed)
        for variant in variants:
            result.append((variant, random.choice([0, 1, 3, 5, 6])))
        
        if len(result) >= target_count:
            break
    
    return result[:target_count]


def generate_simple_combinations(
    datafields: List[str],
    operators: List[str],
    count: int = 500
) -> List[Tuple[str, int]]:
    """
    生成简单但多样化的组合
    
    Args:
        datafields: 数据字段列表
        operators: 操作符列表
        count: 生成数量
    
    Returns:
        (expression, decay)元组列表
    """
    result = []
    
    # 单操作符表达式
    single_op_templates = [
        '{op}({field}, {param})',
        'rank({op}({field}, {param}))',
        '({op}({field}, {param}) / {op}({field}, {param2}))',
    ]
    
    # 双操作符表达式
    dual_op_templates = [
        '{op1}({op2}({field}, {param1}), {param2})',
        '({op1}({field}, {param1}) - {op2}({field}, {param2}))',
        'correlation({op1}({field}, {param1}), {op2}({field}, {param2}), {param3})',
    ]
    
    params = [5, 10, 20, 30, 50, 100, 120, 200, 240]
    
    for _ in range(count):
        # 随机选择模板
        if random.random() < 0.6:
            template = random.choice(single_op_templates)
            op = random.choice(operators)
            field = random.choice(datafields)
            param = random.choice(params)
            param2 = random.choice(params)
            
            expr = template.format(
                op=op, field=field, 
                param=param, param2=param2
            )
        else:
            template = random.choice(dual_op_templates)
            op1 = random.choice(operators)
            op2 = random.choice(operators)
            field = random.choice(datafields)
            param1 = random.choice(params)
            param2 = random.choice(params)
            param3 = random.choice(params[:5])  # correlation用小参数
            
            expr = template.format(
                op1=op1, op2=op2, field=field,
                param1=param1, param2=param2, param3=param3
            )
        
        result.append((expr, random.choice([0, 1, 3, 5, 6])))
    
    return result


if __name__ == "__main__":
    # 测试
    mutator = AlphaMutator()
    
    test_expr = "(close - ts_mean(close, 20))"
    variants = mutator.mutate_expression(test_expr, num_variants=5)
    
    print(f"原始表达式: {test_expr}")
    print(f"\n生成的变体:")
    for i, var in enumerate(variants, 1):
        print(f"{i}. {var}")
