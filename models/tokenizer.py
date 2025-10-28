"""
Alpha表达式分词器
将表达式解析为token序列，用于Transformer训练
"""
import re
from typing import List, Dict, Tuple
import pickle
import os


class AlphaTokenizer:
    """Alpha表达式分词器"""
    
    def __init__(self):
        self.vocab = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_freq = {}
        
    def build_vocab_from_expressions(self, expressions: List[str], min_freq: int = 2):
        """从表达式列表构建词汇表"""
        print(f"构建词汇表，表达式数量: {len(expressions)}")
        
        # 统计token频率
        for expr in expressions:
            tokens = self._tokenize_expression(expr)
            for token in tokens:
                self.token_freq[token] = self.token_freq.get(token, 0) + 1
        
        # 添加高频token到词汇表
        idx = len(self.vocab)
        for token, freq in sorted(self.token_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token
                idx += 1
        
        print(f"词汇表大小: {len(self.vocab)}")
        print(f"Top 10 tokens: {list(self.vocab.keys())[4:14]}")
        
    def _tokenize_expression(self, expression: str) -> List[str]:
        """将表达式分解为token列表"""
        # 移除空格
        expression = expression.replace(' ', '')
        
        tokens = []
        current_token = ''
        depth = 0
        
        for char in expression:
            if char == '(':
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append('(')
                depth += 1
            elif char == ')':
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append(')')
                depth -= 1
            elif char == ',':
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append(',')
            elif char in '+-*/':
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def encode(self, expression: str, max_length: int = 128) -> List[int]:
        """将表达式编码为token ID序列"""
        tokens = self._tokenize_expression(expression)
        
        # 添加起始和结束符
        token_ids = [self.vocab['<SOS>']]
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        token_ids.append(self.vocab['<EOS>'])
        
        # 截断或填充
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [self.vocab['<PAD>']] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID序列解码为表达式"""
        tokens = []
        for idx in token_ids:
            if idx == self.vocab['<PAD>'] or idx == self.vocab['<EOS>']:
                break
            if idx == self.vocab['<SOS>']:
                continue
            if idx in self.inverse_vocab:
                tokens.append(self.inverse_vocab[idx])
            else:
                tokens.append('<UNK>')
        
        return ''.join(tokens)
    
    def extract_features(self, expression: str) -> Dict[str, any]:
        """提取表达式特征"""
        tokens = self._tokenize_expression(expression)
        
        features = {
            'length': len(tokens),
            'depth': self._calculate_depth(expression),
            'num_operators': sum(1 for t in tokens if t.startswith('ts_') or t.startswith('group_')),
            'has_winsorize': 'winsorize' in expression,
            'has_ts_backfill': 'ts_backfill' in expression,
            'has_group_op': any(t.startswith('group_') for t in tokens),
            'has_trade_when': 'trade_when' in expression,
            'num_params': tokens.count(','),
        }
        
        return features
    
    def _calculate_depth(self, expression: str) -> int:
        """计算表达式嵌套深度"""
        max_depth = 0
        current_depth = 0
        
        for char in expression:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth
    
    def save(self, path: str):
        """保存分词器"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'token_freq': self.token_freq,
            }, f)
        print(f"Tokenizer已保存至: {path}")
    
    def load(self, path: str):
        """加载分词器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab']
            self.token_freq = data['token_freq']
        print(f"Tokenizer已加载，词汇表大小: {len(self.vocab)}")
    
    @property
    def vocab_size(self) -> int:
        """词汇表大小"""
        return len(self.vocab)


# 测试代码
if __name__ == "__main__":
    tokenizer = AlphaTokenizer()
    
    # 测试表达式
    test_expressions = [
        "ts_rank(winsorize(ts_backfill(assets, 120), std=4), 22)",
        "group_neutralize(ts_zscore(close, 20), densify(sector))",
        "trade_when(ts_corr(close, volume, 5) > 0, rank(assets), abs(returns) > 0.1)",
    ]
    
    # 构建词汇表
    tokenizer.build_vocab_from_expressions(test_expressions)
    
    # 测试编码解码
    for expr in test_expressions:
        print(f"\n原始表达式: {expr}")
        encoded = tokenizer.encode(expr, max_length=64)
        print(f"编码长度: {len(encoded)}")
        decoded = tokenizer.decode(encoded)
        print(f"解码表达式: {decoded}")
        features = tokenizer.extract_features(expr)
        print(f"特征: {features}")
