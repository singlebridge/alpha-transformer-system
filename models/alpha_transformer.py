"""
Alpha Transformer模型
使用Transformer预测Alpha表达式的Sharpe/Fitness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AlphaTransformerModel(nn.Module):
    """
    Alpha Transformer排序模型
    输入：Alpha表达式的token序列 + 手工特征
    输出：预测的Sharpe/Fitness分数
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 128,
        num_features: int = 8,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 手工特征融合层
        self.feature_fc = nn.Sequential(
            nn.Linear(num_features, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_features, src_mask=None):
        """
        Args:
            src: Tensor [batch_size, seq_len] - token IDs
            src_features: Tensor [batch_size, num_features] - 手工特征
            src_mask: Tensor [seq_len, seq_len] - attention mask
            
        Returns:
            output: Tensor [batch_size, 1] - 预测分数
        """
        # Token嵌入: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=src.device))
        
        # 转换为 [seq_len, batch_size, d_model] 用于Transformer
        embedded = embedded.transpose(0, 1)
        
        # 添加位置编码
        embedded = self.pos_encoder(embedded)
        
        # Transformer编码
        # output: [seq_len, batch_size, d_model]
        encoded = self.transformer_encoder(embedded, src_mask)
        
        # 池化：取所有token的平均
        # [seq_len, batch_size, d_model] -> [batch_size, d_model]
        pooled = encoded.mean(dim=0)
        
        # 处理手工特征
        # [batch_size, num_features] -> [batch_size, d_model//4]
        feat_encoded = self.feature_fc(src_features)
        
        # 融合特征
        # [batch_size, d_model + d_model//4]
        combined = torch.cat([pooled, feat_encoded], dim=1)
        
        # 输出预测
        # [batch_size, 1]
        output = self.output_fc(combined)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """生成attention mask（用于自回归任务，此处暂不需要）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class AlphaRankingLoss(nn.Module):
    """
    排序损失函数
    结合MSE和Ranking Loss
    """
    
    def __init__(self, alpha: float = 0.7, margin: float = 0.5):
        super().__init__()
        self.alpha = alpha  # MSE权重
        self.margin = margin  # Ranking margin
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, 1]
            targets: [batch_size, 1]
        """
        # MSE损失
        mse_loss = self.mse(predictions, targets)
        
        # Pairwise Ranking损失
        # 对于batch中的样本对，如果target_i > target_j，则希望pred_i > pred_j
        batch_size = predictions.size(0)
        if batch_size < 2:
            return mse_loss
        
        ranking_loss = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if targets[i] > targets[j]:
                    # 希望predictions[i] > predictions[j] + margin
                    loss = F.relu(self.margin - (predictions[i] - predictions[j]))
                    ranking_loss += loss
                    count += 1
                elif targets[i] < targets[j]:
                    # 希望predictions[j] > predictions[i] + margin
                    loss = F.relu(self.margin - (predictions[j] - predictions[i]))
                    ranking_loss += loss
                    count += 1
        
        if count > 0:
            ranking_loss = ranking_loss / count
        
        # 组合损失
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ranking_loss
        
        return total_loss


# 测试代码
if __name__ == "__main__":
    # 测试模型
    vocab_size = 1000
    batch_size = 16
    seq_len = 64
    num_features = 8
    
    model = AlphaTransformerModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=128,
        num_features=num_features
    )
    
    # 创建随机输入
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    src_features = torch.randn(batch_size, num_features)
    targets = torch.randn(batch_size, 1)
    
    # 前向传播
    output = model(src, src_features)
    print(f"输入形状: {src.shape}, {src_features.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试损失函数
    criterion = AlphaRankingLoss()
    loss = criterion(output, targets)
    print(f"损失: {loss.item():.4f}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
