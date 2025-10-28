"""
模型训练器
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from models.alpha_transformer import AlphaTransformerModel, AlphaRankingLoss
from data.preprocessor import AlphaDataPreprocessor


class AlphaDataset(Dataset):
    """Alpha数据集"""
    
    def __init__(self, X, X_features, y):
        self.X = torch.LongTensor(X)
        self.X_features = torch.FloatTensor(X_features)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.X_features[idx], self.y[idx]


class AlphaTransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = AlphaRankingLoss(alpha=0.7, margin=0.5)
        self.optimizer = None
        self.scheduler = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # 模型配置信息（用于保存checkpoint）
        self.num_features = 8  # 默认特征数量
        
        # 创建检查点目录
        os.makedirs(config.transformer.model_save_dir, exist_ok=True)
    
    def setup_optimizer(self, learning_rate=None):
        """设置优化器和学习率调度器"""
        lr = learning_rate or config.transformer.learning_rate
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 学习率调度器：WarmupCosine
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.transformer.num_epochs,
            eta_min=1e-6
        )
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (X, X_feat, y) in enumerate(pbar):
            X = X.to(self.device)
            X_feat = X_feat.to(self.device)
            y = y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(X, X_feat)
            
            # 计算损失
            loss = self.criterion(predictions, y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                config.transformer.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, X_feat, y in val_loader:
                X = X.to(self.device)
                X_feat = X_feat.to(self.device)
                y = y.to(self.device)
                
                predictions = self.model(X, X_feat)
                loss = self.criterion(predictions, y)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算相关性
        if len(all_predictions) > 0:
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
        else:
            correlation = 0.0
        
        return avg_loss, correlation
    
    def train(self, train_data, val_data, num_epochs=None):
        """训练模型"""
        epochs = num_epochs or config.transformer.num_epochs
        batch_size = config.transformer.batch_size
        
        # 自动获取特征数量
        self.num_features = train_data['X_features'].shape[1]
        
        # 创建数据加载器
        train_dataset = AlphaDataset(
            train_data['X'], 
            train_data['X_features'], 
            train_data['y']
        )
        val_dataset = AlphaDataset(
            val_data['X'], 
            val_data['X_features'], 
            val_data['y']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Windows系统使用0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # 设置优化器
        if self.optimizer is None:
            self.setup_optimizer()
        
        print(f"\n开始训练，共{epochs}个epoch")
        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_corr = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Correlation: {val_corr:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss, val_corr)
                print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss, val_corr)
        
        print("\n训练完成！")
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename, epoch, val_loss, val_corr):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'val_correlation': val_corr,
            'vocab_size': self.model.embedding.num_embeddings,  # 保存词汇表大小
            'num_features': self.num_features,  # 保存特征数量
        }
        
        filepath = os.path.join(config.transformer.model_save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        filepath = os.path.join(config.transformer.model_save_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"检查点文件不存在: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"加载检查点: {filename}")
        print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_correlation', 'N/A')}")
        
        return True
    
    def predict(self, X, X_features):
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.LongTensor(X).to(self.device)
            X_feat_tensor = torch.FloatTensor(X_features).to(self.device)
            
            predictions = self.model(X_tensor, X_feat_tensor)
            
        return predictions.cpu().numpy()


# 主训练脚本
if __name__ == "__main__":
    print("=== Alpha Transformer 训练 ===\n")
    
    # 加载预处理数据
    preprocessor = AlphaDataPreprocessor()
    
    try:
        data = preprocessor.load_preprocessed_data()
    except:
        print("未找到预处理数据，请先运行 data/collector.py 和 data/preprocessor.py")
        exit(1)
    
    # 创建模型
    model = AlphaTransformerModel(
        vocab_size=preprocessor.tokenizer.vocab_size,
        d_model=config.transformer.d_model,
        nhead=config.transformer.nhead,
        num_layers=config.transformer.num_encoder_layers,
        dim_feedforward=config.transformer.dim_feedforward,
        dropout=config.transformer.dropout,
        max_seq_length=config.transformer.max_seq_length,
        num_features=data['train']['X_features'].shape[1]
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = AlphaTransformerTrainer(model)
    
    # 训练
    train_losses, val_losses = trainer.train(data['train'], data['val'])
    
    # 在测试集上评估
    print("\n=== 测试集评估 ===")
    test_dataset = AlphaDataset(
        data['test']['X'],
        data['test']['X_features'],
        data['test']['y']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_loss, test_corr = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
