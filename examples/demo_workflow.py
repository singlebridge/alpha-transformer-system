"""
完整工作流示例脚本
演示从数据采集到Alpha生成的全流程
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.collector import AlphaDataCollector
from data.preprocessor import AlphaDataPreprocessor
from models.trainer import AlphaTransformerTrainer, AlphaDataset
from models.alpha_transformer import AlphaTransformerModel
from factories.smart_factory import SmartAlphaFactory
from config import config
from torch.utils.data import DataLoader
import pandas as pd


def demo_full_workflow(use_existing_data=False):
    """
    完整工作流演示
    
    Args:
        use_existing_data: 是否使用已有数据（跳过采集步骤）
    """
    
    print("=" * 60)
    print("Alpha Transformer System - 完整工作流演示")
    print("=" * 60)
    
    # ============ Step 1: 数据采集 ============
    print("\n【步骤 1/4】数据采集")
    print("-" * 60)
    
    collector = AlphaDataCollector()
    
    if use_existing_data:
        print("使用已有数据...")
        df = collector.load_existing_data()
        if len(df) == 0:
            print("未找到已有数据，将执行采集")
            use_existing_data = False
    
    if not use_existing_data:
        print("开始采集历史Alpha数据...")
        df = collector.collect_historical_alphas(
            start_date='01-01',
            end_date='12-31',
            min_alphas=100,  # 演示用小数据量
            max_alphas=500
        )
    
    print(f"\n✓ 数据采集完成，共 {len(df)} 条记录")
    collector.get_statistics(df)
    
    # ============ Step 2: 数据预处理 ============
    print("\n【步骤 2/4】数据预处理")
    print("-" * 60)
    
    preprocessor = AlphaDataPreprocessor()
    
    print("开始预处理数据...")
    data = preprocessor.prepare_training_data(
        df,
        target_metric='combined',
        filter_low_quality=True
    )
    
    print(f"\n✓ 数据预处理完成")
    print(f"  训练集: {len(data['train']['y'])} 条")
    print(f"  验证集: {len(data['val']['y'])} 条")
    print(f"  测试集: {len(data['test']['y'])} 条")
    print(f"  词汇表大小: {preprocessor.tokenizer.vocab_size}")
    
    # ============ Step 3: 模型训练 ============
    print("\n【步骤 3/4】模型训练")
    print("-" * 60)
    
    # 创建模型
    model = AlphaTransformerModel(
        vocab_size=preprocessor.tokenizer.vocab_size,
        d_model=128,  # 演示用小模型
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=config.transformer.max_seq_length,
        num_features=data['train']['X_features'].shape[1]
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print("\n开始训练...")
    trainer = AlphaTransformerTrainer(model)
    
    train_losses, val_losses = trainer.train(
        data['train'],
        data['val'],
        num_epochs=10  # 演示用少量epochs
    )
    
    print(f"\n✓ 训练完成")
    print(f"  最佳验证损失: {trainer.best_val_loss:.4f}")
    
    # 测试集评估
    print("\n在测试集上评估...")
    test_dataset = AlphaDataset(
        data['test']['X'],
        data['test']['X_features'],
        data['test']['y']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_loss, test_corr = trainer.validate(test_loader)
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  测试相关性: {test_corr:.4f}")
    
    # ============ Step 4: Alpha生成 ============
    print("\n【步骤 4/4】智能Alpha生成")
    print("-" * 60)
    
    # 创建智能工厂
    factory = SmartAlphaFactory()
    
    # 加载刚训练的模型
    print("加载模型...")
    factory.model = model
    factory.tokenizer = preprocessor.tokenizer
    factory.scaler = preprocessor.scaler
    factory.device = trainer.device
    
    # 获取数据字段
    print("获取数据字段...")
    datafields = factory.wq_client.get_available_datafields()
    
    # 生成并排序
    print("\n生成并排序Alpha...")
    ranked_alphas = factory.generate_and_rank_alphas(
        datafields,
        generation_size=100,  # 演示用小数量
        top_k=20
    )
    
    print(f"\n✓ Alpha生成完成")
    print(f"\n【Top 10 高分Alpha】")
    print("-" * 60)
    
    for i, (expr, decay, score) in enumerate(ranked_alphas[:10]):
        print(f"\n{i+1}. 预测分数: {score:.4f} | Decay: {decay}")
        print(f"   {expr[:100]}...")
    
    # 保存结果
    print("\n保存结果到CSV...")
    result_df = pd.DataFrame([
        {
            'rank': i+1,
            'expression': expr,
            'decay': decay,
            'predicted_score': score
        }
        for i, (expr, decay, score) in enumerate(ranked_alphas)
    ])
    
    result_df.to_csv('demo_generated_alphas.csv', index=False)
    print("✓ 已保存至: demo_generated_alphas.csv")
    
    # ============ 总结 ============
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n下一步建议：")
    print("1. 查看 demo_generated_alphas.csv 中的Alpha表达式")
    print("2. 使用 factory.wq_client.submit_simulations() 回测Top-K Alpha")
    print("3. 使用更多数据（1000-3000条）和更长训练时间获得更好效果")
    print("4. 启动UI界面: python main.py ui")
    
    return {
        'data': data,
        'model': model,
        'trainer': trainer,
        'factory': factory,
        'ranked_alphas': ranked_alphas
    }


def demo_prediction_analysis():
    """
    预测质量分析示例
    需要已有训练好的模型和测试数据
    """
    print("\n" + "=" * 60)
    print("预测质量分析示例")
    print("=" * 60)
    
    # 加载数据
    preprocessor = AlphaDataPreprocessor()
    try:
        data = preprocessor.load_preprocessed_data()
    except:
        print("请先运行完整工作流或训练模型")
        return
    
    # 加载模型
    factory = SmartAlphaFactory()
    if not factory.load_model():
        print("模型加载失败")
        return
    
    # 获取测试集表达式和真实Sharpe
    test_expressions = data['test']['df']['expression'].tolist()[:100]
    test_sharpes = data['test']['df']['sharpe'].tolist()[:100]
    
    # 分析预测质量
    results = factory.analyze_prediction_quality(
        test_expressions,
        test_sharpes
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha Transformer 工作流演示")
    parser.add_argument('--use-existing', action='store_true',
                       help='使用已有数据（跳过采集）')
    parser.add_argument('--analysis-only', action='store_true',
                       help='仅运行预测分析')
    
    args = parser.parse_args()
    
    if args.analysis_only:
        demo_prediction_analysis()
    else:
        demo_full_workflow(use_existing_data=args.use_existing)
