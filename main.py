"""
Alpha Transformer System - 主入口
"""
import argparse
import sys
import os

# 添加父目录到路径以导入machine_lib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ui.app import launch_ui
from data.collector import AlphaDataCollector
from data.preprocessor import AlphaDataPreprocessor
from models.trainer import AlphaTransformerTrainer
from models.alpha_transformer import AlphaTransformerModel
from factories.smart_factory import SmartAlphaFactory
from config import config


def run_data_collection(args):
    """运行数据采集"""
    print("\n=== 数据采集模式 ===")
    collector = AlphaDataCollector()
    
    df = collector.collect_historical_alphas(
        start_date=args.start_date,
        end_date=args.end_date,
        min_alphas=args.min_count,
        max_alphas=args.max_count
    )
    
    collector.get_statistics(df)


def run_preprocessing(args):
    """运行数据预处理"""
    print("\n=== 数据预处理模式 ===")
    
    collector = AlphaDataCollector()
    df = collector.load_existing_data()
    
    if len(df) == 0:
        print("错误：未找到采集的数据，请先运行数据采集")
        return
    
    preprocessor = AlphaDataPreprocessor()
    data = preprocessor.prepare_training_data(
        df,
        target_metric=args.target_metric,
        filter_low_quality=args.filter_low_quality
    )
    
    print("\n预处理完成！")
    print(f"训练集: {len(data['train']['y'])}")
    print(f"验证集: {len(data['val']['y'])}")
    print(f"测试集: {len(data['test']['y'])}")


def run_training(args):
    """运行模型训练"""
    print("\n=== 模型训练模式 ===")
    
    # 加载预处理数据
    preprocessor = AlphaDataPreprocessor()
    try:
        data = preprocessor.load_preprocessed_data()
    except:
        print("错误：未找到预处理数据，请先运行预处理")
        return
    
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
    
    # 训练
    trainer = AlphaTransformerTrainer(model)
    
    if args.resume:
        trainer.load_checkpoint('best_model.pt')
    
    trainer.train(data['train'], data['val'], num_epochs=args.epochs)
    
    # 测试集评估
    from torch.utils.data import DataLoader
    from models.trainer import AlphaDataset
    
    test_dataset = AlphaDataset(
        data['test']['X'],
        data['test']['X_features'],
        data['test']['y']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_loss, test_corr = trainer.validate(test_loader)
    print(f"\n测试集评估:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")


def run_generation(args):
    """运行Alpha生成"""
    print("\n=== Alpha生成模式 ===")
    
    # 创建智能工厂
    factory = SmartAlphaFactory()
    
    # 加载模型
    if not factory.load_model(args.model_path):
        print("错误：模型加载失败")
        return
    
    # 获取数据字段
    print("获取数据字段...")
    df = factory.wq_client.get_available_datafields()
    
    # 生成并排序
    ranked_alphas = factory.generate_and_rank_alphas(
        df,
        generation_size=args.generation_size,
        top_k=args.top_k
    )
    
    # 显示结果
    print(f"\n=== Top {min(20, len(ranked_alphas))} Alphas ===")
    for i, (expr, decay, score) in enumerate(ranked_alphas[:20]):
        print(f"\n{i+1}. Predicted Score: {score:.4f}, Decay: {decay}")
        print(f"   {expr[:120]}...")
    
    # 保存到文件
    if args.save:
        import pandas as pd
        result_df = pd.DataFrame([
            {
                'rank': i+1,
                'expression': expr,
                'decay': decay,
                'predicted_score': score
            }
            for i, (expr, decay, score) in enumerate(ranked_alphas)
        ])
        
        save_path = 'generated_alphas.csv'
        result_df.to_csv(save_path, index=False)
        print(f"\n结果已保存至: {save_path}")


def run_ui(args):
    """运行UI界面"""
    print("\n=== 启动UI界面 ===")
    print(f"访问地址: http://{config.ui.server_name}:{config.ui.server_port}")
    launch_ui()


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Transformer System - 基于深度学习的智能Alpha因子生成系统"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='运行模式')
    
    # UI模式
    parser_ui = subparsers.add_parser('ui', help='启动Web UI界面')
    
    # 数据采集
    parser_collect = subparsers.add_parser('collect', help='采集历史Alpha数据')
    parser_collect.add_argument('--start-date', default='01-01', help='开始日期 (MM-DD)')
    parser_collect.add_argument('--end-date', default='12-31', help='结束日期 (MM-DD)')
    parser_collect.add_argument('--min-count', type=int, default=500, help='最小采集数量')
    parser_collect.add_argument('--max-count', type=int, default=2000, help='最大采集数量')
    
    # 数据预处理
    parser_preprocess = subparsers.add_parser('preprocess', help='预处理数据')
    parser_preprocess.add_argument('--target-metric', choices=['sharpe', 'fitness', 'combined'],
                                   default='combined', help='目标指标')
    parser_preprocess.add_argument('--filter-low-quality', action='store_true',
                                   help='过滤低质量数据')
    
    # 模型训练
    parser_train = subparsers.add_parser('train', help='训练Transformer模型')
    parser_train.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser_train.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    
    # Alpha生成
    parser_generate = subparsers.add_parser('generate', help='生成Alpha因子')
    parser_generate.add_argument('--generation-size', type=int, default=5000,
                                help='生成候选Alpha数量')
    parser_generate.add_argument('--top-k', type=int, default=500,
                                help='返回Top-K个Alpha')
    parser_generate.add_argument('--model-path', default=None,
                                help='模型路径（默认使用best_model.pt）')
    parser_generate.add_argument('--save', action='store_true',
                                help='保存结果到CSV文件')
    
    args = parser.parse_args()
    
    # 根据命令执行对应操作
    if args.command == 'ui':
        run_ui(args)
    elif args.command == 'collect':
        run_data_collection(args)
    elif args.command == 'preprocess':
        run_preprocessing(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'generate':
        run_generation(args)
    else:
        # 默认启动UI
        print("未指定命令，启动UI界面...")
        print("使用 'python main.py --help' 查看所有命令")
        run_ui(args)


if __name__ == "__main__":
    main()
