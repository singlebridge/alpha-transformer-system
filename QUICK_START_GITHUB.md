# 🚀 快速开始 - GitHub版本

如果你是从GitHub克隆的此项目，请按照以下步骤配置：

## 📦 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your_username/alpha-transformer-system.git
cd alpha-transformer-system

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置账户信息

### 方式1: 环境变量（推荐） ⭐

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑.env文件，填入你的WorldQuant Brain账户
# WQ_USERNAME=your_email@example.com
# WQ_PASSWORD=your_password_here
```

### 方式2: 配置文件

```bash
# 1. 复制配置模板
cp config.py.example config.py

# 2. 编辑config.py，在第12-13行填入你的账户信息
# username: str = "your_email@example.com"
# password: str = "your_password_here"
```

⚠️ **重要提示：**
- 不要将包含真实账号密码的文件提交到Git！
- `.env` 和 `config_local.py` 已在 `.gitignore` 中，不会被提交

## 🎯 运行项目

### 启动UI界面
```bash
python main.py ui
```

访问：http://localhost:7860

### 命令行模式
```bash
# 数据采集
python main.py collect --count 100

# 数据预处理
python main.py preprocess

# 模型训练
python main.py train --epochs 30

# Alpha生成
python main.py generate --count 5000
```

## 📚 文档

- 📖 [完整文档](README.md)
- 🎯 [快速入门](QUICKSTART.md)
- 🚀 [质量提升指南](QUALITY_BOOST_GUIDE.md)
- 🏗️ [项目架构](PROJECT_STRUCTURE.md)

## ❓ 常见问题

### Q1: 没有WorldQuant Brain账号怎么办？
A: 访问 https://worldquantbrain.com 注册免费账号

### Q2: 如何准备训练数据？
A: 
1. 先在Tab 1采集历史Alpha数据
2. 在Tab 2预处理，建议勾选"注入种子Alpha"
3. 在Tab 3开始训练

### Q3: 训练需要多长时间？
A: 
- CPU: 约2-5小时（30个epoch）
- GPU: 约30-60分钟（30个epoch）

### Q4: 如何使用种子Alpha提升质量？
A: 在Tab 2预处理时，勾选"注入高质量种子Alpha"，设置占比30%

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**祝你使用愉快！如有问题请提Issue 💬**
