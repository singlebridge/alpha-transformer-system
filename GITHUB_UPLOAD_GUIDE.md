# GitHub上传准备指南

## ⚠️ 上传前必须完成的步骤

### 1. 清理敏感信息 ✅

#### 已完成：
- ✅ 创建了 `config.py.example` 模板文件
- ✅ 创建了 `.env.example` 环境变量模板
- ✅ 更新了 `.gitignore` 排除敏感文件

#### 你需要做：
```bash
# 1. 备份你的config.py（包含真实账号密码）
cp config.py config_local.py

# 2. 用模板替换config.py（或直接删除config.py，让用户自己创建）
cp config.py.example config.py

# 3. 确认.gitignore生效
git status  # 查看是否排除了敏感文件
```

---

### 2. 清理个人数据 ✅

以下文件/文件夹已在 `.gitignore` 中排除，不会上传：

```
data/raw/*.csv                 # 你的历史Alpha数据
data/preprocessed/*.pkl        # 预处理后的数据
checkpoints/*.pt               # 训练好的模型（太大）
logs/                          # 训练日志
__pycache__/                   # Python缓存
.env                           # 环境变量（如果使用）
config_local.py                # 本地配置备份
```

**建议：** 上传前检查一下这些文件夹是否包含个人信息。

---

### 3. 准备示例数据（可选）

如果想让其他人能快速测试，可以准备一些**不含真实数据**的示例：

```python
# 在 data/examples/ 创建示例数据
data/examples/
  ├── sample_alphas.csv        # 10-20条示例Alpha（虚构数据）
  └── README.md                # 说明这是示例数据
```

**示例数据格式：**
```csv
alpha_id,expression,sharpe,fitness,turnover,returns
DEMO_001,(close - ts_mean(close, 20)),1.2,0.8,0.05,0.15
DEMO_002,rank(volume),0.9,0.6,0.03,0.10
```

---

### 4. 检查代码中的硬编码

搜索可能包含个人信息的代码：

```bash
# 搜索email
grep -r "@qq.com" . --exclude-dir=.git

# 搜索密码相关
grep -r "password" . --exclude-dir=.git

# 搜索用户ID
grep -r "JZ27229" . --exclude-dir=.git
```

如果发现硬编码的敏感信息，替换为：
- 环境变量：`os.getenv("WQ_USERNAME")`
- 配置文件：从 `config.py` 读取
- 示例值：`"your_email@example.com"`

---

## 🚀 上传到GitHub

### 方式1：命令行（推荐）

```bash
# 1. 初始化Git仓库（如果还没有）
cd /path/to/alpha_transformer_system
git init

# 2. 添加远程仓库
git remote add origin https://github.com/your_username/alpha-transformer-system.git

# 3. 检查要提交的文件
git status
git diff

# 4. 添加文件
git add .

# 5. 提交
git commit -m "Initial commit: Alpha Transformer System

- Transformer-based alpha factor generation
- Multi-strategy generation framework
- Seed alpha injection mechanism
- Complete UI interface
- Comprehensive documentation"

# 6. 推送到GitHub
git push -u origin main
```

### 方式2：GitHub Desktop

1. 打开 GitHub Desktop
2. 选择 "Add Existing Repository"
3. 选择 `alpha_transformer_system` 文件夹
4. 检查要提交的文件列表
5. 写提交消息
6. 点击 "Publish repository"

---

## 📋 上传清单

在上传前，确认以下事项：

### 必须检查 ✅
- [ ] `config.py` 不包含真实账号密码
- [ ] `data/raw/` 文件夹为空或只有示例数据
- [ ] `.gitignore` 正确配置
- [ ] `README.md` 完整且准确
- [ ] 代码中没有硬编码的敏感信息

### 建议检查
- [ ] 添加了 `LICENSE` 文件（MIT/Apache 2.0）
- [ ] 添加了 `requirements.txt` 依赖列表
- [ ] 添加了 `.env.example` 环境变量模板
- [ ] 添加了 `CONTRIBUTING.md` 贡献指南
- [ ] 更新了文档中的安装说明

### 文件结构检查
```
alpha_transformer_system/
├── README.md ✅
├── LICENSE ⚠️ (需要添加)
├── requirements.txt ⚠️ (需要添加)
├── .gitignore ✅
├── .env.example ✅
├── config.py.example ✅
├── config.py ⚠️ (确认不含敏感信息)
├── main.py ✅
├── data/
│   ├── __init__.py ✅
│   ├── collector.py ✅
│   ├── preprocessor.py ✅
│   ├── seed_alphas.py ✅
│   ├── raw/ (空文件夹或.gitkeep)
│   └── examples/ ⚠️ (建议添加示例数据)
├── models/
├── factories/
├── ui/
├── utils/
├── examples/ ⚠️ (建议添加使用示例)
└── docs/ ✅ (已有多个.md文档)
```

---

## 📝 需要创建的额外文件

### 1. LICENSE 文件

```bash
# 选择MIT License（推荐）
# 访问 https://choosealicense.com/licenses/mit/
# 复制内容，替换 [year] 和 [fullname]
```

### 2. requirements.txt

```bash
# 生成依赖列表
pip freeze > requirements.txt

# 或手动创建（推荐，只包含必需依赖）
```

### 3. CONTRIBUTING.md（可选）

贡献指南，说明如何参与项目开发。

### 4. CHANGELOG.md（可选）

版本更新日志。

---

## 🔒 安全提示

### 如果不小心上传了敏感信息怎么办？

#### 方案1：删除最后一次提交
```bash
git reset --soft HEAD~1  # 撤销提交但保留更改
# 修改文件，移除敏感信息
git add .
git commit -m "Remove sensitive information"
git push --force
```

#### 方案2：使用 BFG Repo-Cleaner
```bash
# 下载 BFG: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --replace-text passwords.txt
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
```

#### 方案3：删除仓库重新上传
1. 在GitHub上删除仓库
2. 清理本地敏感信息
3. 重新创建仓库并上传

⚠️ **重要：** 即使删除了提交，GitHub历史中可能仍有记录。如果泄露了密码，**立即修改密码！**

---

## ✅ 最终检查命令

上传前运行这些命令进行最后检查：

```bash
# 1. 检查git状态
git status

# 2. 查看将要提交的文件
git ls-files

# 3. 搜索敏感信息
grep -r "484807978" .
grep -r "brQZ3p71M68SE" .
grep -r "@qq.com" .

# 4. 检查.gitignore是否生效
git check-ignore -v data/raw/alphas_*.csv
git check-ignore -v config_local.py
git check-ignore -v .env

# 5. 模拟提交（不实际提交）
git add --dry-run .
```

---

## 📞 需要帮助？

如果遇到问题：
1. 查看GitHub帮助文档：https://docs.github.com/
2. 搜索相关问题：https://stackoverflow.com/
3. 确保已备份重要文件

---

**祝上传顺利！🎉**
