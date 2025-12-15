# 🎯 从这里开始

## 👋 欢迎！

我已经帮你整理好了代码结构和训练流程。现在你可以清晰地运行整个训练过程了！

---

## 📚 我创建的文件

为了帮你理清代码，我创建了以下文档：

| 文件 | 用途 | 推荐阅读顺序 |
|------|------|-------------|
| **START_HERE.md** (本文件) | 快速上手指南 | ⭐ 先读这个 |
| **README_QUICK_START.md** | 详细快速开始文档 | ⭐⭐ 重点参考 |
| **WORKFLOW.md** | 完整工作流程说明 | ⭐⭐⭐ 详细参考 |
| **config_template.py** | 配置文件模板 | 用于创建配置 |
| **run_complete_workflow.sh** | 自动化执行脚本 | 可选使用 |
| **check_config.py** | 配置检查脚本 | 运行前检查 |

---

## 🚀 快速开始（3步）

### 第1步：检查并修改配置 (5分钟)

```bash
cd /home/user/triplet

# 打开配置文件
vim src/config.py

# 必须修改的配置：
# 1. 数据路径 DATA_PATHS（csv_path, embedding_dir, 测试集路径）
# 2. GPU设备号 DEVICE
# 3. 确认实验模式 EXPERIMENT_MODE（五折交叉验证时用 "eval_hyperparam_search"）
```

**关键配置示例：**
```python
# src/config.py

# GPU设备
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据路径（改成你的实际路径！）
DATA_PATHS = {
    "csv_path": "/your/path/to/split100.csv",
    "embedding_dir": "/your/path/to/protein_embeddings",
    "external_test_sets": {
        "price": "/your/path/to/price.csv",
        "new": "/your/path/to/new.csv"
    }
}

# 第一步使用这个模式
EXPERIMENT_MODE = "eval_hyperparam_search"
```

---

### 第2步：运行五折交叉验证 (几小时)

```bash
cd src
python main_kfold.py
```

**这一步会：**
- 自动进行5折交叉验证
- 每折独立训练，使用早停（patience=15）
- 在price和new测试集上评估每一折
- 生成聚合的平均结果

**等待完成后，查看结果：**

```bash
# 查看每折训练了多少轮
grep "Total epochs trained:" ../experiments/*KFold*/fold_*/fold.log

# 输出示例：
# fold_1: 45 epochs
# fold_2: 48 epochs
# fold_3: 52 epochs
# fold_4: 46 epochs
# fold_5: 49 epochs
# 平均: (45+48+52+46+49)/5 = 48 epochs

# 查看聚合的测试结果
cat ../experiments/*KFold*/final_cross_validation_summary.json
```

**记住平均的epoch数**（例如：48），你会在第3步用到！

---

### 第3步：全数据最终训练 (几小时)

**修改配置：**

```bash
vim src/config.py
```

改两个地方：

```python
# 1. 改成最终训练模式
EXPERIMENT_MODE = "final_training"

# 2. 改epoch为第2步得到的平均值
TRAIN_CONFIG = {
    ...
    "epochs": 48,  # ← 改成你计算的平均epoch（例如48）
    ...
}
```

**运行最终训练：**

```bash
cd src
python main.py
```

**完成后查看结果：**

```bash
# 查看测试集性能
cat ../experiments/*/final_evaluation_results.json

# 结果文件位置：
# - 模型: experiments/<run_name>/best_model.pth
# - 结果: experiments/<run_name>/final_evaluation_results.json
# - 预测: experiments/<run_name>/predictions_price.csv
# - 预测: experiments/<run_name>/predictions_new.csv
```

---

## 🎓 理解你的代码结构

### 主要入口文件

**1. `src/main_kfold.py` - K折交叉验证**
- 用途：五折交叉验证，找最佳epoch
- 特点：每折独立训练，带早停
- 输出：5折的结果 + 聚合统计

**2. `src/main.py` - 单次训练**
- 用途：最终训练或超参数搜索
- 模式1 (eval_hyperparam_search)：有验证集，带早停
- 模式2 (final_training)：全数据，无早停

**3. `src/trainer.py` - 训练核心**
- 包含：训练循环、早停逻辑、评估流程
- 自动处理：模型保存、损失曲线、结果保存

### 核心模块

```
src/
├── config.py           # 所有配置参数
├── model.py            # EC分类器（简单的MLP）
├── datasets.py         # 对比学习数据集
├── loss.py             # SupCon损失函数
├── samplers.py         # PK采样器
├── evaluation_v2.py    # 评估器（KNN + 投票）
└── data_utils.py       # 数据加载工具
```

---

## 📖 需要更多帮助？

### 快速参考

- **想了解详细流程？** → 阅读 `README_QUICK_START.md`
- **需要完整文档？** → 阅读 `WORKFLOW.md`
- **修改配置困难？** → 参考 `config_template.py`
- **想用自动化脚本？** → 运行 `./run_complete_workflow.sh`

### 常见问题

**Q: 路径找不到怎么办？**
```bash
# 检查你的数据文件在哪里
find /home -name "split100.csv" 2>/dev/null
find /home -name "protein_embeddings" -type d 2>/dev/null

# 然后修改 src/config.py 中的路径
```

**Q: GPU显存不够？**
```python
# 减小batch size，修改 src/config.py:
TRAIN_CONFIG = {
    "P": 8,   # 从16改成8
    "K": 2,   # 从4改成2
    ...
}
```

**Q: 想快速测试流程？**
```python
# 修改 src/config.py:
N_SPLITS = 2  # 只跑2折
TRAIN_CONFIG = {
    "epochs": 10,  # 只训练10轮
    "patience": 3,
    ...
}
```

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] 已安装所有依赖（PyTorch, numpy, sklearn, tqdm等）
- [ ] 已修改 `src/config.py` 中的数据路径
- [ ] 数据文件存在（csv文件和embedding目录）
- [ ] GPU可用（或者已改成CPU模式）
- [ ] 有足够的磁盘空间（实验结果会保存在experiments/目录）

---

## 🎯 总结：你的训练流程

```
┌─────────────────────────────┐
│ 1. 修改 src/config.py        │
│    - 数据路径                │
│    - GPU设备                 │
│    - 模式: eval_hyperparam   │
└─────────────────────────────┘
            ↓
┌─────────────────────────────┐
│ 2. 运行五折交叉验证           │
│    python src/main_kfold.py │
│    → 得到平均epoch: 48       │
└─────────────────────────────┘
            ↓
┌─────────────────────────────┐
│ 3. 修改 src/config.py        │
│    - 模式: final_training    │
│    - epochs: 48             │
└─────────────────────────────┘
            ↓
┌─────────────────────────────┐
│ 4. 运行最终训练              │
│    python src/main.py       │
│    → 得到最终模型和结果      │
└─────────────────────────────┘
```

---

## 🚦 现在就开始！

```bash
# 1. 修改配置
vim src/config.py

# 2. 开始训练
cd src
python main_kfold.py

# 等待完成，然后继续第3步...
```

**祝训练顺利！** 🎉

如果遇到问题，查看详细文档：`README_QUICK_START.md` 和 `WORKFLOW.md`
