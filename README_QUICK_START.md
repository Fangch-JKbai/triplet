# 🚀 快速开始指南

## 📁 项目结构已整理

你的代码已经包含完整的训练流程！主要文件：

```
triplet/
├── src/
│   ├── main_kfold.py      # ← 五折交叉验证（第一步）
│   ├── main.py            # ← 最终训练（第二步）
│   ├── config.py          # ← 所有配置参数
│   ├── trainer.py         # ← 训练核心逻辑
│   ├── model.py           # ← EC分类器模型
│   ├── datasets.py        # ← 数据集类
│   ├── evaluation_v2.py   # ← 评估器
│   └── ...
├── WORKFLOW.md            # ← 详细工作流程文档
├── check_config.py        # ← 配置检查脚本
└── run_complete_workflow.sh  # ← 自动化脚本
```

---

## ⚡ 三种运行方式

### 方式1：自动化脚本（推荐新手）

```bash
./run_complete_workflow.sh
```

这个脚本会：
1. 运行五折交叉验证
2. 分析结果并让你输入合理的epoch
3. 自动修改配置文件
4. 运行最终训练
5. 显示所有结果

---

### 方式2：手动分步执行（推荐有经验用户）

#### 步骤1：检查配置

```bash
# 编辑配置文件（重要！）
vim src/config.py

# 需要检查的关键配置：
# - DATA_PATHS: 确保所有路径正确
# - DEVICE: GPU设备号
# - N_SPLITS: K折数（默认5）
# - EXPERIMENT_MODE: 当前模式
```

#### 步骤2：五折交叉验证

```bash
cd src
python main_kfold.py
```

等待训练完成后，查看结果：

```bash
# 查看每折训练的epoch数
grep "Total epochs trained:" ../experiments/*KFold*/fold_*/fold.log

# 查看聚合结果
cat ../experiments/*KFold*/final_cross_validation_summary.json
```

计算平均early stopping epoch（例如：48）

#### 步骤3：修改配置准备最终训练

编辑 `src/config.py`：

```python
# 改为最终训练模式
EXPERIMENT_MODE = "final_training"

# 修改epoch为第一步得到的平均值
TRAIN_CONFIG = {
    ...
    "epochs": 48,  # ← 改成你计算的平均epoch
    ...
}
```

#### 步骤4：全数据最终训练

```bash
cd src
python main.py
```

训练完成后查看结果：

```bash
# 查看测试结果
cat ../experiments/*/final_evaluation_results.json
```

---

### 方式3：直接运行（如果已经配置好）

```bash
# 五折交叉验证
cd src && python main_kfold.py

# 最终训练（修改config.py后）
cd src && python main.py
```

---

## 🔧 配置文件重点说明

### `src/config.py` 关键配置

#### 1. 实验模式（必改）
```python
EXPERIMENT_MODE = "final_training"  # 或 "eval_hyperparam_search"
```

- `"eval_hyperparam_search"`: 有train/val划分，带早停，会在验证集上调优k和tau
- `"final_training"`: 全数据训练，无早停，使用预设k和tau在测试集评估

#### 2. 数据路径（必改）
```python
DATA_PATHS = {
    "csv_path": "/path/to/your/split100.csv",  # ← 改成你的路径
    "embedding_dir": "/path/to/protein_embeddings",  # ← 改成你的路径
    "external_test_sets": {
        "price": "/path/to/price.csv",  # ← 改成你的路径
        "new": "/path/to/new.csv"  # ← 改成你的路径
    }
}
```

#### 3. 训练参数
```python
TRAIN_CONFIG = {
    "P": 16,              # batch中的类别数
    "K": 4,               # 每个类别的样本数
    "temperature": 0.1,   # 对比学习温度
    "epochs": 1000,       # 最大epoch（有早停时会提前停）
    "learning_rate": 1e-3,
    "patience": 15,       # 早停patience
    ...
}
```

#### 4. K折数量
```python
N_SPLITS = 5  # 五折交叉验证
```

#### 5. 设备
```python
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
```

---

## 📊 理解训练流程

### 为什么要分两步？

**第一步：五折交叉验证**
- 目的：找到最佳的early stopping epoch
- 方法：用5折交叉验证，每折独立训练并早停
- 输出：每折的最佳epoch，聚合的测试性能

**第二步：全数据训练**
- 目的：用全部数据训练最终模型
- 方法：使用第一步得到的epoch数，在全训练集上训练
- 输出：最终模型和测试集性能

### 训练流程图

```
训练数据
    ↓
┌───────────────────┐
│ 第一步：5折交叉验证 │
│  - 每折train/val   │
│  - 带早停机制      │
│  - 记录停止epoch   │
└───────────────────┘
    ↓
  计算平均epoch (例如：48)
    ↓
┌───────────────────┐
│ 第二步：全数据训练  │
│  - 使用全部数据    │
│  - 训练48个epoch  │
│  - 无早停         │
└───────────────────┘
    ↓
在price和new上测试
```

---

## 📈 查看结果

### 交叉验证结果

```bash
# 位置
experiments/<run_name>_KFold/
├── fold_1/
│   ├── best_model.pth
│   ├── loss_curve.png
│   ├── predictions_price.csv
│   └── final_evaluation_results.json
├── fold_2/
├── ...
└── final_cross_validation_summary.json  # ← 聚合结果

# 查看
cat experiments/*KFold/final_cross_validation_summary.json
```

### 最终训练结果

```bash
# 位置
experiments/<run_name>/
├── best_model.pth                    # ← 最终模型
├── final_evaluation_results.json    # ← 测试结果
├── predictions_price.csv             # ← Price预测
├── predictions_new.csv               # ← New预测
├── loss_curve.png                    # ← 损失曲线
└── experiment.log                    # ← 完整日志

# 查看
cat experiments/*/final_evaluation_results.json
```

---

## 🐛 常见问题

### Q1: 路径不存在怎么办？
**A:** 修改 `src/config.py` 中的路径为实际路径

### Q2: GPU内存不足？
**A:** 减小batch size：修改 `TRAIN_CONFIG` 中的 P 和 K（例如：P=8, K=2）

### Q3: 想快速测试？
**A:**
```python
# 修改 config.py
N_SPLITS = 2  # 只跑2折
TRAIN_CONFIG = {
    "epochs": 10,  # 只训练10个epoch
    "patience": 3,
    ...
}
```

### Q4: 如何监控训练进度？
**A:**
```bash
# 实时查看日志
tail -f experiments/<run_name>/experiment.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

### Q5: 训练中断后如何恢复？
**A:** 目前代码不支持断点恢复，需要重新训练。建议：
- 降低epoch数快速测试
- 或在代码中添加checkpoint恢复功能

---

## 💡 最佳实践

1. **第一次运行**：先用小数据集或少量epoch测试流程
2. **GPU选择**：修改 `config.py` 中的 `DEVICE`
3. **日志保存**：所有日志和结果自动保存在 `experiments/` 目录
4. **配置备份**：`run_complete_workflow.sh` 会自动备份配置文件
5. **结果分析**：重点关注 `final_evaluation_results.json` 中的F1和AUC

---

## 📝 示例：完整执行流程

```bash
# 1. 克隆/进入项目目录
cd /home/user/triplet

# 2. 检查和修改配置
vim src/config.py
# 修改数据路径、设备号等

# 3. 运行五折交叉验证
cd src
python main_kfold.py
# 等待完成...

# 4. 查看结果并计算平均epoch
grep "Total epochs trained:" ../experiments/*KFold*/fold_*/fold.log
# 假设得到平均值：48

# 5. 修改配置准备最终训练
vim config.py
# EXPERIMENT_MODE = "final_training"
# epochs = 48

# 6. 运行最终训练
python main.py
# 等待完成...

# 7. 查看最终结果
cat ../experiments/*/final_evaluation_results.json

# 完成！
```

---

## 🎯 下一步

训练完成后，你可以：

1. 分析预测结果：`predictions_price.csv` 和 `predictions_new.csv`
2. 可视化嵌入：使用 `src/vis.py` 或 `src/visualization.py`
3. 比较不同实验：使用 `src/compare_experiments.py`
4. 调整超参数：修改 `config.py` 并重新训练

---

**需要更多帮助？** 查看 `WORKFLOW.md` 获取详细文档！
