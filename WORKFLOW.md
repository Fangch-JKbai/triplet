# 蛋白质EC分类训练流程

## 📋 完整训练流程

### 第一步：五折交叉验证（确定最佳epoch和超参数）

```bash
cd /home/user/triplet/src
python main_kfold.py
```

**这个脚本会做什么：**
1. 将训练数据分成5折
2. 每折独立训练，使用早停机制（patience=15）
3. 每折在最佳checkpoint上评估price和new测试集
4. 最后聚合5折的平均结果

**输出位置：**
- `experiments/<run_name>/fold_1/` - 第1折的结果
- `experiments/<run_name>/fold_2/` - 第2折的结果
- ...
- `experiments/<run_name>/final_cross_validation_summary.json` - 聚合结果

**你需要做的：**
- 运行完成后，查看各折训练到第几个epoch停止（查看日志）
- 计算平均early stopping epoch，这就是你第二步要用的"合理epoch"

---

### 第二步：全数据训练（使用第一步确定的epoch数）

**修改配置文件 `src/config.py`：**

```python
# 1. 修改实验模式为 final_training
EXPERIMENT_MODE = "final_training"

# 2. 修改epoch数为第一步得到的平均epoch（比如假设是50）
TRAIN_CONFIG = {
    "P": 16,
    "K": 4,
    "temperature": 0.1,
    "epochs": 50,  # <--- 改成五折交叉验证的平均epoch
    "learning_rate": 1e-3,
    "grad_clip_norm": 0,
    "patience": 15,  # final_training模式下不使用，但保留
    "scheduler_eta_min": 1e-6,
}

# 3. 确认评估配置（第一步可能已经帮你找到最佳k和tau）
EVAL_CONFIG = {
    "weighted": True,
    "default_k": 3,      # 如果第一步给出更好的k，使用那个值
    "default_tau": 0.35, # 如果第一步给出更好的tau，使用那个值
}
```

**运行训练：**

```bash
cd /home/user/triplet/src
python main.py
```

**这个脚本会做什么：**
1. 使用全部训练数据（无验证集划分）
2. 训练固定的epoch数（无早停）
3. 使用预设的k和tau在price和new测试集上评估

**输出位置：**
- `experiments/<run_name>/best_model.pth` - 最终模型
- `experiments/<run_name>/final_evaluation_results.json` - price和new的测试结果
- `experiments/<run_name>/predictions_price.csv` - price的预测
- `experiments/<run_name>/predictions_new.csv` - new的预测

---

## ⚙️ 当前配置检查

在运行之前，确认 `src/config.py` 中的关键配置：

```python
# 设备（根据你的GPU情况调整）
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 数据路径（确保这些路径存在）
DATA_PATHS = {
    "csv_path": "/home/fangchh/workdir/triplet/data/split100.csv",
    "embedding_dir": "/home/fangchh/workdir/triplet/data/ESM_CPT/protein_embeddings",
    "external_test_sets": {
        "price": "/home/fangchh/workdir/triplet/data/price.csv",
        "new": "/home/fangchh/workdir/triplet/data/new.csv"
    }
}

# K折数量
N_SPLITS = 5

# 随机种子（保证可复现）
SEED = 42
```

---

## 📊 如何确定"合理的epoch"

五折交叉验证完成后，查看日志：

```bash
# 查看每一折的早停信息
grep "Early stopping triggered" experiments/<run_name>/fold_*/fold.log

# 或者查看总结日志
cat experiments/<run_name>/main_experiment.log | grep "Total epochs trained"
```

例如，如果5折分别在第45, 48, 52, 46, 49个epoch停止：
- 平均值：(45+48+52+46+49)/5 = 48
- 使用epoch=48或50作为第二步的训练轮数

---

## 🔍 结果分析

### 交叉验证结果
```bash
cat experiments/<run_name>/final_cross_validation_summary.json
```

### 最终测试结果
```bash
cat experiments/<run_name>/final_evaluation_results.json
```

---

## 💡 常见问题

**Q: 如果数据路径不存在怎么办？**
A: 修改 `src/config.py` 中的路径，指向实际的数据位置

**Q: 五折交叉验证太慢怎么办？**
A: 可以先跑单折测试，修改 `config.py` 中的 `N_SPLITS = 1`

**Q: 如何只在验证集上调优超参数而不做交叉验证？**
A: 修改 `config.py` 设置 `EXPERIMENT_MODE = "eval_hyperparam_search"`，然后运行 `main.py`

**Q: 训练过程中如何监控？**
A: 查看实时日志：`tail -f experiments/<run_name>/experiment.log`
