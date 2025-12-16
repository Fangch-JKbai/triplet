# 🔍 训练监控工具使用指南

## 📁 新增文件

1. **`src/trainer_with_monitoring.py`** - 增强版训练器
   - 每个epoch都在测试集上评估
   - 实时绘制性能曲线
   - 保存训练历史

2. **`src/main_with_monitoring.py`** - 使用监控功能的训练脚本
   - 自动保留20%验证集
   - 每5个epoch评估测试集（可调整）

---

## 🚀 快速开始

### 运行训练（带监控）

```bash
cd src
python main_with_monitoring.py
```

---

## 📊 输出文件

训练完成后，会在 `experiments/<run_name>/` 目录生成：

### 1. 性能监控图 `test_performance_curve.png`

包含4个子图：
- **左上**: Price和New数据集的AUC随epoch变化
- **右上**: Price和New数据集的F1随epoch变化
- **左下**: 训练损失和验证损失曲线
- **右下**: 平均AUC曲线，标注最佳epoch

### 2. 训练历史 `training_history.json`

记录每个epoch的所有指标：
```json
{
  "train_loss": [2.1, 1.5, 1.3, ...],
  "val_loss": [2.0, 1.4, 1.25, ...],
  "epoch": [1, 5, 10, 15, ...],
  "price_auc": [0.65, 0.68, 0.71, ...],
  "price_f1": [0.35, 0.38, 0.42, ...],
  "new_auc": [0.70, 0.71, 0.72, ...],
  "new_f1": [0.55, 0.57, 0.58, ...]
}
```

### 3. 其他文件

- `best_model.pth` - 最佳模型（基于验证损失）
- `final_evaluation_results.json` - 最终测试结果
- `predictions_price.csv` - Price数据集预测
- `predictions_new.csv` - New数据集预测
- `experiment.log` - 完整训练日志

---

## ⚙️ 配置参数

### 调整评估频率

在 `main_with_monitoring.py` 中修改：

```python
EVAL_INTERVAL = 5  # 每5个epoch评估一次

# 改成1表示每个epoch都评估（更详细但更慢）
EVAL_INTERVAL = 1

# 改成10表示每10个epoch评估（更快但不够详细）
EVAL_INTERVAL = 10
```

### 调整验证集比例

在 `main_with_monitoring.py` 中修改：

```python
val_ratio = 0.2  # 20%验证集

# 改成0.1使用10%验证集
val_ratio = 0.1
```

---

## 📈 如何解读曲线

### 正常的训练曲线

```
AUC
│
0.75│         ┌──────┐  最佳epoch
    │        /        \
0.70│       /          \  开始过拟合
    │      /            \
0.65│     /              └─
    │    /
    └────────────────────────► Epoch
    0    50   100  150  200
```

**特征：**
- 前期快速上升
- 中期达到峰值（最佳epoch）
- 后期开始下降（过拟合）

### 异常情况

#### 情况1：一直上升

```
AUC
│                    ┌─
    │               /
0.70│              /
    │             /
0.65│            /
    │           /
    └────────────────────────► Epoch
    0    50   100  150  200
```

**原因：** 训练不足，需要更多epoch

**解决：** 增加 `epochs` 或降低 `patience`

---

#### 情况2：剧烈波动

```
AUC
│
0.75│  ┌─┐  ┌─┐
    │ /   \/   \  /\
0.70│/          \/  \
    │                \
    └────────────────────────► Epoch
```

**原因：** 学习率太高或batch size太小

**解决：** 降低学习率或增加batch size

---

#### 情况3：平坦无变化

```
AUC
│
0.70│────────────────────────
    │
    └────────────────────────► Epoch
```

**原因：** 学习率太小或模型已收敛

**解决：** 增加学习率或检查是否已达最优

---

## 🔍 找最佳Epoch

### 方法1：查看曲线图

打开 `test_performance_curve.png` 右下角子图，红色虚线标注了最佳epoch。

### 方法2：查看日志

训练结束后，日志中会显示：

```
==============================================================
 BEST EPOCH ANALYSIS
==============================================================
Best Epoch (by average test AUC): 185
  Average AUC: 0.7234
  Price AUC: 0.7150
  New AUC: 0.7318
```

### 方法3：分析JSON

```python
import json
import numpy as np

with open('experiments/<run_name>/training_history.json') as f:
    history = json.load(f)

# 计算平均AUC
avg_auc = [(p + n) / 2 for p, n in zip(history['price_auc'], history['new_auc'])]

# 找最大值
best_idx = np.argmax(avg_auc)
best_epoch = history['epoch'][best_idx]

print(f"Best epoch: {best_epoch}")
print(f"Best AUC: {avg_auc[best_idx]:.4f}")
```

---

## 💡 使用建议

### 对比不同配置

运行多次实验，对比不同配置：

```bash
# 实验1：学习率1e-3
# 修改config.py: learning_rate = 1e-3
python main_with_monitoring.py

# 实验2：学习率1e-4
# 修改config.py: learning_rate = 1e-4
python main_with_monitoring.py

# 实验3：更大的dropout
# 修改config.py: dropout_rate = 0.3
python main_with_monitoring.py
```

然后对比各个实验的 `test_performance_curve.png`。

---

### 对比Base vs CPT

```bash
# 训练Base
# 修改config.py: embedding_dir = base路径
python main_with_monitoring.py
# 保存曲线图为: base_curve.png

# 训练CPT
# 修改config.py: embedding_dir = cpt路径
python main_with_monitoring.py
# 保存曲线图为: cpt_curve.png

# 对比两张图，看哪个模型：
# 1. 收敛更快
# 2. 最终性能更好
# 3. 更稳定（波动更小）
```

---

## 🐛 常见问题

### Q: 评估太频繁，训练很慢

**A:** 增加 `EVAL_INTERVAL`：

```python
EVAL_INTERVAL = 10  # 从5改成10
```

---

### Q: 曲线图没有生成

**A:** 检查：
1. matplotlib是否安装：`pip install matplotlib`
2. 权限是否足够
3. 查看日志中的错误信息

---

### Q: 想在训练过程中实时查看曲线

**A:** 可以在另一个终端监控：

```bash
# 监控文件变化
watch -n 5 ls -lh experiments/*/test_performance_curve.png

# 或使用图片查看器自动刷新功能
```

---

## 📝 示例输出

### 训练日志示例

```
==============================================================
 TRAINING WITH TEST SET MONITORING
==============================================================
Total Epochs: 500
Evaluation Interval: Every 5 epoch(s)
Test sets: ['price', 'new']
Using k=3, tau=0.35 for evaluation
Early Stopping Patience: 20
==============================================================

Epoch 1/500 | Train Loss: 2.1234 | Val Loss: 2.0123 | LR: 1.0e-4
  ✓ New best validation loss: 2.0123. Saving model...

Epoch 5/500 | Train Loss: 1.4567 | Val Loss: 1.3456 | LR: 9.8e-5

────────────────────────────────────────────────────────────
 Evaluating on test sets at epoch 5
────────────────────────────────────────────────────────────
  PRICE: AUC=0.6523, F1=0.3456
  NEW: AUC=0.6834, F1=0.5123
────────────────────────────────────────────────────────────

...

Early stopping triggered after 20 epochs with no improvement
Best validation loss: 1.2345
```

---

## 🎯 总结

使用这个监控工具，你可以：

✅ **实时看到**训练过程中测试集性能的变化
✅ **直观发现**过拟合发生的时刻
✅ **准确找到**最佳训练epoch
✅ **对比分析**不同配置的效果
✅ **避免盲目**训练过多或过少的epoch

**现在就试试吧！** 🚀
