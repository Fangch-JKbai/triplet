import matplotlib.pyplot as plt

# --- 您的数据 ---
scores = [0.0776, 0.0788, 0.0779, 0.0792]
models = ['ESM-2', 'ESM_CPT', 'ESM_SFT', 'ESM_ENZ']
colors = ['gray', '#1f77b4', '#ff7f0e', '#2ca02c']

# --- 绘图代码 ---

# 1. 创建图形和坐标轴 (推荐使用 fig, ax 方式)
fig, ax = plt.subplots(figsize=(8, 6))

# 2. 绘制条形图
bars = ax.bar(models, scores, color=colors)

# 3. 设置 Y 轴标签和标题
ax.set_ylabel('Silhouette Score')
ax.set_title('High-Dimensional Cluster Quality')

# 4. 设置 Y 轴范围 (按照您的要求放大差异)
# 这个范围非常窄，可以清晰地看到微小差异
ax.set_ylim([0.075, 0.081])

# 5. [优化] 添加数据标签
# 由于 Y 轴范围很窄，在图顶端直接显示数值非常重要
# fmt='%.4f' 确保标签显示 4 位小数，与您的数据精度一致
ax.bar_label(bars, fmt='%.4f', padding=3)

# 6. [优化] 自动调整布局
# 防止标签（例如 Y 轴标签）被截断
plt.tight_layout()

# 7. 保存图表到本地
save_filename = "silhouette_score_plot.png"
plt.savefig(save_filename)

print(f"图表已保存为: {save_filename}")

# 8. (可选) 如果您想在脚本运行时立即查看图表
# plt.show()