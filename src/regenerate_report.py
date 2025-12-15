import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import os

# 你的实验目录
exp_dir = "experiments/base_1114"

# 加载所有fold的结果
all_fold_results = []
for fold_num in [1, 2,3,4,5]:  # 你有2个fold
    fold_path = f"{exp_dir}/fold_{fold_num}/final_evaluation_results.json"
    if os.path.exists(fold_path):
        with open(fold_path, 'r') as f:
            data = json.load(f)
            all_fold_results.append(data)
        print(f"✅ Loaded fold_{fold_num}")
    else:
        print(f"❌ Missing fold_{fold_num}")

print(f"\nTotal folds loaded: {len(all_fold_results)}")

# 聚合结果
ood_sets = ["price", "new"]
final_report = {}

for ood_set in ood_sets:
    final_report[ood_set] = {}
    all_metrics = defaultdict(list)
    
    for fold_result in all_fold_results:
        results_dict = fold_result.get("results", fold_result)
        
        # 下游任务
        if ood_set in results_dict:
            for key, value in results_dict[ood_set].items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        # 内在质量
        intrinsic_key = f"{ood_set}_intrinsic"
        if intrinsic_key in results_dict:
            for key, value in results_dict[intrinsic_key].items():
                if isinstance(value, (int, float)):
                    all_metrics[f"intrinsic_{key}"].append(value)
    
    # 计算mean和std
    for metric_name, values in all_metrics.items():
        if values:
            mean = np.mean(values)
            std = np.std(values)
            final_report[ood_set][metric_name] = {"mean": mean, "std": std}
            print(f"  {ood_set}.{metric_name}: {mean:.4f} ± {std:.4f}")

# 保存JSON
summary_path = f"{exp_dir}/final_cross_validation_summary.json"
with open(summary_path, 'w') as f:
    json.dump(final_report, f, indent=4)
print(f"\n✅ Saved: {summary_path}")

# 生成Markdown
md_path = f"{exp_dir}/FINAL_REPORT.md"
with open(md_path, 'w') as f:
    f.write("# Comprehensive Embedding Quality Evaluation Report\n\n")
    f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Based on:** 2-fold cross-validation\n\n")
    f.write("---\n\n")
    
    for ood_set, metrics in final_report.items():
        f.write(f"## {ood_set.upper()} Test Set\n\n")
        
        if not metrics:
            f.write("*No data available.*\n\n")
            continue
        
        # 下游任务
        downstream_keys = [k for k in metrics.keys() if not k.startswith('intrinsic_')]
        if downstream_keys:
            f.write("### Downstream Task Performance\n\n")
            f.write("| Metric | Mean | Std |\n")
            f.write("|--------|------|-----|\n")
            for key in sorted(downstream_keys):
                mean = metrics[key]['mean']
                std = metrics[key]['std']
                name = key.replace('_at_fixed_tau_0.35', '').replace('_', ' ').upper()
                f.write(f"| {name} | {mean:.4f} | {std:.4f} |\n")
            f.write("\n")
        
        # 内在质量
        intrinsic_keys = [k for k in metrics.keys() if k.startswith('intrinsic_')]
        if intrinsic_keys:
            f.write("### Intrinsic Embedding Quality\n\n")
            f.write("| Metric | Mean | Std |\n")
            f.write("|--------|------|-----|\n")
            for key in sorted(intrinsic_keys):
                mean = metrics[key]['mean']
                std = metrics[key]['std']
                name = key.replace('intrinsic_', '').replace('_', ' ').title()
                f.write(f"| {name} | {mean:.4f} | {std:.4f} |\n")
            f.write("\n")
        
        f.write("---\n\n")

print(f"✅ Saved: {md_path}")