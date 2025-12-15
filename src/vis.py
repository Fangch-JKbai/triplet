import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap # 导入 UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score # <-- 新增：导入轮廓系数
from matplotlib.lines import Line2D
import os
import glob
import re
import argparse
import math
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from sklearn.metrics.pairwise import rbf_kernel

def parse_ec_prefix(ec_str: str, level: int, default='N/A') -> str:
    if not isinstance(ec_str, str) or not ec_str: return default
    first_ec = ec_str.split(';')[0].strip()
    if not first_ec: return default
    parts = first_ec.split('.')
    valid_parts = []
    for part in parts:
        if part.isdigit(): valid_parts.append(part)
        elif part == '-' or 'n' in part.lower(): valid_parts.append('x')
        else: break
    if not valid_parts: return default
    if level <= 0 or level > 4: level = 4
    requested_parts = valid_parts[:level]
    if len(requested_parts) < level:
         padding = ['x'] * (level - len(requested_parts))
         requested_parts.extend(padding)
    return ".".join(requested_parts)

def load_ec_map(csv_path: str, id_col: str, ec_col: str, ec_level: int) -> Dict[str, str]:
    print(f"Loading labels from {csv_path} (Level: {ec_level})...")
    try:
        df = pd.read_csv(csv_path, sep="\t") 
    except Exception as e:
        print(f"Error reading CSV: {e}"); return {}
    if id_col not in df.columns or ec_col not in df.columns:
        print(f"Error: CSV must contain '{id_col}' and '{ec_col}' columns."); return {}
    ec_map = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col])
        ec_str = str(row[ec_col])
        ec_map[seq_id] = parse_ec_prefix(ec_str, level=ec_level)
    print(f"Loaded {len(ec_map)} ID-to-EC mappings.")
    return ec_map

def load_embeddings(embedding_dir: str, ec_map: Dict[str, str]) -> pd.DataFrame:
    print(f"Loading embeddings from: {embedding_dir}")
    pt_files = glob.glob(os.path.join(embedding_dir, "*.pt"))
    if not pt_files:
        print(f"Warning: No .pt files found in {embedding_dir}"); return pd.DataFrame()
    embeddings_list = []; labels_list = []; ids_list = []
    for pt_file in tqdm(pt_files, desc=f"Loading {os.path.basename(embedding_dir)}"):
        seq_id = os.path.basename(pt_file).replace('.pt', '')
        if seq_id in ec_map:
            label = ec_map[seq_id]
            try:
                data = torch.load(pt_file, map_location='cpu')
                embedding = data['mean_representations'].numpy()
                embeddings_list.append(embedding)
                labels_list.append(label)
                ids_list.append(seq_id)
            except Exception as e:
                print(f"Warning: Could not load {pt_file}: {e}")
    if not embeddings_list:
        print(f"Warning: No matching embeddings found for labels in {embedding_dir}"); return pd.DataFrame()
    df = pd.DataFrame({'seq_id': ids_list, 'ec_prefix': labels_list})
    embedding_df = pd.DataFrame(np.vstack(embeddings_list), index=df.index)
    df = pd.concat([df, embedding_df], axis=1)
    return df

def run_dbscan(df: pd.DataFrame, eps: float, min_samples: int) -> pd.DataFrame:
    if eps <= 0:
        print("DBSCAN filtering skipped (eps=0).")
        return df
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    embedding_cols = df.columns.drop(['seq_id', 'ec_prefix'])
    embeddings = df[embedding_cols].values
    embeddings = StandardScaler().fit_transform(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=1).fit(embeddings)
    labels = db.labels_
    non_noise_indices = np.where(labels != -1)[0]
    original_count = len(df); filtered_count = len(non_noise_indices)
    print(f"DBSCAN: Kept {filtered_count} of {original_count} points ({(100*filtered_count/original_count):.1f}%)")
    return df.iloc[non_noise_indices].reset_index(drop=True)


def main(args):
    # 1. 加载 EC 标签
    ec_map = load_ec_map(args.labels_csv, args.id_col, args.ec_col, ec_level=args.ec_level)
    if not ec_map: return

    # 2. 加载所有模型的嵌入
    all_model_data = []
    for model_name, embed_dir in zip(args.model_names, args.embedding_dirs):
        df = load_embeddings(embed_dir, ec_map)
        if df.empty: continue
        df_filtered = run_dbscan(df, args.dbscan_eps, args.dbscan_min_samples)
        if df_filtered.empty:
            print(f"Warning: DBSCAN filtered all points for {model_name}"); continue
        df_filtered['model_name'] = model_name
        all_model_data.append(df_filtered)
    if not all_model_data:
        print("Error: No valid embedding data loaded. Exiting."); return

    # --- 新增: 第 2.5 步 - 计算轮廓系数 (量化聚类质量) ---
    print("\n--- Silhouette Score (High-Dimensional Cluster Quality) ---")
    for model_df in all_model_data:
        model_name = model_df['model_name'].iloc[0]
        
        # 提取高维嵌入和标签
        embedding_cols = model_df.columns.drop(['seq_id', 'ec_prefix', 'model_name'])
        embeddings = model_df[embedding_cols].values
        labels = model_df['ec_prefix'].values
        
        # 必须先标准化，因为 Silhouette Score 是基于距离的
        embeddings_scaled = StandardScaler().fit_transform(embeddings)
        
        # 检查标签是否多于1个（计算分数所需）
        if len(set(labels)) < 2:
            print(f"  -> Silhouette Score ({model_name}): N/A (Only 1 cluster found)")
            continue

        try:
            score = silhouette_score(embeddings_scaled, labels, metric='euclidean')
            print(f"  -> Silhouette Score ({model_name}): {score:.4f}")
        except Exception as e:
            print(f"  -> Silhouette Score ({model_name}): Error ({e})")
    # --- 结束新增步骤 ---


    # 3. 组合数据并运行降维
    print("\nCombining all embeddings for a unified reduction run...")
    combined_df = pd.concat(all_model_data, ignore_index=True)
    embedding_cols = combined_df.columns.drop(['seq_id', 'ec_prefix', 'model_name'])
    combined_embeddings = combined_df[embedding_cols].values
    print("Standardizing embeddings before reduction...")
    combined_embeddings = StandardScaler().fit_transform(combined_embeddings)

    if args.vis_method.lower() == 'tsne':
        print(f"Running t-SNE on {len(combined_embeddings)} total points (Perplexity: {args.perplexity})...")
        reducer = TSNE(n_components=2, random_state=args.seed, perplexity=args.perplexity, 
                       n_iter=1000, init='pca', learning_rate='auto', n_jobs=1)
        plot_title_prefix = "t-SNE"; plot_title_param = f"Perplexity: {args.perplexity}"
    
    elif args.vis_method.lower() == 'umap':
        print(f"Running UMAP on {len(combined_embeddings)} total points (n_neighbors: {args.umap_neighbors})...")
        reducer = umap.UMAP(n_neighbors=args.umap_neighbors, n_components=2,
                           min_dist=0.1, metric='euclidean', random_state=args.seed)
        plot_title_prefix = "UMAP"; plot_title_param = f"Neighbors: {args.umap_neighbors}"
    else:
        raise ValueError(f"Unknown visualization method: {args.vis_method}. Use 'tsne' or 'umap'.")

    reduced_embeddings = reducer.fit_transform(combined_embeddings)
    combined_df['dim_1'] = reduced_embeddings[:, 0]
    combined_df['dim_2'] = reduced_embeddings[:, 1]
    
    # 4. 保存坐标数据
    if args.output_data_csv:
        print(f"Saving reduction coordinate data to {args.output_data_csv}...")
        try: combined_df.to_csv(args.output_data_csv, index=False)
        except Exception as e: print(f"Error saving data CSV: {e}")

    # 5. 可视化 (Facet Grid)
    print("Generating visualization...")
    unique_labels = sorted(combined_df['ec_prefix'].unique())
    n_colors = len(unique_labels)
    if n_colors <= 10: palette = sns.color_palette("tab10", n_colors)
    elif n_colors <= 20: palette = sns.color_palette("tab20", n_colors)
    else: palette = sns.color_palette("viridis", n_colors)
    color_map = dict(zip(unique_labels, palette))
    if 'N/A' in color_map: color_map['N/A'] = (0.7, 0.7, 0.7)

    num_models = len(args.model_names); cols = args.grid_cols
    rows = math.ceil(num_models / cols)
    fig_height = rows * 5; fig_width = cols * 5 + 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    
    for i, model_name in enumerate(args.model_names):
        ax = axes[i]
        model_data = combined_df[combined_df['model_name'] == model_name]
        if model_data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name); continue
        sns.scatterplot(
            data=model_data, x='dim_1', y='dim_2', hue='ec_prefix',
            palette=color_map, s=15, alpha=0.8, legend=False, ax=ax
        )
        ax.set_title(model_name)
        ax.set_xlabel(f"{plot_title_prefix} 1"); ax.set_ylabel(f"{plot_title_prefix} 2")

    for j in range(num_models, len(axes)): fig.delaxes(axes[j])

    legend_elements = []; max_legend_items = 25
    for i, (label, color) in enumerate(color_map.items()):
        if i < max_legend_items:
            label_text = f"EC {label}" if label != 'N/A' else 'N/A'
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         label=label_text, markerfacecolor=color, markersize=8))
        elif i == max_legend_items:
            legend_elements.append(Line2D([0], [0], marker='', color='w', 
                                         label=f"... ({n_colors - max_legend_items} more)",
                                         markerfacecolor='black', markersize=0)); break
            
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5, (fig_width - (cols*5))/fig_width, 0.0), borderaxespad=0.)
    plot_title = f"{plot_title_prefix} (EC Level: {args.ec_level}, {plot_title_param}, DBSCAN Eps: {args.dbscan_eps})"
    fig.suptitle(plot_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {args.output_image}")

    # 6. MMD 比较 (保持不变, 在完整数据上计算)
    if len(all_model_data) > 1:
        print("\n--- MMD Comparison (on Full Distribution) ---")
        base_df_full = load_embeddings(args.embedding_dirs[0], ec_map)
        if base_df_full.empty:
            print("Error: Cannot load base embeddings for MMD."); return
        embedding_cols = base_df_full.columns.drop(['seq_id', 'ec_prefix'])
        base_embeddings_full = base_df_full[embedding_cols].values
        base_name = args.model_names[0]
        
        for i in range(1, len(args.model_names)):
            comp_name = args.model_names[i]
            print(f"Calculating MMD between {base_name} and {comp_name}...")
            comp_df_full = load_embeddings(args.embedding_dirs[i], ec_map)
            if comp_df_full.empty:
                print(f"  -> Could not load {comp_name} embeddings for MMD."); continue
            comp_embeddings_full = comp_df_full[embedding_cols].values
            mmd_val = calculate_mmd(base_embeddings_full, comp_embeddings_full)
            print(f"  -> MMD^2 ({base_name} vs {comp_name}): {mmd_val:.6f}")

def calculate_mmd(X, Y, gamma=None):
    # (MMD 函数代码保持不变)
    X = np.asarray(X); Y = np.asarray(Y)
    n_samples_x = X.shape[0]; n_samples_y = Y.shape[0]
    if n_samples_x == 0 or n_samples_y == 0: return -1.0
    if X.shape[1] != Y.shape[1]: raise ValueError("Input arrays X and Y must have the same number of features.")
    if gamma is None:
        combined_data = np.vstack((X, Y))
        distances_sq = pairwise_distances(combined_data, metric='sqeuclidean')
        non_zero_distances_sq = distances_sq[distances_sq > 1e-12]
        if len(non_zero_distances_sq) > 0:
            median_dist_sq = np.median(non_zero_distances_sq)
            if median_dist_sq <= 1e-12: gamma = 1.0
            else: gamma = 1.0 / (2 * median_dist_sq) 
        else: gamma = 1.0
    K_XX = rbf_kernel(X, X, gamma=gamma); K_YY = rbf_kernel(Y, Y, gamma=gamma); K_XY = rbf_kernel(X, Y, gamma=gamma)
    m = float(n_samples_x); n = float(n_samples_y)
    if m < 2 or n < 2:
        mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    else:
        term_xx = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
        term_yy = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
        term_xy = np.mean(K_XY)
        mmd2 = term_xx + term_yy - 2 * term_xy
    return max(0, mmd2)


if __name__ == "__main__":
    
    # --- Hardcoded Configuration ---
    class HardcodedArgs: pass
    args = HardcodedArgs()

    # *** 在此处修改您的所有参数 ***
    
    # --- 路径与标签 ---
    args.embedding_dirs = ["/home/fangchh/workdir/triplet/data/base/protein_embeddings", 
                           "/home/fangchh/workdir/triplet/data/ESM_CPT_1117/protein_embeddings",]
    args.model_names = ["ESM-2", "ESM_CPT"]
    args.labels_csv = "/home/fangchh/workdir/triplet/data/split10.csv"
    args.id_col = "Entry"      # 确保这与您 price.csv 中的列名匹配
    args.ec_col = "EC number"  # 确保这与您 price.csv 中的列名匹配
    
    # --- 可视化方法与参数 ---
    args.vis_method = "tsne"  # 可选项: "tsne" 或 "umap"
    args.ec_level = 1         # EC 编号级别 (1, 2, 3, or 4)
    
    args.perplexity = 30      # t-SNE 的 Perplexity (如果 vis_method == "tsne")
    args.umap_neighbors = 20  # UMAP 的 Neighbors (如果 vis_method == "umap")
    
    # --- 过滤与输出 ---
    args.dbscan_eps = 0.0     # 0.0 = 禁用 DBSCAN 过滤
    args.dbscan_min_samples = 5
    
    # 文件名将根据设置动态生成
    args.output_image = f"{args.vis_method}_ec-L{args.ec_level}_models.png"
    args.output_data_csv = f"{args.vis_method}_ec-L{args.ec_level}_coordinates.csv"

    args.seed = 42
    args.grid_cols = 2
    # *** 配置结束 ***

    # (安装 UMAP 和 Silhouette 的提示)
    if args.vis_method.lower() == 'umap':
        try:
            import umap
        except ImportError:
            print("\n" + "="*50)
            print("错误: UMAP 方法需要 'umap-learn' 包。")
            print("请在您的环境中运行: pip install umap-learn")
            print("="*50 + "\n"); exit(1)
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("\n" + "="*50)
        print("错误: 缺少 'scikit-learn' 包。")
        print("请在您的环境中运行: pip install scikit-learn")
        print("="*50 + "\n"); exit(1)


    if len(args.embedding_dirs) != len(args.model_names):
        raise ValueError("The number of embedding_dirs must match the number of model_names.")

    print("--- Running with Hardcoded Configuration ---")
    print(f"Method: {args.vis_method.upper()}")
    print(f"EC Level: {args.ec_level}")
    print(f"DBSCAN Eps: {args.dbscan_eps} (0.0 = Disabled)")
    print(f"Models: {args.model_names}")
    print(f"Labels CSV: {args.labels_csv}")
    print(f"Output Image: {args.output_image}")
    print("------------------------------------------")

    main(args)