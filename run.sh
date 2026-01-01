#!/bin/bash
set -e  # Exit immediately if a command fails

echo "🚀 Starting Embedding Generation Pipeline (Debug Mode: Mutation First)..."

# ============== 配置区 ==============

# 模型列表
MODEL_ORDER=("ESM_SUB" "ESM_CPT_SUB")

# 输入文件
FASTA_FILES=("./data/split100.fasta" "./data/price.fasta" "./data/new.fasta")
CSV_FILE="./data/split100.csv"

# 预定义的突变文件路径 (根据 Python 脚本逻辑: 位于 CSV 同级目录，名为 masked_sequences.fasta)
# 必须与 Python 脚本中的生成路径保持一致
MUTATED_FASTA="./data/masked_sequences.fasta"

# 超参数
BATCH_SIZE=16
DEVICE="cuda:2"

# 数据增强开关
RUN_MUTATION="true"

# 脚本路径
STEP1_SCRIPT="scripts/1_generate_protein_embeddings.py"
STEP2_SCRIPT="scripts/2_calculate_ec_embeddings.py"
STEP3_SCRIPT="scripts/3_create_distance_map.py"

# 输出根目录
EXPERIMENT_ROOT="data"

# ============== 执行流程 ==============

mkdir -p "$EXPERIMENT_ROOT"

for MODEL_NAME in "${MODEL_ORDER[@]}"; do
    echo ""
    echo "================== Processing model: ${MODEL_NAME} =================="
    
    MODEL_RESULT_DIR="${EXPERIMENT_ROOT}/${MODEL_NAME}"
    PROTEIN_EMBED_DIR="${MODEL_RESULT_DIR}/protein_embeddings"
    EC_EMBED_DIR="${MODEL_RESULT_DIR}/ec_embeddings"
    REPORTS_DIR="${MODEL_RESULT_DIR}/reports"
    mkdir -p "$PROTEIN_EMBED_DIR" "$EC_EMBED_DIR" "$REPORTS_DIR"

    # --- Step 1: 生成蛋白质嵌入 (分两步走) ---
    
    # [Step 1.1] 优先处理突变序列 (Debug & Check)
    if [ "${RUN_MUTATION}" == "true" ]; then
        echo "[1/3-A] 🧬 [Priority] Generating & Embedding MUTATED sequences first..."
        
        # 技巧：我们将 --fasta-files 仅指向突变文件
        # 这样脚本会先生成文件，然后立刻只对该文件做 Embedding，速度极快，方便查错
        python "$STEP1_SCRIPT" \
            --model-name "$MODEL_NAME" \
            --fasta-files "$MUTATED_FASTA" \
            --output-dir "$PROTEIN_EMBED_DIR" \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --run-mutation \
            --csv-file "$CSV_FILE"
            
        echo "        ✅ Mutated sequences processed. If no error above, mutation logic is safe."
    else
        echo "[1/3-A] Mutation disabled, skipping priority step."
    fi

    # [Step 1.2] 处理常规大数据集
    echo "[1/3-B] 📁 Processing Standard Datasets (Main Workload)..."
    
    # 检查是否需要运行 (简单的跳过逻辑：如果第一个文件的第一个ID已经有embedding了，可能想跳过? 
    # 这里保留你原本的逻辑，但通常建议依赖 Python 内部的 check，或者覆盖运行)
    
    python "$STEP1_SCRIPT" \
        --model-name "$MODEL_NAME" \
        --fasta-files "${FASTA_FILES[@]}" \
        --output-dir "$PROTEIN_EMBED_DIR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        # 注意：这里不加 --run-mutation，因为上面已经生成并跑过了
    
    echo "[1/3] ✅ All Embeddings for '${MODEL_NAME}' generated."


    # --- Step 2: 计算EC平均嵌入 ---
    echo "[2/3] Calculating EC average embeddings..."
    python "$STEP2_SCRIPT" \
        --csv-path "$CSV_FILE" \
        --embedding-dir "$PROTEIN_EMBED_DIR" \
        --output-dir "$EC_EMBED_DIR"

    # --- Step 3: 创建距离图 ---
    echo "[3/3] Creating distance maps..."
    python "$STEP3_SCRIPT" \
        --ec-embedding-dir "$EC_EMBED_DIR" \
        --output-dir "$REPORTS_DIR"
        
    echo "✅ Model '${MODEL_NAME}' processing complete!"
done

echo ""
echo "🎉 All processes complete!"