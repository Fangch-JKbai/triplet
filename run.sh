#!/bin/bash
set -e  # Exit immediately if a command fails

echo "🚀 Starting Embedding Generation Pipeline (Unified Model Loading)..."

# ============== 配置区（统一管理）==============

# 模型列表（与 EnzSubModelForDownstream 配置一致）
# 可选: ESM_2, ESM_CPT, ESM_SUB, ESM_CPT_SUB
MODEL_ORDER=("ESM_CPT" "ESM_SUB" "ESM_CPT_SUB")

# 输入文件
FASTA_FILES=("./data/split100.fasta" "./data/price.fasta" "./data/new.fasta")
CSV_FILE="./data/split100.csv"

# 超参数
BATCH_SIZE=64
DEVICE="cuda:9"

# 数据增强开关
RUN_MUTATION="true"  # "true" 或 "false"

# Python脚本路径
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

    # --- Step 1: 生成蛋白质嵌入 ---
    if [ ! -d "$PROTEIN_EMBED_DIR" ] || [ -z "$(ls -A "$PROTEIN_EMBED_DIR" 2>/dev/null)" ]; then
        echo "[1/3] 🟡 Embeddings for '${MODEL_NAME}' not found or empty. Generating..."
        
        STEP1_ARGS=(
            "--model-name" "$MODEL_NAME"
            "--fasta-files" "${FASTA_FILES[@]}"
            "--output-dir" "$PROTEIN_EMBED_DIR"
            "--batch-size" "$BATCH_SIZE"
            "--device" "$DEVICE"
        )
        
        # 添加数据增强参数
        if [ "${RUN_MUTATION}" == "true" ]; then
            echo "   -> Data augmentation enabled"
            STEP1_ARGS+=("--run-mutation" "--csv-file" "$CSV_FILE")
        fi
        
        python "$STEP1_SCRIPT" "${STEP1_ARGS[@]}"
        echo "[1/3] ✅ Embeddings for '${MODEL_NAME}' generated."
    else
        echo "[1/3] 🟢 Found existing embeddings for '${MODEL_NAME}'. Skipping generation."
    fi

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
echo "🎉 All processes complete! Results for all models are in: ./${EXPERIMENT_ROOT}/"
echo ""
echo "模型结果目录:"
for MODEL_NAME in "${MODEL_ORDER[@]}"; do
    echo "   - ${EXPERIMENT_ROOT}/${MODEL_NAME}/"
done