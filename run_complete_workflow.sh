#!/bin/bash

# =============================================================================
# 蛋白质EC分类完整训练工作流
# =============================================================================

set -e  # 遇到错误立即停止

echo "========================================"
echo "EC分类完整训练流程"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# =============================================================================
# 第一步：五折交叉验证
# =============================================================================
echo -e "${BLUE}第一步：五折交叉验证（带早停）${NC}"
echo "这个过程会："
echo "  1. 将数据分成5折"
echo "  2. 每折独立训练并使用早停"
echo "  3. 在price和new测试集上评估"
echo "  4. 计算平均结果"
echo ""
read -p "是否开始五折交叉验证？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${RED}用户取消操作${NC}"
    exit 1
fi

echo -e "${GREEN}开始运行 main_kfold.py...${NC}"
cd src
python main_kfold.py

if [ $? -ne 0 ]; then
    echo -e "${RED}五折交叉验证失败！${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 五折交叉验证完成！${NC}"
echo ""

# =============================================================================
# 分析结果并确定合理的epoch数
# =============================================================================
echo -e "${YELLOW}正在分析交叉验证结果...${NC}"

# 查找最新的实验目录
LATEST_EXP=$(ls -td ../experiments/*KFold* 2>/dev/null | head -1)

if [ -z "$LATEST_EXP" ]; then
    echo -e "${RED}找不到交叉验证结果目录${NC}"
    exit 1
fi

echo "实验目录: $LATEST_EXP"
echo ""

# 提取每折的训练epoch数
echo "各折训练的epoch数："
grep -h "Total epochs trained:" "$LATEST_EXP"/fold_*/fold.log 2>/dev/null || echo "无法提取epoch信息"

# 查看聚合结果
if [ -f "$LATEST_EXP/final_cross_validation_summary.json" ]; then
    echo ""
    echo "五折交叉验证聚合结果："
    cat "$LATEST_EXP/final_cross_validation_summary.json"
fi

echo ""
echo -e "${YELLOW}请查看上述结果，确定合理的epoch数${NC}"
echo ""
read -p "请输入要用于最终训练的epoch数（例如：50）: " FINAL_EPOCHS

if [ -z "$FINAL_EPOCHS" ]; then
    echo -e "${RED}未输入epoch数，终止流程${NC}"
    exit 1
fi

# 验证输入是否为数字
if ! [[ "$FINAL_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}输入的epoch数无效！${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}将使用 $FINAL_EPOCHS 个epoch进行最终训练${NC}"

# =============================================================================
# 第二步：修改配置文件
# =============================================================================
echo ""
echo -e "${BLUE}第二步：准备最终训练配置${NC}"
echo "即将修改 config.py:"
echo "  - EXPERIMENT_MODE = 'final_training'"
echo "  - epochs = $FINAL_EPOCHS"
echo ""
read -p "是否继续？[y/N] " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${YELLOW}已跳过最终训练。你可以稍后手动修改config.py并运行main.py${NC}"
    exit 0
fi

# 备份原始配置
cp config.py config.py.backup
echo -e "${GREEN}已备份原始配置到 config.py.backup${NC}"

# 修改配置（使用sed）
sed -i "s/EXPERIMENT_MODE = .*/EXPERIMENT_MODE = \"final_training\"/" config.py
sed -i "s/\"epochs\": [0-9]*,/\"epochs\": $FINAL_EPOCHS,/" config.py

echo -e "${GREEN}✓ 配置文件已更新${NC}"

# =============================================================================
# 第三步：全数据最终训练
# =============================================================================
echo ""
echo -e "${BLUE}第三步：使用全部数据进行最终训练（无早停）${NC}"
echo "训练参数："
echo "  - Epoch数: $FINAL_EPOCHS"
echo "  - 使用全部训练数据"
echo "  - 无验证集，无早停"
echo "  - 将在price和new测试集上评估"
echo ""
read -p "是否开始最终训练？[y/N] " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${YELLOW}已跳过最终训练${NC}"
    echo "配置文件已修改，你可以稍后手动运行: python src/main.py"
    exit 0
fi

echo -e "${GREEN}开始运行 main.py...${NC}"
python main.py

if [ $? -ne 0 ]; then
    echo -e "${RED}最终训练失败！${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 最终训练完成！${NC}"
echo ""

# =============================================================================
# 显示最终结果
# =============================================================================
FINAL_EXP=$(ls -td ../experiments/* 2>/dev/null | head -1)

if [ -f "$FINAL_EXP/final_evaluation_results.json" ]; then
    echo "========================================"
    echo "最终测试结果"
    echo "========================================"
    cat "$FINAL_EXP/final_evaluation_results.json"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 完整训练流程已完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果位置："
echo "  - 交叉验证: $LATEST_EXP"
echo "  - 最终训练: $FINAL_EXP"
echo ""
echo "主要文件："
echo "  - 最终模型: $FINAL_EXP/best_model.pth"
echo "  - 测试结果: $FINAL_EXP/final_evaluation_results.json"
echo "  - Price预测: $FINAL_EXP/predictions_price.csv"
echo "  - New预测: $FINAL_EXP/predictions_new.csv"
