import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

def setup_logging(log_path: str):
    """配置日志记录，同时输出到文件和控制台"""
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 如果已经有处理器了，就先清空，避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建一个流处理器，用于在控制台输出
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def plot_losses(history: Dict[str, List[float]], save_path: str):
    """绘制训练和验证损失曲线并保存为图片"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Loss curve plot saved to {save_path}")

def save_evaluation_results(results: Dict, save_path: str):
    """将最终的评估结果保存为易于阅读和解析的JSON文件"""
    # 我们可以加入一些元数据，让结果文件信息更丰富
    output_data = {
        "metadata": {
            "description": "Final evaluation results for the experiment.",
            "note": "Metrics are macro-averaged over all samples in the test set."
        },
        "results": results
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"Evaluation results saved to {save_path}")

def save_detailed_predictions(records: List[Dict], save_path: str):
    """将详细的逐条预测结果保存为CSV文件，便于分析。"""
    if not records:
        logging.warning("No prediction records to save.")
        return
        
    df = pd.DataFrame(records)
    
    # 将列表格式的标签转换为用分号分隔的字符串，更适合CSV查看
    df['True_Labels'] = df['True_Labels'].apply(lambda x: ';'.join(sorted(x)))
    df['Predicted_Labels'] = df['Predicted_Labels'].apply(lambda x: ';'.join(sorted(x)))
    
    df.to_csv(save_path, index=False, encoding='utf-8')
    logging.info(f"Detailed prediction results saved to {save_path}")