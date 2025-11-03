"""
项目配置文件
LLM 辅助: 本文件由 GitHub Copilot 辅助生成
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = ROOT_DIR / "dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_DATA_DIR = ROOT_DIR / "test"

# 输出目录
OUTPUT_DIR = ROOT_DIR / "runs"
WEIGHTS_DIR = OUTPUT_DIR / "segment"

# 脚本目录
SCRIPTS_DIR = ROOT_DIR / "scripts"

# 模型配置
MODEL_CONFIGS = {
    'nano': {
        'name': 'yolov8n-seg.pt',
        'batch_size': 8,
        'img_size': 640,
        'description': '最小模型,适合8GB显存'
    },
    'small': {
        'name': 'yolov8s-seg.pt',
        'batch_size': 4,
        'img_size': 640,
        'description': '小型模型,需要适当调整batch'
    },
    'medium': {
        'name': 'yolov8m-seg.pt',
        'batch_size': 2,
        'img_size': 640,
        'description': '中型模型,需要12GB+显存'
    }
}

# 训练配置
TRAIN_CONFIG = {
    'epochs': 100,
    'patience': 20,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
}

# SAM 配置
SAM_CONFIG = {
    'model_type': 'vit_b',  # vit_b, vit_l, vit_h
    'checkpoint': 'sam_vit_b_01ec64.pth',
    'points_per_side': 16,
    'pred_iou_thresh': 0.86,
    'stability_score_thresh': 0.92,
}

# 类别配置
CLASSES = {
    0: 'penguin',   # 企鹅
    1: 'seal',      # 海狮/海豹
    2: 'bird',      # 其他鸟类
}

NUM_CLASSES = len(CLASSES)

# 评估配置
EVAL_CONFIG = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.7,
    'boundary_threshold': 2.0,
}

# 日志配置
LOG_CONFIG = {
    'verbose': True,
    'save_period': 10,
    'plots': True,
}


def get_model_config(size: str = 'nano') -> dict:
    """获取模型配置"""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型大小: {size}. 可选: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[size]


def ensure_dirs():
    """确保必要的目录存在"""
    dirs = [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        WEIGHTS_DIR,
        TEST_DATA_DIR,
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("项目配置:")
    print(f"  根目录: {ROOT_DIR}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"\n模型配置:")
    for size, config in MODEL_CONFIGS.items():
        print(f"  {size}: {config['description']}")
    print(f"\n类别数量: {NUM_CLASSES}")
    print(f"类别: {CLASSES}")
