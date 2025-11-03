"""
YOLO模型训练脚本
LLM 辅助: 本文件由 GitHub Copilot 辅助生成
适配 RTX 4060 8GB 显存
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse


def train_yolo_segmentation(
    data_yaml: str = "data/processed/antarctic.yaml",
    model_size: str = "n",  # n=nano, s=small (适合8GB显存)
    epochs: int = 100,
    batch_size: int = 8,  # 适配8GB显存
    img_size: int = 640,
    device: str = "0",
    pretrained: bool = True,
    resume: bool = False,
    project: str = "runs/segment",
    name: str = "antarctic_yolo",
):
    """
    训练YOLO语义分割模型
    
    Args:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 (n/s/m/l/x), n最小适合8GB显存
        epochs: 训练轮数
        batch_size: 批次大小 (8适合8GB显存)
        img_size: 输入图像尺寸
        device: GPU设备ID
        pretrained: 是否使用预训练权重
        resume: 是否从断点继续训练
        project: 项目保存路径
        name: 实验名称
    """
    
    print("=" * 60)
    print("南极动物语义分割 - YOLO训练")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ 未检测到GPU，将使用CPU训练（速度较慢）")
        device = "cpu"
    
    # 加载模型
    if pretrained:
        model_name = f"yolov8{model_size}-seg.pt"
        print(f"\n加载预训练模型: {model_name}")
    else:
        model_name = f"yolov8{model_size}-seg.yaml"
        print(f"\n从头训练模型: {model_name}")
    
    model = YOLO(model_name)
    
    # 训练配置
    print("\n训练配置:")
    print(f"  - 数据集: {data_yaml}")
    print(f"  - 模型大小: YOLOv8{model_size}-seg")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {img_size}")
    print(f"  - Device: {device}")
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        
        # 优化设置（适配8GB显存）
        patience=20,  # 早停耐心值
        save=True,  # 保存检查点
        save_period=10,  # 每10个epoch保存一次
        
        # 数据增强
        augment=True,
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,    # 饱和度增强
        hsv_v=0.4,    # 亮度增强
        degrees=10.0,  # 旋转角度
        translate=0.1, # 平移
        scale=0.5,     # 缩放
        shear=0.0,     # 剪切
        perspective=0.0,  # 透视
        flipud=0.0,    # 上下翻转
        fliplr=0.5,    # 左右翻转
        mosaic=1.0,    # mosaic增强
        mixup=0.0,     # mixup增强
        
        # 优化器设置
        optimizer='AdamW',  # 优化器
        lr0=0.001,      # 初始学习率
        lrf=0.01,       # 最终学习率因子
        momentum=0.937, # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 预热轮数
        warmup_momentum=0.8,  # 预热动量
        
        # 其他
        verbose=True,   # 详细输出
        seed=0,         # 随机种子
        deterministic=True,  # 确定性训练
        
        # 验证
        val=True,       # 训练时验证
        plots=True,     # 生成训练图表
        
        # 性能优化
        amp=True,       # 自动混合精度
        cache=False,    # 不缓存图像（节省内存）
        workers=4,      # 数据加载线程数
        
        # 恢复训练
        resume=resume,  # 是否从上次中断处继续
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    
    # 打印最佳结果
    print(f"\n最佳模型保存位置: {model.trainer.best}")
    print(f"最后模型保存位置: {model.trainer.last}")
    
    return results


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="训练YOLO语义分割模型")
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/antarctic.yaml",
        help="数据集YAML配置文件路径"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型大小: n(nano), s(small), m(medium), l(large), x(xlarge)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="批次大小（8适合8GB显存）"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="输入图像尺寸"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备 (0表示GPU 0, cpu表示CPU)"
    )
    
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="不使用预训练权重"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续训练"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/segment",
        help="项目保存路径"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="antarctic_yolo",
        help="实验名称"
    )
    
    args = parser.parse_args()
    
    # 训练模型
    train_yolo_segmentation(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        pretrained=not args.no_pretrain,
        resume=args.resume,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
